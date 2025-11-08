#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// TODO: Move to utils
#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(codeError);                                               \
    }                                                                     \
  }

namespace {
// void print_device_info() {
//   int device = 0;
//   CHECK_CUDA_ERROR(cudaGetDevice(&device));

//   cudaDeviceProp prop;
//   CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

//   std::cout << "Device: " << prop.name << '\n';
//   std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << '\n';
//   std::cout << "Shared memory per block: " << prop.sharedMemPerBlock
//             << " bytes\n";
// }

std::vector<float> make_matrix(std::size_t n) {
  std::mt19937 gen{36};
  std::uniform_real_distribution<float> dist(-15.0f, 15.0f);

  std::vector<float> matrix(n * n);
  for (std::size_t i = 0; i < n * n; ++i) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

void softmax_row(const float* input_row, float* output_row, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t j = 0; j < n; ++j) {
    float exp_val = std::exp(input_row[j]);
    output_row[j] = exp_val;
    sum += exp_val;
  }
  const float inv_sum = 1.0f / sum;
  for (std::size_t j = 0; j < n; ++j) {
    output_row[j] *= inv_sum;
  }
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> result(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    softmax_row(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  extern __shared__ float sdata[];

  const size_t row = blockIdx.x;
  const size_t tid = threadIdx.x;
  const size_t block_size = blockDim.x;

  const float* in_row = input + row * n;
  float* out_row = output + row * n;

  // 1. expf + запись + сумма
  float local_sum = 0.0f;
  for (size_t j = tid; j < n; j += block_size) {
    float val = expf(in_row[j]);
    out_row[j] = val;
    local_sum += val;
  }

  sdata[tid] = local_sum;
  __syncthreads();

  // 2. Редукция (оптимизированная для block_size = степень двойки)
  for (size_t s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  float total_sum = sdata[0];
  float inv_sum = 1.0f / total_sum;
  __syncthreads();

  // 3. Нормализация
  for (size_t j = tid; j < n; j += block_size) {
    out_row[j] *= inv_sum;
  }
}

void launch_softmax_kernel(const float* d_input, float* d_output, std::size_t n,
                           cudaStream_t stream) {
  int block_size = 1024;  // максимальный размер
  if (n < 32)
    block_size = 32;
  else if (n < 64)
    block_size = 64;
  else if (n < 128)
    block_size = 128;
  else if (n < 256)
    block_size = 256;
  else if (n < 512)
    block_size = 512;
  // иначе 1024

  size_t shared_mem = block_size * sizeof(float);
  softmax_kernel<<<n, block_size, shared_mem, stream>>>(d_input, d_output, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

void warmup_cuda(const std::vector<float>& /*matrix*/, std::size_t /*n*/) {
  float* d_tmp = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, sizeof(float)));
  CHECK_CUDA_ERROR(cudaFree(d_tmp));
}

std::vector<float> run_cuda_simt(const std::vector<float>& matrix,
                                 std::size_t n) {
  const size_t size_bytes = n * n * sizeof(float);
  float *d_in = nullptr, *d_out = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&d_in, size_bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_out, size_bytes));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_in, matrix.data(), size_bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
  launch_softmax_kernel(d_in, d_out, n, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<float> result(n * n);
  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_out, size_bytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  CHECK_CUDA_ERROR(cudaFree(d_in));
  CHECK_CUDA_ERROR(cudaFree(d_out));

  return result;
}

double measure_seconds(const std::function<std::vector<float>()>& work,
                       std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

float max_abs_diff(const std::vector<float>& baseline,
                   const std::vector<float>& candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error(
        "Result size mismatch while validating correctness");
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - candidate[i]));
  }
  return max_diff;
}

// TODO: Create basic utils file
struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << seconds;
  return oss.str();
}

std::string format_diff(float diff) {
  std::ostringstream oss;
  oss << std::defaultfloat << std::setprecision(1) << diff;
  return oss.str();
}

void print_report(std::string_view testName, const RunResult& result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  try {
    // print_device_info();

    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    const auto input = make_matrix(n);
    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    RunResult simt_res;
    try {
      warmup_cuda(input, n);
      simt_res.seconds = measure_seconds(
          [&]() { return run_cuda_simt(input, n); }, simt_res.result);
      simt_res.diff = max_abs_diff(sequential_result, simt_res.result);
      simt_res.success = true;
      // TODO: Compare simt_seconds with the OpenMP+AVX2 timing from practice
      // #1.
    } catch (const std::exception& ex) {
      std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
    }

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("SIMT", simt_res);

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
