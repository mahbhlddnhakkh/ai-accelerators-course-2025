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
std::vector<float> make_matrix(std::size_t n) {
  static std::mt19937 gen_eng{42};

  std::normal_distribution<float> dist{};
  auto gen = [&]() { return dist(gen_eng); };

  std::vector<float> matrix(n * n);

  std::generate(matrix.begin(), matrix.end(), gen);

  return matrix;
}

static inline void process_row(const float* row, std::size_t n, float* out) {
  float rowsum = 0.0f;
  for (std::size_t j = 0; j < n; j++) {
    rowsum += std::exp(row[j]);
  }
  const float inv_rowsum = 1.0f / rowsum;
  for (std::size_t j = 0; j < n; j++) {
    out[j] = std::exp(row[j]) * inv_rowsum;
  }
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> res(n * n);

  for (std::size_t i = 0; i < n; i++) {
    process_row(&matrix[i * n], n, &res[i * n]);
  }

  return res;
}

__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  unsigned int data_row = blockIdx.x * n;
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;

  extern __shared__ float sdata[];
  const float* row = &input[data_row];
  float* out = &output[data_row];

  float tid_sum = 0.0f;
  for (unsigned int j = tid; j < n; j += block_size) {
    tid_sum += __expf(row[j]);
  }
  sdata[tid] = tid_sum;
  __syncthreads();

  for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  float inv_rowsum = 1.0f / sdata[0];

  for (unsigned int j = tid; j < n; j += block_size) {
    out[j] = __expf(row[j]) * inv_rowsum;
  }
}

void warmup_cuda(const std::vector<float>& matrix, std::size_t n) {
  float* d_dummy;
  CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, n * n * sizeof(float)));
  CHECK_CUDA_ERROR(cudaFree(d_dummy));
}

std::vector<float> run_cuda_simt(const std::vector<float>& matrix,
                                 std::size_t n) {
  std::vector<float> res(n * n);

  const size_t sizeInBytes = n * n * sizeof(float);

  float* d_A;
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeInBytes));
  float* d_C;
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeInBytes));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_A, matrix.data(), sizeInBytes, cudaMemcpyHostToDevice));

  // TODO: clip by deviceProp.maxThreadsPerBlock, usually 1024
  const int threads_per_block = 512;
  const int shared_mem_size = threads_per_block * sizeof(float);
  softmax_kernel<<<n, threads_per_block, shared_mem_size>>>(d_A, d_C, n);
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(
      cudaMemcpy(res.data(), d_C, sizeInBytes, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_C));

  return res;
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
