#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
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
  if (n < 2) {
    throw std::runtime_error("make_matrix not implemented");
  }
  std::vector<float> matrix(n * n);

  static std::random_device ran_dev;
  static std::mt19937 ran_eng(ran_dev());
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::generate(matrix.begin(), matrix.end(), [&]() { return dist(ran_eng); });

  return matrix;
}

static void row_calculation_amount(const float* input_row, std::size_t n,
                                   float* output_row) {
  float row_sum = 0.0f;
  for (std::size_t item = 0; item < n; item++) {
    row_sum += exp(input_row[item]);
  }
  for (std::size_t item = 0; item < n; item++) {
    output_row[item] = exp(input_row[item]) / row_sum;
  }
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("Sequential method not implemented");
  }
  std::vector<float> res_matrix(n * n);

  for (std::size_t row = 0; row < n; row++) {
    row_calculation_amount(&matrix[row * n], n, &res_matrix[row * n]);
  }
  return res_matrix;
}

__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  __shared__ float s_data[1];
  if (threadIdx.x == 0) {
    *s_data = 0.0f;
  }
  __syncthreads();

  const int d_offset = blockIdx.x * n;
  const float* row = input + d_offset;
  float* res = output + d_offset;

  float sum_row = 0.0f;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    res[i] = __expf(row[i]);
    sum_row += res[i];
  }
  atomicAdd(s_data, sum_row);
  __syncthreads();

  float divider = 1.0f / *s_data;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    res[i] = res[i] * divider;
  }
}

void launch_softmax_kernel(const float* d_input, float* d_output, std::size_t n,
                           cudaStream_t stream) {
  if (n < 2 || d_input == nullptr || d_output == nullptr) {
    throw std::runtime_error("Invalid parameters for kernel launch");
  }

  const int threads_per_block = 512;
  const int shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<n, threads_per_block, shared_mem_size>>>(d_input, d_output,
                                                            n);
}

void warmup_cuda(const std::vector<float>& matrix, std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("CUDA warm-up not implemented");
  }
  float* d_tmp = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, sizeof(float)));
  CHECK_CUDA_ERROR(cudaFree(d_tmp));
}

std::vector<float> run_cuda_simt(const std::vector<float>& matrix,
                                 std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("CUDA SIMT method not implemented");
  }

  const std::size_t size = n * n * sizeof(float);
  float* d_input = nullptr;
  float* d_output = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_input, matrix.data(), size, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
  launch_softmax_kernel(d_input, d_output, n, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<float> result(n * n);
  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_output, size, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));

  return result;
}

double measure_seconds(std::vector<float> (*work)(),
                       std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

std::vector<float> sequential_work_wrapper(const std::vector<float>* input,
                                           std::size_t n) {
  return run_sequential(*input, n);
}

std::vector<float> cuda_work_wrapper(const std::vector<float>* input,
                                     std::size_t n) {
  return run_cuda_simt(*input, n);
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

void print_report(const std::string& testName, const RunResult& result) {
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

    // Time sequential version
    const auto sequential_start = std::chrono::high_resolution_clock::now();
    sequential_result = run_sequential(input, n);
    const auto sequential_stop = std::chrono::high_resolution_clock::now();
    const double sequential_seconds =
        std::chrono::duration<double>(sequential_stop - sequential_start)
            .count();

    RunResult simt_res;
    try {
      warmup_cuda(input, n);

      // Time CUDA version
      const auto cuda_start = std::chrono::high_resolution_clock::now();
      simt_res.result = run_cuda_simt(input, n);
      const auto cuda_stop = std::chrono::high_resolution_clock::now();
      simt_res.seconds =
          std::chrono::duration<double>(cuda_stop - cuda_start).count();

      simt_res.diff = max_abs_diff(sequential_result, simt_res.result);
      simt_res.success = true;
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