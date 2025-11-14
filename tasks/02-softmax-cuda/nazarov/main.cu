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

// void PrintMatrix(const std::vector<float>& matrix, std::size_t n) {
//   if (matrix.size() != n * n) {
//     std::cerr << "Error: matrix size does not match dimension n×n\n";
//     return;
//   }

//   for (std::size_t i = 0; i < n; ++i) {
//     for (std::size_t j = 0; j < n; ++j) {
//       std::cout << std::setw(8) << std::fixed << std::setprecision(3)
//                 << matrix[i * n + j] << " ";
//     }
//     std::cout << "\n";
//   }
// }

std::vector<float> make_matrix(std::size_t n) {
  std::vector<float> matrix(n * n);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto& x : matrix) {
    x = dist(gen);
  }

  return matrix;
}

void SoftmaxRow(const float* row_begin, float* row_result, std::size_t n) {
  float sum_exp = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    sum_exp += std::exp(row_begin[j]);
  }
  float div_sum_exp = 1 / sum_exp;
  for (std::size_t j = 0; j < n; ++j) {
    row_result[j] = std::exp(row_begin[j]) * div_sum_exp;
  }
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> result(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRow(&matrix[i * n], &result[i * n], n);
  }
  // PrintMatrix(result, n);
  // std::cout << " --------------------------" << std::endl;
  return result;
}

// void launch_softmax_kernel(const float* d_input, float* d_output, std::size_t
// n,
//                            cudaStream_t stream) {
//   throw std::runtime_error("CUDA kernel launch not implemented");
// }

void warmup_cuda(const std::vector<float>& matrix, std::size_t n) {
  float* d_test;
  CHECK_CUDA_ERROR(cudaMalloc(&d_test, n * n * sizeof(float)));
  CHECK_CUDA_ERROR(cudaFree(d_test));
}

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output, std::size_t n) {
  extern __shared__ float sdata[];
  const std::size_t row = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;

  const float* row_input = &input[row * n];
  float* row_output = &output[row * n];

  float local_sum = 0.0f;
  for (std::size_t i = tid; i < n; i += block_size) {
    float ex = expf(row_input[i]);
    local_sum += ex;
    row_output[i] = ex;
  }

  sdata[tid] = local_sum;
  __syncthreads();

  for (unsigned stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  float row_sum = sdata[0];
  __syncthreads();

  for (std::size_t i = tid; i < n; i += block_size) {
    row_output[i] /= row_sum;
  }
}

std::vector<float> run_cuda_simt(const std::vector<float>& matrix,
                                 std::size_t n) {
  std::vector<float> result(n * n);
  size_t size_b = n * n * sizeof(float);
  float* d_in;
  float* d_out;
  CHECK_CUDA_ERROR(cudaMalloc(&d_in, size_b));
  CHECK_CUDA_ERROR(cudaMalloc(&d_out, size_b));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_in, matrix.data(), size_b, cudaMemcpyHostToDevice));

  // const int threads_per_block =
  //     static_cast<int>(std::min<std::size_t>(1024, n));
  const int threads_per_block = 512;
  const int blocks = static_cast<int>(n);
  const std::size_t shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_in, d_out,
                                                                 n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_out, size_b, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_in));
  CHECK_CUDA_ERROR(cudaFree(d_out));
  // PrintMatrix(result, n);
  // std::cout << " --------------------------" << std::endl;
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
    // std::cout << i << " - " << std::abs(baseline[i] - candidate[i]) <<
    // std::endl;
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
