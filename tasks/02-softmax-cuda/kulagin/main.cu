#include <cuda_runtime.h>
#include <bits/stdc++.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
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

inline float rand_range(float a, float b) {
  return ((b - a) * ((float)rand() / (float)RAND_MAX)) + a;
}

void calc_row(const float *row, float *row_res, const std::size_t n) {
  float d = 0.0f;
  for (std::size_t i = 0; i < n; i++) {
    d += std::exp(row[i]);
  }
  d = 1.0f / d;
  for (std::size_t i = 0; i < n; i++) {
    row_res[i] = std::exp(row[i]) * d;
  }
}

std::vector<float> make_matrix(std::size_t n) {
  srand(time(0));
  constexpr float lower = -10.0f, upper = 10.0f;  // Arbitrary numbers
  const std::size_t res_sz = n * n;
  std::vector<float> res(res_sz);
  for (std::size_t i = 0; i < res_sz; i++) {
    res[i] = rand_range(lower, upper);
  }
  return res;
}

std::vector<float> run_sequential(const std::vector<float> &matrix,
                                  std::size_t n) {
  if (matrix.size() != n * n) {
    throw std::runtime_error("Matrix size is not n * n");
  }
  std::vector<float> res(n * n);
  const float *matrix_ptr = matrix.data();
  float *res_ptr = res.data();
  for (std::size_t i = 0; i < n; i++) {
    calc_row(matrix_ptr + i * n, res_ptr + i * n, n);
  }
  return res;
}

__global__ void softmax_kernel(const float *d_input, float *d_output, std::size_t n) {
  __shared__ float s_d[1];
  if (threadIdx.x == 0) {
    *s_d = 0.0f;
  }
  __syncthreads();
  const int d_offset = blockIdx.x * n;
  const float *row = d_input + d_offset;
  float *res = d_output + d_offset;
  float d = 0.0f;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    d += __expf(row[i]);
  }
  atomicAdd(s_d, d);
  __syncthreads();
  float d_inv = 1.0f / *s_d;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    res[i] = __expf(row[i]) * d_inv;
  }
}

std::vector<float> run_cuda_simt(const std::vector<float> &matrix,
                                 std::size_t n) {
  if (n == 0) {
    throw std::runtime_error("n == 0");
  }
  const std::size_t block_size = 512;
  std::size_t sz_b = n * n * sizeof(*matrix.data());
  std::vector<float> res(n * n);
  float *d_m;
  float *d_res;
  CHECK_CUDA_ERROR(cudaMalloc(&d_m, sz_b));
  CHECK_CUDA_ERROR(cudaMalloc(&d_res, sz_b));
  CHECK_CUDA_ERROR(cudaMemcpy(d_m, matrix.data(), sz_b, cudaMemcpyHostToDevice));
  // #define LAB_CUDA_TIMING
#ifdef LAB_CUDA_TIMING
  float elapsedTime = -1.0f;
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));
#endif
  softmax_kernel<<<n, block_size>>>(d_m, d_res, n);
#ifdef LAB_CUDA_TIMING
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "CUDA elapsed time: " << elapsedTime << " ms\n"; // time in ms
#endif
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaMemcpy(res.data(), d_res, sz_b, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(d_m));
  CHECK_CUDA_ERROR(cudaFree(d_res));
  return res;
}

void warmup_cuda(const std::vector<float> &matrix, std::size_t n) {
  constexpr std::size_t div = 2;
  std::size_t _n = n / div ? n > div : 1;
  run_cuda_simt(matrix, _n);
}

double measure_seconds(const std::function<std::vector<float>()> &work,
                       std::vector<float> &result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate) {
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

void print_report(std::string_view testName, const RunResult &result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}
}  // namespace

int main(int argc, char *argv[]) {
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
    } catch (const std::exception &ex) {
      std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
    }

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("SIMT", simt_res);

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
