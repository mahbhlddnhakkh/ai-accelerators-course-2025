#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <omp.h>

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

#include "cutlass/gemm/device/gemm.h"

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
std::vector<__half> make_input_matrix(std::size_t n) {
  if (n < 2) {
    throw std::runtime_error("make_input_matrix not implemented");
  }
  std::vector<__half> matrix(n * n * 2);

  static std::random_device ran_dev;
  static std::mt19937 ran_eng(ran_dev());
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::generate(matrix.begin(), matrix.end(),
                [&]() { return __float2half(dist(ran_eng)); });

  return matrix;
}

static void row_calculation_amount(const float* input_row, std::size_t n,
                                   float* output_row) {
  float row_sum = 0.0f;

#pragma omp simd reduction(+ : row_sum)
  for (std::size_t item = 0; item < n; item++) {
    output_row[item] = expf(input_row[item]);
    row_sum += output_row[item];
  }

  const float divider = 1.0f / row_sum;

#pragma omp simd
  for (std::size_t item = 0; item < n; item++) {
    output_row[item] = output_row[item] * divider;
  }
}

std::vector<float> matrixSquare(const __half* __restrict a_input,
                                const __half* __restrict b_input,
                                std::size_t n) {
  std::vector<float> result(n * n, 0.0f);

  constexpr std::size_t BLOCK_SIZE = 128;

#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
    for (std::size_t j0 = 0; j0 < n; j0 += BLOCK_SIZE) {
      for (std::size_t k0 = 0; k0 < n; k0 += BLOCK_SIZE) {
        std::size_t i_end = std::min(i0 + BLOCK_SIZE, n);
        std::size_t j_end = std::min(j0 + BLOCK_SIZE, n);
        std::size_t k_end = std::min(k0 + BLOCK_SIZE, n);

        for (std::size_t i = i0; i < i_end; ++i) {
          for (std::size_t k = k0; k < k_end; ++k) {
            float a_val = static_cast<float>(a_input[i * n + k]);

#pragma omp simd
            for (std::size_t j = j0; j < j_end; ++j) {
              result[i * n + j] +=
                  a_val * static_cast<float>(b_input[k * n + j]);
            }
          }
        }
      }
    }
  }

  return result;
}
std::vector<float> run_openmp_reference(const std::vector<__half>& matrix,
                                        std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("OpenMP reference not implemented");
  }

  const __half* first_matrix = matrix.data();
  const __half* second_matrix = matrix.data() + n * n;

  std::vector<float> res_matrix = matrixSquare(first_matrix, second_matrix, n);

#pragma omp parallel for
  for (int row = 0; row < n; row++) {
    row_calculation_amount(&res_matrix[row * n], n, &res_matrix[row * n]);
  }

  return res_matrix;
}

__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  extern __shared__ float s_data[];

  const int tid = threadIdx.x;
  const int d_offset = blockIdx.x * n;
  const float* row = input + d_offset;
  float* res = output + d_offset;

  float sum_row = 0.0f;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    res[i] = __expf(row[i]);
    sum_row += res[i];
  }
  s_data[tid] = sum_row;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_data[tid] += s_data[tid + stride];
    }
    __syncthreads();
  }

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_data[threadIdx.x] += s_data[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float divider = 1.0f / *s_data;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    res[i] = res[i] * divider;
  }
}

#define WARP_SIZE 32
#define WMMA_SIZE 16
using namespace nvcuda;
__global__ void GEMMv4(const half* a, const half* b, float* c, std::size_t n) {
  int warp_i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warp_j = (blockIdx.y * blockDim.y + threadIdx.y);

  wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float>
      acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  for (int k = 0; k < n; k += WMMA_SIZE) {
    int a_row = warp_i * WMMA_SIZE;
    int a_col = k;
    int b_row = k;
    int b_col = warp_j * WMMA_SIZE;

    wmma::load_matrix_sync(a_frag, a + a_row * n + a_col, n);
    wmma::load_matrix_sync(b_frag, b + b_row * n + b_col, n);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  int c_row = warp_i * WMMA_SIZE;
  int c_col = warp_j * WMMA_SIZE;

  wmma::store_matrix_sync(c + c_row * n + c_col, acc_frag, n,
                          wmma::mem_row_major);
}
}  // namespace

std::vector<float> run_wmma(const std::vector<__half>& matrix, std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("WMMA method not implemented");
  }

  const std::size_t size = n * n;
  std::vector<float> result(size);

  half *a_input = nullptr, *b_input = nullptr;
  float* c_tmp = nullptr;
  float* d_output = nullptr;

  size_t bytes_half = size * sizeof(half);
  size_t bytes_float = size * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&a_input, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&b_input, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&c_tmp, bytes_float));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes_float));

  CHECK_CUDA_ERROR(
      cudaMemcpy(a_input, matrix.data(), bytes_half, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_input, matrix.data() + size, bytes_half,
                              cudaMemcpyHostToDevice));

  dim3 block_size(4 * WARP_SIZE, 4);
  dim3 block_count((n / block_size.x) * (WARP_SIZE / WMMA_SIZE),
                   (n / block_size.y) / WMMA_SIZE);
  GEMMv4<<<block_count, block_size>>>(a_input, b_input, c_tmp, n);

  const int threads_per_block = 512;
  const int shared_mem_size = threads_per_block * sizeof(float);
  softmax_kernel<<<n, threads_per_block, shared_mem_size>>>(c_tmp, d_output, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_output, bytes_float, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(a_input));
  CHECK_CUDA_ERROR(cudaFree(b_input));
  CHECK_CUDA_ERROR(cudaFree(c_tmp));
  CHECK_CUDA_ERROR(cudaFree(d_output));

  return result;
}

void warmup_wmma(const std::vector<__half>& matrix, std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("WMMA warm-up not implemented");
  }
  run_wmma(matrix, n);
}

cudaError_t CutlassGEMM(const cutlass::half_t* a, const cutlass::half_t* b,
                        float* c, int n) {
  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t,  // A
                                  RowMajor,         // Layout A
                                  cutlass::half_t,  // B
                                  RowMajor,         // Layout B
                                  float,            // C
                                  RowMajor,         // Layout C
                                  float,            // Accumulator
                                  cutlass::arch::OpClassTensorOp,  // Tensor
                                                                   // Cores
                                  cutlass::arch::Sm75              // Tesla T4
                                  >;

  CutlassGemm gemm_operator;

  typename CutlassGemm::Arguments args({n, n, n},    // problem size (M, N, K)
                                       {a, n},       // tensor A
                                       {b, n},       // tensor B
                                       {c, n},       // tensor C
                                       {c, n},       // tensor D
                                       {1.0f, 0.0f}  // alpha, beta
  );

  cutlass::Status status = gemm_operator.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  status = gemm_operator();
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

std::vector<float> run_cutlass(const std::vector<__half>& matrix,
                               std::size_t n) {
  if (n < 2 || matrix.empty()) {
    throw std::runtime_error("Invalid input for CUTLASS method");
  }

  const std::size_t size = n * n;
  std::vector<float> result(size);

  cutlass::half_t *a_input = nullptr, *b_input = nullptr;
  float* c_tmp = nullptr;
  float* d_output = nullptr;

  size_t bytes_half = size * sizeof(cutlass::half_t);
  size_t bytes_float = size * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&a_input, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&b_input, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&c_tmp, bytes_float));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes_float));

  CHECK_CUDA_ERROR(
      cudaMemcpy(a_input, matrix.data(), bytes_half, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_input, matrix.data() + size, bytes_half,
                              cudaMemcpyHostToDevice));

  cudaError_t status = CutlassGEMM(a_input, b_input, c_tmp, n);
  CHECK_CUDA_ERROR(status);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  const int threads_per_block = 512;
  const int shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<n, threads_per_block, shared_mem_size>>>(c_tmp, d_output, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_output, bytes_float, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(a_input));
  CHECK_CUDA_ERROR(cudaFree(b_input));
  CHECK_CUDA_ERROR(cudaFree(c_tmp));
  CHECK_CUDA_ERROR(cudaFree(d_output));

  return result;
}

void warmup_cutlass(const std::vector<__half>& matrix, std::size_t n) {
  if (n < 2 || matrix.empty()) {
    return;  // Просто пропускаем warmup
  }
  run_cutlass(matrix, n);
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

    const auto input = make_input_matrix(n);
    std::vector<float> openmp_result;
    const double openmp_seconds = measure_seconds(
        [&]() { return run_openmp_reference(input, n); }, openmp_result);

    RunResult wmma_res;
    try {
      warmup_wmma(input, n);
      wmma_res.seconds = measure_seconds([&]() { return run_wmma(input, n); },
                                         wmma_res.result);
      wmma_res.diff = max_abs_diff(openmp_result, wmma_res.result);
      wmma_res.success = true;
    } catch (const std::exception& ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    RunResult cutlass_res;
    try {
      warmup_cutlass(input, n);
      cutlass_res.seconds = measure_seconds(
          [&]() { return run_cutlass(input, n); }, cutlass_res.result);
      cutlass_res.diff = max_abs_diff(openmp_result, cutlass_res.result);
      cutlass_res.success = true;
    } catch (const std::exception& ex) {
      std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
    }

    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec\n";
    print_report("WMMA", wmma_res);
    print_report("CUTLASS", cutlass_res);

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
