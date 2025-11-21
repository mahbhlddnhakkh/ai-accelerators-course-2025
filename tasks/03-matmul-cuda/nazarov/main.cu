#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>

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
  std::vector<__half> matrix(2 * n * n);

  std::random_device rd;
  std::mt19937 gen(42);

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : matrix) {
    x = __float2half(dist(gen));
  }
  return matrix;
}

std::vector<float> run_openmp_reference(const std::vector<__half>& matrix,
                                        std::size_t n) {
  const size_t size = n * n;
  const __half* a = matrix.data();
  const __half* b = matrix.data() + size;
  std::vector<float> result(size, 0.0f);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) {
    float* const row = &result[i * n];
    for (size_t k = 0; k < n; ++k) {
      const float a_ik = __half2float(a[i * n + k]);
#pragma omp simd
      for (size_t j = 0; j < n; ++j) {
        row[j] += a_ik * __half2float(b[k * n + j]);
      }
    }

    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (size_t j = 0; j < n; ++j) {
      sum += std::exp(row[j]);
    }

    const float inv_sum = 1.0f / sum;

#pragma omp simd
    for (size_t j = 0; j < n; ++j) {
      row[j] = std::exp(row[j]) * inv_sum;
    }
  }
  return result;
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

#include <mma.h>

using namespace nvcuda;

__global__ void wmma_gemm_kernel(const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 float* __restrict__ C, int Npad) {
  int tile_col = blockIdx.x;
  int tile_row = blockIdx.y;

  const int TILE = 16;
  int row = tile_row * TILE;
  int col = tile_col * TILE;
  //                             row = col = k             хранение по строкам
  wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < Npad; k += TILE) {
    const __half* tile_a_ptr = A + (row * Npad + k);
    const __half* tile_b_ptr = B + (k * Npad + col);

    wmma::load_matrix_sync(a_frag, tile_a_ptr, Npad);
    wmma::load_matrix_sync(b_frag, tile_b_ptr, Npad);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* tile_c_ptr = C + (row * Npad + col);
  wmma::store_matrix_sync(tile_c_ptr, c_frag, Npad, wmma::mem_row_major);
}

std::vector<float> run_wmma(const std::vector<__half>& matrix, std::size_t n) {
  if (matrix.size() != 2ull * n * n) {
    throw std::runtime_error(
        "Input matrix size must be exactly 2*n*n (A then B)");
  }

  const int TILE = 16;
  int N = static_cast<int>(n);
  // округление вверх до кратного 16
  int Npad = ((N + TILE - 1) / TILE) * TILE;

  __half* d_A = nullptr;
  __half* d_B = nullptr;
  float* d_C = nullptr;
  float* d_softmax = nullptr;

  size_t bytes_half_padded = static_cast<size_t>(Npad * Npad) * sizeof(__half);
  size_t bytes_f_padded = static_cast<size_t>(Npad * Npad) * sizeof(float);
  size_t bytes_f_N = static_cast<size_t>(N * N) * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_f_padded));
  CHECK_CUDA_ERROR(cudaMalloc(&d_softmax, bytes_f_N));

  CHECK_CUDA_ERROR(cudaMemset(d_A, 0, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMemset(d_B, 0, bytes_half_padded));

  const __half* h_A = matrix.data();
  const __half* h_B = matrix.data() + static_cast<size_t>(N * N);

  // Копирпование матрицы A(n×n) в левый верхний угол d_A(Npad×Npad)
  CHECK_CUDA_ERROR(cudaMemcpy2D(d_A, Npad * sizeof(__half), h_A,
                                N * sizeof(__half), N * sizeof(__half), N,
                                cudaMemcpyHostToDevice));
  // Копирпование матрицы B(n×n) в левый верхний угол d_B(Npad×Npad)
  CHECK_CUDA_ERROR(cudaMemcpy2D(d_B, Npad * sizeof(__half), h_B,
                                N * sizeof(__half), N * sizeof(__half), N,
                                cudaMemcpyHostToDevice));

  dim3 grid(Npad / TILE, Npad / TILE);
  dim3 block(32, 1, 1);
  wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, Npad);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaMemcpy2D(d_softmax, N * sizeof(float), d_C,
                                Npad * sizeof(float), N * sizeof(float), N,
                                cudaMemcpyDeviceToDevice));

  const int threads_per_block = 512;
  const int blocks = N;
  const std::size_t shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_softmax,
                                                                 d_softmax, N);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::vector<float> result(n * n);
  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_softmax, bytes_f_N, cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_softmax);

  return result;
}

void warmup_wmma(const std::vector<__half>& matrix, std::size_t n) {
  run_wmma(matrix, n);
}

std::vector<float> run_cutlass(const std::vector<__half>& matrix,
                               std::size_t n) {
  if (matrix.size() != 2ull * n * n) {
    throw std::runtime_error(
        "Input matrix size must be exactly 2*n*n (A then B)");
  }

  const int TILE = 16;
  const int N = static_cast<int>(n);
  const int Npad = ((N + TILE - 1) / TILE) * TILE;

  __half *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_softmax = nullptr;

  size_t bytes_half_padded = static_cast<size_t>(Npad * Npad) * sizeof(__half);
  size_t bytes_float_padded = static_cast<size_t>(Npad * Npad) * sizeof(float);
  size_t bytes_float_N = n * n * sizeof(float);

  // выделение памяти
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_float_padded));
  CHECK_CUDA_ERROR(cudaMalloc(&d_softmax, bytes_float_N));
  // заполнение нулями
  CHECK_CUDA_ERROR(cudaMemset(d_A, 0, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMemset(d_B, 0, bytes_half_padded));
  CHECK_CUDA_ERROR(cudaMemset(d_C, 0, bytes_float_padded));

  const __half* h_A = matrix.data();
  const __half* h_B = matrix.data() + (size_t)N * N;
  // копирование
  CHECK_CUDA_ERROR(cudaMemcpy2D(d_A, Npad * sizeof(__half), h_A,
                                N * sizeof(__half), N * sizeof(__half), N,
                                cudaMemcpyHostToDevice));

  CHECK_CUDA_ERROR(cudaMemcpy2D(d_B, Npad * sizeof(__half), h_B,
                                N * sizeof(__half), N * sizeof(__half), N,
                                cudaMemcpyHostToDevice));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;

  Gemm gemm_op;

  // размерности (m*n*k) для перемножения, равны тк матрицы квадратные
  cutlass::gemm::GemmCoord problem_size(Npad, Npad, Npad);
  // шаги от начала одной строки дол начала следующей
  // A[i][j] = *(d_A + i * lda + j) и тд
  int lda = Npad, ldb = Npad, ldc = Npad;
  // C = alpha * A·B + betta * C = A*B
  float alpha = 1.0f, beta = 0.0f;

  typename Gemm::Arguments arguments{
      problem_size,
      {reinterpret_cast<cutlass::half_t*>(d_A), lda},
      {reinterpret_cast<cutlass::half_t*>(d_B), ldb},
      {d_C, ldc},
      {d_C, ldc},  // C = D
      {alpha, beta}};

  // запуск
  cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM failed: " +
                             std::to_string(static_cast<int>(status)));
  }
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Копируем матрицу для softmax (Npad -> N)
  CHECK_CUDA_ERROR(cudaMemcpy2D(d_softmax, N * sizeof(float), d_C,
                                Npad * sizeof(float), N * sizeof(float), N,
                                cudaMemcpyDeviceToDevice));

  // Softmax
  const int threads_per_block = 512;
  const int blocks = N;
  const size_t shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_softmax,
                                                                 d_softmax, N);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::vector<float> result(N * N);
  CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_softmax, bytes_float_N,
                              cudaMemcpyDeviceToHost));

  // Освобождаем память
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_softmax);

  return result;
}

void warmup_cutlass(const std::vector<__half>& matrix, std::size_t n) {
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
