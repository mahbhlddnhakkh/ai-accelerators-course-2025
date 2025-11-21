#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
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
  std::mt19937 gen{36};
  std::uniform_real_distribution<float> dist(-15.0f, 15.0f);
  std::vector<__half> matrix(n * n);
  for (std::size_t i = 0; i < n * n; ++i) {
    matrix[i] = __float2half(dist(gen));
  }
  return matrix;
}

void softmax_row_basic(const float* input_row, float* output_row,
                       std::size_t n) {
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

std::vector<float> run_openmp_reference(const std::vector<__half>& matrix_a_h,
                                        std::size_t n) {
  const std::vector<__half>& matrix_b_h = matrix_a_h;
  std::vector<float> matrix_a_f(n * n);
  std::vector<float> matrix_b_f(n * n);
  for (std::size_t i = 0; i < n * n; ++i) {
    matrix_a_f[i] = __half2float(matrix_a_h[i]);
    matrix_b_f[i] = __half2float(matrix_b_h[i]);
  }
  std::vector<float> result(n * n);

#pragma omp parallel
  {
    std::vector<float> temp_row_c(n, 0.0f);

#pragma omp for
    for (std::size_t i = 0; i < n; ++i) {
      float* row_c = temp_row_c.data();

      for (std::size_t k = 0; k < n; ++k) {
        const float a_val = matrix_a_f[i * n + k];
        for (std::size_t j = 0; j < n; ++j) {
          row_c[j] += a_val * matrix_b_f[k * n + j];
        }
      }
      softmax_row_basic(row_c, &result[i * n], n);
    }
  }
  return result;
}

__global__ void softmax_kernel(const float* input, float* output,
                               std::size_t n) {
  extern __shared__ float sdata[];

  const std::size_t row = blockIdx.x;
  const std::size_t tid = threadIdx.x;
  const std::size_t block_size = blockDim.x;

  const float* in_row = input + row * n;
  float* out_row = output + row * n;

  float local_sum = 0.0f;
  for (std::size_t j = tid; j < n; j += block_size) {
    float val = expf(in_row[j]);
    out_row[j] = val;
    local_sum += val;
  }

  sdata[tid] = local_sum;
  __syncthreads();

  for (std::size_t s = block_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  float total_sum = sdata[0];
  float inv_sum = 1.0f / total_sum;
  __syncthreads();

  for (std::size_t j = tid; j < n; j += block_size) {
    out_row[j] *= inv_sum;
  }
}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int WARP_SIZE = 32;

__global__ void matmul_wmma_kernel(const __half* a, const __half* b, float* c,
                                   int n) {
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int warpN = blockIdx.y;
  int num_tiles_m = (n + WMMA_M - 1) / WMMA_M;
  int num_tiles_n = (n + WMMA_N - 1) / WMMA_N;

  if (warpM >= num_tiles_m || warpN >= num_tiles_n) {
    return;
  }

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                         nvcuda::wmma::row_major>
      frag_a;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                         nvcuda::wmma::row_major>
      frag_b;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      frag_c;

  nvcuda::wmma::fill_fragment(frag_c, 0.0f);

  int a_row_start = warpM * WMMA_M;
  int b_col_start = warpN * WMMA_N;

  for (int k = 0; k < n; k += WMMA_K) {
    nvcuda::wmma::load_matrix_sync(frag_a, a + (a_row_start * n + k), n);
    nvcuda::wmma::load_matrix_sync(frag_b, b + (k * n + b_col_start), n);

    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
  }

  int c_row_start = warpM * WMMA_M;
  int c_col_start = warpN * WMMA_N;

  nvcuda::wmma::store_matrix_sync(c + (c_row_start * n + c_col_start), frag_c,
                                  n, nvcuda::wmma::mem_row_major);
}

void run_matmul_wmma_and_softmax(const __half* d_a, const __half* d_b,
                                 float* d_result, float* d_temp_c, int n) {
  dim3 block_size_matmul(4 * WARP_SIZE, 1);
  dim3 grid_size_matmul((n + 63) / 64, (n + 15) / 16);

  matmul_wmma_kernel<<<grid_size_matmul, block_size_matmul>>>(d_a, d_b,
                                                              d_temp_c, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  int threads_per_block_softmax = 256;
  if (threads_per_block_softmax > n) {
    threads_per_block_softmax = n;
  }
  dim3 block_size_softmax(threads_per_block_softmax);
  dim3 grid_size_softmax(n);

  std::size_t shared_mem_size = threads_per_block_softmax * sizeof(float);
  softmax_kernel<<<grid_size_softmax, block_size_softmax, shared_mem_size>>>(
      d_temp_c, d_result, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void warmup_wmma(const std::vector<__half>& matrix_a_h, std::size_t n) {
  // Предполагаем, что B = A
  const std::vector<__half>& matrix_b_h = matrix_a_h;

  // 1. Выделение памяти на GPU
  __half *d_a = nullptr, *d_b = nullptr;
  float *d_result = nullptr, *d_temp_c = nullptr;

  try {
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, n * n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, n * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_c, n * n * sizeof(float)));

    // 2. Копирование данных на GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, matrix_a_h.data(), n * n * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, matrix_b_h.data(), n * n * sizeof(__half),
                                cudaMemcpyHostToDevice));

    // 3. Вызов функции запуска GEMM и Softmax
    run_matmul_wmma_and_softmax(d_a, d_b, d_result, d_temp_c, n);

    // 4. Освобождение памяти на GPU
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaFree(d_temp_c));
  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_result) cudaFree(d_result);
    if (d_temp_c) cudaFree(d_temp_c);
    throw;
  }
}

std::vector<float> run_wmma(const std::vector<__half>& matrix_a_h,
                            std::size_t n) {
  // Предполагаем, что B = A
  const std::vector<__half>& matrix_b_h = matrix_a_h;

  // 1. Выделение памяти на GPU
  __half *d_a = nullptr, *d_b = nullptr;
  float *d_result = nullptr, *d_temp_c = nullptr;

  try {
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, n * n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, n * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_c, n * n * sizeof(float)));

    // 2. Копирование данных на GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, matrix_a_h.data(), n * n * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, matrix_b_h.data(), n * n * sizeof(__half),
                                cudaMemcpyHostToDevice));

    // 3. Вызов функции запуска GEMM и Softmax
    run_matmul_wmma_and_softmax(d_a, d_b, d_result, d_temp_c, n);

    // 4. Копирование результата на хост
    std::vector<float> h_result(n * n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_result,
                                n * n * sizeof(float), cudaMemcpyDeviceToHost));

    // 5. Освобождение памяти на GPU
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaFree(d_temp_c));

    return h_result;
  } catch (...) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_result) cudaFree(d_result);
    if (d_temp_c) cudaFree(d_temp_c);
    throw;
  }
}

void run_matmul_cutlass_and_softmax(const cutlass::half_t* d_a_padded,
                                    const cutlass::half_t* d_b_padded,
                                    float* d_result, float* d_temp_c_padded,
                                    int n, int n_padded) {
  // 1. Запуск CUTLASS GEMM на *пadded* матрицах
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using Layout = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, Layout, cutlass::half_t, Layout, ElementOutput, Layout,
      ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;

  typename Gemm::Arguments args(
      {n_padded, n_padded, n_padded},  // Размер задачи (padded)
      {d_a_padded, n_padded},          // A matrix (padded)
      {d_b_padded, n_padded},          // B matrix (padded)
      {d_temp_c_padded, n_padded},     // C matrix (source for epilogue, padded)
      {d_temp_c_padded, n_padded},     // D matrix (output destination, padded)
      {ElementCompute(1.0f), ElementCompute(0.0f)}  // alpha, beta
  );

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM cannot implement these arguments: " +
                             std::string(cutlassGetStatusString(status)));
  }

  // Запуск GEMM
  status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM launch failed: " +
                             std::string(cutlassGetStatusString(status)));
  }
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 2. Копирование результата GEMM (padded) в буфер для softmax (original size)
  // Используем cudaMemcpy2D для копирования n строк по n элементов float,
  // из матрицы размером n_padded x n_padded (шаг n_padded) в n x n (шаг n).
  CHECK_CUDA_ERROR(cudaMemcpy2D(d_result, n * sizeof(float), d_temp_c_padded,
                                n_padded * sizeof(float), n * sizeof(float), n,
                                cudaMemcpyDeviceToDevice));

  // 3. Запуск softmax на буфере оригинального размера
  int threads_per_block_softmax = 256;
  if (threads_per_block_softmax > static_cast<int>(n)) {
    threads_per_block_softmax = static_cast<int>(n);
  }
  dim3 block_size_softmax(threads_per_block_softmax);
  dim3 grid_size_softmax(n);  // n блоков для n строк
  std::size_t shared_mem_size = threads_per_block_softmax * sizeof(float);
  softmax_kernel<<<grid_size_softmax, block_size_softmax, shared_mem_size>>>(
      d_result, d_result, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void warmup_cutlass(const std::vector<__half>& matrix_a_h, std::size_t n) {
  const std::vector<__half>& matrix_b_h = matrix_a_h;
  const int TILE = 16;
  const int n_int = static_cast<int>(n);
  const int n_padded =
      ((n_int + TILE - 1) / TILE) * TILE;  // Округление вверх до кратного 16

  cutlass::half_t *d_a_padded = nullptr, *d_b_padded = nullptr;
  float *d_result = nullptr, *d_temp_c_padded = nullptr;

  try {
    std::size_t bytes_half_padded =
        static_cast<std::size_t>(n_padded) * n_padded * sizeof(cutlass::half_t);
    std::size_t bytes_float_padded =
        static_cast<std::size_t>(n_padded) * n_padded * sizeof(float);
    std::size_t bytes_float_n = n * n * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_a_padded, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_padded, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, bytes_float_n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_c_padded, bytes_float_padded));

    CHECK_CUDA_ERROR(cudaMemset(d_a_padded, 0, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMemset(d_b_padded, 0, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMemset(d_temp_c_padded, 0, bytes_float_padded));

    std::vector<cutlass::half_t> temp_a_h(matrix_a_h.size());
    std::vector<cutlass::half_t> temp_b_h(matrix_b_h.size());
    std::transform(
        matrix_a_h.begin(), matrix_a_h.end(), temp_a_h.begin(),
        [](const __half& h) { return static_cast<cutlass::half_t>(h); });
    std::transform(
        matrix_b_h.begin(), matrix_b_h.end(), temp_b_h.begin(),
        [](const __half& h) { return static_cast<cutlass::half_t>(h); });

    CHECK_CUDA_ERROR(
        cudaMemcpy2D(d_a_padded, n_padded * sizeof(cutlass::half_t),
                     temp_a_h.data(), n * sizeof(cutlass::half_t),
                     n * sizeof(cutlass::half_t), n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy2D(d_b_padded, n_padded * sizeof(cutlass::half_t),
                     temp_b_h.data(), n * sizeof(cutlass::half_t),
                     n * sizeof(cutlass::half_t), n, cudaMemcpyHostToDevice));

    run_matmul_cutlass_and_softmax(d_a_padded, d_b_padded, d_result,
                                   d_temp_c_padded, n_int, n_padded);

    CHECK_CUDA_ERROR(cudaFree(d_a_padded));
    CHECK_CUDA_ERROR(cudaFree(d_b_padded));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaFree(d_temp_c_padded));
  } catch (...) {
    if (d_a_padded) cudaFree(d_a_padded);
    if (d_b_padded) cudaFree(d_b_padded);
    if (d_result) cudaFree(d_result);
    if (d_temp_c_padded) cudaFree(d_temp_c_padded);
    throw;
  }
}

std::vector<float> run_cutlass(const std::vector<__half>& matrix_a_h,
                               std::size_t n) {
  const std::vector<__half>& matrix_b_h = matrix_a_h;
  const int TILE = 16;
  const int n_int = static_cast<int>(n);
  const int n_padded = ((n_int + TILE - 1) / TILE) * TILE;

  cutlass::half_t *d_a_padded = nullptr, *d_b_padded = nullptr;
  float *d_result = nullptr, *d_temp_c_padded = nullptr;

  try {
    std::size_t bytes_half_padded =
        static_cast<std::size_t>(n_padded) * n_padded * sizeof(cutlass::half_t);
    std::size_t bytes_float_padded =
        static_cast<std::size_t>(n_padded) * n_padded * sizeof(float);
    std::size_t bytes_float_n = n * n * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_a_padded, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b_padded, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, bytes_float_n));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_c_padded, bytes_float_padded));

    CHECK_CUDA_ERROR(cudaMemset(d_a_padded, 0, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMemset(d_b_padded, 0, bytes_half_padded));
    CHECK_CUDA_ERROR(cudaMemset(d_temp_c_padded, 0, bytes_float_padded));

    std::vector<cutlass::half_t> temp_a_h(matrix_a_h.size());
    std::vector<cutlass::half_t> temp_b_h(matrix_b_h.size());
    std::transform(
        matrix_a_h.begin(), matrix_a_h.end(), temp_a_h.begin(),
        [](const __half& h) { return static_cast<cutlass::half_t>(h); });
    std::transform(
        matrix_b_h.begin(), matrix_b_h.end(), temp_b_h.begin(),
        [](const __half& h) { return static_cast<cutlass::half_t>(h); });

    CHECK_CUDA_ERROR(
        cudaMemcpy2D(d_a_padded, n_padded * sizeof(cutlass::half_t),
                     temp_a_h.data(), n * sizeof(cutlass::half_t),
                     n * sizeof(cutlass::half_t), n, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy2D(d_b_padded, n_padded * sizeof(cutlass::half_t),
                     temp_b_h.data(), n * sizeof(cutlass::half_t),
                     n * sizeof(cutlass::half_t), n, cudaMemcpyHostToDevice));

    run_matmul_cutlass_and_softmax(d_a_padded, d_b_padded, d_result,
                                   d_temp_c_padded, n_int, n_padded);

    std::vector<float> h_result(n * n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_result, bytes_float_n,
                                cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_a_padded));
    CHECK_CUDA_ERROR(cudaFree(d_b_padded));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaFree(d_temp_c_padded));

    return h_result;
  } catch (...) {
    if (d_a_padded) cudaFree(d_a_padded);
    if (d_b_padded) cudaFree(d_b_padded);
    if (d_result) cudaFree(d_result);
    if (d_temp_c_padded) cudaFree(d_temp_c_padded);
    throw;
  }
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
