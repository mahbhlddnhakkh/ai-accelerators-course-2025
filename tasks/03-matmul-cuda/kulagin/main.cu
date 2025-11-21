#ifndef CUDA_NO_BFLOAT16
#define CUDA_NO_BFLOAT16 1
#endif

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
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

constexpr bool measure_time_internal_all = true;
constexpr bool separate_measure_all = true;
constexpr bool exclude_softmax = false;

inline __half rand_range(float a, float b) {
  return __float2half(((b - a) * ((float)rand() / (float)RAND_MAX)) + a);
}

std::vector<__half> make_input_matrix(std::size_t n) {
  srand(time(0));
  constexpr float lower = -1.0f, upper = 1.0f;  // Arbitrary numbers
  const std::size_t res_sz = n * n * 2;
  std::vector<__half> res(res_sz);
  for (std::size_t i = 0; i < res_sz; i++) {
    res[i] = rand_range(lower, upper);
  }
  return res;
}

std::vector<float> run_openmp_reference(const std::vector<__half> &matrix,
                                        std::size_t n) {
  if (n == 0) {
    throw std::runtime_error("n == 0");
  }
  std::vector<float> res(n * n, 0.0f);
  float *res_ptr = res.data();
  const __half *a = matrix.data();
  const __half *b = matrix.data() + n * n;
#pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    float *res_row = res_ptr + i * n;
    for (size_t k = 0; k < n; k++) {
      const float a_elem = __half2float(a[i * n + k]);
      const __half *b_row = b + k * n;
      for (size_t j = 0; j < n; j++) {
        res_row[j] += a_elem * __half2float(b_row[j]);
      }
    }
    if (!exclude_softmax) {
      float d = 0.0f;
      for (size_t j = 0; j < n; j++) {
        d += std::exp(res_row[j]);
      }
      d = 1.0f / d;
      for (size_t j = 0; j < n; j++) {
        res_row[j] = std::exp(res_row[j]) * d;
      }
    }
  }
  return res;
}

__global__ void softmax_kernel(const float *input, float *output, size_t n) {
  __shared__ float s_d[1];
  if (threadIdx.x == 0) {
    *s_d = 0.0f;
  }
  __syncthreads();
  const int d_offset = blockIdx.x * n;
  const float *row = input + d_offset;
  float *res = output + d_offset;
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

const int WARP_SIZE = 32;
const int WMMA_SIZE = 16;

__global__ void wmma_gemm(const __half *a, const __half *b, float *output,
                          size_t n) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * WMMA_SIZE;
  int col = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_SIZE;
  if (row >= n || col >= n) {
    return;
  }
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_SIZE, WMMA_SIZE,
                         WMMA_SIZE, __half, nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_SIZE, WMMA_SIZE,
                         WMMA_SIZE, __half, nvcuda::wmma::row_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_SIZE, WMMA_SIZE,
                         WMMA_SIZE, float>
      acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
  for (int i = 0; i < n; i += WMMA_SIZE) {
    nvcuda::wmma::load_matrix_sync(a_frag, a + row * n + i, n);
    nvcuda::wmma::load_matrix_sync(b_frag, b + col + i * n, n);
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }
  nvcuda::wmma::store_matrix_sync(output + row * n + col, acc_frag, n,
                                  nvcuda::wmma::mem_row_major);
}

#define upper_div(a, b) (((a) + (b) - 1) / (b))

template <bool measure_time_internal = measure_time_internal_all,
          bool separate_measure = separate_measure_all>
std::vector<float> run_wmma(const std::vector<__half> &matrix, std::size_t n) {
  if (n <= WMMA_SIZE) {
    return run_openmp_reference(matrix, n);
  }
  std::vector<float> res(n * n, 0.0f);
  size_t szb_half = n * n * sizeof(*matrix.data());
  size_t szb = n * n * sizeof(*res.data());
  __half *a_dev = nullptr;
  __half *b_dev = nullptr;
  float *res_gemm_dev = nullptr;
  float *res_dev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&a_dev, szb_half));
  CHECK_CUDA_ERROR(cudaMalloc(&b_dev, szb_half));
  CHECK_CUDA_ERROR(cudaMalloc(&res_gemm_dev, szb));
  CHECK_CUDA_ERROR(cudaMalloc(&res_dev, szb));

  CHECK_CUDA_ERROR(
      cudaMemcpy(a_dev, matrix.data(), szb_half, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_dev, matrix.data() + n * n, szb_half,
                              cudaMemcpyHostToDevice));
  dim3 block_size(4 * WARP_SIZE, 4);
  dim3 block_count((upper_div(n, block_size.x)) * (WARP_SIZE / WMMA_SIZE),
                   (upper_div(upper_div(n, block_size.y), WMMA_SIZE)));

  float elapsedTime = -1.0f;
  cudaEvent_t start, stop;
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
  }
  wmma_gemm<<<block_count, block_size>>>(a_dev, b_dev, res_gemm_dev, n);
  if (measure_time_internal && separate_measure) {
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "wmma_gemm elapsed time: " << elapsedTime << " ms\n";
    CHECK_CUDA_ERROR(cudaEventRecord(start));
  }
  softmax_kernel<<<n, 512>>>(res_gemm_dev, res_dev, n);
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    if (separate_measure)
      std::cout << "softmax elapsed time: " << elapsedTime << " ms\n";
    else
      std::cout << "wmma_gemm + softmax elapsed time: " << elapsedTime
                << " ms\n";
  }
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  if (exclude_softmax) {
    res_dev = res_gemm_dev;
  }
  CHECK_CUDA_ERROR(
      cudaMemcpy(res.data(), res_dev, szb, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(a_dev));
  CHECK_CUDA_ERROR(cudaFree(b_dev));
  CHECK_CUDA_ERROR(cudaFree(res_gemm_dev));
  CHECK_CUDA_ERROR(cudaFree(res_dev));
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  }
  return res;
}

void warmup_wmma(const std::vector<__half> &matrix, std::size_t n) {
  for (size_t i = 0; i < 3; i++) {
    (void)run_wmma<false>(matrix, n);
  }
}

template <bool measure_time_internal = false>
cudaError_t CutlassGEMM(const cutlass::half_t *a, const cutlass::half_t *b,
                        float *c, int n) {
  using RowMajor = cutlass::layout::RowMajor;
  using OpClassTensorOp = cutlass::arch::OpClassTensorOp;
  using Sm75 = cutlass::arch::Sm75;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t,  //  Data-type of A matrix
                                  RowMajor,         //  Layout of A matrix
                                  cutlass::half_t,  //  Data-type of B matrix
                                  RowMajor,         //  Layout of B matrix
                                  float,            //  Data-type of C matrix
                                  RowMajor,         //  Layout of C matrix
                                  float,            //  Element Accumulator
                                  OpClassTensorOp,  //  Tag indicating Tensor
                                                    //  Cores
                                  Sm75>;            //  SM architecture

  CutlassGemm::Arguments args({n, n, n},  //  Gemm Problem dimensions
                              {a, n},     //  Tensor-ref for source matrix A
                              {b, n},     //  Tensor-ref for source matrix B
                              {c, n},     //  Tensor-ref for source matrix C
                              {c, n},   //  Tensor-ref for destination matrix D
                              {1, 0});  //  Scalars used in the Epilogue

  CutlassGemm gemm_operator;
  float elapsedTime = -1.0f;
  cudaEvent_t start, stop;
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
  }
  cutlass::Status status = gemm_operator(args);
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "cutlass_gemm elapsed time: " << elapsedTime << " ms\n";
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  }
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <bool measure_time_internal = measure_time_internal_all,
          bool separate_measure = separate_measure_all>
std::vector<float> run_cutlass(const std::vector<__half> &matrix,
                               std::size_t n) {
  if (n <= WMMA_SIZE) {
    return run_openmp_reference(matrix, n);
  }
  const size_t block_size = 512;
  std::vector<float> res(n * n, 0.0f);
  size_t szb_half = n * n * sizeof(*matrix.data());
  size_t szb = n * n * sizeof(*res.data());
  cutlass::half_t *a_dev = nullptr;
  cutlass::half_t *b_dev = nullptr;
  float *res_gemm_dev = nullptr;
  float *res_dev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&a_dev, szb_half));
  CHECK_CUDA_ERROR(cudaMalloc(&b_dev, szb_half));
  CHECK_CUDA_ERROR(cudaMalloc(&res_gemm_dev, szb));
  CHECK_CUDA_ERROR(cudaMalloc(&res_dev, szb));

  CHECK_CUDA_ERROR(
      cudaMemcpy(a_dev, matrix.data(), szb_half, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_dev, matrix.data() + n * n, szb_half,
                              cudaMemcpyHostToDevice));

  float elapsedTime = -1.0f;
  cudaEvent_t start, stop;
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    if (!separate_measure) {
      CHECK_CUDA_ERROR(cudaEventRecord(start));
    }
  }

  CHECK_CUDA_ERROR(CutlassGEMM < measure_time_internal &&
                   separate_measure > (a_dev, b_dev, res_gemm_dev, n));

  if (measure_time_internal && separate_measure) {
    CHECK_CUDA_ERROR(cudaEventRecord(start));
  }
  softmax_kernel<<<n, block_size>>>(res_gemm_dev, res_dev, n);
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    if (separate_measure)
      std::cout << "softmax elapsed time: " << elapsedTime << " ms\n";
    else
      std::cout << "cutlass_gemm + softmax elapsed time: " << elapsedTime
                << " ms\n";
  }
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  if (exclude_softmax) {
    res_dev = res_gemm_dev;
  }
  CHECK_CUDA_ERROR(
      cudaMemcpy(res.data(), res_dev, szb, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(a_dev));
  CHECK_CUDA_ERROR(cudaFree(b_dev));
  CHECK_CUDA_ERROR(cudaFree(res_dev));
  CHECK_CUDA_ERROR(cudaFree(res_gemm_dev));
  if (measure_time_internal) {
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  }
  return res;
}

void warmup_cutlass(const std::vector<__half> &matrix, std::size_t n) {
  for (size_t i = 0; i < 3; i++) {
    (void)run_cutlass<false>(matrix, n);
  }
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
    // check for nan
    if (baseline[i] != baseline[i] || candidate[i] != candidate[i]) {
      max_diff = +INFINITY;
    }
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
    } catch (const std::exception &ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    RunResult cutlass_res;
    try {
      warmup_cutlass(input, n);
      cutlass_res.seconds = measure_seconds(
          [&]() { return run_cutlass(input, n); }, cutlass_res.result);
      cutlass_res.diff = max_abs_diff(openmp_result, cutlass_res.result);
      cutlass_res.success = true;
    } catch (const std::exception &ex) {
      std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
    }

    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec\n";
    print_report("WMMA", wmma_res);
    print_report("CUTLASS", cutlass_res);

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
