#ifndef UTILS_H
#define UTILS_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>

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

#define BLOCK_SIZE 32
#define THREADS_PER_BLOCK 512

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

struct timer {
  std::chrono::high_resolution_clock::time_point t_start;
  timer();
  ~timer();
  void reset();
  double elapsed();
};

struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

template <typename T>
void print_matrix(const std::vector<T> &matrix, const std::size_t n);
void make_input_matrix(std::vector<__half> &matrix, std::size_t n);
void print_report(std::string_view testName, const RunResult &result);

double measure_seconds(const std::function<void()> &work);

float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate);

std::string format_time(double seconds);
std::string format_diff(float diff);

void run_openmp_reference(const std::vector<__half> &input_A,
                          const std::vector<__half> &input_B,
                          std::vector<float> &output, const std::size_t n);

void run_matrix_mult_gpu_ver_1(const std::vector<__half> &input_A,
                               const std::vector<__half> &input_B,
                               std::vector<float> &output, const std::size_t n);
__global__ void GPU_MATMUL_V1(const __half *input, float *output,
                              const std::size_t n);

__global__ void GPU_MATMUL_V2(const __half *input, float *output,
                              const std::size_t n);
void run_matrix_mult_gpu_ver_2(const std::vector<__half> &input_A,
                               const std::vector<__half> &input_B,
                               std::vector<float> &output, const std::size_t n);

void warmup_wmma(const std::vector<__half> &input_A,
                 const std::vector<__half> &input_B, std::vector<float> &output,
                 const std::size_t n);
__global__ void WMMA_kernel(const __half *input, float *output,
                            const std::size_t n);
void run_wmma(const std::vector<__half> &input_A,
              const std::vector<__half> &input_B, std::vector<float> &output,
              const std::size_t n);

void warmup_cutlass(const std::vector<__half> &input_A,
                    const std::vector<__half> &input_B,
                    std::vector<float> &output, const std::size_t n);
void run_cutlass(const std::vector<__half> &input_A,
                 const std::vector<__half> &input_B, std::vector<float> &output,
                 std::size_t n);

__global__ void softmax_kernel(float *d_matrix, size_t n);
void launch_softmax_kernel(float *d_matrix, size_t n);

#endif  // !UTILS_H
