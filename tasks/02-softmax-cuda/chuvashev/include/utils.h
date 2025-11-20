#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#ifndef CHECK_CUDA_ERROR
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
#endif

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

// Utils
void make_matrix(std::size_t n, std::vector<float> &matrix);
void print_matrix(const std::size_t n, const std::vector<float> &matrix);
void print_report(std::string_view testName, const RunResult &result);
std::string format_diff(float diff);
std::string format_time(double seconds);
float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate);
double measure_seconds(const std::function<void()> &work);

// Sequential
void run_sequential(const std::vector<float> &input, std::vector<float> &output,
                    std::size_t n);

// CUDA
__global__ void warmup_kernel(float *d_matrix, const std::size_t n);
__global__ void softmax_kernel(float *d_matrix, size_t n);
void warmup_cuda(const std::vector<float> &matrix, std::size_t n);
void launch_softmax_kernel(float *d_matrix, size_t n);
void run_cuda_simt(const std::vector<float> &input, std::vector<float> &output,
                   std::size_t n);

#endif  // !UTILS_H