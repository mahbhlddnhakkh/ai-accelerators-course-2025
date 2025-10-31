#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
__m256 exp256_ps(__m256 x);

void PrintMatrix(const std::vector<float> &matrix, std::size_t n) {
  if (matrix.size() != n * n) {
    std::cerr << "Error: matrix size does not match dimension n×n\n";
    return;
  }

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                << matrix[i * n + j] << " ";
    }
    std::cout << "\n";
  }
}
std::vector<float> make_matrix(std::size_t n) {
  std::vector<float> matrix(n * n);

  std::random_device rd;
  std::mt19937 gen(15);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto &x : matrix) {
    x = dist(gen);
  }

  return matrix;
}

void SoftmaxRow(const float *row_begin, float *row_result, std::size_t n) {
  float sum_exp = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    sum_exp += std::exp(row_begin[j]);
  }
  float div_sum_exp = 1 / sum_exp;
  for (std::size_t j = 0; j < n; ++j) {
    row_result[j] = std::exp(row_begin[j]) * div_sum_exp;
  }
}

std::vector<float> run_sequential(const std::vector<float> &matrix,
                                  std::size_t n) {
  std::vector<float> result(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRow(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

std::vector<float> run_openmp(const std::vector<float> &matrix, std::size_t n) {
  std::vector<float> result(n * n);
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRow(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

// Горизонтальная сумма 8 float в __m256 -> float
static inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);    // выделяем левые 4
  __m128 hi = _mm256_extractf128_ps(v, 1);  // выделяем правые 4
  // сложение поэлементно
  __m128 sum128 = _mm_add_ps(lo, hi);  // -> [a0+a4, a1+a5, a2+a6, a3+a7]
  // сложение соседних
  sum128 = _mm_hadd_ps(sum128, sum128);  // -> [a+b, c+d, a+b, c+d]
  sum128 = _mm_hadd_ps(sum128, sum128);  // -> [total, total, total, total]
  return _mm_cvtss_f32(sum128);          // берём первое
}

// запись вектора в массив
static inline void storeu256_ps(float *dst, __m256 v) {
  _mm256_storeu_ps(dst, v);
}
// взять вектор из массива
static inline __m256 loadu256_ps(const float *src) {
  return _mm256_loadu_ps(src);
}

void SoftmaxRowSimd(const float *row_begin, float *row_result, std::size_t n) {
  if (n == 0) return;
  std::size_t i = 0;
  float sum_exp = 0.0;
  // считаем экспоненты (первый проход)
  for (; i + 7 < n; i += 8) {
    __m256 v = loadu256_ps(row_begin + i);  // вектор из указателя xi
    __m256 e = exp256_ps(v);  // вектор экспонент e^(xi)
    storeu256_ps(row_result + i, e);  // записываем числители дробей e^(xi)
    sum_exp += hsum256_ps(e);  // суммируем e^(xi)
  }
  // остаток скалярно
  for (; i < n; ++i) {
    float s = std::exp(row_begin[i]);
    row_result[i] = s;
    sum_exp += s;
  }
  // Нормализация: умножаем все значения на 1/sum_exp (второй проход)
  float inv_sum = 1.0f / sum_exp;
  __m256 inv_vec = _mm256_set1_ps(inv_sum);
  i = 0;
  for (; i + 7 < n; i += 8) {
    __m256 e = loadu256_ps(row_result + i);
    __m256 r = _mm256_mul_ps(e, inv_vec);
    storeu256_ps(row_result + i, r);
  }
  for (; i < n; ++i) {
    row_result[i] *= inv_sum;
  }
}

std::vector<float> run_simd(const std::vector<float> &matrix, std::size_t n) {
  std::vector<float> result(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRowSimd(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

std::vector<float> run_openmp_simd(const std::vector<float> &matrix,
                                   std::size_t n) {
  std::vector<float> result(n * n);
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRowSimd(&matrix[i * n], &result[i * n], n);
  }
  return result;
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

RunResult run_test_case(const std::function<std::vector<float>()> &runner,
                        const std::vector<float> &baseline,
                        std::string_view methodName) {
  RunResult result;
  try {
    result.seconds = measure_seconds(runner, result.result);
    result.diff = max_abs_diff(baseline, result.result);
    result.success = true;
  } catch (const std::exception &ex) {
    std::cerr << methodName << " method failed: " << ex.what() << '\n';
  }
  return result;
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
    // PrintMatrix(input, n);

    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);
    // std::cout << std::endl;
    // PrintMatrix(sequential_result, n);
    auto omp_res = run_test_case([&] { return run_openmp(input, n); },
                                 sequential_result, "OpenMP");
    auto simd_res = run_test_case([&] { return run_simd(input, n); },
                                  sequential_result, "SIMD");
    auto omp_simd_res = run_test_case([&] { return run_openmp_simd(input, n); },
                                      sequential_result, "OpenMP + SIMD");

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("OpenMP", omp_res);
    print_report("SIMD", simd_res);
    print_report("OpenMP + SIMD", omp_simd_res);

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

namespace {
__m256 exp256_ps(__m256 x) {
  /* Modified code. The original code is here:
    https://github.com/reyoung/avx_mathfun

     AVX implementation of exp
     Based on "sse_mathfun.h", by Julien Pommier
     http://gruntthepeon.free.fr/ssemath/
     Copyright (C) 2012 Giovanni Garberoglio
     Interdisciplinary Laboratory for Computational Science (LISC)
     Fondazione Bruno Kessler and University of Trento
     via Sommarive, 18
     I-38123 Trento (Italy)
    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
    (this is the zlib license)
  */
  /*
    To increase the compatibility across different compilers the original code
    is converted to plain AVX2 intrinsics code without ingenious macro's, gcc
    style alignment attributes etc. The modified code requires AVX2
  */
  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
  __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375);
  __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
  tmp = _mm256_floor_ps(fx);
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);
  tmp = _mm256_mul_ps(fx, cephes_exp_C1);
  __m256 z = _mm256_mul_ps(fx, cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);
  z = _mm256_mul_ps(x, x);

  __m256 y = cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
}  // namespace