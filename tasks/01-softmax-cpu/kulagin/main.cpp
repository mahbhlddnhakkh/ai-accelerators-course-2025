#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

__m256 exp256_ps(__m256 x);

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

void calc_row_simd(const float *row, float *row_res, const std::size_t n) {
  __m256 m_d = _mm256_setzero_ps();
  __m256 tmp;
  float d = 0.0f;
  const std::size_t tail_start = n - n % 8;
  const std::size_t tail_stop = n - 8;
  for (std::size_t i = 0; i < tail_stop; i += 8) {
    tmp = _mm256_loadu_ps(row + i);
    tmp = exp256_ps(tmp);
    m_d = _mm256_add_ps(m_d, tmp);
  }
  for (std::size_t i = tail_start; i < n; i++) {
    d += std::exp(row[i]);
  }

  m_d = _mm256_hadd_ps(m_d, m_d);
  tmp = _mm256_permute_ps(m_d, 0b10110001);  // 1 0 3 2
  m_d = _mm256_add_ps(m_d, tmp);
  tmp = _mm256_permute2f128_ps(m_d, m_d, 0b00100001);  // 1 and 2
  m_d = _mm256_add_ps(m_d, tmp);
  tmp = _mm256_set1_ps(d);
  m_d = _mm256_add_ps(m_d, tmp);
  tmp = _mm256_set1_ps(1.0f);
  m_d = _mm256_div_ps(tmp, m_d);
  d = _mm256_cvtss_f32(m_d);

  for (std::size_t i = 0; i < tail_stop; i += 8) {
    tmp = _mm256_loadu_ps(row + i);
    tmp = exp256_ps(tmp);
    tmp = _mm256_mul_ps(tmp, m_d);
    _mm256_storeu_ps(row_res + i, tmp);
  }
  for (std::size_t i = tail_start; i < n; i++) {
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

std::vector<float> run_openmp(const std::vector<float> &matrix, std::size_t n) {
  if (matrix.size() != n * n) {
    throw std::runtime_error("Matrix size is not n * n");
  }
  std::vector<float> res(n * n);
  const float *matrix_ptr = matrix.data();
  float *res_ptr = res.data();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; i++) {
    calc_row(matrix_ptr + i * n, res_ptr + i * n, n);
  }
  return res;
}

std::vector<float> run_simd(const std::vector<float> &matrix, std::size_t n) {
  if (matrix.size() != n * n) {
    throw std::runtime_error("Matrix size is not n * n");
  }
  if (n < 8) {
    throw std::runtime_error("Must n >= 8");
  }
  std::vector<float> res(n * n);
  const float *matrix_ptr = matrix.data();
  float *res_ptr = res.data();
  for (std::size_t i = 0; i < n; i++) {
    calc_row_simd(matrix_ptr + i * n, res_ptr + i * n, n);
  }
  return res;
}

std::vector<float> run_openmp_simd(const std::vector<float> &matrix,
                                   std::size_t n) {
  if (matrix.size() != n * n) {
    throw std::runtime_error("Matrix size is not n * n");
  }
  if (n < 8) {
    throw std::runtime_error("Must n >= 8");
  }
  std::vector<float> res(n * n);
  const float *matrix_ptr = matrix.data();
  float *res_ptr = res.data();
#pragma omp parallel for
  for (std::size_t i = 0; i < n; i++) {
    calc_row_simd(matrix_ptr + i * n, res_ptr + i * n, n);
  }
  return res;
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

    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

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
  //  https://stackoverflow.com/questions/48863719
  /* Modified code from this source: https://github.com/reyoung/avx_mathfun

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
    style alignment attributes etc. Moreover, the part
    "express exp(x) as exp(g+ n*log(2))" has been significantly simplified.
    This modified code is not thoroughly tested!
  */

  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
  __m256 inv_LOG2EF = _mm256_set1_ps(0.693147180559945f);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
  __m256 fx;
  __m256i imm0;
  __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 z = _mm256_mul_ps(fx, inv_LOG2EF);
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
