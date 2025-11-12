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

std::vector<float> make_matrix(std::size_t n) {
  std::vector<float> matrix(n * n);

  static std::random_device ran_dev;
  static std::mt19937 ran_eng(ran_dev());
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::generate(matrix.begin(), matrix.end(), [&]() { return dist(ran_eng); });

  return matrix;
}

static void row_calculation_amount(const float* input_row, std::size_t n,
                                   float* output_row) {
  float row_sum = 0.0f;
  for (std::size_t item = 0; item < n; item++) {
    row_sum += exp(input_row[item]);
  }
  const float divider = 1 / row_sum;
  for (std::size_t item = 0; item < n; item++) {
    output_row[item] = exp(input_row[item]) / row_sum;
  }
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> res_matrix(n * n);

  for (std::size_t row = 0; row < n; row++) {
    row_calculation_amount(&matrix[row * n], n, &res_matrix[row * n]);
  }
  return res_matrix;
}

std::vector<float> run_openmp(const std::vector<float>& matrix, std::size_t n) {
  std::vector<float> res_matrix(n * n);

#pragma omp parallel for
  for (int row = 0; row < n; row++) {
    row_calculation_amount(&matrix[row * n], n, &res_matrix[row * n]);
  }
  return res_matrix;
}

static std::size_t AVX_FLOAT_COUNT = 8;

__m256 exp256_ps(__m256 x);

static void row_calculation_amount_simd(const float* input_row, std::size_t n,
                                        float* output_row) {
  __m256 sum8 = _mm256_setzero_ps();
  std::size_t item_counter = 0;

  for (; item_counter + AVX_FLOAT_COUNT <= n; item_counter += AVX_FLOAT_COUNT) {
    __m256 items = _mm256_loadu_ps(&input_row[item_counter]);
    items = exp256_ps(items);
    sum8 = _mm256_add_ps(sum8, items);
    _mm256_storeu_ps(&output_row[item_counter], items);
  }

  alignas(32) float temp_sum[8];
  _mm256_store_ps(temp_sum, sum8);

  float sum = 0.0f;
  for (int i = 0; i < AVX_FLOAT_COUNT; i++) {
    sum += temp_sum[i];
  }

  float tail_sum = 0.0f;
  for (; item_counter < n; item_counter++) {
    float exp_val = exp(input_row[item_counter]);
    output_row[item_counter] = exp_val;
    tail_sum += exp_val;
  }

  float total_sum = sum + tail_sum;
  const float divider = 1.0f / total_sum;
  const __m256 divider_vec = _mm256_set1_ps(divider);

  item_counter = 0;
  for (; item_counter + AVX_FLOAT_COUNT <= n; item_counter += AVX_FLOAT_COUNT) {
    __m256 items = _mm256_loadu_ps(&output_row[item_counter]);
    items = _mm256_mul_ps(items, divider_vec);
    _mm256_storeu_ps(&output_row[item_counter], items);
  }

  for (; item_counter < n; item_counter++) {
    output_row[item_counter] *= divider;
  }
}

std::vector<float> run_simd(const std::vector<float>& matrix, std::size_t n) {
  std::vector<float> res_matrix(n * n);

  for (std::size_t row = 0; row < n; row++) {
    row_calculation_amount_simd(&matrix[row * n], n, &res_matrix[row * n]);
  }
  return res_matrix;
}

std::vector<float> run_openmp_simd(const std::vector<float>& matrix,
                                   std::size_t n) {
  std::vector<float> res_matrix(n * n);

#pragma omp parallel for
  for (int row = 0; row < n; row++) {
    row_calculation_amount_simd(&matrix[row * n], n, &res_matrix[row * n]);
  }
  return res_matrix;
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

RunResult run_test_case(const std::function<std::vector<float>()>& runner,
                        const std::vector<float>& baseline,
                        std::string_view methodName) {
  RunResult result;
  try {
    result.seconds = measure_seconds(runner, result.result);
    result.diff = max_abs_diff(baseline, result.result);
    result.success = true;
  } catch (const std::exception& ex) {
    std::cerr << methodName << " method failed: " << ex.what() << '\n';
  }
  return result;
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
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

namespace {

// AVX1-совместимая реализация экспоненты
__m256 exp256_ps(__m256 x) {
  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
  __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375f);
  __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4f);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);

  __m256 one = _mm256_set1_ps(1.0f);
  __m256 half = _mm256_set1_ps(0.5f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  __m256 fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_add_ps(fx, half);

  __m128i fx_low = _mm_cvtps_epi32(_mm256_extractf128_ps(fx, 0));
  __m128i fx_high = _mm_cvtps_epi32(_mm256_extractf128_ps(fx, 1));
  fx = _mm256_set_m128(_mm_cvtepi32_ps(fx_high), _mm_cvtepi32_ps(fx_low));

  __m256 tmp = _mm256_mul_ps(fx, cephes_exp_C1);
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

  __m128i imm0_low = _mm_cvtps_epi32(_mm256_extractf128_ps(fx, 0));
  __m128i imm0_high = _mm_cvtps_epi32(_mm256_extractf128_ps(fx, 1));

  imm0_low = _mm_add_epi32(imm0_low, _mm_set1_epi32(0x7f));
  imm0_high = _mm_add_epi32(imm0_high, _mm_set1_epi32(0x7f));

  imm0_low = _mm_slli_epi32(imm0_low, 23);
  imm0_high = _mm_slli_epi32(imm0_high, 23);

  __m256 pow2n =
      _mm256_set_m128(_mm_castsi128_ps(imm0_high), _mm_castsi128_ps(imm0_low));

  y = _mm256_mul_ps(y, pow2n);
  return y;
}
}  // namespace