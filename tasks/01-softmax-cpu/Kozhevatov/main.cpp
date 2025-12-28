/**
 * @file main.cpp
 * @brief Реализация Softmax с оптимизациями (AVX, OpenMP)
 *
 * Программа вычисляет Softmax для каждой строки матрицы n×n.
 * Реализованы 4 метода:
 * 1. Sequential - базовая скалярная реализация
 * 2. OpenMP - многопоточная параллелизация
 * 3. SIMD - векторизация с использованием AVX2 инструкций
 * 4. OpenMP+SIMD - гибридный подход
 *
 * @param[in] argc Количество аргументов командной строки
 * @param[in] argv Аргументы командной строки
 * @return EXIT_SUCCESS при успешном выполнении, EXIT_FAILURE при ошибке
 *
 * Пример использования:
 * @code{.sh}
 * ./softmax_cpu 1024           # Тест с матрицей 1024x1024
 * ./softmax_cpu --test         # Запуск тестов корректности
 * ./softmax_cpu --debug 8      # Отладка с матрицей 8x8
 * @endcode
 */

#include <immintrin.h>  // AVX инструкции (Intel Intrinsics)
#include <omp.h>        // OpenMP для параллелизации

#include <algorithm>  // Для std::max, std::min
#include <chrono>  // Для измерения времени: high_resolution_clock
#include <cmath>       // Математические функции: exp, abs
#include <cstdlib>     // Для EXIT_SUCCESS, EXIT_FAILURE
#include <functional>  // Для std::function (коллбэки)
#include <iomanip>  // Для форматирования вывода: setprecision, fixed
#include <iostream>  // Основной ввод-вывод: cout, cerr
#include <random>  // Генерация случайных чисел: mt19937, uniform_real_distribution
#include <sstream>  // Для форматирования строк: ostringstream
#include <stdexcept>  // Исключения: runtime_error, invalid_argument
#include <string>     // Строки std::string
#include <string_view>  // std::string_view (легковесная замена const char*)
#include <vector>  // Динамический массив std::vector

namespace {
// Быстрая векторная экспонента для AVX (аппроксимация полиномом)
// Основана на алгоритме из библиотеки "sse_mathfun.h" (Julien Pommier)
// https://github.com/RJVB/sse_mathfun/blob/master/sse_mathfun.h

static inline __m256 exp256_ps(__m256 x) {
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

  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

// Генерация тестовой матрицы
std::vector<float> make_matrix(std::size_t n) {
  std::vector<float> matrix(n * n);
  std::mt19937 gen(15);  // Фиксированный seed для воспроизводимости
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (auto& x : matrix) {
    x = dist(gen);
  }
  return matrix;
}

// Softmax для одной строки (скалярная версия)
void SoftmaxRow(const float* row_begin, float* row_result, std::size_t n) {
  float sum_exp = 0.0f;
  for (std::size_t j = 0; j < n; ++j) {
    sum_exp += std::exp(row_begin[j]);
  }

  // Защита от деления на ноль (хотя маловероятно при exp(x) > 0)
  if (sum_exp == 0.0f) {
    std::fill(row_result, row_result + n, 1.0f / n);
    return;
  }

  float div_sum_exp = 1.0f / sum_exp;
  for (std::size_t j = 0; j < n; ++j) {
    row_result[j] = std::exp(row_begin[j]) * div_sum_exp;
  }
}

// Сумма 8 float в векторе AVX
static inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(sum128);
}

// Вспомогательные функции для работы с AVX
static inline void storeu256_ps(float* dst, __m256 v) {
  _mm256_storeu_ps(dst, v);
}
static inline __m256 loadu256_ps(const float* src) {
  return _mm256_loadu_ps(src);
}

// Softmax для одной строки (векторизованная версия)
// УСЛОВИЕ ЦИКЛА ПРАВИЛЬНОЕ: i + 7 < n эквивалентно i < n - 7
// При n = 8: i=0 -> 0+7<8=true, i=8 -> 8+7<8=false → обработаны 8 элементов
void SoftmaxRowSimd(const float* row_begin, float* row_result, std::size_t n) {
  if (n == 0) return;

  std::size_t i = 0;
  float sum_exp = 0.0f;

  // Первый проход: вычисляем экспоненты и их сумму
  for (; i + 7 < n; i += 8) {
    __m256 v = loadu256_ps(row_begin + i);
    __m256 e = exp256_ps(v);
    storeu256_ps(row_result + i, e);
    sum_exp += hsum256_ps(e);
  }

  // Обработка хвоста (оставшиеся элементы)
  for (; i < n; ++i) {
    float s = std::exp(row_begin[i]);
    row_result[i] = s;
    sum_exp += s;
  }

  // Защита от деления на ноль
  if (sum_exp == 0.0f) {
    float val = 1.0f / n;
    std::fill(row_result, row_result + n, val);
    return;
  }

  // Второй проход: нормализация
  float inv_sum = 1.0f / sum_exp;
  __m256 inv_vec = _mm256_set1_ps(inv_sum);
  i = 0;

  // Векторизованная нормализация
  for (; i + 7 < n; i += 8) {
    __m256 e = loadu256_ps(row_result + i);
    __m256 r = _mm256_mul_ps(e, inv_vec);
    storeu256_ps(row_result + i, r);
  }

  // Нормализация хвоста
  for (; i < n; ++i) {
    row_result[i] *= inv_sum;
  }
}

// Реализации для разных методов
std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> result(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRow(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

std::vector<float> run_openmp(const std::vector<float>& matrix, std::size_t n) {
  std::vector<float> result(n * n);
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRow(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

std::vector<float> run_simd(const std::vector<float>& matrix, std::size_t n) {
  std::vector<float> result(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRowSimd(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

std::vector<float> run_openmp_simd(const std::vector<float>& matrix,
                                   std::size_t n) {
  std::vector<float> result(n * n);
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    SoftmaxRowSimd(&matrix[i * n], &result[i * n], n);
  }
  return result;
}

// Измерение времени выполнения
double measure_seconds(const std::function<std::vector<float>()>& work,
                       std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

// Проверка корректности: максимальная разница
float max_abs_diff(const std::vector<float>& baseline,
                   const std::vector<float>& candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error("Result size mismatch");
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - candidate[i]));
  }
  return max_diff;
}

// Структура для хранения результатов теста
struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

// Форматирование вывода
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

// Запуск одного теста с проверкой
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

// Тестирование корректности SIMD реализации для различных размеров
void test_simd_correctness() {
  std::cout << "\n=== Тестирование корректности SIMD реализации ===\n";
  std::cout
      << "Проверяем граничные случаи (размеры, кратные 8 и некратные):\n\n";

  // Критические размеры для проверки
  std::vector<std::size_t> test_sizes = {1,  2,  3,  4,  5,   6,   7,
                                         8,  9,  15, 16, 17,  31,  32,
                                         33, 63, 64, 65, 127, 128, 129};

  bool all_tests_passed = true;

  for (std::size_t n : test_sizes) {
    std::cout << "n = " << std::setw(3) << n << ": ";

    auto matrix = make_matrix(n);
    std::vector<float> result_scalar(n * n);
    std::vector<float> result_simd(n * n);

    // Скалярная версия
    for (std::size_t i = 0; i < n; ++i) {
      SoftmaxRow(&matrix[i * n], &result_scalar[i * n], n);
    }

    // SIMD версия
    for (std::size_t i = 0; i < n; ++i) {
      SoftmaxRowSimd(&matrix[i * n], &result_simd[i * n], n);
    }

    // Проверка максимальной разницы
    float max_diff = max_abs_diff(result_scalar, result_simd);

    // Проверка, что суммы строк равны 1 (с небольшой погрешностью)
    bool row_sums_correct = true;
    for (std::size_t i = 0; i < n; ++i) {
      float sum_simd = 0.0f;
      for (std::size_t j = 0; j < n; ++j) {
        sum_simd += result_simd[i * n + j];
      }
      if (std::abs(sum_simd - 1.0f) > 1e-5f) {
        row_sums_correct = false;
        break;
      }
    }

    if (max_diff < 1e-5f && row_sums_correct) {
      std::cout << "✅ ОК (diff = " << std::scientific << max_diff << ")\n";
    } else {
      std::cout << "❌ ПРОБЛЕМА (diff = " << std::scientific << max_diff;
      if (!row_sums_correct) std::cout << ", суммы строк не равны 1";
      std::cout << ")\n";
      all_tests_passed = false;
    }
  }

  if (all_tests_passed) {
    std::cout << "\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ! SIMD реализация корректна.\n";
  } else {
    std::cout << "\n❌ ЕСТЬ ПРОБЛЕМЫ! Требуется отладка SIMD реализации.\n";
  }
}

// Проверка конкретного размера матрицы (для отладки)
void debug_single_test(std::size_t n) {
  std::cout << "\n=== Отладочная информация для n = " << n << " ===\n";
  std::cout << "Количество элементов в строке: " << n << "\n";
  std::cout << "Векторных итераций (по 8 элементов): " << (n + 7) / 8 << "\n";
  std::cout << "Остаточных элементов (хвост): " << n % 8 << "\n\n";

  // Создаем матрицу
  auto matrix = make_matrix(n);

  // Обрабатываем первую строку с отладочным выводом
  std::vector<float> scalar_result(n);
  std::vector<float> simd_result(n);

  SoftmaxRow(&matrix[0], &scalar_result[0], n);
  SoftmaxRowSimd(&matrix[0], &simd_result[0], n);

  std::cout << "Первая строка матрицы (первые 10 элементов):\n";
  for (std::size_t j = 0; j < std::min(n, static_cast<std::size_t>(10)); ++j) {
    std::cout << "  Элемент " << j << ": input=" << std::setprecision(4)
              << matrix[j] << ", scalar=" << scalar_result[j]
              << ", SIMD=" << simd_result[j]
              << ", diff=" << std::abs(scalar_result[j] - simd_result[j])
              << "\n";
  }

  // Проверяем суммы
  float sum_scalar = 0.0f, sum_simd = 0.0f;
  for (std::size_t j = 0; j < n; ++j) {
    sum_scalar += scalar_result[j];
    sum_simd += simd_result[j];
  }

  std::cout << "\nСумма скалярного результата: " << sum_scalar << "\n";
  std::cout << "Сумма SIMD результата: " << sum_simd << "\n";
  std::cout << "Разница сумм: " << std::abs(sum_scalar - sum_simd) << "\n";
}
}  // namespace

int main(int argc, char* argv[]) {
  // Если запуск с флагом --test, выполняем тестирование
  if (argc == 2 && std::string(argv[1]) == "--test") {
    test_simd_correctness();
    return EXIT_SUCCESS;
  }

  // Если запуск с флагом --debug N, выполняем отладку для конкретного размера
  if (argc == 3 && std::string(argv[1]) == "--debug") {
    std::size_t n = static_cast<std::size_t>(std::stoul(argv[2]));
    debug_single_test(n);
    return EXIT_SUCCESS;
  }

  // Обычный режим работы
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      std::cerr << "       " << argv[0] << " --test     (запуск всех тестов)\n";
      std::cerr << "       " << argv[0]
                << " --debug N  (отладка для размера N)\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    const auto input = make_matrix(n);

    // Базовая последовательная версия
    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    // Тестируем оптимизированные версии
    auto omp_res = run_test_case([&] { return run_openmp(input, n); },
                                 sequential_result, "OpenMP");
    auto simd_res = run_test_case([&] { return run_simd(input, n); },
                                  sequential_result, "SIMD");
    auto omp_simd_res = run_test_case([&] { return run_openmp_simd(input, n); },
                                      sequential_result, "OpenMP + SIMD");

    // Вывод результатов
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