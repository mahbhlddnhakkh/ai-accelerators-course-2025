/**
 * @file matmul_softmax.cu
 * @brief Реализация матричного умножения с последующим Softmax с использованием
 * различных вычислительных методов
 *
 * Программа вычисляет матричное умножение двух квадратных матриц размера n×n с
 * последующим применением Softmax к каждой строке результата. Реализованы 3
 * метода:
 * 1. OpenMP - многопоточная параллелизация на CPU (эталонная реализация)
 * 2. WMMA API - использование Tensor Cores через CUDA WMMA API
 * 3. CUTLASS API - использование высокопроизводительной библиотеки CUTLASS
 *
 * @param[in] argc Количество аргументов командной строки
 * @param[in] argv Аргументы командной строки (размер матрицы n)
 * @return EXIT_SUCCESS при успешном выполнении, EXIT_FAILURE при ошибке
 *
 * @note Входные матрицы имеют тип half (FP16), выходные данные - тип float
 * (FP32)
 * @note Время измеряется с включением всех операций (копирование данных,
 * вычисления)
 *
 * Пример использования:
 * @code{.sh}
 * ./matmul_softmax 1024           # Тест с матрицей 1024x1024
 * @endcode
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
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

/**
 * @def CHECK_CUDA_ERROR
 * @brief Макрос для проверки ошибок CUDA
 *
 * @param callable CUDA API вызов
 *
 * @details При обнаружении ошибки выводит подробную информацию и завершает
 * программу
 */
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

/**
 * @brief Генерация входных матриц случайных значений
 *
 * @param n Размер квадратных матриц (n×n)
 * @return std::vector<__half> Вектор, содержащий две матрицы размера n×n в
 * формате half
 *
 * @details Генерируются две матрицы A и B, которые объединяются в один вектор:
 *          [A(n×n) | B(n×n)]. Значения равномерно распределены в диапазоне
 * [-1.0, 1.0]
 */
std::vector<__half> make_input_matrix(std::size_t n) {
  std::vector<__half> matrix(2 * n * n);

  std::random_device rd;
  std::mt19937 gen(42);  // Фиксированный seed для воспроизводимости
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (auto& x : matrix) {
    x = __float2half(dist(gen));
  }

  return matrix;
}

/**
 * @brief Эталонная реализация матричного умножения и Softmax с использованием
 * OpenMP
 *
 * @param matrix Входной вектор, содержащий две матрицы A и B
 * @param n Размер матриц (n×n)
 * @return std::vector<float> Результат умножения A×B с применением Softmax к
 * каждой строке
 *
 * @details Реализация включает:
 *          - Параллелизация по строкам с помощью OpenMP
 *          - Векторизация внутренних циклов с помощью SIMD директив
 *          - Построчное применение Softmax после умножения
 */
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

/**
 * @brief CUDA ядро для вычисления Softmax по строкам
 *
 * @param[in] input Входная матрица (уже вычисленные экспоненты)
 * @param[out] output Выходная матрица (нормализованные значения)
 * @param n Размер матрицы (n×n)
 *
 * @details Каждый блок обрабатывает одну строку матрицы:
 *          - Потоки вычисляют частичные суммы экспонент
 *          - Используется shared memory для редукции сумм
 *          - Результат нормализуется делением на сумму строки
 */
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

  // Параллельная редукция для суммирования всех local_sum
  for (unsigned stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  float row_sum = sdata[0];
  __syncthreads();

  // Нормализация
  for (std::size_t i = tid; i < n; i += block_size) {
    row_output[i] /= row_sum;
  }
}

/**
 * @brief Ядро WMMA для матричного умножения с использованием Tensor Cores
 *
 * @param[in] A Входная матрица A (тип half)
 * @param[in] B Входная матрица B (тип half)
 * @param[out] C Выходная матрица C (тип float)
 * @param M Количество строк в матрице A и C
 * @param N Количество столбцов в матрице B и C
 * @param K Количество столбцов в A и строк в B
 *
 * @details Использует WMMA API для задействования Tensor Cores:
 *          - Работает с фрагментами 16x16x16
 *          - Каждый блок обрабатывает один тайл 16x16
 *          - Использует row-major layout для всех матриц
 */
__global__ void wmma_gemm_kernel(const __half* A, const __half* B, float* C,
                                 int M, int N, int K) {
  // Размер tile для WMMA
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  // Определяем, какой tile обрабатывает этот блок
  const int tileM = (blockIdx.x * WMMA_M) / WMMA_M;
  const int tileN = (blockIdx.y * WMMA_N) / WMMA_N;

  if (tileM * WMMA_M >= M || tileN * WMMA_N >= N) {
    return;
  }

  // Объявляем фрагменты для WMMA
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                         nvcuda::wmma::row_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         float>
      c_frag;

  // Инициализируем аккумулятор нулями
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);

  // Умножение матриц с использованием Tensor Cores
  for (int k = 0; k < K; k += WMMA_K) {
    int aRow = tileM * WMMA_M;
    int aCol = k;
    int bRow = k;
    int bCol = tileN * WMMA_N;

    // Проверка границ
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      nvcuda::wmma::load_matrix_sync(a_frag, &A[aRow * K + aCol], K);
      nvcuda::wmma::load_matrix_sync(b_frag, &B[bRow * N + bCol], N);
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  // Сохраняем результат
  int cRow = tileM * WMMA_M;
  int cCol = tileN * WMMA_N;
  if (cRow < M && cCol < N) {
    nvcuda::wmma::store_matrix_sync(&C[cRow * N + cCol], c_frag, N,
                                    nvcuda::wmma::mem_row_major);
  }
}

/**
 * @brief Реализация матричного умножения и Softmax с использованием WMMA API
 *
 * @param matrix Входные матрицы A и B в формате half
 * @param n Размер матриц (n×n)
 * @return std::vector<float> Результат с применением Softmax
 *
 * @throws std::runtime_error Если размер входных данных некорректен
 *
 * @details Особенности реализации:
 *          - Автоматическое выравнивание размеров до кратного 16 для WMMA
 *          - Копирование данных с паддингом и обратно
 *          - Последовательный вызов: GEMM → Softmax
 */
std::vector<float> run_wmma(const std::vector<__half>& matrix, std::size_t n) {
  if (matrix.size() != 2ull * n * n) {
    throw std::runtime_error(
        "Input matrix size must be exactly 2*n*n (A then B)");
  }

  int N = static_cast<int>(n);

  // Для WMMA размер должен быть кратен 16
  const int WMMA_TILE = 16;
  int N_aligned = ((N + WMMA_TILE - 1) / WMMA_TILE) * WMMA_TILE;

  __half* d_A = nullptr;
  __half* d_B = nullptr;
  float* d_C = nullptr;
  float* d_temp = nullptr;

  size_t bytes_half =
      static_cast<size_t>(N_aligned * N_aligned) * sizeof(__half);
  size_t bytes_float =
      static_cast<size_t>(N_aligned * N_aligned) * sizeof(float);
  size_t bytes_float_original = static_cast<size_t>(N * N) * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_float));
  CHECK_CUDA_ERROR(cudaMalloc(&d_temp, bytes_float_original));

  const __half* h_A = matrix.data();
  const __half* h_B = matrix.data() + static_cast<size_t>(N * N);

  // Инициализируем память нулями
  CHECK_CUDA_ERROR(cudaMemset(d_A, 0, bytes_half));
  CHECK_CUDA_ERROR(cudaMemset(d_B, 0, bytes_half));
  CHECK_CUDA_ERROR(cudaMemset(d_C, 0, bytes_float));

  // Копируем данные с выравниванием
  for (int i = 0; i < N; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_A + i * N_aligned, h_A + i * N,
                                N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B + i * N_aligned, h_B + i * N,
                                N * sizeof(__half), cudaMemcpyHostToDevice));
  }

  // Настройка grid и block для WMMA
  dim3 blockDim(32, 8);  // 256 потоков на блок (для WMMA 16x16)
  dim3 gridDim((N_aligned + 15) / 16, (N_aligned + 15) / 16);

  wmma_gemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N_aligned, N_aligned,
                                          N_aligned);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Копируем результат без паддинга обратно
  for (int i = 0; i < N; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_temp + i * N, d_C + i * N_aligned,
                                N * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // Применяем softmax
  const int threads_per_block = 256;
  const int blocks = N;
  const size_t shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_temp, d_temp,
                                                                 N);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::vector<float> result(n * n);
  CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_temp, bytes_float_original,
                              cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_temp);

  return result;
}

/**
 * @brief Функция разогрева для WMMA реализации
 *
 * @param matrix Входные матрицы
 * @param n Размер матриц
 *
 * @details Выполняет один проход вычислений для инициализации кэша и компиляции
 * ядра
 */
void warmup_wmma(const std::vector<__half>& matrix, std::size_t n) {
  run_wmma(matrix, n);
}

/**
 * @brief Реализация матричного умножения и Softmax с использованием CUTLASS API
 *
 * @param matrix Входные матрицы A и B в формате half
 * @param n Размер матриц (n×n)
 * @return std::vector<float> Результат с применением Softmax
 *
 * @throws std::runtime_error Если размер входных данных некорректен или CUTLASS
 * завершается с ошибкой
 *
 * @details Особенности реализации:
 *          - Использует Tensor Cores через CUTLASS
 *          - Автоматическое выравнивание размеров до кратного 8
 *          - Оптимизированная работа с памятью
 *          - Поддержка архитектур Turing (Sm75) и Ampere (Sm80)
 */
std::vector<float> run_cutlass(const std::vector<__half>& matrix,
                               std::size_t n) {
  if (matrix.size() != 2ull * n * n) {
    throw std::runtime_error(
        "Input matrix size must be exactly 2*n*n (A then B)");
  }

  const int N = static_cast<int>(n);

  // Для CUTLASS с Tensor Cores нужен размер кратен 8 (для FP16)
  const int CUTLASS_TILE = 8;
  int N_aligned = ((N + CUTLASS_TILE - 1) / CUTLASS_TILE) * CUTLASS_TILE;

  __half *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_temp = nullptr;

  size_t bytes_half =
      static_cast<size_t>(N_aligned * N_aligned) * sizeof(__half);
  size_t bytes_float =
      static_cast<size_t>(N_aligned * N_aligned) * sizeof(float);
  size_t bytes_float_original = static_cast<size_t>(N * N) * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_half));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_float));
  CHECK_CUDA_ERROR(cudaMalloc(&d_temp, bytes_float_original));

  // Инициализируем память нулями
  CHECK_CUDA_ERROR(cudaMemset(d_A, 0, bytes_half));
  CHECK_CUDA_ERROR(cudaMemset(d_B, 0, bytes_half));
  CHECK_CUDA_ERROR(cudaMemset(d_C, 0, bytes_float));

  const __half* h_A = matrix.data();
  const __half* h_B = matrix.data() + static_cast<size_t>(N * N);

  // Копируем данные с выравниванием
  for (int i = 0; i < N; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_A + i * N_aligned, h_A + i * N,
                                N * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B + i * N_aligned, h_B + i * N,
                                N * sizeof(__half), cudaMemcpyHostToDevice));
  }

  // Используем CUTLASS для умножения с Tensor Cores
  // Sm75 - Turing, Sm80 - Ampere, подставьте свою архитектуру
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,                 // Тип A
      cutlass::layout::RowMajor,       // Layout A
      cutlass::half_t,                 // Тип B
      cutlass::layout::RowMajor,       // Layout B
      float,                           // Тип C
      cutlass::layout::RowMajor,       // Layout C
      float,                           // Тип аккумулятора
      cutlass::arch::OpClassTensorOp,  // Используем Tensor Cores
      cutlass::arch::Sm75  // Архитектура (Sm75 для Turing, Sm80 для Ampere)
      >;

  cutlass::gemm::GemmCoord problem_size(N_aligned, N_aligned, N_aligned);

  typename Gemm::Arguments arguments(
      problem_size, {reinterpret_cast<cutlass::half_t*>(d_A), N_aligned},
      {reinterpret_cast<cutlass::half_t*>(d_B), N_aligned}, {d_C, N_aligned},
      {d_C, N_aligned}, {1.0f, 0.0f}  // alpha, beta
  );

  Gemm gemm_op;

  // Инициализируем GEMM операцию
  cutlass::Status status = gemm_op.initialize(arguments);

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS initialization failed: " +
                             std::to_string(static_cast<int>(status)));
  }

  // Запускаем GEMM
  status = gemm_op();

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS GEMM failed: " +
                             std::to_string(static_cast<int>(status)));
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // Копируем результат без паддинга
  for (int i = 0; i < N; ++i) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_temp + i * N, d_C + i * N_aligned,
                                N * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // Применяем softmax
  const int threads_per_block = 256;
  const int blocks = N;
  const size_t shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_temp, d_temp,
                                                                 N);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  std::vector<float> result(N * N);
  CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_temp, bytes_float_original,
                              cudaMemcpyDeviceToHost));

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_temp);

  return result;
}

/**
 * @brief Функция разогрева для CUTLASS реализации
 *
 * @param matrix Входные матрицы
 * @param n Размер матриц
 *
 * @details Выполняет один проход вычислений для инициализации кэша и компиляции
 * ядра
 */
void warmup_cutlass(const std::vector<__half>& matrix, std::size_t n) {
  run_cutlass(matrix, n);
}

/**
 * @brief Измерение времени выполнения функции
 *
 * @param work Функция для выполнения и измерения времени
 * @param result_store Ссылка для сохранения результата работы функции
 * @return double Время выполнения в секундах
 */
double measure_seconds(const std::function<std::vector<float>()>& work,
                       std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

/**
 * @brief Вычисление максимальной абсолютной разницы между двумя векторами
 *
 * @param baseline Эталонный вектор (результат OpenMP)
 * @param candidate Проверяемый вектор (результат WMMA или CUTLASS)
 * @return float Максимальная абсолютная разница
 *
 * @throws std::runtime_error Если размеры векторов не совпадают
 */
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

/**
 * @brief Форматирование времени для вывода
 *
 * @param seconds Время в секундах
 * @return std::string Отформатированная строка с фиксированной точностью
 */
std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6) << seconds;
  return oss.str();
}

/**
 * @brief Форматирование разницы в научной нотации
 *
 * @param diff Значение разницы
 * @return std::string Отформатированная строка в формате scientific
 */
std::string format_diff_scientific(float diff) {
  if (diff == 0.0f) {
    return "0.00e+00";
  }

  std::ostringstream oss;
  int exponent = static_cast<int>(std::floor(std::log10(std::abs(diff))));
  float value = diff * std::pow(10.0f, -exponent);

  oss << std::fixed << std::setprecision(2) << value << "e";
  if (exponent >= 0) {
    oss << "+";
  }
  oss << exponent;

  return oss.str();
}

}  // namespace

/**
 * @brief Главная функция программы
 *
 * Обрабатывает аргументы командной строки, запускает вычисления
 * тремя методами и выводит результаты с временем выполнения и точностью.
 */
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

    // Запуск эталонной реализации (OpenMP)
    std::vector<float> openmp_result;
    const double openmp_seconds = measure_seconds(
        [&]() { return run_openmp_reference(input, n); }, openmp_result);

    std::vector<float> wmma_result;
    double wmma_seconds = 0.0;
    float wmma_diff = 0.0f;
    bool wmma_success = false;

    // Запуск WMMA реализации
    try {
      warmup_wmma(input, n);
      wmma_seconds =
          measure_seconds([&]() { return run_wmma(input, n); }, wmma_result);
      wmma_diff = max_abs_diff(openmp_result, wmma_result);
      wmma_success = true;
    } catch (const std::exception& ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    std::vector<float> cutlass_result;
    double cutlass_seconds = 0.0;
    float cutlass_diff = 0.0f;
    bool cutlass_success = false;

    // Запуск CUTLASS реализации
    try {
      warmup_cutlass(input, n);
      cutlass_seconds = measure_seconds([&]() { return run_cutlass(input, n); },
                                        cutlass_result);
      cutlass_diff = max_abs_diff(openmp_result, cutlass_result);
      cutlass_success = true;
    } catch (const std::exception& ex) {
      std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
    }

    // Вывод результатов
    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec"
              << std::endl;

    if (wmma_success) {
      std::cout << "WMMA: " << format_time(wmma_seconds)
                << " sec (diff: " << format_diff_scientific(wmma_diff) << ")"
                << std::endl;
    }

    if (cutlass_success) {
      std::cout << "CUTLASS: " << format_time(cutlass_seconds)
                << " sec (diff: " << format_diff_scientific(cutlass_diff) << ")"
                << std::endl;
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}