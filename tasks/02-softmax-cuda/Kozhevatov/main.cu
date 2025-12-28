/**
 * @file main.cu
 * @brief CUDA реализация Softmax
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Макрос для проверки ошибок CUDA
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
 * @brief Генерация случайной квадратной матрицы заданного размера
 * @param n Размер матрицы (n x n)
 * @return Вектор размером n*n, содержащий элементы матрицы в row-major порядке
 * @note Используется случайное начальное значение (random_device) для различных
 * запусков
 */
std::vector<float> make_matrix(std::size_t n) {
  std::vector<float> matrix(n * n);

  static std::random_device ran_dev;
  static std::mt19937 ran_eng(ran_dev());
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  for (std::size_t i = 0; i < n * n; ++i) {
    matrix[i] = dist(ran_eng);
  }

  return matrix;
}

/**
 * @brief Вычисление Softmax для одной строки (CPU реализация)
 * @param input_row Указатель на входную строку
 * @param output_row Указатель на выходную строку
 * @param n Длина строки
 * @details Вычисляет сумму экспонент всех элементов строки,
 *          затем нормализует каждый элемент делением на эту сумму
 */
static void row_calculation_amount(const float* input_row, std::size_t n,
                                   float* output_row) {
  float row_sum = 0.0f;
  for (std::size_t item = 0; item < n; item++) {
    row_sum += exp(input_row[item]);
  }
  for (std::size_t item = 0; item < n; item++) {
    output_row[item] = exp(input_row[item]) / row_sum;
  }
}

/**
 * @brief Последовательная (CPU) реализация Softmax для всей матрицы
 * @param matrix Входная матрица в row-major порядке
 * @param n Размер матрицы (n x n)
 * @return Матрица после применения Softmax к каждой строке
 * @details Применяет row_calculation_amount к каждой строке матрицы независимо
 */
std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> res_matrix(n * n);

  for (std::size_t row = 0; row < n; row++) {
    row_calculation_amount(&matrix[row * n], n, &res_matrix[row * n]);
  }
  return res_matrix;
}

/**
 * @brief CUDA ядро для вычисления Softmax
 * @param input Указатель на входные данные в глобальной памяти GPU
 * @param output Указатель на выходные данные в глобальной памяти GPU
 * @param n Размер матрицы (n x n)
 * @details Каждый блок обрабатывает одну строку матрицы.
 *          Использует shared memory для редукции суммы экспонент.
 *          Выполняет параллельное вычисление экспонент и их суммы,
 *          затем нормализует значения.
 */
__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  extern __shared__ float s_data[];

  const int row_idx = blockIdx.x;
  const int tid = threadIdx.x;

  const float* row = input + row_idx * n;
  float* res = output + row_idx * n;

  // Фаза 1: Вычисление экспонент и частичных сумм
  float thread_sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    float val = __expf(row[i]);
    res[i] = val;  // Сохраняем экспоненту
    thread_sum += val;
  }

  // Сохраняем частичную сумму потока
  s_data[tid] = thread_sum;
  __syncthreads();

  // Фаза 2: Редукция для получения общей суммы строки
  // Используем tree reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_data[tid] += s_data[tid + stride];
    }
    __syncthreads();
  }

  // Фаза 3: Нормализация
  float row_sum = s_data[0];
  if (row_sum != 0.0f) {  // Избегаем деления на 0
    float inv_row_sum = 1.0f / row_sum;
    for (int i = tid; i < n; i += blockDim.x) {
      res[i] *= inv_row_sum;
    }
  }
}

/**
 * @brief Запуск CUDA ядра для вычисления Softmax
 * @param d_input Указатель на входные данные в памяти GPU
 * @param d_output Указатель на выходные данные в памяти GPU
 * @param n Размер матрицы (n x n)
 * @param stream CUDA поток для асинхронного выполнения
 * @details Конфигурирует параметры запуска ядра: количество блоков, потоков,
 *          выделяет shared memory и запускает ядро.
 */
void launch_softmax_kernel(const float* d_input, float* d_output, std::size_t n,
                           cudaStream_t stream) {
  // Лучше адаптировать количество потоков под размер строки
  int threads_per_block = 256;
  if (n < 256) {
    threads_per_block = 64;
  }
  // Округляем до степени двойки для эффективной редукции
  threads_per_block = min(512, max(32, (int)pow(2, ceil(log2(n)))));

  const int shared_mem_size = threads_per_block * sizeof(float);

  softmax_kernel<<<n, threads_per_block, shared_mem_size, stream>>>(
      d_input, d_output, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
}

/**
 * @brief Прогрев GPU для стабилизации производительности
 * @param matrix Входная матрица (не используется, оставлен для совместимости)
 * @param n Размер матрицы (не используется, оставлен для совместимости)
 * @details Выполняет простые операции на GPU для инициализации контекста
 *          и стабилизации частот перед основными измерениями.
 */
void warmup_cuda(const std::vector<float>& matrix, std::size_t n) {
  // Параметры matrix и n не используются, но оставлены для совместимости
  (void)matrix;
  (void)n;

  float* d_tmp = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, sizeof(float)));
  CHECK_CUDA_ERROR(cudaFree(d_tmp));
}

/**
 * @brief CUDA реализация Softmax с использованием SIMT подхода
 * @param matrix Входная матрица в row-major порядке
 * @param n Размер матрицы (n x n)
 * @return Матрица после применения Softmax к каждой строке
 * @details Копирует данные на GPU, выполняет ядро, копирует результаты обратно.
 *          Использует один CUDA поток для асинхронного выполнения.
 */
std::vector<float> run_cuda_simt(const std::vector<float>& matrix,
                                 std::size_t n) {
  const std::size_t size = n * n * sizeof(float);
  float* d_input = nullptr;
  float* d_output = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_input, matrix.data(), size, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
  launch_softmax_kernel(d_input, d_output, n, stream);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::vector<float> result(n * n);
  CHECK_CUDA_ERROR(
      cudaMemcpy(result.data(), d_output, size, cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));

  return result;
}

/**
 * @brief Вычисление максимальной абсолютной разницы между двумя векторами
 * @param baseline Опорный вектор (обычно результат CPU реализации)
 * @param candidate Вектор для сравнения (обычно результат GPU реализации)
 * @return Максимальная абсолютная разница между соответствующими элементами
 * @throws std::runtime_error если размеры векторов не совпадают
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
 * @brief Структура для хранения результатов выполнения
 */
struct RunResult {
  std::vector<float> result;
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

/**
 * @brief Форматирование времени для вывода
 * @param seconds Время в секундах
 * @return Строковое представление времени с фиксированной точностью (5 знаков)
 */
std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(5) << seconds;
  return oss.str();
}

/**
 * @brief Форматирование разницы для вывода
 * @param diff Значение разницы
 * @return Строковое представление разницы с научной нотацией (5 знаков)
 */
std::string format_diff(float diff) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(5) << diff;
  return oss.str();
}

/**
 * @brief Вывод отчета о выполнении теста
 * @param testName Название теста
 * @param result Результаты выполнения
 */
void print_report(const std::string& testName, const RunResult& result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}
}  // namespace

/**
 * @brief Основная функция программы
 * @param argc Количество аргументов командной строки
 * @param argv Массив аргументов командной строки
 * @return EXIT_SUCCESS при успешном выполнении, EXIT_FAILURE при ошибке
 * @details Программа принимает один аргумент - размер матрицы n (n x n).
 *          Выполняет последовательную и CUDA реализации Softmax,
 *          сравнивает результаты и выводит время выполнения.
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

    const auto input = make_matrix(n);
    std::vector<float> sequential_result;

    // Time sequential version
    const auto sequential_start = std::chrono::high_resolution_clock::now();
    sequential_result = run_sequential(input, n);
    const auto sequential_stop = std::chrono::high_resolution_clock::now();
    const double sequential_seconds =
        std::chrono::duration<double>(sequential_stop - sequential_start)
            .count();

    RunResult simt_res;
    try {
      warmup_cuda(input, n);

      // Time CUDA version
      const auto cuda_start = std::chrono::high_resolution_clock::now();
      simt_res.result = run_cuda_simt(input, n);
      const auto cuda_stop = std::chrono::high_resolution_clock::now();
      simt_res.seconds =
          std::chrono::duration<double>(cuda_stop - cuda_start).count();

      simt_res.diff = max_abs_diff(sequential_result, simt_res.result);
      simt_res.success = true;
    } catch (const std::exception& ex) {
      std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
    }

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("SIMT", simt_res);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor
              << "\n";
    return EXIT_SUCCESS;

  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }
}