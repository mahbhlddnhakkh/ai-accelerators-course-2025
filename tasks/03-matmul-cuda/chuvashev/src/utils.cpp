#include <utils.h>

// Utils funcitons
std::chrono::high_resolution_clock::time_point t_start;
timer::timer() { t_start = std::chrono::high_resolution_clock::now(); }
timer::~timer() {}
void timer::reset() { t_start = std::chrono::high_resolution_clock::now(); }
double timer::elapsed() {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             std::chrono::high_resolution_clock::now() - t_start)
      .count();
}

template <typename T>
void print_matrix(const std::vector<T> &matrix, const std::size_t n) {
  for (std::size_t idx_i = 0; idx_i < n; ++idx_i) {
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      std::cout << (float)(matrix[idx_i * n + idx_j]) << "\t";
    }
    std::cout << "\n";
  }
}
void make_input_matrix(std::vector<__half> &matrix, std::size_t n) {
  // throw std::runtime_error("make_input_matrix not implemented");
  std::random_device rd;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 gen(rd());
#pragma omp parallel for
  for (int idx = 0; idx < n * n; ++idx) {
    matrix[idx] = __float2half(dist(gen));
  }
}
void print_report(std::string_view testName, const RunResult &result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}

double measure_seconds(const std::function<void()> &work) {
  const auto start = std::chrono::high_resolution_clock::now();
  work();
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

void run_openmp_reference(const std::vector<__half> &input_A,
                          const std::vector<__half> &input_B,
                          std::vector<float> &output, const std::size_t n) {
  // throw std::runtime_error("OpenMP reference not implemented");

  int block_size = 32;
  int count_of_blocks = (n + block_size - 1) / block_size;

#pragma omp parallel for
  for (int ii = 0; ii < count_of_blocks; ++ii) {
    int i_start = ii * block_size;
    int i_end = (std::min)((ii + 1) * block_size, (int)n);
    for (int jj = 0; jj < count_of_blocks; ++jj) {
      int j_start = jj * block_size;
      int j_end = (std::min)((jj + 1) * block_size, (int)n);
      for (int kk = 0; kk < count_of_blocks; ++kk) {
        int k_start = kk * block_size;
        int k_end = (std::min)((kk + 1) * block_size, (int)n);
        for (int i = i_start; i < i_end; ++i) {
          for (int k = k_start; k < k_end; ++k) {
            float value = __half2float(input_A[i * n + k]);
            for (int j = j_start; j < j_end; ++j) {
              output[i * n + j] += value * __half2float(input_B[k * n + j]);
            }
          }
        }
      }
    }
  }

#pragma omp parallel for
  for (int idx_i = 0; idx_i < n; ++idx_i) {
    float current_row_sum = 0;
    for (std::size_t idx = 0; idx < n; ++idx) {
      float current_value = std::exp(output[idx_i * n + idx]);
      current_row_sum += current_value;
      output[idx_i * n + idx] = current_value;
    }
    current_row_sum = 1.0f / current_row_sum;
    for (std::size_t idx = 0; idx < n; ++idx) {
      output[idx_i * n + idx] *= current_row_sum;
    }
  }
}