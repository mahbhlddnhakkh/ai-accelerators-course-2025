#include <utils.h>

std::chrono::high_resolution_clock::time_point t_start;
timer::timer() { t_start = std::chrono::high_resolution_clock::now(); }
timer::~timer() {}
void timer::reset() { t_start = std::chrono::high_resolution_clock::now(); }
double timer::elapsed() {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             std::chrono::high_resolution_clock::now() - t_start)
      .count();
}

void make_matrix(std::size_t n, std::vector<float> &matrix) {
  // throw std::runtime_error("make_matrix not implemented");
  std::random_device rd;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 gen(rd());
  std::size_t size = n * n;
#pragma omp parallel for
  for (int idx = 0; idx < size; ++idx) {
    matrix[idx] = dist(gen);
  }
}
void print_matrix(const std::size_t n, const std::vector<float> &matrix) {
  for (std::size_t idx_i = 0; idx_i < n; ++idx_i) {
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      std::cout << matrix[idx_i * n + idx_j] << "\t";
    }
    std::cout << "\n";
  }
}
double measure_seconds(const std::function<void()> &work) {
  timer timer;
  work();
  return timer.elapsed();
}
float max_abs_diff(const std::vector<float> &baseline,
                   const std::vector<float> &candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error(
        "Result size mismatch while validating correctness");
  }
  float max_diff = 0.0f;
#pragma omp parallel for
  for (int i = 0; i < baseline.size(); ++i) {
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
void print_report(std::string_view testName, const RunResult &result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}

// Sequential
void run_sequential(const std::vector<float> &input, std::vector<float> &output,
                    std::size_t n) {
  // throw std::runtime_error("Sequential method not implemented");

  for (std::size_t idx_i = 0; idx_i < n; ++idx_i) {
    float current_sum = 0.0f;
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      float current_value = std::exp(input[idx_i * n + idx_j]);
      current_sum += current_value;
      output[idx_i * n + idx_j] = current_value;
    }
    current_sum = 1.0f / current_sum;
    for (std::size_t idx_j = 0; idx_j < n; ++idx_j) {
      output[idx_i * n + idx_j] *= current_sum;
    }
  }
}