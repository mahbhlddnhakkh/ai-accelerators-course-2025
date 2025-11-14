#include <simd_utils.h>
#include <utils.h>

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

    // creating matrix
    std::vector<float> input(n * n, 0);
    make_matrix(n, input);

    // sequential calculating matrix
    std::vector<float> sequential_result(n * n, 0);
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, sequential_result, n); });
    // print_matrix(n, sequential_result);
    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";

    {
      RunResult simt_res;
      simt_res.result.resize(n * n, 0);
      try {
        warmup_cuda(input, n);
        simt_res.seconds = measure_seconds(
            [&]() { return run_cuda_simt(input, simt_res.result, n); });
        simt_res.diff = max_abs_diff(sequential_result, simt_res.result);
        simt_res.success = true;
        // print_matrix(n, simt_res.result);
        // TODO: Compare simt_seconds with the OpenMP+AVX2 timing from practice
        // #1.
      } catch (const std::exception &ex) {
        std::cerr << "CUDA SIMT method failed: " << ex.what() << '\n';
      }
      print_report("SIMT", simt_res);
    }

    {
      RunResult simd_res;
      simd_res.result.resize(n * n, 0);
      try {
        simd_res.seconds = measure_seconds(
            [&]() { return run_openmp_simd(input, simd_res.result, n); });
        simd_res.diff = max_abs_diff(sequential_result, simd_res.result);
        simd_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "OpenMP + SIMD method failed: " << ex.what() << '\n';
      }
      print_report("OpenMP_SIMD", simd_res);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
