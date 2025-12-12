#include <utils.h>

int main(int argc, char *argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    if (n % 32 != 0) {
      n = ((n / 32) + 1) * 32;
    }

    std::vector<__half> input_A(n * n, 0);
    std::vector<__half> input_B(n * n, 0);

    make_input_matrix(input_A, n);
    make_input_matrix(input_B, n);

    // print_matrix<__half>(input_A, n);
    // print_matrix<__half>(input_B, n);

    std::vector<float> openmp_result(n * n, 0);
    const double openmp_seconds = measure_seconds([&]() {
      return run_openmp_reference(input_A, input_B, openmp_result, n);
    });
    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec\n";
    // print_matrix<float>(openmp_result, n);

    {
      RunResult mmgv1_res;
      mmgv1_res.result.resize(n * n, 0);
      try {
        run_matrix_mult_gpu_ver_1(input_A, input_B, mmgv1_res.result, n);
        mmgv1_res.result.clear();
        mmgv1_res.result.resize(n * n, 0);
        mmgv1_res.seconds = measure_seconds([&]() {
          return run_matrix_mult_gpu_ver_1(input_A, input_B, mmgv1_res.result,
                                           n);
        });
        mmgv1_res.diff = max_abs_diff(openmp_result, mmgv1_res.result);
        mmgv1_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "MMGV1 method failed: " << ex.what() << '\n';
      }
      print_report("MMGV1", mmgv1_res);
      // print_matrix<float>(mmgv1_res.result, n);
    }

    {
      RunResult mmgv2_res;
      mmgv2_res.result.resize(n * n, 0);
      try {
        run_matrix_mult_gpu_ver_2(input_A, input_B, mmgv2_res.result, n);
        mmgv2_res.result.clear();
        mmgv2_res.result.resize(n * n, 0);
        mmgv2_res.seconds = measure_seconds([&]() {
          return run_matrix_mult_gpu_ver_2(input_A, input_B, mmgv2_res.result,
                                           n);
        });
        mmgv2_res.diff = max_abs_diff(openmp_result, mmgv2_res.result);
        mmgv2_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "MMGV2 method failed: " << ex.what() << '\n';
      }
      print_report("MMGV2", mmgv2_res);
      // print_matrix<float>(mmgv1_res.result, n);
    }

    {
      RunResult wmma_res;
      wmma_res.result.resize(n * n, 0);
      try {
        if (n % 32 != 0)
          throw std::runtime_error("WMMA isn't working with this size.");
        warmup_wmma(input_A, input_B, wmma_res.result, n);
        wmma_res.result.clear();
        wmma_res.result.resize(n * n, 0);
        wmma_res.seconds = measure_seconds(
            [&]() { return run_wmma(input_A, input_B, wmma_res.result, n); });
        wmma_res.diff = max_abs_diff(openmp_result, wmma_res.result);
        wmma_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "WMMA method failed: " << ex.what() << '\n';
      }
      print_report("WMMA", wmma_res);
    }

    {
      RunResult cutlass_res;
      cutlass_res.result.resize(n * n, 0);
      try {
        if (n % 32 != 0)
          throw std::runtime_error("CUTLASS isn't working with this size.");
        warmup_cutlass(input_A, input_B, cutlass_res.result, n);
        cutlass_res.result.clear();
        cutlass_res.result.resize(n * n, 0);
        cutlass_res.seconds = measure_seconds([&]() {
          return run_cutlass(input_A, input_B, cutlass_res.result, n);
        });
        cutlass_res.diff = max_abs_diff(openmp_result, cutlass_res.result);
        cutlass_res.success = true;
      } catch (const std::exception &ex) {
        std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
      }
      print_report("CUTLASS", cutlass_res);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
