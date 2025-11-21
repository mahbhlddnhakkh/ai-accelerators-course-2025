#include <cutlass/gemm/device/gemm.h>
#include <utils.h>

void warmup_cutlass(const std::vector<__half> &input,
                    std::vector<float> &output, const std::size_t n) {
  // throw std::runtime_error("CUTLASS warm-up not implemented");
  run_cutlass(input, output, n);
}

cudaError_t CutlassGEMM(const cutlass::half_t *input, float *output,
                        const std::size_t n) {
  using RowMajor = cutlass::layout::RowMajor;
  using OpClassTesorOp = cutlass::arch::OpClassTensorOp;
  using Sm75 = cutlass::arch::Sm75;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t, RowMajor, cutlass::half_t,
                                  RowMajor, float, RowMajor, float,
                                  OpClassTesorOp, Sm75>;

  CutlassGemm::Arguments args(
      {cutlass::gemm::GemmCoord::Index(n), cutlass::gemm::GemmCoord::Index(n),
       cutlass::gemm::GemmCoord::Index(n)},
      {input, n}, {input, n}, {output, n}, {output, n}, {1, 0});

  CutlassGemm gemm_operator;
  cutlass::Status status = gemm_operator(args);
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

void run_cutlass(const std::vector<__half> &input, std::vector<float> &output,
                 std::size_t n) {
  // throw std::runtime_error("CUTLASS method not implemented");

  cutlass::half_t *d_input;
  float *d_output;

  CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * n * sizeof(__half)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * n * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data(), n * n * sizeof(__half),
                              cudaMemcpyHostToDevice));

  timer timer;
  cudaError_t status = CutlassGEMM(d_input, d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  double time = timer.elapsed();
  std::cout << "Time of work CUTLASS's kernel: " << time << std::endl;

  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks(n);
  softmax_kernel<<<blocks, threads_per_block>>>(d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, n * n * sizeof(float),
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_input));
  CHECK_CUDA_ERROR(cudaFree(d_output));
}