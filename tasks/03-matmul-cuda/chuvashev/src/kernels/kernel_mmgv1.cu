#include <utils.h>

__global__ void GPU_MATMUL_V1(const __half *input_A, const __half *input_B,
                              float *output, const std::size_t n) {
  std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n || j >= n) return;

  float sum = 0.0f;

  for (std::size_t idx = 0; idx < n; ++idx) {
    sum +=
        __half2float(input_A[i * n + idx]) * __half2float(input_B[idx * n + j]);
  }

  output[i * n + j] = sum;
}

void run_matrix_mult_gpu_ver_1(const std::vector<__half> &input_A,
                               const std::vector<__half> &input_B,
                               std::vector<float> &output,
                               const std::size_t n) {
  cudaSetDevice(0);

  __half *d_input_A;
  __half *d_input_B;
  float *d_output;

  CHECK_CUDA_ERROR(cudaMalloc(&d_input_A, sizeof(__half) * n * n));
  CHECK_CUDA_ERROR(cudaMalloc(&d_input_B, sizeof(__half) * n * n));
  CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(float) * n * n));

  CHECK_CUDA_ERROR(cudaMemcpy(d_input_A, input_A.data(), sizeof(__half) * n * n,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_input_B, input_B.data(), sizeof(__half) * n * n,
                              cudaMemcpyHostToDevice));

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((n + block_size.x - 1) / block_size.x,
                 (n + block_size.y - 1) / block_size.y);
  timer timer;
  GPU_MATMUL_V1<<<grid_size, block_size>>>(d_input_A, d_input_B, d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  double time = timer.elapsed();
  std::cout << "Time of work MMGV1's kernel: " << time << std::endl;

  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks(n);
  softmax_kernel<<<blocks, threads_per_block>>>(d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaMemcpy(output.data(), d_output, sizeof(float) * n * n,
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(d_input_A));
  CHECK_CUDA_ERROR(cudaFree(d_input_B));
  CHECK_CUDA_ERROR(cudaFree(d_output));
}