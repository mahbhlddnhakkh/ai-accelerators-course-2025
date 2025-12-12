#include <utils.h>

__global__ void GPU_MATMUL_V2(const __half *input_A, const __half *input_B,
                              float *output, const std::size_t n) {
  std::size_t block_row = blockIdx.y;
  std::size_t block_col = blockIdx.x;

  std::size_t local_row = threadIdx.y;
  std::size_t local_col = threadIdx.x;

  std::size_t row = block_row * blockDim.y + local_row;
  std::size_t col = block_col * blockDim.x + local_col;

  if (row >= n || col >= n) return;

  if (local_row >= BLOCK_SIZE || local_col >= BLOCK_SIZE) return;

  __shared__ float block_a[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float block_b[BLOCK_SIZE * BLOCK_SIZE];

  float sum = 0.0f;

  for (std::size_t block = 0; block < gridDim.x; ++block) {
    block_a[local_row * BLOCK_SIZE + local_col] =
        (input_A[row * n + block * BLOCK_SIZE + local_col]);
    block_b[local_row * BLOCK_SIZE + local_col] =
        (input_B[col + (block * BLOCK_SIZE + local_row) * n]);

    __syncthreads();

    for (std::size_t k = 0; k < BLOCK_SIZE; ++k) {
      sum += (block_a[local_row * BLOCK_SIZE + k]) *
             (block_b[k * BLOCK_SIZE + local_col]);
    }

    __syncthreads();
  }

  output[row * n + col] = sum;
}

void run_matrix_mult_gpu_ver_2(const std::vector<__half> &input_A,
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
  GPU_MATMUL_V2<<<grid_size, block_size>>>(d_input_A, d_input_B, d_output, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  double time = timer.elapsed();
  std::cout << "Time of work MMGV2's kernel: " << time << std::endl;

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