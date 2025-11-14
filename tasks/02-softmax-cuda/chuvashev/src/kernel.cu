#include <utils.h>

// CUDA

__global__ void warmup_kernel(float *d_matrix, const std::size_t n) {
  std::size_t row = blockIdx.y;
  std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    std::size_t index_of_start_elem = row * n + col;
    d_matrix[index_of_start_elem] = 2.0f * d_matrix[index_of_start_elem];
  }
}
void warmup_cuda(const std::vector<float> &matrix, std::size_t n) {
  // throw std::runtime_error("CUDA warm-up not implemented");
  CHECK_CUDA_ERROR(cudaSetDevice(0));
  std::size_t byte_size_memory = n * n * sizeof(float);
  float *d_matrix;
  CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, byte_size_memory));
  CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(), byte_size_memory,
                              cudaMemcpyHostToDevice));
  dim3 threds_per_block(1024);
  dim3 blocks(n / 1024, n);
  // timer timer;
  warmup_kernel<<<blocks, threds_per_block>>>(d_matrix, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  /*double time = timer.elapsed();
  std::cout << time << "\tGB/s: "
            << n * n * sizeof(float) * 2.0f /
                   (1024.0f * 1024.0f * 1024.0f * time)
            << std::endl;*/
  CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

__global__ void softmax_kernel(float *d_matrix, size_t n) {
  size_t row = blockIdx.x;

  __shared__ float smem[THREADS_PER_BLOCK];
  smem[threadIdx.x] = 0.0f;
  __syncthreads();

  if (row < n) {
    for (std::size_t col = threadIdx.x; col < n; col += blockDim.x) {
      size_t index = row * n + col;
      smem[threadIdx.x] += expf(d_matrix[index]);
    }
  }

  __syncthreads();

  /*if (threadIdx.x == 0) {
    float local_sum = 0.0f;
    for (std::size_t idx = 0; idx < THREADS_PER_BLOCK; ++idx) {
      local_sum += smem[idx];
    }
    smem[0] = local_sum;
  }
  __syncthreads();*/

  for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smem[threadIdx.x] += smem[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (row < n) {
    for (std::size_t col = threadIdx.x; col < n; col += blockDim.x) {
      size_t index = row * n + col;
      d_matrix[index] = expf(d_matrix[index]) / smem[0];
    }
  }
}
void launch_softmax_kernel(float *d_matrix, size_t n) {
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks(n);

  timer timer;
  softmax_kernel<<<blocks, threads_per_block>>>(d_matrix, n);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  double time = timer.elapsed();
  /*std::cout << "CUDA: " << time << "\tGB/s: "
            << n * n * sizeof(float) * 3.0f /
                   (1024.0 * 1024.0f * 1024.0f * time)
            << std::endl;*/

  std::cout << "CUDA_alg: " << time << " sec (GB/s: "
            << n * n * sizeof(float) * 3.0f /
                   (1024.0 * 1024.0f * 1024.0f * time)
            << ")\n";
}

void run_cuda_simt(const std::vector<float> &input, std::vector<float> &output,
                   std::size_t n) {
  CHECK_CUDA_ERROR(cudaSetDevice(0));

  size_t byte_size = n * n * sizeof(float);
  float *d_matrix;

  CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, byte_size));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_matrix, input.data(), byte_size, cudaMemcpyHostToDevice));

  launch_softmax_kernel(d_matrix, n);

  CHECK_CUDA_ERROR(
      cudaMemcpy(output.data(), d_matrix, byte_size, cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaFree(d_matrix));
}