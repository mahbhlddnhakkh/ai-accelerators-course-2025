#include <utils.h>

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