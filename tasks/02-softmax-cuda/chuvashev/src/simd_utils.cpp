#include <simd_utils.h>

// OpenMP_SIMD
__m256 exp256_ps(__m256 x) {
  /* Modified code. The original code is here:
    https://github.com/reyoung/avx_mathfun

     AVX implementation of exp
     Based on "sse_mathfun.h", by Julien Pommier
     http://gruntthepeon.free.fr/ssemath/
     Copyright (C) 2012 Giovanni Garberoglio
     Interdisciplinary Laboratory for Computational Science (LISC)
     Fondazione Bruno Kessler and University of Trento
     via Sommarive, 18
     I-38123 Trento (Italy)
    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
    (this is the zlib license)
  */
  /*
    To increase the compatibility across different compilers the original code
    is converted to plain AVX2 intrinsics code without ingenious macro's, gcc
    style alignment attributes etc. The modified code requires AVX2
  */
  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
  __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375);
  __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
  tmp = _mm256_floor_ps(fx);
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);
  tmp = _mm256_mul_ps(fx, cephes_exp_C1);
  __m256 z = _mm256_mul_ps(fx, cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);
  z = _mm256_mul_ps(x, x);

  __m256 y = cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
inline float calc_sum_of_exp_vec(__m256 vec_of_8_exps) {
  __m128 hiQuad = _mm256_extractf128_ps(vec_of_8_exps, 1);
  __m128 loQuad = _mm256_castps256_ps128(vec_of_8_exps);
  __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  __m128 loDual = sumQuad;
  __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  __m128 sumDual = _mm_add_ps(loDual, hiDual);
  __m128 lo = sumDual;
  __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  __m128 sum = _mm_add_ss(lo, hi);
  return _mm_cvtss_f32(sum);
}
void calculate_row_simd(const float *address_input, float *address_output,
                        std::size_t n) {
  float current_sum = 0;

  std::size_t idx_j = 0;

  for (; idx_j + 8 <= n; idx_j += 8) {
    __m256 vec_of_8_elems = _mm256_loadu_ps(&address_input[idx_j]);
    __m256 vec_of_8_exps = exp256_ps(vec_of_8_elems);
    _mm256_storeu_ps(&address_output[idx_j], vec_of_8_exps);
    current_sum += calc_sum_of_exp_vec(vec_of_8_exps);
  }
  for (; idx_j < n; ++idx_j) {
    float exp = std::exp(address_input[idx_j]);
    address_output[idx_j] = exp;
    current_sum += exp;
  }

  idx_j = 0;
  current_sum = 1.0f / current_sum;

  for (; idx_j + 8 <= n; idx_j += 8) {
    __m256 vec_of_8_elems = _mm256_loadu_ps(&address_output[idx_j]);
    __m256 vec_of_8_sums = _mm256_set1_ps(current_sum);
    __m256 vec_of_8_results = _mm256_mul_ps(vec_of_8_elems, vec_of_8_sums);
    _mm256_storeu_ps(&address_output[idx_j], vec_of_8_results);
  }

  for (; idx_j < n; ++idx_j) {
    address_output[idx_j] *= current_sum;
  }
}
void run_openmp_simd(const std::vector<float> &input,
                     std::vector<float> &output, std::size_t n) {
  // throw std::runtime_error("OpenMP + SIMD method not implemented");
#pragma omp parallel for
  for (int idx_i = 0; idx_i < n; ++idx_i) {
    calculate_row_simd(&input[idx_i * n], &output[idx_i * n], n);
  }
}
