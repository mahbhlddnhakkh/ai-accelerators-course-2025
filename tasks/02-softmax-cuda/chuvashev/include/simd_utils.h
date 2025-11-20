#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#include <immintrin.h>

#include <cmath>
#include <vector>

// OpenMP_SIMD
__m256 exp256_ps(__m256 x);
inline float calc_sum_of_exp_vec(__m256 vec_of_8_exps);
void calculate_row_simd(const float *address_input, float *address_output,
                        std::size_t n);
void run_openmp_simd(const std::vector<float> &input,
                     std::vector<float> &output, std::size_t n);

#endif  // !SIMD_UTILS_H