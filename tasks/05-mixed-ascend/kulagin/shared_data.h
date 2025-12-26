#pragma once

constexpr int N_m = 256;
constexpr int TILING_BASE_M = 64;
constexpr long SOFTMAX_TILE_NUM = 2;

constexpr long SOFTMAX_USE_CORE_NAIVE = 16;

static_assert(N_m % TILING_BASE_M == 0, "N_m is not dividable by TILING_BASE_M");
static_assert(N_m % SOFTMAX_TILE_NUM == 0, "N_m is not dividable by softmax_tile_num");

#define USE_NAIVE_IMPL 1
