/**
 * @file softmax_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr int32_t BATCH_SIZE = 2048;
constexpr int32_t FEATURE_DIM = 2048;
constexpr int32_t TOTAL_LENGTH = BATCH_SIZE * FEATURE_DIM;
constexpr int32_t USE_CORE_NUM = 16;
constexpr int32_t ROWS_PER_CORE = BATCH_SIZE / USE_CORE_NUM;
constexpr int32_t BLOCK_LENGTH = ROWS_PER_CORE * FEATURE_DIM;
constexpr int32_t TILES_PER_ROW = 2;
constexpr int32_t TILE_LENGTH = FEATURE_DIM / TILES_PER_ROW;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelAdd {
 public:
  __aicore__ inline KernelAdd() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    xGm.SetGlobalBuffer((__gm__ T*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(),
                        BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ T*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(),
                        BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(tmpQueue, (TILE_LENGTH + TILE_LENGTH) * sizeof(T));
  }
  __aicore__ inline void Process() {
    AscendC::LocalTensor<T> tmpLocalFull = tmpQueue.Get<T>();
    AscendC::LocalTensor<T> tmpLocal = tmpLocalFull[0];
    AscendC::LocalTensor<T> tmpLocal2 = tmpLocalFull[TILE_LENGTH];

    // TODO: handle multiple rows in one iter
    for (int32_t r = 0; r < ROWS_PER_CORE; r++) {
      int32_t row_offset = r * FEATURE_DIM;

      AscendC::Duplicate(tmpLocal, T(0), TILE_LENGTH);

      for (int32_t i = 0; i < TILES_PER_ROW; i++) {
        CopyIn(i, row_offset);
        Compute(i, tmpLocal, tmpLocal2);
      }

      // AscendC::SumParams sumParams = {
      //     1, (TILE_LENGTH * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T),
      //     TILE_LENGTH};
      // AscendC::Sum(tmpLocal2, tmpLocal, sumParams);

      const uint32_t shape[] = {1, TILE_LENGTH};
      AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR>(tmpLocal2, tmpLocal,
                                                          shape, true);

      T sum_val = tmpLocal2.GetValue(0);
      AscendC::Duplicate(tmpLocal, sum_val, TILE_LENGTH);

      for (int32_t i = 0; i < TILES_PER_ROW; i++) {
        CopyIn(i, row_offset);
        Compute2(i, tmpLocal);
        CopyOut(i, row_offset);
      }
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress, int32_t row_offset) {
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[row_offset + progress * TILE_LENGTH],
                      TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute(int32_t progress,
                                 AscendC::LocalTensor<T>& tmpLocal,
                                 AscendC::LocalTensor<T>& tmpLocal2) {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::Exp(tmpLocal2, xLocal, TILE_LENGTH);
    AscendC::Add(tmpLocal, tmpLocal, tmpLocal2, TILE_LENGTH);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void Compute2(int32_t progress,
                                  AscendC::LocalTensor<T>& tmpLocal) {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);
    AscendC::Div(zLocal, xLocal, tmpLocal, TILE_LENGTH);
    outQueueZ.EnQue<T>(zLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress, int32_t row_offset) {
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopy(zGm[row_offset + progress * TILE_LENGTH], zLocal,
                      TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpQueue;
  AscendC::GlobalTensor<T> xGm;
  AscendC::GlobalTensor<T> zGm;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z) {
  KernelAdd<float> op;
  op.Init(x, z);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* z) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif
