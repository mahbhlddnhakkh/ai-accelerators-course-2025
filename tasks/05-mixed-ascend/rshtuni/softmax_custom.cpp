/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr long BLOCK_DIM = 16;
constexpr long SIZE = 96;

constexpr int32_t TOTAL_LENGTH = SIZE * SIZE;  // total length of data
constexpr int32_t ROWS_PER_BLOCK = SIZE / BLOCK_DIM;
constexpr int32_t BLOCK_LENGTH = SIZE;  // length computed of each core
constexpr int32_t TILE_NUM = 2;         // split data into 2 tiles for each core
constexpr int32_t BUFFER_NUM = 2;       // tensor num for each queue
constexpr int32_t TILE_LENGTH =
    BLOCK_LENGTH / TILE_NUM /
    BUFFER_NUM;  // separate to 2 parts, due to double buffer

class KernelSoftmax {
 public:
  __aicore__ inline KernelSoftmax() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    xGm.SetGlobalBuffer((__gm__ float*)x + BLOCK_LENGTH * ROWS_PER_BLOCK *
                                               AscendC::GetBlockIdx(),
                        BLOCK_LENGTH * BLOCK_DIM);
    zGm.SetGlobalBuffer((__gm__ float*)z + BLOCK_LENGTH * ROWS_PER_BLOCK *
                                               AscendC::GetBlockIdx(),
                        BLOCK_LENGTH * BLOCK_DIM);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));

    pipe.InitBuffer(tmpBufCalc, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
  }

  __aicore__ inline void Process() {
    tmpCalc = tmpBufCalc.Get<float>();
    tmp = tmpBuf.Get<float>();

    for (int32_t row = 0; row < ROWS_PER_BLOCK; row++) {
      int32_t row_offset = row * BLOCK_LENGTH;

      AscendC::Duplicate(tmpCalc, 0.0f, TILE_LENGTH);

      for (uint32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
        CopyIn(i, row_offset);
        ComputeSum(i);
      }

      constexpr uint32_t shape[] = {1, TILE_LENGTH};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmp, tmpCalc,
                                                              shape, true);

      constexpr uint32_t dst_shape[] = {TILE_LENGTH};
      constexpr uint32_t src_shape[] = {1};
      AscendC::Broadcast<float, 1, 0>(tmpCalc, tmp, dst_shape, src_shape);

      for (uint32_t i = 0; i < TILE_NUM * BUFFER_NUM; i++) {
        CopyIn(i, row_offset);
        ComputeSoftmax(i);
        CopyOut(i, row_offset);
      }
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress, int32_t row_offset) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[row_offset + progress * TILE_LENGTH],
                      TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void ComputeSum(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);

    AscendC::Add(tmpCalc, tmpCalc, xLocal, TILE_LENGTH);

    inQueueX.FreeTensor(xLocal);
  }

  __aicore__ inline void ComputeSoftmax(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);
    AscendC::Div(zLocal, xLocal, tmpCalc, TILE_LENGTH);

    outQueueZ.EnQue<float>(zLocal);
    inQueueX.FreeTensor(xLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress, int32_t row_offset) {
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[row_offset + progress * TILE_LENGTH], zLocal,
                      TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf, tmpBufCalc;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> zGm;
  AscendC::LocalTensor<float> tmpCalc;
  AscendC::LocalTensor<float> tmp;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z) {
  KernelSoftmax op;
  op.Init(x, z);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* z) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif
