#include "shared_data.h"

#if MY_SOFTMAX_IMPL == 2

#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = N * N;  // total length of data
constexpr int32_t ROWS_PER_BLOCK = N / USE_CORE_NUM;
constexpr int32_t BLOCK_LENGTH = N;  // length computed of each core
constexpr int32_t TILE_NUM = 2;      // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;    // tensor num for each queue
constexpr int32_t TILE_LENGTH =
    BLOCK_LENGTH / TILE_NUM /
    BUFFER_NUM;  // separate to 2 parts, due to double buffer
constexpr int32_t TILE_CALCULATED = TILE_NUM * BUFFER_NUM;

static_assert(TILE_LENGTH % 8 == 0,
              "TILE_LENGTH must be dividable by 8 (32 bytes)");
static_assert(BLOCK_LENGTH % (BUFFER_NUM * TILE_NUM) == 0,
              "BLOCK_LENGTH must be dividable by BUFFER_NUM * TILE_NUM");

class KernelSoftmax {
 public:
  __aicore__ inline KernelSoftmax() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    xGm.SetGlobalBuffer((__gm__ float*)x + BLOCK_LENGTH * ROWS_PER_BLOCK *
                                               AscendC::GetBlockIdx(),
                        BLOCK_LENGTH * USE_CORE_NUM);
    zGm.SetGlobalBuffer((__gm__ float*)z + BLOCK_LENGTH * ROWS_PER_BLOCK *
                                               AscendC::GetBlockIdx(),
                        BLOCK_LENGTH * USE_CORE_NUM);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueTmp, BUFFER_NUM, 1 * sizeof(float));

    pipe.InitBuffer(tmpBufCalc, TILE_CALCULATED * sizeof(float));
    pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = TILE_CALCULATED;
    tmpCalc = tmpBufCalc.Get<float>();
    tmp = tmpBuf.Get<float>();
    for (int32_t row = 0; row < ROWS_PER_BLOCK; row++) {
      for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn1(i, row);
        Compute1(i, row);
        CopyOut1(i, row);
      }

      AscendC::LocalTensor<float> tmpRed = outQueueTmp.AllocTensor<float>();
      constexpr uint32_t shape[] = {1, TILE_CALCULATED};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmpRed, tmpCalc,
                                                              shape, true);
      constexpr uint32_t dst_shape[] = {TILE_LENGTH};
      constexpr uint32_t src_shape[] = {1};
      AscendC::Broadcast<float, 1, 0>(tmp, tmpRed, dst_shape, src_shape);
      outQueueTmp.FreeTensor(tmpRed);

      for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn2(i, row);
        Compute2(i, row);
        CopyOut2(i, row);
      }
    }
  }

 private:
  __aicore__ inline void CopyIn1(int32_t progress, int32_t row) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[row * BLOCK_LENGTH + progress * TILE_LENGTH],
                      TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute1(int32_t progress, int32_t row) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> tmpCalcLocal = outQueueTmp.AllocTensor<float>();
    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);
    constexpr uint32_t shape[] = {1, TILE_LENGTH};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
        tmpCalcLocal, xLocal, shape, true);
    outQueueTmp.EnQue<float>(tmpCalcLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut1(int32_t progress, int32_t row) {
    AscendC::LocalTensor<float> tmpCalcLocal = outQueueTmp.DeQue<float>();
    // AscendC::DataCopy(tmpCalc[progress], tmpCalcLocal, 1);
    tmpCalc.SetValue(progress, tmpCalcLocal.GetValue(0));
    outQueueTmp.FreeTensor(tmpCalcLocal);
  }

  __aicore__ inline void CopyIn2(int32_t progress, int32_t row) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[row * BLOCK_LENGTH + progress * TILE_LENGTH],
                      TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute2(int32_t progress, int32_t row) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);
    AscendC::Div(zLocal, xLocal, tmp, TILE_LENGTH);
    outQueueZ.EnQue<float>(zLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut2(int32_t progress, int32_t row) {
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[row * BLOCK_LENGTH + progress * TILE_LENGTH], zLocal,
                      TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ, outQueueTmp;
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

#endif
