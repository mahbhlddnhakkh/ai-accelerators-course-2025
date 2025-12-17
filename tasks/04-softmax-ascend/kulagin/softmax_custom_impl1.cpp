#include "shared_data.h"

#if MY_SOFTMAX_IMPL == 1

#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = N * N;  // total length of data
constexpr int32_t USE_CORE_NUM = N;      // num of core used
constexpr int32_t BLOCK_LENGTH =
    TOTAL_LENGTH / USE_CORE_NUM;   // length computed of each core
constexpr int32_t TILE_NUM = 8;    // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
constexpr int32_t TILE_LENGTH =
    BLOCK_LENGTH / TILE_NUM /
    BUFFER_NUM;  // separate to 2 parts, due to double buffer
constexpr int32_t TILE_CALCULATED = TILE_NUM * BUFFER_NUM;

static_assert(BLOCK_LENGTH % (TILE_NUM * BUFFER_NUM) == 0, "");
static_assert(BLOCK_LENGTH % BUFFER_NUM == 0,
              "BLOCK_LENGTH must be aligned with BUFFER_NUM");
static_assert(TILE_LENGTH % 8 == 0, "TILE_LENGTH must be 8 (floats)");
static_assert(TILE_CALCULATED % 8 == 0, "TILE_CALCULATED must be 8 (floats)");

class KernelSoftmax {
 public:
  __aicore__ inline KernelSoftmax() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR z) {
    xGm.SetGlobalBuffer(
        (__gm__ float*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    zGm.SetGlobalBuffer(
        (__gm__ float*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(float));

    pipe.InitBuffer(tmpBufRed, sizeof(float));
    pipe.InitBuffer(tmpBufCalc, TILE_CALCULATED * sizeof(float));
    pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpBufExp, TILE_LENGTH * sizeof(float));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = TILE_CALCULATED;
    tmpCalc = tmpBufCalc.Get<float>();
    tmp = tmpBuf.Get<float>();
    for (uint32_t i = 0; i < loopCount; i++) {
      CopyIn1(i);
      Compute1(i);
      CopyOut1(i);
    }

    AscendC::LocalTensor<float> tmpRed = tmpBufRed.Get<float>();
    constexpr uint32_t shape[] = {1, TILE_CALCULATED};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmpRed, tmpCalc,
                                                            shape, true);
    constexpr uint32_t dst_shape[] = {TILE_LENGTH};
    constexpr uint32_t src_shape[] = {1};
    AscendC::Broadcast<float, 1, 0>(tmp, tmpRed, dst_shape, src_shape);

    for (uint32_t i = 0; i < loopCount; i++) {
      CopyIn2(i);
      Compute2(i);
      CopyOut2(i);
    }
  }

 private:
  __aicore__ inline void CopyIn1(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute1(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> xLocalExp = tmpBufExp.Get<float>();
    AscendC::LocalTensor<float> tmpRed = tmpBufRed.Get<float>();
    AscendC::Exp(xLocalExp, xLocal, TILE_LENGTH);
    constexpr uint32_t shape[] = {1, TILE_LENGTH};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmpRed, xLocalExp,
                                                            shape, true);
    // AscendC::DataCopy((tmpCalc)[progress], tmpRed, 1);
    tmpCalc.SetValue(progress, tmpRed.GetValue(0));
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut1(int32_t progress) {}

  __aicore__ inline void CopyIn2(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
    inQueueX.EnQue(xLocal);
  }
  __aicore__ inline void Compute2(int32_t progress) {
    AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> xLocalExp = tmpBufExp.Get<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Exp(xLocalExp, xLocal, TILE_LENGTH);
    AscendC::Div(zLocal, xLocalExp, tmp, TILE_LENGTH);
    outQueueZ.EnQue<float>(zLocal);
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut2(int32_t progress) {
    AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
    outQueueZ.FreeTensor(zLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf, tmpBufCalc, tmpBufRed,
      tmpBufExp;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> tmpGm;
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
