/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "shared_data.h"

#if USE_NAIVE_IMPL == 1

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

/**
 * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
 * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
 * @param  tilingGM: tiling gm addr.
 * @retval None
 */
__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

  for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
  return;
}

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulLeakyKernel {
 public:
  __aicore__ inline MatmulLeakyKernel(){};
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                              GM_ADDR workspace, const TCubeTiling& tiling,
                              AscendC::TPipe* pipe);
  __aicore__ inline void Process(AscendC::TPipe* pipe);

  __aicore__ inline void MatmulCompute();
  __aicore__ inline void LeakyReluCompute();
  __aicore__ inline void CopyOut(uint32_t count);
  __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling& tiling,
                                    int32_t& offsetA, int32_t& offsetB,
                                    int32_t& offsetC, int32_t& offsetBias);

  Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
         MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
      matmulObj;

  AscendC::GlobalTensor<aType> aGlobal;
  AscendC::GlobalTensor<bType> bGlobal;
  AscendC::GlobalTensor<cType> cGlobal;
  AscendC::GlobalTensor<biasType> biasGlobal;
  AscendC::LocalTensor<cType> reluOutLocal;
  TCubeTiling tiling;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue_;
};

/**
 * @brief  Set matmulLeaky input and output gm addr of current core.
 * @param  a: A matrix gm addr.
 * @param  b: B matrix gm addr.
 * @param  bias: Bias gm addr.
 * @param  c: C matrix gm addr.
 * @param  workspace: Temporary gm space addr required by matmul calc.
 * @param  tiling: matmul tiling data.
 * @param  pipe: Global memory and sync management TPipe object.
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
    const TCubeTiling& tiling, AscendC::TPipe* pipe) {
  this->tiling = tiling;
  aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType*>(a),
                          tiling.M * tiling.Ka);
  bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType*>(b),
                          tiling.Kb * tiling.N);
  cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c),
                          tiling.M * tiling.N);
  biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias),
                             tiling.N);

  int32_t offsetA, offsetB, offsetC, offsetBias;
  CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC,
             offsetBias);  // Calculate the gm offset based on the blockidx.
  aGlobal = aGlobal[offsetA];
  bGlobal = bGlobal[offsetB];
  cGlobal = cGlobal[offsetC];
  biasGlobal = biasGlobal[offsetBias];
  pipe->InitBuffer(
      reluOutQueue_, 1,
      tiling.baseM * tiling.baseN * sizeof(cType));  // Init output buffer.
}

/**
 * @brief  Main process of matmul calculation
 * @param  pipe: Global memory and sync management TPipe object.
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::Process(
    AscendC::TPipe* pipe) {
  uint32_t computeRound = 0;

#ifdef CUSTOM_ASCEND310P
  // Set temp UB space when on ASCEND310P
  AscendC::TBuf<> tmpMMFormatUb;
  AscendC::LocalTensor<uint8_t> mmformatUb;
  pipe->InitBuffer(tmpMMFormatUb, tiling.baseM * tiling.baseN * sizeof(cType));
  mmformatUb =
      tmpMMFormatUb.Get<uint8_t>(tiling.baseM * tiling.baseN * sizeof(cType));
  matmulObj.SetLocalWorkspace(mmformatUb);
#endif
  matmulObj.SetTensorA(aGlobal);
  matmulObj.SetTensorB(bGlobal);
  matmulObj.SetBias(biasGlobal);
  while (matmulObj.template Iterate<true>()) {  // Once Iterate, compute baseM *
                                                // baseN, sync is set true here.
    MatmulCompute();                            // Get matmul compute result.
    LeakyReluCompute();                         // Compute leakyRelu.
    CopyOut(computeRound);  // Copy leakyRelu out result to GM.
    computeRound++;
  }
  matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute() {
  reluOutLocal = reluOutQueue_.AllocTensor<cType>();
  matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute() {
  // LeakyRelu(reluOutLocal, reluOutLocal, (cType)0.001,
  //           tiling.baseM * tiling.baseN);
  reluOutQueue_.EnQue(reluOutLocal);
}

/**
 * @brief  Copy leakyRelu out result to GM.
 * @param  count: Iterate count(once Iterate, compute baseM * baseN).
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count) {
  reluOutQueue_.DeQue<cType>();
  const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
  const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
  uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N +
                          count / roundM * tiling.baseN);
  AscendC::DataCopyParams copyParam = {
      (uint16_t)tiling.baseM,
      (uint16_t)(tiling.baseN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0,
      (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) /
                 AscendC::DEFAULT_C0_SIZE)};
  DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
  reluOutQueue_.FreeTensor(reluOutLocal);
}

/**
 * @brief  Calculate the gm offset based on the blockidx.
 * @param  blockIdx: Current Core blockidx.
 * @param  tiling: Matmul tiling data.
 * @param  offsetA: Gm offset of A matrix.
 * @param  offsetB: Gm offset of B matrix.
 * @param  offsetC: Gm offset of C matrix.
 * @param  offsetBias: Gm offset of Bias matrix.
 * @retval None
 */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(
    int32_t blockIdx, const TCubeTiling& tiling, int32_t& offsetA,
    int32_t& offsetB, int32_t& offsetC, int32_t& offsetBias) {
  auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
  auto mCoreIndx = blockIdx % mSingleBlocks;
  auto nCoreIndx = blockIdx / mSingleBlocks;

  offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
  offsetB = nCoreIndx * tiling.singleCoreN;
  offsetC = mCoreIndx * tiling.N * tiling.singleCoreM +
            nCoreIndx * tiling.singleCoreN;
  offsetBias = nCoreIndx * tiling.singleCoreN;
}

/**
 * @brief  matmul_leakyrelu kernel function entry
 * @param  a: A matrix gm addr.
 * @param  b: B matrix gm addr.
 * @param  bias: Bias gm addr.
 * @param  c: Out gm addr.
 * @param  workspace: Temporary gm space addr required by matmul calc.
 * @param  tilingGm: Tiling data addr.
 * @retval None
 */
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b,
                                                    GM_ADDR bias, GM_ADDR c,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  AscendC::TPipe pipe;
  TCubeTiling tiling;
  CopyTiling(&tiling, tilingGm);

  MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
  matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
  REGIST_MATMUL_OBJ(
      &pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj,
      &matmulLeakyKernel.tiling);  // Initialize the matmul object.
  matmulLeakyKernel.Process(&pipe);
}

//  Softmax

constexpr int32_t USE_CORE_NUM = SOFTMAX_USE_CORE_NAIVE;
constexpr int32_t TOTAL_LENGTH = N_m * N_m;  // total length of data
constexpr int32_t ROWS_PER_BLOCK = N_m / USE_CORE_NUM;
constexpr int32_t BLOCK_LENGTH = N_m;  // length computed of each core
constexpr int32_t TILE_NUM =
    SOFTMAX_TILE_NUM;              // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
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

    pipe.InitBuffer(tmpBufRed, sizeof(float));
    pipe.InitBuffer(tmpBufCalc, TILE_CALCULATED * sizeof(float));
    pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = TILE_CALCULATED;
    tmpCalc = tmpBufCalc.Get<float>();
    tmp = tmpBuf.Get<float>();
    for (int32_t row = 0; row < ROWS_PER_BLOCK; row++) {
      for (int32_t i = 0; i < loopCount; i++) {
        CopyIn1(i, row);
        Compute1(i, row);
        CopyOut1(i, row);
      }

      AscendC::LocalTensor<float> tmpRed = tmpBufRed.Get<float>();
      constexpr uint32_t shape[] = {1, TILE_CALCULATED};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmpRed, tmpCalc,
                                                              shape, true);
      constexpr uint32_t dst_shape[] = {TILE_LENGTH};
      constexpr uint32_t src_shape[] = {1};
      AscendC::Broadcast<float, 1, 0>(tmp, tmpRed, dst_shape, src_shape);

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
    AscendC::LocalTensor<float> tmpRed = tmpBufRed.Get<float>();
    AscendC::Exp(xLocal, xLocal, TILE_LENGTH);
    constexpr uint32_t shape[] = {1, TILE_LENGTH};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(tmpRed, xLocal,
                                                            shape, true);
    // AscendC::DataCopy((tmpCalc)[progress], tmpRed, 1);
    tmpCalc.SetValue(progress, tmpRed.GetValue(0));
    inQueueX.FreeTensor(xLocal);
  }
  __aicore__ inline void CopyOut1(int32_t progress, int32_t row) {}

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
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf, tmpBufCalc, tmpBufRed;
  AscendC::GlobalTensor<float> xGm;
  AscendC::GlobalTensor<float> tmpGm;
  AscendC::GlobalTensor<float> zGm;
  AscendC::LocalTensor<float> tmpCalc;
  AscendC::LocalTensor<float> tmp;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z) {
  KernelSoftmax op;
  op.Init(x, z);
  //  return;
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                       uint8_t* z) {
  softmax_custom<<<blockDim, nullptr, stream>>>(x, z);
}
#endif

#endif
