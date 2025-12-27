/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

/**
 * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
 * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
 * @param  localMemSize: Temporary local memory size required by matmul calc.
 * @param  tilingGM: tiling gm addr.
 * @retval None
 */
__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize,
                                  GM_ADDR tilingGM) {
  uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

  for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
  localMemSize =
      *reinterpret_cast<__gm__ uint64_t *>(tilingGM + sizeof(TCubeTiling));
  return;
}

/**
 * @brief  Calculate the gm offset and tail size based on the blockidx.
 * @param  blockIdx: Current Core blockidx.
 * @param  tiling: Matmul tiling data.
 * @param  offsetA: Gm offset of A matrix.
 * @param  offsetB: Gm offset of B matrix.
 * @param  offsetC: Gm offset of C matrix.
 * @param  tailM: SingleCoreM size of tail core.
 * @param  tailN: SingleCoreN size of tail core.
 * @param  isTransA: A matrix transpose.
 * @param  isTransB: B matrix transpose.
 * @retval None
 */
__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling,
                                    int &offsetA, int &offsetB, int &offsetC,
                                    int &tailM, int &tailN, bool isTransA,
                                    bool isTransB) {
  uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
  uint32_t mCoreIndx = blockIdx % mSingleBlocks;
  uint32_t nCoreIndx = blockIdx / mSingleBlocks;

  offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
  if (isTransA) {
    offsetA = mCoreIndx * tiling.singleCoreM;
  }
  offsetB = nCoreIndx * tiling.singleCoreN;
  if (isTransB) {
    offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
  }
  offsetC = mCoreIndx * tiling.N * tiling.singleCoreM +
            nCoreIndx * tiling.singleCoreN;

  tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
  tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;

  tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
  tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

/**
 * @brief  matmul kernel function entry
 * @param  a: A matrix gm addr.
 * @param  b: B matrix gm addr.
 * @param  c: C matrix gm addr.
 * @param  workspace: Temporary gm space addr required by matmul calc.
 * @param  tilingGm: Tiling data addr.
 * @retval None
 */
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b,
                                                    GM_ADDR c,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tilingGm) {
  using A_T = half;
  using B_T = half;
  using C_T = float;

  AscendC::TPipe pipe;
  TCubeTiling tiling;
  uint64_t localMemSize = 0;
  CopyTiling(&tiling, localMemSize, tilingGm);

  AscendC::GlobalTensor<A_T> aGlobal;
  AscendC::GlobalTensor<B_T> bGlobal;
  AscendC::GlobalTensor<C_T> cGlobal;
  aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a),
                          tiling.M * tiling.Ka);
  bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b),
                          tiling.Ka * tiling.N);
  cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c),
                          tiling.M * tiling.N);

  int offsetA = 0;
  int offsetB = 0;
  int offsetC = 0;
  bool isTransA = false;
  bool isTransB = false;

  int tailM = 0;
  int tailN = 0;
  // Calculate the gm offset and tail size based on the blockidx.
  CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN,
               isTransA, isTransB);

  auto gmA = aGlobal[offsetA];
  auto gmB = bGlobal[offsetB];
  auto gmC = cGlobal[offsetC];

  Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
         MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>>
      mm;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm,
                    &tiling);  // Initialize the matmul object.
#ifdef CUSTOM_ASCEND310P
  // Set temp UB space when on ASCEND310P.
  AscendC::TBuf<> tmpMMFormatUb;
  AscendC::LocalTensor<uint8_t> mmFormatUb;
  pipe.InitBuffer(tmpMMFormatUb, localMemSize);
  mmFormatUb = tmpMMFormatUb.Get<uint8_t>(localMemSize);
  mm.SetLocalWorkspace(mmFormatUb);
#endif
  mm.SetTensorA(gmA, isTransA);
  mm.SetTensorB(gmB, isTransB);
  mm.SetTail(tailM, tailN);
  mm.IterateAll(gmC);
  mm.End();

  pipe.Destroy();
}
