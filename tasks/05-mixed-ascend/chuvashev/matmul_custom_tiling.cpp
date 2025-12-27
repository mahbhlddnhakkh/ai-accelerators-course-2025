/**
 * @file matmul_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;
using namespace std;

/**
 * @brief  Generate matmul tiling.
 * @param  socVersion: Platform socversion.
 * @param  tilingBuf data buffer.
 */
void GenerateTiling(const char *socVersion, uint8_t *tilingBuf, uint32_t n) {
  int M = n;
  int N = n;
  int K = n;

  TPosition leftPosition = TPosition::GM;
  CubeFormat leftFormat = CubeFormat::ND;
  DataType leftDtype = DataType::DT_FLOAT16;
  bool isTransA = false;

  TPosition rightPosition = TPosition::GM;
  CubeFormat rightFormat = CubeFormat::ND;
  DataType rightDtype = DataType::DT_FLOAT16;
  bool isTransB = false;

  TPosition resultPosition = TPosition::GM;
  CubeFormat resultFormat = CubeFormat::ND;
  DataType resultDtype = DataType::DT_FLOAT;

  bool isBias = false;

  int usedCoreNum = 2;
  int32_t baseM = 16;
  int32_t baseN = 16;

  optiling::TCubeTiling tilingData;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

  tilingApi.SetDim(usedCoreNum);  // Set the number of cores that participate in
                                  // multi-core computaion is 2.
  tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
  tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
  tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

  tilingApi.SetOrgShape(M, N, K);
  tilingApi.SetShape(M, N, K);
  tilingApi.SetFixSplit(baseM, baseN,
                        -1);  // Set the fixed baseM=128, baseN=256.
  tilingApi.SetBias(isBias);
  tilingApi.SetBufferSpace(-1, -1, -1);

  int64_t res = tilingApi.GetTiling(tilingData);  // Get matmul tiling data.
  if (res == -1) {
    std::cout << "gen tiling failed" << std::endl;
  }
  uint32_t tcubeTilingSize = tilingData.GetDataSize();
  tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

  uint64_t localMemSize;
  ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,
                                  localMemSize);
  *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
  return;
}
