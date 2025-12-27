/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_custom.h"
#include "aclrtlaunch_softmax_custom.h"
extern void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                              uint8_t* z);
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                              uint8_t*);
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z);
#endif

extern void GenerateTiling(const char* socVersion, uint8_t* tilingBuf);

constexpr long BLOCK_DIM = 16;
constexpr long SIZE = 96;

int32_t main(int32_t argc, char* argv[]) {
  const char* socVersion = SOC_VERSION;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

  size_t aFileSize = SIZE * SIZE * sizeof(float);
  size_t bFileSize = SIZE * SIZE * sizeof(float);
  size_t biasFileSize = SIZE * sizeof(float);
  size_t matmulOutputSize = SIZE * SIZE * sizeof(float);
  size_t softmaxOutputSize = SIZE * SIZE * sizeof(float);

  size_t tilingFileSize = sizeof(TCubeTiling);
  size_t userWorkspaceSize = 0;
  size_t systemWorkspaceSize =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;

  uint8_t* tilingBuf = (uint8_t*)malloc(tilingFileSize);
  GenerateTiling(socVersion, tilingBuf);

#ifdef CUSTOM_ASCEND310P
  uint32_t matmulBlockDim = 2;
#else
  uint32_t matmulBlockDim = 1;
#endif

  uint32_t softmaxBlockDim = BLOCK_DIM;

#ifdef ASCENDC_CPU_DEBUG
  uint8_t* a = (uint8_t*)AscendC::GmAlloc(aFileSize);
  uint8_t* b = (uint8_t*)AscendC::GmAlloc(bFileSize);
  uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasFileSize);
  uint8_t* matmul_output = (uint8_t*)AscendC::GmAlloc(matmulOutputSize);
  uint8_t* softmax_output = (uint8_t*)AscendC::GmAlloc(softmaxOutputSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);

  ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
  ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
  ReadFile("./input/bias.bin", biasFileSize, bias, biasFileSize);
  memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);

  ICPU_RUN_KF(matmul_custom, matmulBlockDim, a, b, bias, matmul_output,
              workspace, tiling);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(softmax_custom, softmaxBlockDim, matmul_output, softmax_output);

  WriteFile("./output/matmul_output.bin", matmul_output, matmulOutputSize);
  WriteFile("./output/softmax_output.bin", softmax_output, softmaxOutputSize);

  AscendC::GmFree((void*)a);
  AscendC::GmFree((void*)b);
  AscendC::GmFree((void*)bias);
  AscendC::GmFree((void*)matmul_output);
  AscendC::GmFree((void*)softmax_output);
  AscendC::GmFree((void*)tiling);
  AscendC::GmFree((void*)workspace);
#else
  CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t* inputAHost;
  uint8_t* inputADevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputAHost), aFileSize));
  CHECK_ACL(
      aclrtMalloc((void**)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
  CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t* inputBHost;
  uint8_t* inputBDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputBHost), bFileSize));
  CHECK_ACL(
      aclrtMalloc((void**)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
  CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t* inputBiasHost;
  uint8_t* inputBiasDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputBiasHost), biasFileSize));
  CHECK_ACL(aclrtMalloc((void**)&inputBiasDevice, biasFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
  CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost,
                        biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t* matmulOutputHost;
  uint8_t* matmulOutputDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&matmulOutputHost), matmulOutputSize));
  CHECK_ACL(aclrtMalloc((void**)&matmulOutputDevice, matmulOutputSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  uint8_t* softmaxOutputHost;
  uint8_t* softmaxOutputDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&softmaxOutputHost), softmaxOutputSize));
  CHECK_ACL(aclrtMalloc((void**)&softmaxOutputDevice, softmaxOutputSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  uint8_t* tilingHost;
  uint8_t* tilingDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingFileSize));
  CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize,
                        ACL_MEMCPY_HOST_TO_HOST));
  CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost,
                        tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  uint8_t* workspaceDevice;
  CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACLRT_LAUNCH_KERNEL(matmul_custom)
  (matmulBlockDim, stream, inputADevice, inputBDevice, inputBiasDevice,
   matmulOutputDevice, workspaceDevice, tilingDevice);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  softmax_custom_do(softmaxBlockDim, stream, matmulOutputDevice,
                    softmaxOutputDevice);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(matmulOutputHost, matmulOutputSize, matmulOutputDevice,
                        matmulOutputSize, ACL_MEMCPY_DEVICE_TO_HOST));
  CHECK_ACL(aclrtMemcpy(softmaxOutputHost, softmaxOutputSize,
                        softmaxOutputDevice, softmaxOutputSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  WriteFile("./output/matmul_output.bin", matmulOutputHost, matmulOutputSize);
  WriteFile("./output/softmax_output.bin", softmaxOutputHost,
            softmaxOutputSize);

  CHECK_ACL(aclrtFree(inputADevice));
  CHECK_ACL(aclrtFreeHost(inputAHost));
  CHECK_ACL(aclrtFree(inputBDevice));
  CHECK_ACL(aclrtFreeHost(inputBHost));
  CHECK_ACL(aclrtFree(inputBiasDevice));
  CHECK_ACL(aclrtFreeHost(inputBiasHost));
  CHECK_ACL(aclrtFree(matmulOutputDevice));
  CHECK_ACL(aclrtFreeHost(matmulOutputHost));
  CHECK_ACL(aclrtFree(softmaxOutputDevice));
  CHECK_ACL(aclrtFreeHost(softmaxOutputHost));
  CHECK_ACL(aclrtFree(tilingDevice));
  CHECK_ACL(aclrtFreeHost(tilingHost));
  CHECK_ACL(aclrtFree(workspaceDevice));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif

  free(tilingBuf);
  return 0;
}
