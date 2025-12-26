/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "shared_data.h"
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
//#include "aclrtlaunch_matmul_leakyrelu_custom.h"
#if USE_NAIVE_IMPL == 0
#include "aclrtlaunch_matmul_softmax_custom.h"
#else
#include "aclrtlaunch_matmul_custom.h"
extern void softmax_custom_do(uint32_t blockDim, void* stream, uint8_t* x,
                              uint8_t* z);
#endif
#else
#include "tikicpulib.h"
#if USE_NAIVE_IMPL == 0
extern "C" void matmul_softmax_custom(uint8_t*, uint8_t*, uint8_t*, uint8_t*,
                                        uint8_t*, uint8_t*);
#else
extern "C" void matmul_custom(uint8_t *a, uint8_t *b, uint8_t *bias, uint8_t *c, uint8_t *workspace, uint8_t *tiling);
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z);
#endif
#endif
extern void GenerateTiling(const char* socVersion, uint8_t* tilingBuf);

int32_t main1(int32_t argc, char* argv[]) {
  const char* socVersion = SOC_VERSION;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  //size_t aFileSize = 262144 * sizeof(int16_t);
  //size_t bFileSize = 163840 * sizeof(int16_t);
  //size_t cFileSize = 655360 * sizeof(float);
  //size_t biasFileSize = 640 * sizeof(float);
  size_t aFileSize = N_m * N_m * sizeof(int16_t);
  size_t bFileSize = N_m * N_m * sizeof(int16_t);
  size_t cFileSize = N_m * N_m * sizeof(float);
  size_t biasFileSize = N_m * sizeof(float);
  size_t tilingFileSize = sizeof(TCubeTiling);
  size_t userWorkspaceSize = 0;
  size_t systemWorkspaceSize =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
  uint8_t* tilingBuf = (uint8_t*)malloc(tilingFileSize);
  GenerateTiling(socVersion, tilingBuf);
#ifdef CUSTOM_ASCEND310P
  uint32_t blockDim = 2;
#else
  uint32_t blockDim = 1;
#endif

#ifdef ASCENDC_CPU_DEBUG
  uint8_t* a = (uint8_t*)AscendC::GmAlloc(aFileSize);
  uint8_t* b = (uint8_t*)AscendC::GmAlloc(bFileSize);
  uint8_t* bias = (uint8_t*)AscendC::GmAlloc(biasFileSize);
  uint8_t* c = (uint8_t*)AscendC::GmAlloc(cFileSize);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);

  ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
  ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
  ReadFile("./input/bias.bin", biasFileSize, bias, biasFileSize);
  memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);
#if USE_NAIVE_IMPL == 0
  ICPU_RUN_KF(matmul_softmax_custom, blockDim, a, b, bias, c, workspace,
              tiling);
#else
  ICPU_RUN_KF(matmul_custom, blockDim, a, b, bias, c, workspace,
              tiling);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(softmax_custom, SOFTMAX_USE_CORE_NAIVE, c, c);
#endif
  WriteFile("./output/output.bin", c, cFileSize);
  AscendC::GmFree((void*)a);
  AscendC::GmFree((void*)b);
  AscendC::GmFree((void*)bias);
  AscendC::GmFree((void*)c);
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

  uint8_t* outputCHost;
  uint8_t* outputCDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&outputCHost), cFileSize));
  CHECK_ACL(aclrtMalloc((void**)&outputCDevice, cFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  uint8_t* inputBiasHost;
  uint8_t* inputBiasDevice;
  CHECK_ACL(aclrtMallocHost((void**)(&inputBiasHost), biasFileSize));
  CHECK_ACL(aclrtMalloc((void**)&inputBiasDevice, biasFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
  CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost,
                        biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

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

#if USE_NAIVE_IMPL == 0
  ACLRT_LAUNCH_KERNEL(matmul_softmax_custom)
  (blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice,
   workspaceDevice, tilingDevice);
#else
  uint8_t *outputCDevice2;
  CHECK_ACL(aclrtMalloc((void**)&outputCDevice2, cFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACLRT_LAUNCH_KERNEL(matmul_custom)
  (blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice,
   workspaceDevice, tilingDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));
  
  softmax_custom_do(SOFTMAX_USE_CORE_NAIVE, stream, outputCDevice, outputCDevice);
#endif

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtFree(inputADevice));
  CHECK_ACL(aclrtFreeHost(inputAHost));
  CHECK_ACL(aclrtFree(inputBDevice));
  CHECK_ACL(aclrtFreeHost(inputBHost));
  CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
#if USE_NAIVE_IMPL == 0
  WriteFile("./output/output.bin", outputCHost, cFileSize);
#else
  WriteFile("./output/output.bin", outputCHost, cFileSize);
#endif
#if USE_NAIVE_IMPL == 1
  CHECK_ACL(aclrtFree(outputCDevice2));
#endif
  CHECK_ACL(aclrtFree(outputCDevice));
  CHECK_ACL(aclrtFreeHost(outputCHost));
  CHECK_ACL(aclrtFree(inputBiasDevice));
  CHECK_ACL(aclrtFreeHost(inputBiasHost));
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

int32_t main(int32_t argc, char* argv[]) {
  int ret = main1(argc, argv);
  if (ret != 0) {
    return ret;
  }
  return ret;
  // EXTREME NAIVE VERSION
#if USE_NAIVE_IMPL == 1 && !defined(ASCENDC_CPU_DEBUG)
  uint32_t blockDim = SOFTMAX_USE_CORE_NAIVE;
  size_t inputByteSize = N_m * N_m * sizeof(float);
  size_t outputByteSize = N_m * N_m * sizeof(float);

  CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *xHost, *zHost;
  uint8_t *xDevice, *zDevice;

  CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
  CHECK_ACL(
      aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./output/output2.bin", inputByteSize, xHost, inputByteSize);

  CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  softmax_custom_do(blockDim, stream, xDevice, zDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output.bin", zHost, outputByteSize);

  CHECK_ACL(aclrtFree(xDevice));
  CHECK_ACL(aclrtFree(zDevice));
  CHECK_ACL(aclrtFreeHost(xHost));
  CHECK_ACL(aclrtFreeHost(zHost));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif
  return 0;
}
