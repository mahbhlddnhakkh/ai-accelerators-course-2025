#include <iostream>

#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG

#include "acl/acl.h"
#include "aclrtlaunch_exp_custom.h"
#include "aclrtlaunch_matmul_custom.h"

extern void exp_custom_do(uint32_t block_dim, void *stream, uint8_t *x,
                          uint8_t *y, uint8_t *tiling);

#else

#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *a, uint8_t *b, uint8_t *c,
                              uint8_t *workspace, uint8_t *tiling);
extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR tiling);

#endif

extern void GenerateTiling(const char *socVersion, uint8_t *tilingBuf,
                           uint32_t n);

struct TileInfo {
  uint32_t N;
  uint32_t M;
  uint32_t num_of_ai_cores;
  uint32_t tile_length;  // длина одного тайла (В байтах)
  uint32_t sizeof_type;  // размер 1 элемента (В байтах)
  uint32_t count_of_based_blocks;
  uint32_t count_of_cutted_blocks;
  uint32_t based_rows_per_block;
  uint32_t cutted_rows_per_block;
  uint32_t elems_per_tile;  // кол-во элементов на каждый тайл (НЕ в байтах)
  uint32_t tiles_per_row;
  uint32_t
      length_last_tile;  // кол-во элементов на последнем тайле (НЕ в байтах)
  uint32_t length_last_tile_align;  // кол-во элементов на последнем тайле с
                                    // выравниванием по 32 байтам (НЕ в байтах)
  uint32_t buffer_num;
};

void GenerateTilingSoftMax(uint32_t n, TileInfo &tiling) {
  tiling.N = n;
  tiling.buffer_num = 1;

  tiling.num_of_ai_cores = 1;

  tiling.tile_length =
      512 / tiling.buffer_num;  // учитываем, что DoubleBuffering
  tiling.sizeof_type = sizeof(float);

  std::size_t bytes = n * tiling.sizeof_type;
  std::size_t size_of_vec = 32;
  tiling.M = tiling.N;

  if (bytes % size_of_vec != 0) {
    std::size_t cut_bytes = bytes % size_of_vec;
    std::size_t additional_bytes = size_of_vec - cut_bytes;

    tiling.M +=
        additional_bytes / tiling.sizeof_type;  // выровняли по 32 байтам
  }

  uint32_t remainder_rows =
      n % tiling.num_of_ai_cores;  // кол-во строк, которые не останутся не
                                   // обработанными

  if (remainder_rows == 0) {  // каждый ai-core получает одинковое число строк
    tiling.count_of_based_blocks = tiling.num_of_ai_cores;
    tiling.count_of_cutted_blocks = 0;
    tiling.based_rows_per_block = n / tiling.num_of_ai_cores;
    tiling.cutted_rows_per_block = 0;
  } else {  // все блоки получают n / tiling.num_of_ai_cores строк, а также
            // remainder_rows блоков получают дополнительно по 1 строке
    tiling.count_of_based_blocks = remainder_rows;
    tiling.count_of_cutted_blocks = tiling.num_of_ai_cores - remainder_rows;
    tiling.based_rows_per_block = n / tiling.num_of_ai_cores + 1;
    tiling.cutted_rows_per_block = n / tiling.num_of_ai_cores;
  }

  if (tiling.M == tiling.N)  // данные выровнены по 32 байтам (нужно учесть, что
                             // DobuleBuffering)
  {
    tiling.elems_per_tile = tiling.tile_length / tiling.sizeof_type;
    tiling.tiles_per_row =
        (tiling.N + tiling.elems_per_tile - 1) /
        tiling.elems_per_tile;  // тут используется длина не alignутой строки
    tiling.length_last_tile = (tiling.N % tiling.elems_per_tile == 0)
                                  ? tiling.elems_per_tile
                                  : (tiling.N % tiling.elems_per_tile);
    tiling.length_last_tile_align = tiling.length_last_tile;
  } else {
    tiling.elems_per_tile = tiling.tile_length / tiling.sizeof_type;
    tiling.tiles_per_row =
        (tiling.N + tiling.elems_per_tile - 1) /
        tiling.elems_per_tile;  // тут используется длина не alignутой строки
    tiling.length_last_tile = (tiling.N % tiling.elems_per_tile == 0)
                                  ? tiling.elems_per_tile
                                  : (tiling.N % tiling.elems_per_tile);
    tiling.length_last_tile_align =
        (tiling.M - tiling.N) + tiling.length_last_tile % tiling.elems_per_tile;
  }
}

void print_tile_info(TileInfo &tiling) {
  std::cout << "=== TileInfo ===" << std::endl;
  std::cout << std::endl;

  std::cout << "0. M: " << (tiling.M) << std::endl;
  std::cout << "   Назначение: количество столбцов" << std::endl;
  std::cout << std::endl;

  std::cout << "1. N: " << (tiling.N) << std::endl;
  std::cout << "   Назначение: количество строк" << std::endl;
  std::cout << std::endl;

  std::cout << "2. num_of_ai_cores: " << (tiling.num_of_ai_cores) << std::endl;
  std::cout << "   Назначение: количество AI Core для обработки" << std::endl;
  std::cout << std::endl;

  std::cout << "3. tile_length: " << (tiling.tile_length) << std::endl;
  std::cout << "   Назначение: длина одного тайла в БАЙТАХ (обычно 512)"
            << std::endl;
  std::cout << std::endl;

  std::cout << "4. sizeof_type: " << (tiling.sizeof_type) << std::endl;
  std::cout << "   Назначение: размер одного элемента в БАЙТАХ" << std::endl;
  std::cout << std::endl;

  std::cout << "5. count_of_based_blocks: " << (tiling.count_of_based_blocks)
            << std::endl;
  std::cout
      << "   Назначение: количество блоков с based_rows_per_block строками"
      << std::endl;
  std::cout << std::endl;

  std::cout << "6. count_of_cutted_blocks: " << (tiling.count_of_cutted_blocks)
            << std::endl;
  std::cout
      << "   Назначение: количество блоков с cutted_rows_per_block строками"
      << std::endl;
  std::cout << std::endl;

  std::cout << "7. based_rows_per_block: " << (tiling.based_rows_per_block)
            << std::endl;
  std::cout << "   Назначение: строк в based-блоках" << std::endl;
  std::cout << std::endl;

  std::cout << "8. cutted_rows_per_block: " << (tiling.cutted_rows_per_block)
            << std::endl;
  std::cout << "   Назначение: строк в cutted-блоках" << std::endl;
  std::cout << std::endl;

  std::cout << "9. elems_per_tile: " << (tiling.elems_per_tile) << std::endl;
  std::cout << "   Назначение: элементов на тайл (НЕ байты!)" << std::endl;
  std::cout << "   Формула: tile_length / sizeof_type" << std::endl;
  std::cout << std::endl;

  std::cout << "10. tiles_per_row: " << (tiling.tiles_per_row) << std::endl;
  std::cout << "    Назначение: тайлов на строку" << std::endl;
  std::cout << "    Формула: (N + elems_per_tile - 1) / elems_per_tile"
            << std::endl;
  std::cout << std::endl;

  std::cout << "11. length_last_tile: " << (tiling.length_last_tile)
            << std::endl;
  std::cout << "    Назначение: элементов в последнем тайле (НЕ байты!)"
            << std::endl;
  std::cout << "    Формула: N % elems_per_tile (или elems_per_tile если 0)"
            << std::endl;
  std::cout << std::endl;

  std::cout << "12. length_last_tile_align: " << (tiling.length_last_tile_align)
            << std::endl;
  std::cout << "    Назначение: элементов в последнем тайле с выравниванием"
            << std::endl;
  std::cout << std::endl;
}

int32_t main(int32_t argc, char *argv[]) {
  uint32_t n = 512;
  if (argc > 1) {
    int parsed = std::atoi(argv[1]);

    if (parsed > 0) {
      n = static_cast<uint32_t>(parsed);
    } else {
      std::cerr << "Invalid argument n, using default value: 512\n";
    }
  }
  const char *socVersion = SOC_VERSION;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
  size_t input_file_size =
      n * n * sizeof(uint16_t);  // 16 bit -> 2 bytes of half
  size_t output_file_size = n * n * sizeof(float);

  // matmul TCubeTiling + localMemorySize

  size_t tilingFileSize = sizeof(TCubeTiling) + sizeof(uint64_t);
  size_t userWorkspaceSize = 0;
  size_t systemWorkspaceSize =
      static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
  size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
  uint8_t *tilingBuf = (uint8_t *)malloc(tilingFileSize);
  GenerateTiling(socVersion, tilingBuf, n);
#ifdef CUSTOM_ASCEND310P
  uint32_t blockDim = 2;
#else
  uint32_t blockDim = 1;
#endif

  TileInfo tiling_softmax;
  GenerateTilingSoftMax(n, tiling_softmax);
  print_tile_info(tiling_softmax);

#ifdef ASCENDC_CPU_DEBUG

  uint8_t *a = (uint8_t *)AscendC::GmAlloc(input_file_size);
  uint8_t *b = (uint8_t *)AscendC::GmAlloc(input_file_size);
  uint8_t *c = (uint8_t *)AscendC::GmAlloc(output_file_size);
  uint8_t *output = (uint8_t *)AscendC::GmAlloc(output_file_size);
  uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
  uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

  ReadFile("./input/A.bin", input_file_size, a, input_file_size);
  ReadFile("./input/B.bin", input_file_size, b, input_file_size);
  memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);

  ICPU_RUN_KF(matmul_custom, blockDim, a, b, c, workspace, tiling);

  WriteFile("./output/output_mult.bin", c, output_file_size);

  ICPU_RUN_KF(exp_custom, tiling_softmax.num_of_ai_cores, c, output,
              (uint8_t *)(&tiling_softmax));

  WriteFile("./output/output.bin", output, output_file_size);

  AscendC::GmFree((void *)a);
  AscendC::GmFree((void *)b);
  AscendC::GmFree((void *)c);
  AscendC::GmFree((void *)output);
  AscendC::GmFree((void *)workspace);
  AscendC::GmFree((void *)tiling);
#else

  CHECK_ACL(aclInit(nullptr));
  uint32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  // a_matrix
  uint8_t *a_host;
  uint8_t *a_device;
  CHECK_ACL(aclrtMallocHost((void **)(&a_host), input_file_size));
  CHECK_ACL(aclrtMalloc((void **)&a_device, input_file_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/A.bin", input_file_size, a_host, input_file_size);
  CHECK_ACL(aclrtMemcpy(a_device, input_file_size, a_host, input_file_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // b_matrix
  uint8_t *b_host;
  uint8_t *b_device;
  CHECK_ACL(aclrtMallocHost((void **)(&b_host), input_file_size));
  CHECK_ACL(aclrtMalloc((void **)&b_device, input_file_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./input/B.bin", input_file_size, b_host, input_file_size);
  CHECK_ACL(aclrtMemcpy(b_device, input_file_size, b_host, input_file_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // workspace
  uint8_t *workspace_device;
  CHECK_ACL(aclrtMalloc((void **)&workspace_device, workspaceSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  // tiling matmul
  uint8_t *tiling_matmul_host;
  uint8_t *tiling_matmul_device;
  CHECK_ACL(aclrtMallocHost((void **)(&tiling_matmul_host), tilingFileSize));
  CHECK_ACL(aclrtMalloc((void **)&tiling_matmul_device, tilingFileSize,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tiling_matmul_host, tilingFileSize, tilingBuf,
                        tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
  CHECK_ACL(aclrtMemcpy(tiling_matmul_device, tilingFileSize,
                        tiling_matmul_host, tilingFileSize,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // c_matrix
  uint8_t *c_device;
  uint8_t *c_host;
  CHECK_ACL(aclrtMallocHost((void **)(&c_host), output_file_size));
  CHECK_ACL(aclrtMalloc((void **)&c_device, output_file_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACLRT_LAUNCH_KERNEL(matmul_custom)
  (blockDim, stream, a_device, b_device, c_device, workspace_device,
   tiling_matmul_device);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(c_host, output_file_size, c_device, output_file_size,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  WriteFile("./output/output_matmul_c.bin", c_host, output_file_size);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtFree(a_device));
  CHECK_ACL(aclrtFreeHost(a_host));
  CHECK_ACL(aclrtFree(b_device));
  CHECK_ACL(aclrtFreeHost(b_host));
  CHECK_ACL(aclrtFree(workspace_device));
  CHECK_ACL(aclrtFree(tiling_matmul_device));
  CHECK_ACL(aclrtFreeHost(tiling_matmul_host));

  // tiling softmax
  uint8_t *tiling_softmax_device;
  CHECK_ACL(aclrtMalloc((void **)&tiling_softmax_device, sizeof(TileInfo),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tiling_softmax_device, sizeof(TileInfo),
                        (uint8_t *)(&tiling_softmax), sizeof(TileInfo),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  // output matrix
  uint8_t *output_host;
  uint8_t *output_device;

  CHECK_ACL(aclrtMallocHost((void **)(&output_host), output_file_size));
  CHECK_ACL(aclrtMalloc((void **)&output_device, output_file_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACLRT_LAUNCH_KERNEL(exp_custom)
  (tiling_softmax.num_of_ai_cores, stream, c_device, output_device,
   tiling_softmax_device);
  // exp_custom_do(tiling_softmax.num_of_ai_cores, stream, c_device,
  // output_device, tiling_softmax_device);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(output_host, output_file_size, output_device,
                        output_file_size, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output.bin", output_host, output_file_size);

  CHECK_ACL(aclrtFree(c_device));
  CHECK_ACL(aclrtFreeHost(c_host));
  CHECK_ACL(aclrtFree(output_device));
  CHECK_ACL(aclrtFreeHost(output_host));
  CHECK_ACL(aclrtFree(tiling_softmax_device));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif
  free(tilingBuf);
  return 0;
}