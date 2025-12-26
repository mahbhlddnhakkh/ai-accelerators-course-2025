#include <iostream>

#include "data_utils.h"

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
  uint32_t length_last_tile_align;  // выравниванием по 32 байтам (НЕ в байтах)
  uint32_t buffer_num;
};

void GenerateTilingData(uint32_t n, TileInfo& tiling) {
  tiling.N = n;
  tiling.buffer_num = 2;

  if (tiling.N < 16) {
    tiling.num_of_ai_cores = tiling.N;
  } else {
    tiling.num_of_ai_cores = 16;
  }

  tiling.tile_length =
      1024 / tiling.buffer_num;  // учитываем, что DoubleBuffering
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

void print_tile_info(TileInfo& tiling) {
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

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void exp_custom_do(uint32_t block_dim, void* stream, uint8_t* x,
                          uint8_t* y, uint8_t* tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR tiling);
#endif

int main(int argc, char* argv[]) {
  uint32_t n = 2048;

  if (argc > 1) {
    int parsed = std::atoi(argv[1]);

    if (parsed > 0) {
      n = static_cast<uint32_t>(parsed);
    } else {
      std::cerr << "Invalid argument n, using default value: 2048\n";
    }
  }

  TileInfo tiling;
  GenerateTilingData(n, tiling);

  std::size_t input_count_of_bytes = tiling.N * tiling.M * sizeof(float);
  // std::size_t output_count_of_bytes = tiling.N * tiling.N * sizeof(float);
  std::size_t output_count_of_bytes = tiling.N * tiling.M * sizeof(float);

  std::cout << "SIZE IN MB: " << (float)input_count_of_bytes / (1024.0 * 1024.0)
            << std::endl;

  print_tile_info(tiling);

#ifdef ASCENDC_CPU_DEBUG

  uint8_t* x = (uint8_t*)AscendC::GmAlloc(input_count_of_bytes);
  uint8_t* y = (uint8_t*)AscendC::GmAlloc(output_count_of_bytes);

  ReadFile("./input/input_x.bin", input_count_of_bytes, x,
           input_count_of_bytes);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);

  ICPU_RUN_KF(exp_custom, tiling.num_of_ai_cores, x, y, (uint8_t*)(&tiling));

  WriteFile("./output/output_y.bin", y, output_count_of_bytes);

  AscendC::GmFree(x);
  AscendC::GmFree(y);

#else

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *x_host, *y_host;
  uint8_t *x_device, *y_device, *tiling_device;

  CHECK_ACL(aclrtMallocHost((void**)&x_host, input_count_of_bytes));
  CHECK_ACL(aclrtMallocHost((void**)&y_host, output_count_of_bytes));

  ReadFile("./input/input_x.bin", input_count_of_bytes, x_host,
           input_count_of_bytes);

  CHECK_ACL(aclrtMalloc((void**)&x_device, input_count_of_bytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&y_device, output_count_of_bytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&tiling_device, sizeof(TileInfo),
                        ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMemcpy(x_device, input_count_of_bytes, x_host,
                        input_count_of_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(tiling_device, sizeof(TileInfo), (uint8_t*)(&tiling),
                        sizeof(TileInfo), ACL_MEMCPY_HOST_TO_DEVICE));

  exp_custom_do(tiling.num_of_ai_cores, stream, x_device, y_device,
                tiling_device);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(y_host, output_count_of_bytes, y_device,
                        output_count_of_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output_y.bin", y_host, output_count_of_bytes);

  CHECK_ACL(aclrtFree(x_device));
  CHECK_ACL(aclrtFree(y_device));
  CHECK_ACL(aclrtFree(tiling_device));
  CHECK_ACL(aclrtFreeHost(x_host));
  CHECK_ACL(aclrtFreeHost(y_host));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtResetDevice(0));
  CHECK_ACL(aclFinalize());

#endif

  return 0;
}
