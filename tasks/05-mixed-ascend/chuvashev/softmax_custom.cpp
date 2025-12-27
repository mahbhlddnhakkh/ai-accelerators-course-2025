#include "kernel_operator.h"

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

__aicore__ inline float ReduceSum(AscendC::LocalTensor<float> &src,
                                  uint32_t reduce_len) {
  float sum = 0;
  for (uint32_t i = 0; i < reduce_len; ++i) {
    sum += src.GetValue(i);
  }
  return sum;
}

class KernelSoftmax {
 private:
  uint32_t N;
  uint32_t M;
  uint32_t num_of_ai_cores;
  uint32_t tile_length;
  uint32_t sizeof_type;
  uint32_t count_of_based_blocks;
  uint32_t count_of_cutted_blocks;
  uint32_t based_rows_per_block;
  uint32_t cutted_rows_per_block;
  uint32_t elems_per_tile;
  uint32_t tiles_per_row;
  uint32_t length_last_tile;
  uint32_t length_last_tile_align;
  uint32_t buffer_num;

  // uint32_t global_offset_x = 0;
  // uint32_t global_offset_y = 0;
  uint32_t global_offset = 0;
  uint32_t count_of_rows = 0;

  AscendC::TPipe *pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> in_queue_x;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_queue_y;

  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_sum;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_exp;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_reduce;
  // AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_for_div;

  AscendC::GlobalTensor<float> x_global;
  AscendC::GlobalTensor<float> y_global;

 public:
  __aicore__ inline KernelSoftmax(TileInfo *tile_ptr) {
    this->N = tile_ptr->N;
    this->M = tile_ptr->M;
    this->num_of_ai_cores = tile_ptr->num_of_ai_cores;
    this->tile_length = tile_ptr->tile_length;
    this->sizeof_type = tile_ptr->sizeof_type;
    this->count_of_based_blocks = tile_ptr->count_of_based_blocks;
    this->count_of_cutted_blocks = tile_ptr->count_of_cutted_blocks;
    this->based_rows_per_block = tile_ptr->based_rows_per_block;
    this->cutted_rows_per_block = tile_ptr->cutted_rows_per_block;
    this->elems_per_tile = tile_ptr->elems_per_tile;
    this->tiles_per_row = tile_ptr->tiles_per_row;
    this->length_last_tile = tile_ptr->length_last_tile;
    this->length_last_tile_align = tile_ptr->length_last_tile_align;
    this->buffer_num = tile_ptr->buffer_num;
  }

  __aicore__ inline void Init(AscendC::TPipe *p, GM_ADDR x, GM_ADDR y) {
    pipe = p;
    uint32_t block_idx = AscendC::GetBlockIdx();

    if (block_idx < count_of_based_blocks) {  // считаем смещенеи по строкам
      global_offset =
          block_idx * based_rows_per_block * M;  // тут с обработкой padding'a
      // global_offset_x = block_idx * based_rows_per_block * M; // тут с
      // обработкой padding'a global_offset_y = block_idx * based_rows_per_block
      // * N; // тут именно N (так как выходная матрица должна быть без
      // padding'a)
      count_of_rows = based_rows_per_block;
    } else {
      global_offset = count_of_based_blocks * based_rows_per_block * M;
      // global_offset_x = count_of_based_blocks * based_rows_per_block * M;
      // global_offset_y = count_of_based_blocks * based_rows_per_block * N;
      global_offset +=
          (block_idx - count_of_based_blocks) * cutted_rows_per_block * M;
      // global_offset_x += (block_idx - count_of_based_blocks) *
      // cutted_rows_per_block * M; global_offset_y += (block_idx -
      // count_of_based_blocks) * cutted_rows_per_block * N;
      count_of_rows = cutted_rows_per_block;
    }

    // x_global.SetGlobalBuffer((__gm__ float*)x + global_offset_x,
    // count_of_rows * M * sizeof(float)); y_global.SetGlobalBuffer((__gm__
    // float*)y + global_offset_y, count_of_rows * N * sizeof(float));

    x_global.SetGlobalBuffer((__gm__ float *)x + global_offset,
                             count_of_rows * M * sizeof(float));
    y_global.SetGlobalBuffer((__gm__ float *)y + global_offset,
                             count_of_rows * M * sizeof(float));

    pipe->InitBuffer(in_queue_x, buffer_num,
                     tile_length);  // с учетом DoubleBuffering
    pipe->InitBuffer(out_queue_y, buffer_num,
                     tile_length);  // с учетом DoubleBuffering

    pipe->InitBuffer(buffer_for_sum, elems_per_tile * sizeof(float));
    pipe->InitBuffer(buffer_for_exp, elems_per_tile * sizeof(float));
    // pipe->InitBuffer(buffer_for_div, elems_per_tile * sizeof(float));
    pipe->InitBuffer(buffer_for_reduce, sizeof(float));
  }

  __aicore__ inline void CopyIn(uint32_t offset, uint32_t elems) {
    AscendC::LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();

    AscendC::DataCopy(x_local, x_global[offset], elems);

    in_queue_x.EnQue(x_local);
  }

  __aicore__ inline void ComputeExpsAndSum(uint32_t aligned_elems,
                                           uint32_t actual_elems,
                                           AscendC::LocalTensor<float> &exps,
                                           AscendC::LocalTensor<float> &sums) {
    AscendC::LocalTensor<float> x_local = in_queue_x.DeQue<float>();

    // exp для alignутой строки
    AscendC::Exp(exps, x_local, aligned_elems);

    if (aligned_elems != actual_elems) {
      uint64_t mask0 =
          ((uint64_t)1 << aligned_elems) - ((uint64_t)1 << actual_elems);
      uint64_t mask[2] = {mask0, 0};
      AscendC::Duplicate(exps, 0.0f, mask, 1, 1, 1);
    }

    AscendC::Add(sums, sums, exps, aligned_elems);

    in_queue_x.FreeTensor(x_local);
  }

  __aicore__ inline void DivideOnExps(uint32_t aligned_elems,
                                      uint32_t actual_elems,
                                      AscendC::LocalTensor<float> &exps,
                                      AscendC::LocalTensor<float> &sums) {
    AscendC::LocalTensor<float> x_local = in_queue_x.DeQue<float>();
    AscendC::LocalTensor<float> y_local = out_queue_y.AllocTensor<float>();

    // AscendC::LocalTensor<float> div = buffer_for_div.Get<float>();

    AscendC::Exp(exps, x_local, aligned_elems);

    if (aligned_elems != actual_elems) {
      uint64_t mask0 =
          ((uint64_t)1 << aligned_elems) - ((uint64_t)1 << actual_elems);
      uint64_t mask[2] = {mask0, 0};

      AscendC::Duplicate(exps, 0.0f, mask, 1, 1, 1);
    }

    // AscendC::Div(y_local, exps, div, aligned_elems);
    AscendC::Div(y_local, exps, sums, aligned_elems);

    in_queue_x.FreeTensor(x_local);
    out_queue_y.EnQue(y_local);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t aligned_elems) {
    // uint32_t offset = r * N + t * elems_per_tile;

    AscendC::LocalTensor<float> y_local = out_queue_y.DeQue<float>();

    AscendC::DataCopy(y_global[offset], y_local, aligned_elems);

    out_queue_y.FreeTensor(y_local);
  }

  __aicore__ inline void Process() {
    AscendC::LocalTensor<float> sum_tensor =
        buffer_for_sum.Get<float>();  // тензор для хранения суммы exp по всем
                                      // тайлам в строке
    AscendC::LocalTensor<float> exp_tensor =
        buffer_for_exp.Get<float>();  // тензор для хранения exp текущего тайла
    AscendC::LocalTensor<float> reduce_scalar =
        buffer_for_reduce.Get<float>();  // хранит значение редуцированной суммы
    // AscendC::LocalTensor<float> div =
    //     buffer_for_div.Get<float>();  // хранит элементы, на которые будем
    //                                   // делить тайлы в строке

    for (uint32_t r = 0; r < count_of_rows; ++r) {
      AscendC::Duplicate(sum_tensor, 0.0f, elems_per_tile);

      uint32_t row_start_addres =
          r * M;  // тут именно M так как мы работаем с выровненной матрицей

      for (uint32_t t = 0; t < tiles_per_row; ++t) {
        uint32_t offset = row_start_addres + t * elems_per_tile;
        uint32_t aligned_elems =
            (t == tiles_per_row - 1) ? length_last_tile_align : elems_per_tile;
        uint32_t actual_elems =
            (t == tiles_per_row - 1) ? length_last_tile : elems_per_tile;

        CopyIn(offset, aligned_elems);
        ComputeExpsAndSum(aligned_elems, actual_elems, exp_tensor, sum_tensor);
      }

      const uint32_t shape[] = {1, elems_per_tile};
      //       float value = 0.0f;
      // #ifdef CUSTOM_ASCEND310P
      //       value = ReduceSum(sum_tensor, shape[0]);
      // #else
      //       AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
      //           reduce_scalar, sum_tensor, shape, true);
      //       value = reduce_scalar.GetValue(0);
      // #endif
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(
          reduce_scalar, sum_tensor, shape, true);
      float value = reduce_scalar.GetValue(0);
      // AscendC::Duplicate(div, value, elems_per_tile);
      AscendC::Duplicate(sum_tensor, value, elems_per_tile);

      for (uint32_t t = 0; t < tiles_per_row; ++t) {
        uint32_t offset = row_start_addres + t * elems_per_tile;
        uint32_t aligned_elems =
            (t == tiles_per_row - 1) ? length_last_tile_align : elems_per_tile;
        uint32_t actual_elems =
            (t == tiles_per_row - 1) ? length_last_tile : elems_per_tile;

        CopyIn(offset, aligned_elems);
        DivideOnExps(aligned_elems, actual_elems, exp_tensor, sum_tensor);
        CopyOut(offset, aligned_elems);
      }
    }
  }
};

extern "C" __global__ __aicore__ void exp_custom(GM_ADDR x, GM_ADDR y,
                                                 GM_ADDR tiling) {
  TileInfo tile;

  tile.N = ((__gm__ TileInfo *)tiling)->N;
  tile.M = ((__gm__ TileInfo *)tiling)->M;
  tile.num_of_ai_cores = ((__gm__ TileInfo *)tiling)->num_of_ai_cores;
  tile.tile_length = ((__gm__ TileInfo *)tiling)->tile_length;
  tile.sizeof_type = ((__gm__ TileInfo *)tiling)->sizeof_type;
  tile.count_of_based_blocks =
      ((__gm__ TileInfo *)tiling)->count_of_based_blocks;
  tile.count_of_cutted_blocks =
      ((__gm__ TileInfo *)tiling)->count_of_cutted_blocks;
  tile.based_rows_per_block = ((__gm__ TileInfo *)tiling)->based_rows_per_block;
  tile.cutted_rows_per_block =
      ((__gm__ TileInfo *)tiling)->cutted_rows_per_block;
  tile.elems_per_tile = ((__gm__ TileInfo *)tiling)->elems_per_tile;
  tile.tiles_per_row = ((__gm__ TileInfo *)tiling)->tiles_per_row;
  tile.length_last_tile = ((__gm__ TileInfo *)tiling)->length_last_tile;
  tile.length_last_tile_align =
      ((__gm__ TileInfo *)tiling)->length_last_tile_align;
  tile.buffer_num = ((__gm__ TileInfo *)tiling)->buffer_num;

  KernelSoftmax op(&tile);

  AscendC::TPipe pipe;
  op.Init(&pipe, x, y);
  op.Process();

  pipe.Destroy();
}

#ifndef ASCENDC_CPU_DEBUG
void exp_custom_do(uint32_t block_dim, void *stream, uint8_t *x, uint8_t *y,
                   uint8_t *tiling) {
  exp_custom<<<block_dim, nullptr, stream>>>(x, y, tiling);
}
#endif