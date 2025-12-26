#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
SIZE = 64

def softmax(_matrix):
    mx_exp = np.exp(_matrix)
    row_sum = np.sum(mx_exp, axis=1).reshape(SIZE, 1)
    return mx_exp/row_sum

def gen_golden_data_simple():
    input_matrix = np.random.uniform(-1, 1, [SIZE, SIZE]).astype(np.float32)
    golden = softmax(input_matrix).astype(np.float32)

    input_matrix.tofile("./input/input_matrix.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
