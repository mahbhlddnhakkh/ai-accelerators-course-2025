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
import os
SIZE = 96

def softmax(_matrix):
    mx_exp = np.exp(_matrix)
    row_sum = np.sum(mx_exp, axis=1).reshape(SIZE, 1)
    return mx_exp/row_sum

def gen_golden_data():
    alpha = 0.001

    input_a = np.random.randint(-1, 1, [SIZE, SIZE]).astype(np.float16) * alpha
    input_b = np.random.randint(-1, 1, [SIZE, SIZE]).astype(np.float16) * alpha
    input_bias = np.random.randint(-1, 1, [SIZE]).astype(np.float32)
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    #golden = np.where(golden >= 0, golden, golden * alpha)
    golden1 = softmax(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    golden.tofile("./output/golden.bin")
    golden1.tofile("./output/golden_softmax.bin")


if __name__ == "__main__":
    gen_golden_data()
