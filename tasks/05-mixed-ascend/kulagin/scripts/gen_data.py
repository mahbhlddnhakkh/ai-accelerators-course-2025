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

def softmax(m):
    n = m.shape[0]
    m_exp = np.exp(m)
    m_exp_row_sum = np.sum(m_exp, axis=1).reshape((n, 1))
    return m_exp / m_exp_row_sum

def gen_golden_data():
    _N = 256

    M = 1024
    N = 640
    K = 256
    M = _N
    N = _N
    K = _N

    #input_a = np.random.randint(1, 10, [M, K]).astype(np.float16)
    #input_b = np.random.randint(1, 10, [K, N]).astype(np.float16)
    #input_bias = np.random.randint(1, 10, [N]).astype(np.float32)
    input_a = np.random.uniform(-1.0, 1.0, [N, N]).reshape(N, N).astype(np.float16)
    input_b = np.random.uniform(-1.0, 1.0, [N, N]).reshape(N, N).astype(np.float16)
    input_bias = np.random.uniform(-1.0, 1.0, [N]).astype(np.float32)
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    #golden = np.where(golden >= 0, golden, golden * 0.001)
    golden = softmax(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
