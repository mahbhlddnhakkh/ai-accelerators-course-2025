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

def softmax(m):
    n = m.shape[0]
    m_exp = np.exp(m)
    m_exp_row_sum = np.sum(m_exp, axis=1).reshape((n, 1))
    return m_exp / m_exp_row_sum

def gen_golden_data_simple():
    N = 64
    input_ = np.random.uniform(-10.0, 10.0, [N, N]).reshape(N, N).astype(np.float32)
    golden = softmax(input_).astype(np.float32)

    input_.tofile("./input/input.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
