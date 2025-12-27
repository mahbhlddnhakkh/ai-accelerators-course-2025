import numpy as np
import os
import argparse

def gen_golden_data():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="matrix size N x N")
    
    args = parser.parse_args()

    N = args.N

    alpha = 0.001

    input_a = np.random.randint(1, 10, [N, N]).astype(np.float16) * alpha
    input_b = np.random.randint(1, 10, [N, N]).astype(np.float16) * alpha

    matmul = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32))).astype(np.float32)
    
    exp_vals = np.exp(matmul)

    softmax_actual = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    
    input_a.tofile("./input/A.bin")
    input_b.tofile("./input/B.bin")

    softmax_actual.tofile("./output/golden.bin")
    matmul.tofile("./output/golden_mult.bin")


if __name__ == "__main__":
    gen_golden_data()
