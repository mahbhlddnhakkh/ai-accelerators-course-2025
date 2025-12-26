import numpy as np
import argparse

def gen_golden_data_simple():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="matrix size N x N")
    
    args = parser.parse_args()

    N = args.N
    M = N
    sizeof_value = 4
    sizeof_vec = 32

    bytes_length = N * sizeof_value

    if bytes_length % sizeof_vec != 0:
        bytes_of_cut = bytes_length % sizeof_vec
        additional_bytes = sizeof_vec - bytes_of_cut
        M = N + additional_bytes // sizeof_value

    input_x = np.random.uniform(-1, 1, (N, M)).astype(np.float32)

    input_actual = input_x[:, :N]

    exp_vals = np.exp(input_actual)

    softmax_actual = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    # input_x = np.full((N, N), 0).astype(np.float32)
    golden = np.zeros((N, M), dtype=np.float32)
    golden[:, :N] = softmax_actual

    print(f"Shape of input matrix: {input_x.shape}" )
    print(f"Shape of golden matrix: {golden.shape}" )

    # print(input_x[0:40])
    # print(golden[0:40])

    input_x.reshape(-1).tofile("./input/input_x.bin")
    golden.reshape(-1).tofile("./output/golden.bin")

    # np.savetxt("./input/input_x.txt", input_x, fmt='%.6f', delimiter='\t')
    # np.savetxt("./output/golden.txt", golden, fmt='%.6f', delimiter='\t')

if __name__ == "__main__":
    gen_golden_data_simple()
