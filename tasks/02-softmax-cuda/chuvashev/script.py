import subprocess
import matplotlib.pyplot as plt

name_of_proc = "output.exe"
name_of_bat = "script.bat"

values = ["1024", "2048", "4096", "10000", "15000", "20000"]
threads = ["64", "128", "256", "512", "1024"]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'teal', 'magenta', 'olive', 'coral', 'navy']

all_data = {}

cmd_compile = [name_of_bat, "1024"]
result_compile = subprocess.run(cmd_compile, capture_output=True, text=True)

sequential_times = []
simd_times = []
for value in values:
    cmd = [name_of_proc, value]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    
    lines = result.stdout.split('\n')
    
    line_seq = lines[0].split(' ')
    sequential_times.append(float(line_seq[1]))
    
    line_opm_avx2 = lines[3].split(' ')
    simd_times.append(float(line_opm_avx2[1]))

all_data['sequential'] = sequential_times
all_data['simd'] = simd_times

for thread in threads:
    cmd_compile = [name_of_bat, thread]
    result_compile = subprocess.run(cmd_compile, capture_output=True, text=True)
    simt_times = []
    cuda_alg_times = []
    for value in values:
        cmd = [name_of_proc, value]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        
        lines = result.stdout.split('\n')

        line_simt = lines[1].split(' ')
        simt_times.append(float(line_simt[1]))
        
        line_cuda_alg = lines[2].split(' ')
        cuda_alg_times.append(float(line_cuda_alg[1]))

    all_data[f'cuda_{thread}'] = cuda_alg_times
    all_data[f'simt_{thread}'] = simt_times

sizes = [int(v) for v in values]

plt.figure(figsize=(12, 8))

plt.plot(sizes, all_data['simd'], marker='x', label='OpenMP + SIMD', color=colors[0], linewidth=2)
plt.plot(sizes, all_data['sequential'], marker='x', label='Sequential', color=colors[1], linewidth=2)

temp = 2
for thread in threads:
    plt.plot(sizes, all_data[f'cuda_{thread}'], marker='x', label=f"CUDA TPB={thread}", color=colors[temp])
    temp += 1
    plt.plot(sizes, all_data[f'simt_{thread}'], marker='x', label=f"SIMT TPB={thread}", color=colors[temp])
    temp += 1

plt.xlabel("Количество элементов (n)")
plt.ylabel("Время (сек)")
plt.title("Сравнение всех методов вычисления Softmax")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("softmax_all_methods.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sizes, all_data['sequential'], marker='o', label='Sequential', color='red', linewidth=2)
plt.plot(sizes, all_data['simd'], marker='s', label='OpenMP + SIMD', color='blue', linewidth=2)

plt.xlabel("Количество элементов (n)")
plt.ylabel("Время (сек)")
plt.title("Сравнение Sequential и SIMD методов")
plt.legend()
plt.grid(True)
plt.savefig("softmax_simd_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 6))

for i, thread in enumerate(threads):
    plt.plot(sizes, all_data[f'cuda_{thread}'], marker='o', label=f"CUDA TPB={thread}", color=colors[i])

plt.xlabel("Количество элементов (n)")
plt.ylabel("Время (сек)")
plt.title("Сравнение CUDA методов с разными блоками")
plt.legend()
plt.grid(True)
plt.savefig("softmax_cuda_comparison.png", dpi=300)
plt.show()

plt.figure(figsize=(12, 8))

for i, thread in enumerate(threads):
    plt.plot(sizes, all_data[f'cuda_{thread}'], marker='o', label=f"CUDA TPB={thread}", color=colors[i])
    plt.plot(sizes, all_data[f'simt_{thread}'], marker='s', linestyle='--', label=f"SIMT TPB={thread}", color=colors[i])

plt.xlabel("Количество элементов (n)")
plt.ylabel("Время (сек)")
plt.title("Сравнение CUDA и SIMT методов")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("softmax_cuda_vs_simt.png", dpi=300, bbox_inches='tight')
plt.show()