# AI Accelerators Course

Hands-on programming labs for exploring how classic deep-learning primitives run on different accelerator backends.

## Repository Layout
- `tasks/01-softmax-cpu/` - CPU reference implementation of softmax with a runnable example target.
- `tasks/02-softmax-cuda/` - CUDA port of the softmax kernel plus a simple harness.
- `tasks/03-matmul-cuda/` - CUDA matrix multiplication exercise and demo driver.
- `tasks/04-softmax-ascend/` - Softmax operators targeting Huawei Ascend hardware.
- `tasks/05-mixed-ascend/` - Mixed matrix multiplication + Softmax Ascend exercises building on the softmax work.

## Build & Run
All tasks are optional CMake targets controlled by feature flags:

```bash
cmake -S . -B build -DENABLE_CPU=ON -DENABLE_CUDA=ON -DENABLE_ASCEND=OFF
cmake --build build
```

Each task exposes its sample executable under `tasks/<task-name>/<your_id>`. After building, binaries live in `build/tasks/<task-name>/<your_id>/`. Toggle the `ENABLE_*` switches to limit the build to the environments available on your machine.
