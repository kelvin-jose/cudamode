# cudamode

This repository tracks my progress in learning CUDA programming for GPU acceleration and parallel computation. It includes various experiments, organized modules, and sample kernels to explore different CUDA optimization techniques.

---

## 📁 Repository Structure

### `matrix_vector_warp/`

Experiments involving warp-level operations and matrix-vector multiplications.

- `include/`
  - `utils.cuh`: Utility functions for warp-level operations.
  - `warp_op.cuh`: Warp-specific operation headers.
- `kernel/`
  - `warp_op.cu`: CUDA implementation of warp-level operations.
  - `main.cu`: Driver code to execute the warp-based matrix-vector operation.
- `Makefile`: Build instructions for the warp-level module.

---

### `SGEMM/`

Single-precision General Matrix Multiply (SGEMM) experiments using different optimization techniques.

- `include/`
  - `kernel_00_naive.cuh`: Naive SGEMM implementation.
  - `kernel_01_coalesced.cuh`: Memory coalescing-optimized version.
  - `kernel_02_shared_mem.cuh`: Shared memory optimized version.
- `kernels.cuh`: Common kernel utilities.
- `main.cu`: Host code to test the SGEMM kernels.

---

### Other CUDA Programs

Standalone CUDA programs exploring various operations.

- `layernorm.cu`: Layer normalization implementation.
- `matrix_add.cu`: Matrix addition kernel.
- `matrix_mult.cu`: Matrix multiplication (likely basic).
- `matrix_transpose.cu`: Matrix transposition.
- `matrix_vector_mul.cu`: Matrix-vector multiplication.
- `vector_add.cu`: Vector addition example.

---

## 🔧 Build & Run

Each directory may have its own `Makefile`. To build a specific module:

```bash
cd matrix_vector_warp
make
./main
```

---

## 🧠 Topics Covered

- Warp-level parallelism
- Memory coalescing
- Shared memory optimization
- SGEMM kernel design
- CUDA memory hierarchy
- Basic matrix operations (add, multiply, transpose)
- Layer normalization

---

## 📌 Notes

This repository is intended as a personal learning space. Contributions or suggestions are welcome, but the primary goal is educational exploration.

---

## 📄 License

This project is not currently licensed. Feel free to explore and adapt the code for personal learning purposes.

