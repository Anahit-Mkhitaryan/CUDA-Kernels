
```markdown
# CUDA Kernels Homework

This repository explores three CUDA kernel strategies and analyzes their performance with respect to warp divergence.
## 📘 Description

The goal of this project is to explore and analyze how warp divergence affects performance in CUDA kernels. We implement three kernels:

1. **No Divergence** – All threads take the same execution path.
2. **Divergence** – Threads in a warp follow different paths based on thread index parity.
3. **Warp-Aligned** – Conditional logic aligned to warp boundaries to minimize divergence.

## 📂 Files

| File                | Description                                                |
|---------------------|------------------------------------------------------------|
| `no_divergence.cu`  | CUDA kernel where all threads follow the same path         |
| `divergence.cu`     | Kernel with warp divergence (`if (idx % 2 == 0)`)          |
| `warp_aligned.cu`   | Optimized kernel with warp-aligned branching               |
| `report.tex`        | LaTeX report with full analysis, execution times, and charts (optional) |
| `README.md`         | This file                                                  |

## ⚙️ How to Run

You can run this project in **Google Colab** with CUDA support:

1. Upload the `.cu` file you'd like to run.
2. Use the following magic to compile and execute:
   ```cpp
   %%writefile your_kernel_file.cu
   // paste your code here
   ```

   ```cpp
   !nvcc your_kernel_file.cu -o kernel_exec
   !./kernel_exec
   ```

3. Make sure your Colab runtime is set to use a **GPU**:  
   *Runtime → Change runtime type → Hardware accelerator: GPU*

## 🧪 Performance Results

Each kernel was executed 6 times. Execution times were measured using CUDA events. Results showed:

- **No Divergence**: Lowest average time.
- **Divergence**: Highest time due to warp serialization.
- **Warp-Aligned**: Improved performance by avoiding intra-warp divergence.

### Summary of Average Execution Times

| Kernel            | Mean Time (ms) | Mean w/o Outlier (ms) |
|-------------------|----------------|------------------------|
| No Divergence     | 7.63           | 7.36                   |
| Divergence        | 8.93           | 7.61                   |
| Warp-Aligned      | 8.08           | 7.49                   |

## 📌 Conclusion

Warp divergence can lead to significant performance penalties in GPU programming. Aligning control logic with warp boundaries improves performance by reducing branching costs.

