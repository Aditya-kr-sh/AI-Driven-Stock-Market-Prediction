# Ramanujan Universe (RU) HPC Cluster Execution Guide

This document describes the environment configuration, compilation directives, job submission workflow, and performance reporting framework to run the stock prediction platform on the Ramanujan Universe Linux HPC Cluster.

---

## 🏗️ 1. Environment Diagnostics & Setup

Before executing workloads on the cluster, verify system capabilities and library dependencies:

1. **Verify environment module compatibility**:
   Check which compiler versions, Python packages, and CUDA environments are available on your cluster node:
   ```bash
   module avail gcc
   module avail cuda
   module avail python
   ```
2. **Run Diagnostics**:
   Execute our diagnostic script to inspect CPU resources, GCC support, CUDA drivers, and package imports:
   ```bash
   python scripts/check_hpc_environment.py
   ```
   *Verify that CUDA Availability is reported as `True` and the active GPU device name and VRAM size match cluster expectations.*

---

## ⚙️ 2. Cython & OpenMP C-Extension Compilation

The accelerated technical indicators pipeline requires compiling our Cython source code on the cluster nodes using GCC:

1. **Clean prior builds**:
   ```bash
   rm -f ai_engine/features/*.so
   rm -rf build/
   ```
2. **Trigger Compilation**:
   Run the unified setup driver to compile the C-extensions using the cluster's GCC compiler with OpenMP flags activated automatically:
   ```bash
   python setup.py build_ext --inplace
   ```
3. **Verify loading**:
   Ensure the newly compiled shared library (`.so`) is imported correctly:
   ```bash
   python -c "from ai_engine.features import is_openmp_available; print('OpenMP Active:', is_openmp_available())"
   ```

---

## 🚀 3. Batch Job Scheduling via Slurm

Large-scale training and benchmarks should be run asynchronously using the cluster's Slurm workload scheduler:

1. **Instantiate the Job Script**:
   Copy the parameterized Slurm template:
   ```bash
   cp scripts/submit_job.sh.template scripts/submit_job.sh
   chmod +x scripts/submit_job.sh
   ```
2. **Configure Placeholders**:
   Open `scripts/submit_job.sh` and customize the placeholder elements (`<...>`) based on your cluster project profile:
   - **`--partition`**: Target partition (e.g. `gpu`, `compute`).
   - **`--gres`**: Request GPU allocation (e.g. `gpu:1`, `gpu:a100:1`).
   - **`--cpus-per-task`**: Thread count limit mapped directly to OpenMP loops (e.g., `12` or `16`).
   - **`--mem`**: Memory bounds allocated to nodes (e.g., `32G` or `64G`).
   - **`--time`**: Job time limits (e.g., `04:00:00`).
   - **Module commands**: Uncomment and write the target modules to load compiler and CUDA packages (e.g., `module load gcc/11.2.0 cuda/12.1.1`).
3. **Submit Workloads**:
   Queue the batch execution run:
   ```bash
   sbatch scripts/submit_job.sh
   ```
4. **Monitor Progress**:
   Check allocation status:
   ```bash
   squeue -u $USER
   ```

---

## 📊 4. Performance Logs & Reports

Upon completion, the pipeline outputs two summary reports in the `docs/` directory:

### A. Feature Pipeline Benchmarks (`docs/openmp_hpc_benchmark_report.txt`)
Contains execution scaling timelines of technical indicator computations run on the complete available NIFTY-50 historical series, comparing:
- Python/NumPy Baseline.
- OpenMP Single Thread.
- OpenMP Multi-threaded (scaling up to available cores allocation).

### B. Final Model Evaluations (`docs/final_model_evaluation_report.txt`)
Details performance metrics (MSE, RMSE, MAE, R², Directional Accuracy, Pearson Correlation) and training/inference execution timelines for the XGBoost, LSTM, and Transformer architectures.

### C. Reproducibility Logs
Both reports record execution metadata to guarantee academic reproducibility:
- **Git commit hash**: Traces execution source.
- **Random seed**: Initialized seeds (default: 42) for Python, NumPy, and PyTorch (CUDA).
- **Hardware Profile**: CPU model, logical thread counts, compiler (GCC) details, and GPU card/VRAM specification.
- **Timestamp**: Execution logging times.
