"""
HPC Environment Diagnostics Tool.
Queries and reports platform details, package versions, CPU cores, 
GCC availability, OpenMP compilation status, and CUDA/GPU hardware specs.
"""

import sys
import os
import platform
import subprocess
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn

def get_cpu_info() -> str:
    """Safely retrieves CPU model name on Windows and Linux systems."""
    try:
        if platform.system() == "Windows":
            # Query registry
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(val).strip()
        elif platform.system() == "Linux":
            # Query proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown Processor"


def get_gcc_version() -> str:
    """Queries GCC version using CLI commands safely."""
    try:
        res = subprocess.run(["gcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return res.stdout.split("\n")[0]
    except Exception:
        pass
    return "GCC not found in system PATH"


def check_openmp_status() -> str:
    """Verifies whether Cython OpenMP module was compiled and loads successfully."""
    try:
        from ai_engine.features import is_openmp_available
        if is_openmp_available():
            return "Active (OpenMPEngine compiled and loaded successfully)"
        else:
            return "Inactive (PythonEngine fallback active; Cython module not loaded)"
    except Exception as e:
        return f"Checking error: {e}"


def run_diagnostics():
    print("=========================================================================")
    print("                   HPC DIAGNOSTICS & SYSTEM REPORT                       ")
    print("=========================================================================")
    
    # 1. Platform Summary
    print("\n--- Platform Summary ---")
    print(f"  Operating System : {platform.system()} ({platform.release()} - Version {platform.version()})")
    print(f"  Python Version   : {sys.version}")
    print(f"  Host Architecture: {platform.machine()}")

    # 2. Package Versions
    print("\n--- Dependency Check ---")
    print(f"  PyTorch          : {torch.__version__}")
    print(f"  NumPy            : {np.__version__}")
    print(f"  Pandas           : {pd.__version__}")
    print(f"  XGBoost          : {xgb.__version__}")
    print(f"  Scikit-Learn     : {sklearn.__version__}")

    # 3. CPU Capabilities
    print("\n--- CPU Resources ---")
    print(f"  Model Name       : {get_cpu_info()}")
    print(f"  Cores (Logical)  : {os.cpu_count() or 1}")
    print(f"  Cores (Physical) : {psutil_cores() if 'psutil' in sys.modules else 'Unknown'}")

    # 4. GCC & OpenMP Compiling status
    print("\n--- C Compiler & Parallelization ---")
    print(f"  GCC Compiler Info: {get_gcc_version()}")
    print(f"  OpenMP Status    : {check_openmp_status()}")

    # 5. Hardware Acceleration (CUDA & GPU)
    print("\n--- GPU Hardware Acceleration ---")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available   : {cuda_avail}")
    if cuda_avail:
        print(f"  PyTorch CUDA Ver : {torch.version.cuda}")
        print(f"  Active Device ID : {torch.cuda.current_device()}")
        print(f"  GPU Model Name   : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Total GPU Memory : {total_mem:.2f} GB")
    else:
        print("  CUDA is unavailable; deep learning models will run on CPU.")
        
    print("\n=========================================================================")
    print("                SYSTEM CHECK COMPLETE FOR PILOT RUN                      ")
    print("=========================================================================")


def psutil_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1
    except Exception:
        return 1

if __name__ == "__main__":
    # Ensure psutil is imported for physical core checking if possible
    try:
        import psutil
    except ImportError:
        pass
    run_diagnostics()
