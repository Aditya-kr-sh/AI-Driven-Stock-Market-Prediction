"""
Benchmarking framework for quantitative and parallel computing tasks.
Records execution time, process-level CPU utilization, and RAM usage deltas.
Provides JSON serialization for comparative modeling (Python vs. OpenMP vs. CUDA)
with system metadata, history rotation, and forward-compatible GPU metric fields.
"""

import time
import os
import gc
import json
import platform
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import psutil
from ai_engine.utils.config import settings
from ai_engine.utils.logging import logger

@dataclass
class BenchmarkResult:
    """Dataclass holding metrics for a single benchmark run."""
    name: str
    implementation: str  # e.g., 'python', 'openmp', 'cuda'
    elapsed_time_sec: float
    start_memory_mb: float
    end_memory_mb: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: str
    system_metadata: Dict[str, Any]  # Stores platform, CPU, python version, and git hash
    gpu_metrics: Optional[Dict[str, Any]] = None  # Forward-compatible field for CUDA/GPU profiling


def get_system_metadata() -> Dict[str, Any]:
    """Compiles local CPU, system platform, python engine version, and git commit details."""
    git_hash = "unknown"
    try:
        # Fetch current commit hash silently
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        pass

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_info": platform.processor() or "unknown",
        "cpu_count": psutil.cpu_count(logical=True),
        "git_commit_hash": git_hash,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


class BenchmarkTracker:
    """
    Maintains benchmark history, limits file size via log rotation limits,
    and persists records to data/benchmark_results.json.
    """
    _results: List[BenchmarkResult] = []

    @classmethod
    def add_result(cls, result: BenchmarkResult) -> None:
        """Adds a result to the history, applies history rotation limits, and serializes to disk."""
        # Load prior results from disk to append if the in-memory array is empty
        if not cls._results:
            loaded_data = cls.load_results()
            for item in loaded_data:
                try:
                    cls._results.append(
                        BenchmarkResult(
                            name=item["name"],
                            implementation=item["implementation"],
                            elapsed_time_sec=item["elapsed_time_sec"],
                            start_memory_mb=item["start_memory_mb"],
                            end_memory_mb=item["end_memory_mb"],
                            memory_delta_mb=item["memory_delta_mb"],
                            cpu_percent=item["cpu_percent"],
                            timestamp=item["timestamp"],
                            system_metadata=item.get("system_metadata", {}),
                            gpu_metrics=item.get("gpu_metrics")
                        )
                    )
                except Exception:
                    pass  # Skip corrupted or obsolete schemas

        cls._results.append(result)

        # Rotate history to prevent unlimited growth (configured in settings)
        max_records = getattr(settings, "MAX_BENCHMARK_RECORDS", 100)
        if len(cls._results) > max_records:
            cls._results = cls._results[-max_records:]

        cls.save_results()

    @classmethod
    def get_results(cls) -> List[BenchmarkResult]:
        """Returns the current in-memory benchmark history list."""
        return cls._results

    @classmethod
    def clear_results(cls) -> None:
        """Clears all benchmark history records both in-memory and on disk."""
        cls._results = []
        filepath = settings.BASE_PATH / "data" / "benchmark_results.json"
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception as e:
                logger.error(f"Failed to clear benchmark file: {e}")

    @classmethod
    def save_results(cls) -> None:
        """Serializes historical results to data/benchmark_results.json."""
        filepath = settings.BASE_PATH / "data" / "benchmark_results.json"
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            serializable_data = [asdict(r) for r in cls._results]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to serialize benchmark results to {filepath}: {e}")

    @classmethod
    def load_results(cls) -> List[Dict[str, Any]]:
        """Loads and returns historical results saved on disk."""
        filepath = settings.BASE_PATH / "data" / "benchmark_results.json"
        if not filepath.exists():
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read benchmark results from {filepath}: {e}")
            return []


class BenchmarkContext:
    """
    Context manager to wrap code blocks for performance monitoring.
    Automatically captures time, RAM delta, average CPU load, and system metadata.
    """

    def __init__(self, name: str, implementation: str):
        """
        Args:
            name: The descriptive identifier of the task (e.g. 'RSI Calculation').
            implementation: The technology label (e.g. 'python', 'openmp', 'cuda').
        """
        self.name = name
        self.implementation = implementation.lower()
        self.process = psutil.Process(os.getpid())
        self.start_time: float = 0.0
        self.start_mem: float = 0.0
        self.end_time: float = 0.0
        self.end_mem: float = 0.0

    def __enter__(self):
        # Force garbage collection to ensure stable memory measurements
        gc.collect()
        
        self.start_mem = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
        self.process.cpu_percent(interval=None)  # Warm up CPU tracker
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.end_mem = self.process.memory_info().rss / (1024 * 1024)

        elapsed = self.end_time - self.start_time
        mem_delta = self.end_mem - self.start_mem
        cpu_usage = self.process.cpu_percent(interval=None)

        result = BenchmarkResult(
            name=self.name,
            implementation=self.implementation,
            elapsed_time_sec=elapsed,
            start_memory_mb=self.start_mem,
            end_memory_mb=self.end_mem,
            memory_delta_mb=mem_delta,
            cpu_percent=cpu_usage,
            timestamp=datetime.utcnow().isoformat() + "Z",
            system_metadata=get_system_metadata(),
            gpu_metrics=None  # Can be populated dynamically in future phases
        )

        logger.info(
            f"Benchmark '{self.name}' [{self.implementation}] completed: "
            f"Time = {elapsed:.4f}s | "
            f"Memory Delta = {mem_delta:+.2f} MB | "
            f"CPU = {cpu_usage:.1f}%"
        )

        BenchmarkTracker.add_result(result)


def benchmark(name: str, implementation: str):
    """
    Decorator for wrapping functions to run benchmarks automatically.
    
    Example:
        @benchmark("Technical Indicator Engine", "openmp")
        def calculate_features():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with BenchmarkContext(name=name, implementation=implementation):
                return func(*args, **kwargs)
        return wrapper
    return decorator
