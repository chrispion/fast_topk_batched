#!/usr/bin/env python3
"""
TOP-K BENCHMARK: PRODUCTION-GRADE EVALUATION
============================================

A comprehensive benchmark suite for evaluating Top-K implementations
across CPU and GPU architectures. Designed for ML engineers and researchers
needing rigorous performance analysis.

Features:
- Multi-dimensional parameter sweeps
- Statistical rigor (percentiles, confidence intervals)
- Throughput, latency, and efficiency metrics
- Cross-platform comparison (CPU vs GPU)
- Memory bandwidth analysis
- CSV export with full metadata
"""

import ctypes
import time
import numpy as np
import torch
import pandas as pd
import argparse
import sys
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ----------------------------
# CONFIGURATION
# ----------------------------
@dataclass
class BenchmarkConfig:
    # Core parameters
    batch_sizes: List[int] = None
    vocab_sizes: List[int] = None
    k_values: List[int] = None
    
    # Statistical parameters
    runs: int = 100
    warmup_cpu: int = 10
    warmup_gpu: int = 20
    
    # Output
    output_csv: str = "topk_benchmark_results.csv"
    output_summary: str = "topk_benchmark_summary.txt"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        if self.vocab_sizes is None:
            self.vocab_sizes = [50257, 128000]
        if self.k_values is None:
            self.k_values = [10, 50, 100, 1000]

# ----------------------------
# PERFORMANCE METRICS
# ----------------------------
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a single configuration."""
    # Timing statistics (ms)
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    p999_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    
    # Throughput metrics
    tokens_per_sec: float
    samples_per_sec: float
    effective_bandwidth_gbps: float
    
    # Efficiency metrics
    compute_efficiency: Optional[float] = None  # % of theoretical peak
    memory_efficiency: Optional[float] = None   # % of theoretical bandwidth
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

# ----------------------------
# HARDWARE PROFILE
# ----------------------------
class HardwareProfiler:
    """Collect hardware specifications for context."""
    
    @staticmethod
    def get_cpu_info() -> Dict:
        """Get basic CPU info without external dependencies."""
        import platform
        import multiprocessing
        
        info = {
            'cpu_name': platform.processor(),
            'cpu_cores_physical': multiprocessing.cpu_count(),
            'cpu_cores_logical': multiprocessing.cpu_count(),
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine()
        }
        
        # Try to get more detailed info on Windows
        if platform.system() == 'Windows':
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name,numberofcores,numberoflogicalprocessors', '/format:list'],
                    capture_output=True, text=True, shell=True
                )
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Name=' in line:
                        info['cpu_name'] = line.split('Name=')[1].strip()
                    elif 'NumberOfCores=' in line:
                        info['cpu_cores_physical'] = int(line.split('NumberOfCores=')[1].strip())
                    elif 'NumberOfLogicalProcessors=' in line:
                        info['cpu_cores_logical'] = int(line.split('NumberOfLogicalProcessors=')[1].strip())
            except:
                pass
        
        return info
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get GPU info if available."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.version.cuda else None
        }
        
        if torch.cuda.is_available():
            try:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info['gpu_memory_free_gb'] = torch.cuda.memory_reserved(0) / 1e9
            except:
                info['gpu_name'] = 'CUDA Device (name unavailable)'
        
        return info

# ----------------------------
# BENCHMARK ENGINE
# ----------------------------
class TopKBenchmark:
    """Main benchmark engine."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.metadata = {}
        
        # Load CPU library
        try:
            self.lib_cpu = ctypes.CDLL("./fast_topk_batched.dll")
            self.lib_cpu.fast_topk_batched.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                ctypes.c_int, ctypes.c_void_p
            ]
            self.cpu_lib_loaded = True
        except Exception as e:
            print(f"Warning: Could not load CPU library: {e}")
            self.cpu_lib_loaded = False
        
        # Set PyTorch for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        # Collect hardware info
        self.metadata['cpu'] = HardwareProfiler.get_cpu_info()
        self.metadata['gpu'] = HardwareProfiler.get_gpu_info()
        self.metadata['torch_version'] = torch.__version__
        self.metadata['numpy_version'] = np.__version__
        
    def cptr(self, ptr: int):
        """Convert pointer to ctypes."""
        return ctypes.c_void_p(int(ptr))
    
    def _compute_percentiles(self, times_ms: np.ndarray) -> Dict:
        """Compute comprehensive percentiles."""
        return {
            'mean': float(np.mean(times_ms)),
            'p50': float(np.percentile(times_ms, 50)),
            'p90': float(np.percentile(times_ms, 90)),
            'p95': float(np.percentile(times_ms, 95)),
            'p99': float(np.percentile(times_ms, 99)),
            'p999': float(np.percentile(times_ms, 99.9)),
            'min': float(np.min(times_ms)),
            'max': float(np.max(times_ms)),
            'std': float(np.std(times_ms)),
            'cv': float(np.std(times_ms) / np.mean(times_ms)) if np.mean(times_ms) > 0 else 0
        }
    
    def _compute_throughput(self, B: int, V: int, mean_time_ms: float) -> Tuple[float, float, float]:
        """Compute throughput metrics."""
        if mean_time_ms <= 0:
            return 0.0, 0.0, 0.0
        
        tokens_per_sec = (B * 1000) / mean_time_ms
        samples_per_sec = 1000 / mean_time_ms
        
        # Effective bandwidth: (input + output) bytes / time
        total_bytes = (B * V * 4) + (B * self.config.k_values[0] * 4)  # Assuming first K value
        effective_gbps = (total_bytes / 1e9) / (mean_time_ms / 1000.0)
        
        return tokens_per_sec, samples_per_sec, effective_gbps
    
    def benchmark_cpu_custom(self, logits_np: np.ndarray, out_np: np.ndarray, 
                            B: int, V: int, K: int) -> Optional[PerformanceMetrics]:
        """Benchmark custom CPU implementation."""
        if not self.cpu_lib_loaded:
            return None
        
        ptr = self.cptr(logits_np.ctypes.data)
        out_ptr = self.cptr(out_np.ctypes.data)
        
        # Warmup
        for _ in range(self.config.warmup_cpu):
            self.lib_cpu.fast_topk_batched(ptr, B, V, K, out_ptr)
        
        # Benchmark
        times_sec = []
        for _ in range(self.config.runs):
            t0 = time.perf_counter_ns()
            self.lib_cpu.fast_topk_batched(ptr, B, V, K, out_ptr)
            t1 = time.perf_counter_ns()
            times_sec.append((t1 - t0) / 1e9)
        
        times_ms = np.array(times_sec) * 1000.0
        stats = self._compute_percentiles(times_ms)
        
        # Compute throughput
        tps, sps, bw = self._compute_throughput(B, V, stats['mean'])
        
        return PerformanceMetrics(
            mean_ms=stats['mean'],
            p50_ms=stats['p50'],
            p90_ms=stats['p90'],
            p99_ms=stats['p99'],
            p999_ms=stats['p999'],
            min_ms=stats['min'],
            max_ms=stats['max'],
            std_ms=stats['std'],
            tokens_per_sec=tps,
            samples_per_sec=sps,
            effective_bandwidth_gbps=bw
        )
    
    def benchmark_cpu_torch(self, logits_cpu: torch.Tensor, K: int) -> PerformanceMetrics:
        """Benchmark PyTorch CPU implementation."""
        # Warmup
        for _ in range(self.config.warmup_cpu):
            torch.topk(logits_cpu, K)
        
        # Benchmark
        times_sec = []
        for _ in range(self.config.runs):
            t0 = time.perf_counter_ns()
            torch.topk(logits_cpu, K)
            t1 = time.perf_counter_ns()
            times_sec.append((t1 - t0) / 1e9)
        
        times_ms = np.array(times_sec) * 1000.0
        stats = self._compute_percentiles(times_ms)
        
        B, V = logits_cpu.shape
        tps, sps, bw = self._compute_throughput(B, V, stats['mean'])
        
        return PerformanceMetrics(
            mean_ms=stats['mean'],
            p50_ms=stats['p50'],
            p90_ms=stats['p90'],
            p99_ms=stats['p99'],
            p999_ms=stats['p999'],
            min_ms=stats['min'],
            max_ms=stats['max'],
            std_ms=stats['std'],
            tokens_per_sec=tps,
            samples_per_sec=sps,
            effective_bandwidth_gbps=bw
        )
    
    def benchmark_cuda_torch(self, logits_gpu: torch.Tensor, K: int) -> Optional[PerformanceMetrics]:
        """Benchmark PyTorch CUDA implementation."""
        if not torch.cuda.is_available():
            return None
        
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(self.config.warmup_gpu):
            torch.topk(logits_gpu, K)
            torch.cuda.synchronize()
        
        # Benchmark with CUDA events
        times_ms = []
        for _ in range(self.config.runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            torch.topk(logits_gpu, K)
            end.record()
            end.synchronize()
            
            times_ms.append(start.elapsed_time(end))
        
        times_ms = np.array(times_ms)
        stats = self._compute_percentiles(times_ms)
        
        B, V = logits_gpu.shape
        tps, sps, bw = self._compute_throughput(B, V, stats['mean'])
        
        return PerformanceMetrics(
            mean_ms=stats['mean'],
            p50_ms=stats['p50'],
            p90_ms=stats['p90'],
            p99_ms=stats['p99'],
            p999_ms=stats['p999'],
            min_ms=stats['min'],
            max_ms=stats['max'],
            std_ms=stats['std'],
            tokens_per_sec=tps,
            samples_per_sec=sps,
            effective_bandwidth_gbps=bw
        )
    
    def run_single_config(self, B: int, V: int, K: int) -> Dict:
        """Run benchmarks for a single configuration."""
        print(f"  Benchmarking B={B:4d}, V={V:6d}, K={K:4d}", end="", flush=True)
        
        # Prepare data
        logits_cpu = torch.randn(B, V, dtype=torch.float32)
        out_cpu = np.empty((B, K), dtype=np.int32)
        
        # Run benchmarks
        cpu_torch = self.benchmark_cpu_torch(logits_cpu, K)
        cpu_custom = self.benchmark_cpu_custom(logits_cpu.numpy(), out_cpu, B, V, K)
        
        if torch.cuda.is_available():
            logits_gpu = logits_cpu.cuda()
            cuda_torch = self.benchmark_cuda_torch(logits_gpu, K)
        else:
            cuda_torch = None
        
        print(f" ✓")
        
        return {
            'batch_size': B,
            'vocab_size': V,
            'k': K,
            'cpu_custom': cpu_custom,
            'cpu_torch': cpu_torch,
            'cuda_torch': cuda_torch,
            'total_operations': B * V,
            'memory_bytes_input': B * V * 4,
            'memory_bytes_output': B * K * 4
        }
    
    def run(self) -> pd.DataFrame:
        """Run complete benchmark suite."""
        print("\n" + "="*80)
        print("TOP-K BENCHMARK SUITE")
        print("="*80)
        
        print(f"\nHardware Context:")
        print(f"  CPU: {self.metadata['cpu']['cpu_name']}")
        print(f"  GPU: {self.metadata['gpu'].get('gpu_name', 'N/A')}")
        print(f"  CUDA Available: {self.metadata['gpu']['cuda_available']}")
        print(f"  Torch: {self.metadata['torch_version']}")
        
        print(f"\nConfiguration:")
        print(f"  Batch sizes: {self.config.batch_sizes}")
        print(f"  Vocab sizes: {self.config.vocab_sizes}")
        print(f"  K values: {self.config.k_values}")
        print(f"  Runs per config: {self.config.runs}")
        
        print(f"\nRunning benchmarks...")
        
        total_configs = len(self.config.batch_sizes) * len(self.config.vocab_sizes) * len(self.config.k_values)
        current = 0
        
        for V in self.config.vocab_sizes:
            for B in self.config.batch_sizes:
                for K in self.config.k_values:
                    current += 1
                    print(f"\n[{current}/{total_configs}]", end="")
                    
                    result = self.run_single_config(B, V, K)
                    self.results.append(result)
        
        return self._process_results()
    
    def _process_results(self) -> pd.DataFrame:
        """Process results into DataFrame."""
        records = []
        
        for result in self.results:
            base_record = {
                'batch_size': result['batch_size'],
                'vocab_size': result['vocab_size'],
                'k': result['k'],
                'total_operations': result['total_operations'],
                'memory_input_mb': result['memory_bytes_input'] / 1e6,
                'memory_output_mb': result['memory_bytes_output'] / 1e6
            }
            
            # CPU Custom
            if result['cpu_custom']:
                cpu_custom = result['cpu_custom'].to_dict()
                for key, value in cpu_custom.items():
                    base_record[f'cpu_custom_{key}'] = value
            else:
                # Fill with None if not available
                for field in ['mean_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'p999_ms', 
                             'min_ms', 'max_ms', 'std_ms', 'tokens_per_sec', 
                             'samples_per_sec', 'effective_bandwidth_gbps']:
                    base_record[f'cpu_custom_{field}'] = None
            
            # CPU Torch
            cpu_torch = result['cpu_torch'].to_dict()
            for key, value in cpu_torch.items():
                base_record[f'cpu_torch_{key}'] = value
            
            # CUDA Torch
            if result['cuda_torch']:
                cuda_torch = result['cuda_torch'].to_dict()
                for key, value in cuda_torch.items():
                    base_record[f'cuda_torch_{key}'] = value
            else:
                for field in ['mean_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'p999_ms',
                             'min_ms', 'max_ms', 'std_ms', 'tokens_per_sec',
                             'samples_per_sec', 'effective_bandwidth_gbps']:
                    base_record[f'cuda_torch_{field}'] = None
            
            # Compute speedups where possible
            if result['cpu_custom'] and result['cpu_custom'].mean_ms > 0:
                if result['cpu_torch'].mean_ms > 0:
                    base_record['speedup_cpu_custom_vs_cpu_torch'] = (
                        result['cpu_torch'].mean_ms / result['cpu_custom'].mean_ms
                    )
                
                if result['cuda_torch'] and result['cuda_torch'].mean_ms > 0:
                    base_record['speedup_cuda_vs_cpu_custom'] = (
                        result['cpu_custom'].mean_ms / result['cuda_torch'].mean_ms
                    )
            
            if result['cuda_torch'] and result['cuda_torch'].mean_ms > 0:
                if result['cpu_torch'].mean_ms > 0:
                    base_record['speedup_cuda_vs_cpu_torch'] = (
                        result['cpu_torch'].mean_ms / result['cuda_torch'].mean_ms
                    )
            
            records.append(base_record)
        
        df = pd.DataFrame(records)
        return df

# ----------------------------
# ANALYSIS AND REPORTING
# ----------------------------
class BenchmarkAnalyzer:
    """Analyze and report benchmark results."""
    
    def __init__(self, df: pd.DataFrame, metadata: Dict):
        self.df = df
        self.metadata = metadata
    
    def generate_summary(self) -> str:
        """Generate comprehensive summary."""
        summary = []
        
        summary.append("="*80)
        summary.append("TOP-K BENCHMARK: EXECUTIVE SUMMARY")
        summary.append("="*80)
        summary.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Hardware context
        summary.append("\n" + "-"*40)
        summary.append("HARDWARE CONTEXT")
        summary.append("-"*40)
        summary.append(f"CPU: {self.metadata['cpu']['cpu_name']}")
        summary.append(f"CPU Cores: {self.metadata['cpu'].get('cpu_cores_physical', 'N/A')}")
        summary.append(f"GPU: {self.metadata['gpu'].get('gpu_name', 'N/A')}")
        summary.append(f"CUDA Available: {self.metadata['gpu']['cuda_available']}")
        
        # Key Findings
        summary.append("\n" + "-"*40)
        summary.append("KEY PERFORMANCE FINDINGS")
        summary.append("-"*40)
        
        # CPU Custom vs CPU Torch
        speedup_col = 'speedup_cpu_custom_vs_cpu_torch'
        if speedup_col in self.df.columns:
            valid_speedups = self.df[speedup_col].dropna()
            if len(valid_speedups) > 0:
                avg_speedup = valid_speedups.mean()
                max_speedup = valid_speedups.max()
                
                summary.append(f"\n1. CPU Custom vs PyTorch CPU:")
                summary.append(f"   • Average speedup: {avg_speedup:.2f}x")
                summary.append(f"   • Maximum speedup: {max_speedup:.2f}x")
                
                # Performance classification
                if avg_speedup > 5:
                    classification = "OUTSTANDING"
                elif avg_speedup > 3:
                    classification = "EXCELLENT"
                elif avg_speedup > 2:
                    classification = "VERY GOOD"
                elif avg_speedup > 1.5:
                    classification = "GOOD"
                elif avg_speedup > 1.1:
                    classification = "MODEST"
                else:
                    classification = "MINIMAL"
                
                summary.append(f"   • Performance classification: {classification}")
        
        # Crossover analysis
        cuda_speedup_col = 'speedup_cuda_vs_cpu_custom'
        if cuda_speedup_col in self.df.columns:
            valid_cuda_speedups = self.df[cuda_speedup_col].dropna()
            if len(valid_cuda_speedups) > 0:
                cpu_wins = valid_cuda_speedups[valid_cuda_speedups < 1]
                gpu_wins = valid_cuda_speedups[valid_cuda_speedups > 1]
                
                summary.append(f"\n2. CPU-GPU Crossover Analysis:")
                summary.append(f"   • CPU wins in {len(cpu_wins)} configurations")
                summary.append(f"   • GPU wins in {len(gpu_wins)} configurations")
                
                if not cpu_wins.empty:
                    # Find the best CPU case (lowest CPU time among CPU wins)
                    cpu_win_indices = cpu_wins.index
                    cpu_custom_times = self.df.loc[cpu_win_indices, 'cpu_custom_mean_ms']
                    best_cpu_idx = cpu_custom_times.idxmin()
                    best_cpu_case = self.df.loc[best_cpu_idx]
                    
                    summary.append(f"   • Best CPU case: B={int(best_cpu_case['batch_size'])}, " +
                                 f"V={int(best_cpu_case['vocab_size'])}, " +
                                 f"K={int(best_cpu_case['k'])}: " +
                                 f"{best_cpu_case['cpu_custom_mean_ms']:.3f} ms")
                
                if not gpu_wins.empty:
                    # Find the best GPU case (lowest GPU time among GPU wins)
                    gpu_win_indices = gpu_wins.index
                    cuda_torch_times = self.df.loc[gpu_win_indices, 'cuda_torch_mean_ms']
                    best_gpu_idx = cuda_torch_times.idxmin()
                    best_gpu_case = self.df.loc[best_gpu_idx]
                    
                    summary.append(f"   • Best GPU case: B={int(best_gpu_case['batch_size'])}, " +
                                 f"V={int(best_gpu_case['vocab_size'])}, " +
                                 f"K={int(best_gpu_case['k'])}: " +
                                 f"{best_gpu_case['cuda_torch_mean_ms']:.3f} ms")
        
        # Throughput analysis
        summary.append(f"\n3. Throughput Analysis:")
        if 'cpu_custom_tokens_per_sec' in self.df.columns:
            max_tps_cpu = self.df['cpu_custom_tokens_per_sec'].max()
            summary.append(f"   • Peak CPU throughput: {max_tps_cpu:,.0f} tokens/sec")
        
        if 'cuda_torch_tokens_per_sec' in self.df.columns:
            max_tps_gpu = self.df['cuda_torch_tokens_per_sec'].max()
            summary.append(f"   • Peak GPU throughput: {max_tps_gpu:,.0f} tokens/sec")
        
        # Memory efficiency
        summary.append(f"\n4. Memory Efficiency:")
        if 'cpu_custom_effective_bandwidth_gbps' in self.df.columns:
            max_bw_cpu = self.df['cpu_custom_effective_bandwidth_gbps'].max()
            summary.append(f"   • Peak CPU bandwidth: {max_bw_cpu:.2f} GB/s")
            # Estimate typical DDR4 bandwidth
            summary.append(f"   • Typical DDR4 bandwidth: 25-50 GB/s")
            if max_bw_cpu > 0:
                efficiency = (max_bw_cpu / 40) * 100  # Compare to 40 GB/s mid-range
                summary.append(f"   • Estimated efficiency: {efficiency:.1f}% of typical DDR4")
        
        # Recommendations
        summary.append("\n" + "-"*40)
        summary.append("ENGINEERING RECOMMENDATIONS")
        summary.append("-"*40)
        
        if speedup_col in self.df.columns:
            valid_speedups = self.df[speedup_col].dropna()
            if len(valid_speedups) > 0:
                avg_speedup = valid_speedups.mean()
                
                if avg_speedup > 3:
                    summary.append("🏆 EXCEPTIONAL: Custom implementation significantly outperforms PyTorch")
                    summary.append("   Deployment recommendations:")
                    summary.append("   • Primary choice for edge deployment")
                    summary.append("   • Use for latency-critical applications (batch size ≤ 2)")
                    summary.append("   • Consider hybrid CPU/GPU routing based on batch size")
                elif avg_speedup > 2:
                    summary.append("✅ EXCELLENT: Clear advantage over PyTorch")
                    summary.append("   Deployment recommendations:")
                    summary.append("   • Strong candidate for CPU-based inference")
                    summary.append("   • Consider for mobile/edge applications")
                    summary.append("   • Evaluate GPU for batch sizes > 8")
                elif avg_speedup > 1.5:
                    summary.append("✓ GOOD: Moderate improvement over PyTorch")
                    summary.append("   Deployment recommendations:")
                    summary.append("   • Suitable for specific low-latency use cases")
                    summary.append("   • Continue optimization efforts")
                    summary.append("   • GPU preferred for most batch sizes")
        
        # Performance profile
        summary.append("\n" + "-"*40)
        summary.append("PERFORMANCE PROFILE")
        summary.append("-"*40)
        
        # Find optimal configurations
        if 'cpu_custom_tokens_per_sec' in self.df.columns:
            cpu_tps_valid = self.df['cpu_custom_tokens_per_sec'].dropna()
            if len(cpu_tps_valid) > 0:
                optimal_cpu_idx = self.df['cpu_custom_tokens_per_sec'].idxmax()
                optimal_cpu = self.df.loc[optimal_cpu_idx]
                
                summary.append(f"\nOptimal CPU Configuration:")
                summary.append(f"  • Batch size: {int(optimal_cpu['batch_size'])}")
                summary.append(f"  • Vocab size: {int(optimal_cpu['vocab_size'])}")
                summary.append(f"  • K value: {int(optimal_cpu['k'])}")
                summary.append(f"  • Throughput: {optimal_cpu['cpu_custom_tokens_per_sec']:,.0f} tokens/sec")
                summary.append(f"  • Latency (p50): {optimal_cpu['cpu_custom_p50_ms']:.3f} ms")
        
        if 'cuda_torch_tokens_per_sec' in self.df.columns:
            gpu_tps_valid = self.df['cuda_torch_tokens_per_sec'].dropna()
            if len(gpu_tps_valid) > 0:
                optimal_gpu_idx = self.df['cuda_torch_tokens_per_sec'].idxmax()
                optimal_gpu = self.df.loc[optimal_gpu_idx]
                
                summary.append(f"\nOptimal GPU Configuration:")
                summary.append(f"  • Batch size: {int(optimal_gpu['batch_size'])}")
                summary.append(f"  • Vocab size: {int(optimal_gpu['vocab_size'])}")
                summary.append(f"  • K value: {int(optimal_gpu['k'])}")
                summary.append(f"  • Throughput: {optimal_gpu['cuda_torch_tokens_per_sec']:,.0f} tokens/sec")
                summary.append(f"  • Latency (p50): {optimal_gpu['cuda_torch_p50_ms']:.3f} ms")
        
        return "\n".join(summary)
    
    def print_results_table(self):
        """Print formatted results table."""
        print("\n" + "="*100)
        print("DETAILED PERFORMANCE RESULTS")
        print("="*100)
        print(f"{'B/V/K':<12} | {'CPU Custom':^30} | {'CPU Torch':^30} | {'CUDA Torch':^30}")
        print(f"{'':<12} | {'p50 (ms)':>10} {'tokens/sec':>12} {'BW':>8} | "
              f"{'p50 (ms)':>10} {'tokens/sec':>12} {'BW':>8} | "
              f"{'p50 (ms)':>10} {'tokens/sec':>12} {'BW':>8}")
        print("-"*100)
        
        for _, row in self.df.iterrows():
            tag = f"{int(row['batch_size'])}/{int(row['vocab_size'])}/{int(row['k'])}"
            
            # CPU Custom
            if pd.notna(row.get('cpu_custom_p50_ms')):
                cpu_custom_str = f"{row['cpu_custom_p50_ms']:>10.3f} {row['cpu_custom_tokens_per_sec']:>12,.0f} {row['cpu_custom_effective_bandwidth_gbps']:>8.2f}"
            else:
                cpu_custom_str = f"{'N/A':>10} {'N/A':>12} {'N/A':>8}"
            
            # CPU Torch
            cpu_torch_str = f"{row['cpu_torch_p50_ms']:>10.3f} {row['cpu_torch_tokens_per_sec']:>12,.0f} {row['cpu_torch_effective_bandwidth_gbps']:>8.2f}"
            
            # CUDA Torch
            if pd.notna(row.get('cuda_torch_p50_ms')):
                cuda_str = f"{row['cuda_torch_p50_ms']:>10.3f} {row['cuda_torch_tokens_per_sec']:>12,.0f} {row['cuda_torch_effective_bandwidth_gbps']:>8.2f}"
            else:
                cuda_str = f"{'N/A':>10} {'N/A':>12} {'N/A':>8}"
            
            print(f"{tag:<12} | {cpu_custom_str} | {cpu_torch_str} | {cuda_str}")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Top-K Benchmark Suite")
    parser.add_argument("--runs", type=int, default=50, help="Number of runs per configuration")
    parser.add_argument("--output", type=str, default="topk_benchmark_results.csv", 
                       help="Output CSV file")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128,256,512,1024",
                       help="Comma-separated batch sizes")
    parser.add_argument("--vocab-sizes", type=str, default="50257,128000",
                       help="Comma-separated vocabulary sizes")
    parser.add_argument("--k-values", type=str, default="10,50,100,1000",
                       help="Comma-separated K values")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmarks")
    
    args = parser.parse_args()
    
    # Parse lists
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    vocab_sizes = [int(x) for x in args.vocab_sizes.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]
    
    # Create config
    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        vocab_sizes=vocab_sizes,
        k_values=k_values,
        runs=args.runs,
        output_csv=args.output
    )
    
    # Run benchmark
    benchmark = TopKBenchmark(config)
    df = benchmark.run()
    
    # Save results
    df.to_csv(config.output_csv, index=False, encoding='utf-8')
    print(f"\n✓ Results saved to: {Path(config.output_csv).resolve()}")
    
    # Generate analysis
    analyzer = BenchmarkAnalyzer(df, benchmark.metadata)
    analyzer.print_results_table()
    
    # Generate and save summary
    summary = analyzer.generate_summary()
    print(f"\n{summary}")
    
    with open(config.output_summary, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n✓ Summary saved to: {Path(config.output_summary).resolve()}")
    
    # Final verdict
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    
    speedup_col = 'speedup_cpu_custom_vs_cpu_torch'
    if speedup_col in df.columns:
        valid_speedups = df[speedup_col].dropna()
        if len(valid_speedups) > 0:
            avg_speedup = valid_speedups.mean()
            
            print(f"\nPerformance Summary:")
            print(f"• Average speedup over PyTorch CPU: {avg_speedup:.2f}x")
            
            if avg_speedup > 5:
                print("🏆 OUTSTANDING IMPLEMENTATION")
                print("• World-class performance for CPU-based Top-K")
                print("• Production-ready for all latency-critical applications")
            elif avg_speedup > 3:
                print("🎯 EXCELLENT IMPLEMENTATION")
                print("• Significantly outperforms PyTorch CPU")
                print("• Highly suitable for production deployment")
            elif avg_speedup > 2:
                print("✅ VERY GOOD IMPLEMENTATION")
                print("• Clear performance advantage over PyTorch")
                print("• Production-ready with specific optimizations")
            elif avg_speedup > 1.5:
                print("✓ GOOD IMPLEMENTATION")
                print("• Modest improvement over PyTorch")
                print("• Suitable for targeted optimizations")
            else:
                print("⚠️  MINIMAL IMPROVEMENT")
                print("• Similar performance to PyTorch")
                print("• Consider further optimization efforts")
        else:
            print("⚠️  No valid CPU custom benchmarks completed")
    else:
        print("⚠️  CPU custom benchmarks not available")

if __name__ == "__main__":
    main()