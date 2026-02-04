# bench_kernels.py
# Usage examples:
#   python bench_kernels.py --kernel softmax --M 8192 --Ns 256 512 1024 2048 4096
#   python bench_kernels.py --kernel rmsnorm --M 8192 --Ns 1024 2048 4096
#
# Notes:
# - softmax: compares Triton vs Helion vs torch.softmax (if Helion is installed).
# - rmsnorm: compares CuTe DSL vs torch.nn.functional.rms_norm (if CuTe DSL is installed).
#
# You should place this script next to:
#   - triton_softmax.py (defines softmax_triton)
#   - helion_softmax.py (defines softmax_two_pass)
#   - cutedsl_rmsnorm.py (defines rms_norm_ host launcher)  [optional]
#
import argparse
import csv
import importlib
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    import triton
    import triton.testing
except Exception as e:
    raise RuntimeError("This benchmark script requires Triton installed.") from e


# -------------------------
# Utilities
# -------------------------
@dataclass
class BenchResult:
    name: str
    M: int
    N: int
    dtype: str
    ms: float
    gb_s: Optional[float] = None


def _bytes_moved_softmax(M: int, N: int, dtype: torch.dtype) -> int:
    # Conservative: read X + write Y (ignores intermediate reads/writes in registers/shared)
    return 2 * M * N * torch.tensor([], dtype=dtype).element_size()


def _bytes_moved_rmsnorm(M: int, N: int, dtype: torch.dtype) -> int:
    # Conservative: read X + read W + write Y
    # W is length N, but reused across rows; still count once for a lower bound.
    # For a per-kernel lower bound: read X + write Y.
    elem = torch.tensor([], dtype=dtype).element_size()
    return (M * N * elem) + (M * N * elem)  # read X + write Y


def bench_fn(name: str, fn: Callable[[], None], warmup_ms=25, rep_ms=100) -> float:
    # Returns mean time in milliseconds by default
    ms = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean")
    return float(ms)


def try_import(module: str, symbol: str):
    try:
        m = importlib.import_module(module)
        return getattr(m, symbol)
    except Exception:
        return None


def set_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


# -------------------------
# Softmax benchmark
# -------------------------
def run_softmax(M: int, N: int, dtype: torch.dtype, device: str,
                warmup_ms: int, rep_ms: int) -> List[BenchResult]:
    results: List[BenchResult] = []
    x = torch.randn((M, N), device=device, dtype=dtype)

    # Baseline: torch.softmax
    def torch_run():
        torch.softmax(x, dim=1)

    ms = bench_fn("torch.softmax", torch_run, warmup_ms, rep_ms)
    gb_s = _bytes_moved_softmax(M, N, dtype) / (ms * 1e-3) / 1e9
    results.append(BenchResult("torch.softmax", M, N, str(dtype).replace("torch.", ""), ms, gb_s))

    # Triton
    softmax_triton = try_import("triton_softmax", "softmax_triton")
    if softmax_triton is not None:
        def triton_run():
            softmax_triton(x)
        ms = bench_fn("triton.softmax", triton_run, warmup_ms, rep_ms)
        gb_s = _bytes_moved_softmax(M, N, dtype) / (ms * 1e-3) / 1e9
        results.append(BenchResult("triton.softmax", M, N, str(dtype).replace("torch.", ""), ms, gb_s))
    else:
        print("[skip] triton_softmax.softmax_triton not found. Put triton_softmax.py рядом или поправьте import.")

    # Helion
    softmax_two_pass = try_import("helion_softmax", "softmax_two_pass")
    if softmax_two_pass is not None:
        def helion_run():
            softmax_two_pass(x)
        ms = bench_fn("helion.softmax", helion_run, warmup_ms, rep_ms)
        gb_s = _bytes_moved_softmax(M, N, dtype) / (ms * 1e-3) / 1e9
        results.append(BenchResult("helion.softmax", M, N, str(dtype).replace("torch.", ""), ms, gb_s))
    else:
        print("[skip] helion_softmax.softmax_two_pass not found or Helion not installed.")

    return results


# -------------------------
# RMSNorm benchmark (CuTe DSL vs torch)
# -------------------------
def run_rmsnorm(M: int, N: int, dtype: torch.dtype, device: str,
               warmup_ms: int, rep_ms: int, eps: float) -> List[BenchResult]:
    results: List[BenchResult] = []
    x = torch.randn((M, N), device=device, dtype=dtype)
    w = torch.randn((N,), device=device, dtype=dtype)
    y = torch.empty_like(x)

    # Torch baseline (PyTorch has functional RMSNorm in recent versions; fall back to manual if missing)
    has_rms = hasattr(torch.nn.functional, "rms_norm")

    def torch_run():
        if has_rms:
            torch.nn.functional.rms_norm(x, (N,), w, eps=eps)
        else:
            # manual: x * rsqrt(mean(x^2)+eps) * w
            denom = torch.rsqrt((x.float().pow(2).mean(dim=1, keepdim=True)) + eps)
            (x.float() * denom).to(dtype) * w

    ms = bench_fn("torch.rmsnorm", torch_run, warmup_ms, rep_ms)
    gb_s = _bytes_moved_rmsnorm(M, N, dtype) / (ms * 1e-3) / 1e9
    results.append(BenchResult("torch.rmsnorm", M, N, str(dtype).replace("torch.", ""), ms, gb_s))

    # CuTe DSL
    # Expect a host launcher rms_norm_(mX, mW, mY, num_tokens, hidden_dim, eps)
    rms_norm_ = try_import("cutedsl_rmsnorm", "rms_norm_")
    if rms_norm_ is None:
        print("[skip] cutedsl_rmsnorm.rms_norm_ not found or CuTe DSL not installed.")
        return results

    # CuTe runtime tensor wrappers
    try:
        from cutlass.cute.runtime import from_dlpack
    except Exception:
        print("[skip] cutlass.cute.runtime.from_dlpack not available.")
        return results

    mX = from_dlpack(x, assumed_align=16)
    mW = from_dlpack(w, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)

    def cute_run():
        rms_norm_(mX, mW, mY, M, N, eps)

    ms = bench_fn("cute.rmsnorm", cute_run, warmup_ms, rep_ms)
    gb_s = _bytes_moved_rmsnorm(M, N, dtype) / (ms * 1e-3) / 1e9
    results.append(BenchResult("cute.rmsnorm", M, N, str(dtype).replace("torch.", ""), ms, gb_s))

    return results


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["softmax", "rmsnorm"], default="softmax")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--Ns", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup_ms", type=int, default=25)
    parser.add_argument("--rep_ms", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--out_csv", type=str, default="bench_results.csv")
    args = parser.parse_args()

    set_seed()
    device = "cuda"
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    all_rows: List[BenchResult] = []
    for N in args.Ns:
        print(f"\n=== {args.kernel} | M={args.M}, N={N}, dtype={args.dtype} ===")
        if args.kernel == "softmax":
            rows = run_softmax(args.M, N, dtype, device, args.warmup_ms, args.rep_ms)
        else:
            rows = run_rmsnorm(args.M, N, dtype, device, args.warmup_ms, args.rep_ms, args.eps)

        # Pretty print
        for r in rows:
            if r.gb_s is None:
                print(f"{r.name:14s}: {r.ms:8.3f} ms")
            else:
                print(f"{r.name:14s}: {r.ms:8.3f} ms | {r.gb_s:8.1f} GB/s (lower bound)")
        all_rows.extend(rows)

    # Save CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "M", "N", "dtype", "ms", "gb_s_lower_bound"])
        for r in all_rows:
            w.writerow([r.name, r.M, r.N, r.dtype, f"{r.ms:.6f}", "" if r.gb_s is None else f"{r.gb_s:.3f}"])

    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
