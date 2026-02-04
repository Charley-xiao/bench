# bench.py
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import triton.testing as tt

from w4a16.quant_utils import quantize_int4_weight_only
from w4a16.baseline import w4a16_baseline
from w4a16.triton_w4a16 import w4a16_triton  # your split-k (partials+reduce) + store(sk=1)
from w4a16.triton_w4a16_dp import w4a16_triton_dp  # paper DP fused baseline


def try_import_helion():
    try:
        from w4a16.helion_w4a16 import w4a16_helion
        return w4a16_helion
    except Exception:
        return None


def try_import_cute():
    try:
        from w4a16.cutedsl_w4a16 import w4a16_cutedsl
        return w4a16_cutedsl
    except Exception:
        return None


def do_bench_ms(fn: Callable[[], None], warmup_ms: int, rep_ms: int) -> float:
    # warmup/rep are in milliseconds. return_mode="mean" returns mean runtime. :contentReference[oaicite:5]{index=5}
    return float(tt.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean"))


def choose_split_k_heuristic(M: int) -> int:
    # Must match w4a16_triton() heuristic in your triton_w4a16.py
    if M <= 32:
        return 4
    if M <= 128:
        return 2
    return 1


def make_inputs(M: int, K: int, N: int, group_size: int, seed: int = 0):
    device = "cuda"
    assert K % group_size == 0
    torch.manual_seed(seed)
    A = torch.randn((M, K), device=device, dtype=torch.float16)
    torch.manual_seed(seed + 1)
    W = torch.randn((K, N), device=device, dtype=torch.float16)
    packed, scales, zeros = quantize_int4_weight_only(W, group_size=group_size, symmetric=False)
    return A, packed, scales, zeros


@dataclass
class Record:
    name: str
    regime: str
    M: int
    K: int
    N: int
    group_size: int
    split_k: str
    ms: float


def bench_one(
    name: str,
    fn: Callable[[], None],
    warmup_ms: int,
    rep_ms: int,
) -> float:
    # Make sure kernels are compiled / cached before timing
    fn()
    torch.cuda.synchronize()
    return do_bench_ms(fn, warmup_ms, rep_ms)


def bench_case(
    regime: str,
    M: int,
    K: int,
    N: int,
    group_size: int,
    warmup_ms: int,
    rep_ms: int,
    with_helion: bool,
    with_cute: bool,
    seed: int,
    records: List[Record],
):
    A, packed, scales, zeros = make_inputs(M, K, N, group_size, seed=seed)

    # Implementations
    helion_impl = try_import_helion() if with_helion else None
    cute_impl = try_import_cute() if with_cute else None

    # --- Naive baseline: dequant then matmul (likely dominated by dequant work)
    def run_naive_baseline():
        w4a16_baseline(A, packed, scales, zeros, N=N, group_size=group_size)

    # --- Paper baseline: DP fused kernel (no Split-K)
    def run_dp_fused():
        w4a16_triton_dp(A, packed, scales, zeros, N=N, group_size=group_size)

    # --- Your fused kernels
    def run_triton_store():
        w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size, force_split_k=1)

    split_k_h = choose_split_k_heuristic(M)

    def run_triton_heuristic():
        w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size, force_split_k=None)

    def run_triton_sk2():
        w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size, force_split_k=2)

    def run_triton_sk4():
        w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size, force_split_k=4)

    print(f"\n[{regime}] M={M} K={K} N={N} gs={group_size}")
    print(f"  split_k heuristic(None) -> {split_k_h}")
    if os.environ.get("TRITON_PRINT_AUTOTUNING", "") == "1":
        # Triton will print the best autotune config for kernels decorated with @triton.autotune. :contentReference[oaicite:6]{index=6}
        print("  TRITON_PRINT_AUTOTUNING=1 (DP fused baseline uses @autotune; config will print when first tuned)")

    # Benchmark in a consistent order
    ms_naive = bench_one("baseline_dequant_matmul", run_naive_baseline, warmup_ms, rep_ms)
    ms_dp = bench_one("triton_dp_fused", run_dp_fused, warmup_ms, rep_ms)
    ms_store = bench_one("triton_store", run_triton_store, warmup_ms, rep_ms)
    ms_auto = bench_one("triton_heuristic", run_triton_heuristic, warmup_ms, rep_ms)

    print(f"  baseline_dequant_matmul : {ms_naive:.4f} ms")
    print(f"  triton_dp_fused (paper) : {ms_dp:.4f} ms")
    print(f"  triton_store (sk=1)     : {ms_store:.4f} ms")
    print(f"  triton_heuristic(None)  : {ms_auto:.4f} ms")

    records.append(Record("baseline_dequant_matmul", regime, M, K, N, group_size, "n/a", ms_naive))
    records.append(Record("triton_dp_fused",         regime, M, K, N, group_size, "1",   ms_dp))
    records.append(Record("triton_store",            regime, M, K, N, group_size, "1",   ms_store))
    records.append(Record("triton_heuristic",        regime, M, K, N, group_size, str(split_k_h), ms_auto))

    # Optional explicit split-k sweeps (decode-focused)
    # Split-K is designed to help when M is small (decode). It may hurt for large M due to reduction overhead. :contentReference[oaicite:7]{index=7}
    if M <= 128:
        ms_sk2 = bench_one("triton_splitk", run_triton_sk2, warmup_ms, rep_ms)
        print(f"  triton_splitk(2)        : {ms_sk2:.4f} ms")
        records.append(Record("triton_splitk", regime, M, K, N, group_size, "2", ms_sk2))

    if M <= 64:
        ms_sk4 = bench_one("triton_splitk", run_triton_sk4, warmup_ms, rep_ms)
        print(f"  triton_splitk(4)        : {ms_sk4:.4f} ms")
        records.append(Record("triton_splitk", regime, M, K, N, group_size, "4", ms_sk4))

    # Optional external implementations
    if helion_impl is not None:
        def run_helion():
            helion_impl(A, packed, scales, zeros, N=N, group_size=group_size)
        ms_h = bench_one("helion_fused", run_helion, warmup_ms, rep_ms)
        print(f"  helion_fused            : {ms_h:.4f} ms")
        records.append(Record("helion_fused", regime, M, K, N, group_size, "", ms_h))

    if cute_impl is not None:
        def run_cute():
            cute_impl(A, packed, scales, zeros, N=N, group_size=group_size)
        ms_c = bench_one("cutedsl_fused", run_cute, warmup_ms, rep_ms)
        print(f"  cutedsl_fused           : {ms_c:.4f} ms")
        records.append(Record("cutedsl_fused", regime, M, K, N, group_size, "", ms_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["decode", "prefill", "both"], default="both")
    ap.add_argument("--decode_Ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--prefill_Ms", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--Ns", type=int, nargs="+", default=[4096, 8192, 11008])
    ap.add_argument("--group_size", type=int, default=128)

    # Defaults: slightly larger warmup can reduce underestimation issues for do_bench in some cases. :contentReference[oaicite:8]{index=8}
    ap.add_argument("--warmup_ms", type=int, default=100)
    ap.add_argument("--rep_ms", type=int, default=200)

    ap.add_argument("--out", type=str, default="bench_results.csv")
    ap.add_argument("--with_helion", action="store_true")
    ap.add_argument("--with_cute", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    records: List[Record] = []
    Ns = [int(x) for x in args.Ns]
    decode_Ms = [int(x) for x in args.decode_Ms]
    prefill_Ms = [int(x) for x in args.prefill_Ms]

    if args.regime in ("decode", "both"):
        for N in Ns:
            for M in decode_Ms:
                bench_case(
                    "decode", M, args.K, N, args.group_size,
                    args.warmup_ms, args.rep_ms,
                    args.with_helion, args.with_cute,
                    args.seed,
                    records
                )

    if args.regime in ("prefill", "both"):
        for N in Ns:
            for M in prefill_Ms:
                bench_case(
                    "prefill", M, args.K, N, args.group_size,
                    args.warmup_ms, args.rep_ms,
                    args.with_helion, args.with_cute,
                    args.seed,
                    records
                )

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "regime", "M", "K", "N", "group_size", "split_k", "ms"])
        for r in records:
            w.writerow([r.name, r.regime, r.M, r.K, r.N, r.group_size, r.split_k, r.ms])

    print(f"\nSaved {args.out}")
    print("Tip: TRITON_PRINT_AUTOTUNING=1 will print selected configs for @autotune kernels. (DP fused baseline uses @autotune.)")  # :contentReference[oaicite:9]{index=9}


if __name__ == "__main__":
    main()
