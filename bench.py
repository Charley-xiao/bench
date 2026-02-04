# bench.py
import argparse
import csv
import os
from typing import List, Tuple

import torch
import triton.testing

from w4a16.quant_utils import quantize_int4_weight_only
from w4a16.baseline import w4a16_baseline
from w4a16.triton_w4a16 import w4a16_triton, _w4a16_splitk_atomic_kernel


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


def bench_one(fn, warmup_ms=25, rep_ms=100) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean"))


def parse_int_list(xs: List[int]) -> List[int]:
    # already ints, but keep as helper for future
    return [int(x) for x in xs]


def format_best_config() -> str:
    """
    Triton autotuned kernels expose `best_config` after the first time autotuning runs.
    There is also TRITON_PRINT_AUTOTUNING=1 which prints best configs automatically. :contentReference[oaicite:1]{index=1}
    """
    cfg = getattr(_w4a16_splitk_atomic_kernel, "best_config", None)
    if cfg is None:
        return "<none yet>"
    # cfg is a triton.Config; print its meta-params + compiler params (warps/stages)
    try:
        meta = dict(cfg.kwargs)
    except Exception:
        meta = {}
    return f"{meta} | num_warps={cfg.num_warps}, num_stages={cfg.num_stages}"


def run_regime(
    regime_name: str,
    Ms: List[int],
    K: int,
    Ns: List[int],
    group_size: int,
    dtype: torch.dtype,
    out_rows: List[Tuple],
    warmup_ms: int,
    rep_ms: int,
    include_helion: bool,
    include_cute: bool,
):
    device = "cuda"
    helion_impl = try_import_helion() if include_helion else None
    cute_impl = try_import_cute() if include_cute else None

    # For each N, weights are the same across M in this benchmark run (fair + faster).
    for N in Ns:
        torch.manual_seed(0)
        W = torch.randn((K, N), device=device, dtype=dtype)
        packed, scales, zeros = quantize_int4_weight_only(W, group_size=group_size, symmetric=False)

        for M in Ms:
            torch.manual_seed(0)
            A = torch.randn((M, K), device=device, dtype=dtype)

            # --- Baseline: dequant + matmul
            def run_baseline():
                w4a16_baseline(A, packed, scales, zeros, N=N, group_size=group_size)

            # --- Triton fused (Split-K + atomic)
            def run_triton():
                w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size)

            # Warmup Triton once to trigger autotune and cache selection for this (M,N,K,group_size)
            run_triton()
            torch.cuda.synchronize()

            chosen = format_best_config()
            print(f"\n[{regime_name}] M={M} K={K} N={N} | Triton best_config = {chosen}")

            ms_base = bench_one(run_baseline, warmup_ms=warmup_ms, rep_ms=rep_ms)
            ms_tri = bench_one(run_triton, warmup_ms=warmup_ms, rep_ms=rep_ms)

            print(f"  baseline_dequant_matmul: {ms_base:.4f} ms")
            print(f"  triton_fused_w4a16     : {ms_tri:.4f} ms")

            out_rows.append(("baseline_dequant_matmul", regime_name, M, K, N, group_size, ms_base, chosen))
            out_rows.append(("triton_fused_w4a16",     regime_name, M, K, N, group_size, ms_tri,  chosen))

            # Optional Helion
            if helion_impl is not None:
                def run_helion():
                    helion_impl(A, packed, scales, zeros, N=N, group_size=group_size)

                # warmup
                run_helion()
                torch.cuda.synchronize()

                ms_h = bench_one(run_helion, warmup_ms=warmup_ms, rep_ms=rep_ms)
                print(f"  helion_fused_w4a16     : {ms_h:.4f} ms")
                out_rows.append(("helion_fused_w4a16", regime_name, M, K, N, group_size, ms_h, ""))

            # Optional CuTe DSL
            if cute_impl is not None:
                def run_cute():
                    cute_impl(A, packed, scales, zeros, N=N, group_size=group_size)

                # warmup
                run_cute()
                torch.cuda.synchronize()

                ms_c = bench_one(run_cute, warmup_ms=warmup_ms, rep_ms=rep_ms)
                print(f"  cute_fused_w4a16       : {ms_c:.4f} ms")
                out_rows.append(("cute_fused_w4a16", regime_name, M, K, N, group_size, ms_c, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["decode", "prefill", "both"], default="both")
    ap.add_argument("--decode_Ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--prefill_Ms", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--K", type=int, default=8192)
    ap.add_argument("--Ns", type=int, nargs="+", default=[4096, 8192, 11008])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--dtype", choices=["fp16"], default="fp16")
    ap.add_argument("--warmup_ms", type=int, default=25)
    ap.add_argument("--rep_ms", type=int, default=100)
    ap.add_argument("--out", type=str, default="bench_results.csv")
    ap.add_argument("--with_helion", action="store_true")
    ap.add_argument("--with_cute", action="store_true")
    ap.add_argument("--print_autotuning", action="store_true",
                    help="Set TRITON_PRINT_AUTOTUNING=1 so Triton prints best config after tuning.")
    args = ap.parse_args()

    if args.print_autotuning:
        # Triton prints autotuning results if this env var is set. :contentReference[oaicite:2]{index=2}
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    dtype = torch.float16
    Ns = parse_int_list(args.Ns)
    decode_Ms = parse_int_list(args.decode_Ms)
    prefill_Ms = parse_int_list(args.prefill_Ms)

    rows: List[Tuple] = []
    if args.regime in ("decode", "both"):
        run_regime(
            "decode", decode_Ms, args.K, Ns, args.group_size, dtype,
            rows, args.warmup_ms, args.rep_ms, args.with_helion, args.with_cute
        )
    if args.regime in ("prefill", "both"):
        run_regime(
            "prefill", prefill_Ms, args.K, Ns, args.group_size, dtype,
            rows, args.warmup_ms, args.rep_ms, args.with_helion, args.with_cute
        )

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "regime", "M", "K", "N", "group_size", "ms", "triton_best_config"])
        w.writerows(rows)

    print(f"\nSaved {args.out}")
    print("Tip: the first run for each (M,N,...) includes autotuning overhead; timing uses do_bench after warmup.")


if __name__ == "__main__":
    main()
