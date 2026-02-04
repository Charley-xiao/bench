# plot_results.py
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def pretty_method(row) -> str:
    name = str(row["name"])
    sk = str(row["split_k"]) if "split_k" in row else ""

    if name == "baseline_dequant_matmul":
        return "baseline (dequant+matmul)"
    if name == "triton_dp_fused":
        return "triton DP fused (paper baseline)"
    if name == "triton_store":
        return "triton store (sk=1)"
    if name == "triton_heuristic":
        return f"triton heuristic (sk={sk})"
    if name == "triton_splitk":
        return f"triton split-k (sk={sk})"
    if name == "helion_fused":
        return "helion fused"
    if name == "cutedsl_fused":
        return "cuTe DSL fused"
    return f"{name}{'' if sk in ('', 'n/a') else f' (sk={sk})'}"


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["M", "K", "N", "group_size"]:
        df[col] = df[col].astype(int)
    df["ms"] = df["ms"].astype(float)
    df["split_k"] = df["split_k"].astype(str)
    df["method"] = df.apply(pretty_method, axis=1)
    return df


def grid_plot_latency(df: pd.DataFrame, outpath: Path, ylog: bool = False):
    regimes = [r for r in ["decode", "prefill"] if r in df["regime"].unique()]
    Ns = sorted(df["N"].unique().tolist())

    fig, axes = plt.subplots(
        len(regimes), len(Ns),
        figsize=(5.4 * len(Ns), 4.0 * len(regimes)),
        squeeze=False
    )

    for i, regime in enumerate(regimes):
        for j, N in enumerate(Ns):
            ax = axes[i][j]
            sub = df[(df["regime"] == regime) & (df["N"] == N)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            for method, g in sub.groupby("method"):
                g = g.sort_values("M")
                ax.plot(g["M"], g["ms"], marker="o", linewidth=1.8, label=method)

            ax.set_title(f"{regime} | N={N}")
            ax.set_xlabel("M")
            ax.set_ylabel("latency (ms)")
            ax.grid(True, alpha=0.25)
            if ylog:
                ax.set_yscale("log")

    # global legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)

    fig.suptitle("", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")


def grid_plot_speedup(df: pd.DataFrame, outpath: Path, baseline_name: str, title: str):
    key = ["regime", "M", "K", "N", "group_size"]
    base = df[df["name"] == baseline_name][key + ["ms"]].rename(columns={"ms": "ms_base"})
    merged = df.merge(base, on=key, how="left")
    merged = merged[merged["ms_base"].notna()].copy()
    merged["speedup"] = merged["ms_base"] / merged["ms"]

    regimes = [r for r in ["decode", "prefill"] if r in merged["regime"].unique()]
    Ns = sorted(merged["N"].unique().tolist())

    fig, axes = plt.subplots(
        len(regimes), len(Ns),
        figsize=(5.4 * len(Ns), 4.0 * len(regimes)),
        squeeze=False
    )

    for i, regime in enumerate(regimes):
        for j, N in enumerate(Ns):
            ax = axes[i][j]
            sub = merged[(merged["regime"] == regime) & (merged["N"] == N)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            # Drop the baseline itself (speedup=1) to reduce clutter
            sub = sub[sub["name"] != baseline_name]

            for method, g in sub.groupby("method"):
                g = g.sort_values("M")
                ax.plot(g["M"], g["speedup"], marker="o", linewidth=1.8, label=method)

            ax.axhline(1.0, linestyle="--", linewidth=1.2)
            ax.set_title(f"{regime} | N={N}")
            ax.set_xlabel("M")
            ax.set_ylabel("speedup (×)")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="bench_results.csv")
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--ylog", action="store_true", help="log-scale latency axis")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)

    print("Found regimes:", sorted(df["regime"].unique().tolist()))
    print("Found Ns:", sorted(df["N"].unique().tolist()))
    print("Found methods:", sorted(df["method"].unique().tolist()))

    grid_plot_latency(df, outdir / ("latency_grid_log.png" if args.ylog else "latency_grid.png"), ylog=args.ylog)

    # Speedup vs naive baseline (dequant + matmul)
    grid_plot_speedup(
        df,
        outdir / "speedup_vs_naive_baseline.png",
        baseline_name="baseline_dequant_matmul",
        title=""
    )

    # Speedup vs DP fused baseline (paper-style baseline)
    grid_plot_speedup(
        df,
        outdir / "speedup_vs_dp_fused.png",
        baseline_name="triton_dp_fused",
        title=""
    )


if __name__ == "__main__":
    main()
