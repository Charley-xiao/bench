import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="bench_results.csv")
    ap.add_argument("--out", type=str, default="bench_plot.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Plot latency vs N for each kernel name
    pt = df.pivot_table(index="N", columns="name", values="ms", aggfunc="mean").sort_index()
    ax = pt.plot(marker="o")
    ax.set_xlabel("N (output features)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("W4A16 GEMM latency vs N")
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
