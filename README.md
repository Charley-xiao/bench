# W4A16 INT4 Weight-Only GEMM (Decode-Fused)

This repo implements a decode-fused INT4 weight-only GEMM:
- A: FP16 activations [M, K]
- Bq: packed INT4 weights [K, N] stored as uint8 with 2 weights/byte
- scales, zeros: per-group affine quantization parameters (group along K)

Implementations:
- Triton: fused decode + matmul
- Helion: tiled Python kernel (compiled to Triton)
- CuTe DSL: layout-explicit kernel (experimental)

## Install
You need a CUDA GPU.

### Python deps
pip install torch triton matplotlib pandas

Optional:
pip install helion  (if available in your environment)
CuTe DSL requires CUTLASS Python / cutlass bindings per its docs.

## Quickstart
python -m w4a16.test_correctness
python bench.py --M 16 --K 4096 --Ns 4096 8192 --dtype fp16
python plot_results.py --csv bench_results.csv

```bash
TRITON_PRINT_AUTOTUNING=1 python bench.py --K 8192 --Ns 4096 8192 11008 --out bench_results.csv

python plot_results.py --csv bench_results.csv --outdir plots

```