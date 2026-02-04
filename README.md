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
TRITON_PRINT_AUTOTUNING=1 python bench.py \
  --regime decode \
  --K 8192 \
  --Ns 4096 8192 11008 \
  --decode_Ms 1 2 4 8 16 32 \
  --group_size 128 \
  --warmup_ms 25 --rep_ms 100

TRITON_PRINT_AUTOTUNING=1 python bench.py \
  --regime prefill \
  --K 8192 \
  --Ns 4096 8192 11008 \
  --prefill_Ms 128 256 512 1024 \
  --group_size 128

TRITON_PRINT_AUTOTUNING=1 python bench.py --K 8192 --Ns 4096 8192 11008

TRITON_PRINT_AUTOTUNING=1 python bench.py --regime both --with_helion
# or
TRITON_PRINT_AUTOTUNING=1 python bench.py --regime both --with_cute

```