import torch
from w4a16.quant_utils import quantize_int4_weight_only
from w4a16.baseline import w4a16_baseline
from w4a16.triton_w4a16 import w4a16_triton

def main():
    torch.manual_seed(0)
    device = "cuda"

    M = 16
    K = 1024
    N = 2048
    group_size = 128

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    W = torch.randn((K, N), device=device, dtype=torch.float16)

    packed, scales, zeros = quantize_int4_weight_only(W, group_size=group_size, symmetric=False)

    # Baseline (dequant then matmul)
    C_ref = w4a16_baseline(A, packed, scales, zeros, N=N, group_size=group_size)

    # Triton fused
    C_tri = w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size)

    # We expect some quantization error; compare against baseline dequant path
    torch.testing.assert_close(C_tri, C_ref, rtol=2e-2, atol=2e-2)
    print("✅ Triton correctness vs baseline ok")

if __name__ == "__main__":
    main()
