import torch
from w4a16.quant_utils import quantize_int4_weight_only
from w4a16.baseline import w4a16_baseline
from w4a16.triton_w4a16 import w4a16_triton

def check_finite(x, name):
    if not torch.isfinite(x).all():
        bad = x[~torch.isfinite(x)]
        print(f"[{name}] contains non-finite values: count={bad.numel()} "
              f"min={bad.min().item() if bad.numel() else 'n/a'} "
              f"max={bad.max().item() if bad.numel() else 'n/a'}")
        return False
    return True

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

    # 1) Force SPLIT_K=1 (should NOT need atomic accumulation)
    C_tri_1 = w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size, force_split_k=1)
    ok1 = check_finite(C_tri_1, "triton_splitk1")
    if ok1:
        torch.testing.assert_close(C_tri_1, C_ref, rtol=2e-2, atol=2e-2)
        print("✅ Triton correctness vs baseline ok (force_split_k=1)")

    # 2) Let autotune choose (likely SPLIT_K>1) — tests atomic path
    C_tri_auto = w4a16_triton(A, packed, scales, zeros, N=N, group_size=group_size, force_split_k=None)
    ok2 = check_finite(C_tri_auto, "triton_auto")
    if ok2:
        torch.testing.assert_close(C_tri_auto, C_ref, rtol=2e-2, atol=2e-2)
        print("✅ Triton correctness vs baseline ok (autotune)")

if __name__ == "__main__":
    main()
