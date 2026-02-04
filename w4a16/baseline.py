import torch
from .quant_utils import unpack_int4


def dequant_int4(
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    N: int,
    group_size: int = 128,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize packed int4 weights back to FP16/FP32.
    packed_w: [K, ceil(N/2)] uint8
    scales:   [K//gs, N] fp16
    zeros:    [K//gs, N] uint8
    returns:  W_hat [K, N] out_dtype
    """
    assert packed_w.dtype == torch.uint8
    K = packed_w.shape[0]
    assert K % group_size == 0
    G = K // group_size

    q = unpack_int4(packed_w, N).to(torch.int32)  # [K, N]
    out = torch.empty((K, N), device=packed_w.device, dtype=torch.float32)

    for g in range(G):
        k0 = g * group_size
        k1 = (g + 1) * group_size
        scale = scales[g, :].float()[None, :]  # [1, N]
        zero = zeros[g, :].to(torch.int32)[None, :]  # [1, N]
        out[k0:k1, :] = (q[k0:k1, :] - zero) * scale

    return out.to(out_dtype)


def w4a16_baseline(
    A: torch.Tensor,            # [M, K] fp16
    packed_w: torch.Tensor,     # [K, ceil(N/2)] uint8
    scales: torch.Tensor,       # [K//gs, N] fp16
    zeros: torch.Tensor,        # [K//gs, N] uint8
    N: int,
    group_size: int = 128,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Baseline: dequantize W then do A @ W.
    """
    assert A.ndim == 2 and A.is_cuda
    M, K = A.shape
    W = dequant_int4(packed_w, scales, zeros, N=N, group_size=group_size, out_dtype=A.dtype)
    C = A @ W  # [M, N]
    return C.to(out_dtype)
