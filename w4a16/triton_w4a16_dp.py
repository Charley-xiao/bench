# w4a16/triton_w4a16_dp.py
import torch
import triton
import triton.language as tl

# DP fused baseline (paper-style "traditional data-parallel" tiling):
# 2D grid over (M_tiles, N_tiles), each program computes a full C tile and reduces over K.
# This is the baseline Split-K is compared against in "SplitK vs Data Parallel" framing.

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K", "Np", "GROUP_SIZE"],
)
@triton.jit
def _w4a16_dp_fused_kernel(
    A_ptr, Bp_ptr, S_ptr, Z_ptr, C_ptr,
    # runtime sizes
    M, N, K,
    # packed N dimension (ceil(N/2))
    Np,
    # strides (passed as constexpr for specialization)
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bpk: tl.constexpr, stride_bpn2: tl.constexpr,
    stride_sg: tl.constexpr, stride_sn: tl.constexpr,
    stride_zg: tl.constexpr, stride_zn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # quant
    GROUP_SIZE: tl.constexpr,
    # tiling
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm_in = rm < M
    rn_in = rn < N

    # Optional alignment hints: apply to tensors, not constexpr ints.
    # tl.multiple_of is a compiler hint for vectorization/alignment. :contentReference[oaicite:1]{index=1}
    tl.multiple_of(rm, 1)
    tl.multiple_of(rn, 1)

    # Safe indices for pointer arithmetic
    rn_safe = tl.where(rn_in, rn, 0)
    n2 = rn_safe // 2
    n2_in = n2 < Np
    n2_safe = tl.where(n2_in, n2, 0)
    is_hi = (rn_safe & 1).to(tl.int1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # DP: full K reduction inside this program
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        rk_in = rk < K
        rk_safe = tl.where(rk_in, rk, 0)

        # A tile [BM, BK]
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk_safe[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=rm_in[:, None] & rk_in[None, :], other=0.0).to(tl.float16)

        # Packed weights bytes [BK, BN] (2 int4 per uint8)
        bp_ptrs = Bp_ptr + rk_safe[:, None] * stride_bpk + n2_safe[None, :] * stride_bpn2
        bp = tl.load(bp_ptrs, mask=rk_in[:, None] & rn_in[None, :] & n2_in[None, :], other=0).to(tl.uint8)

        lo = bp & 0x0F
        hi = (bp >> 4) & 0x0F
        q = tl.where(is_hi[None, :], hi, lo).to(tl.int32)

        # Group index
        g = (rk_safe // GROUP_SIZE).to(tl.int32)

        # scales/zeros [BK, BN]
        s_ptrs = S_ptr + g[:, None] * stride_sg + rn_safe[None, :] * stride_sn
        z_ptrs = Z_ptr + g[:, None] * stride_zg + rn_safe[None, :] * stride_zn
        sz_mask = rk_in[:, None] & rn_in[None, :]

        s = tl.load(s_ptrs, mask=sz_mask, other=0.0).to(tl.float16)
        z = tl.load(z_ptrs, mask=sz_mask, other=0).to(tl.uint8).to(tl.int32)

        b = (q - z).to(tl.float16) * s  # dequant -> fp16

        acc += tl.dot(a, b)

    # Store output
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn_safe[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=rm_in[:, None] & rn_in[None, :])


def w4a16_triton_dp(
    A: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    N: int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    DP fused baseline (paper-style baseline): fused dequant + GEMM, no Split-K.
    Autotuning uses @triton.autotune. :contentReference[oaicite:2]{index=2}
    """
    assert A.is_cuda and packed_w.is_cuda and scales.is_cuda and zeros.is_cuda
    assert A.dtype == torch.float16
    assert packed_w.dtype == torch.uint8
    assert scales.dtype == torch.float16
    assert zeros.dtype == torch.uint8

    M, K = A.shape
    assert K % group_size == 0

    # packed_w is [K, ceil(N/2)]
    Np = packed_w.shape[1]

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _w4a16_dp_fused_kernel[grid](
        A, packed_w, scales, zeros, C,
        M, N, K,
        Np,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bpk=packed_w.stride(0), stride_bpn2=packed_w.stride(1),
        stride_sg=scales.stride(0), stride_sn=scales.stride(1),
        stride_zg=zeros.stride(0), stride_zn=zeros.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        GROUP_SIZE=group_size,
    )
    return C
