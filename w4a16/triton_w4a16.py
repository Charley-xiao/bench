# w4a16/triton_w4a16.py
import torch
import triton
import triton.language as tl


# ----------------------------
# Autotune configs (A100-ish)
# ----------------------------
# Notes:
# - BLOCK_K should divide common LLM K (4096/5120/8192).
# - SPLIT_K helps when M is small (decode regime).
# - num_stages controls software pipelining; 3-5 are common starting points.
# - num_warps: 4 or 8 usually reasonable for BN 64/128.
_AUTOTUNE_CONFIGS = [
    # No split-K / light tiling
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 32, "SPLIT_K": 1}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 32, "SPLIT_K": 1}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1}, num_warps=8, num_stages=5),

    # Split-K variants for skinny M (decode)
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 32, "SPLIT_K": 2}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 2}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 32, "SPLIT_K": 4}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 4}, num_warps=8, num_stages=4),

    # A bit larger K tile (sometimes good on Ampere)
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64, "SPLIT_K": 2}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 2}, num_warps=8, num_stages=4),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K", "GROUP_SIZE"],
)
@triton.jit
def _w4a16_splitk_atomic_kernel(
    A_ptr,               # *fp16, [M, K]
    Bp_ptr,              # *uint8, [K, ceil(N/2)] packed int4
    S_ptr,               # *fp16,  [K//GROUP_SIZE, N]
    Z_ptr,               # *uint8, [K//GROUP_SIZE, N]
    Cacc_ptr,            # *fp32,  [M, N] accumulation buffer (must be zeroed)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bpk: tl.constexpr,
    stride_bpn2: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_zg: tl.constexpr,
    stride_zn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Decode-fused W4A16 GEMM with Split-K and atomic reduction into FP32 Cacc.

    Grid:
      pid_m: tile along M
      pid_n: tile along N
      pid_sk: split along K (0..SPLIT_K-1)

    Each (pid_m, pid_n, pid_sk) computes partial sums over k = pid_sk*BLOCK_K + t*(BLOCK_K*SPLIT_K).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_sk = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this split-k slice
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Split-K loop over K tiles:
    # k0 starts at pid_sk*BLOCK_K and jumps by BLOCK_K*SPLIT_K.
    for k0 in range(pid_sk * BLOCK_K, K, BLOCK_K * SPLIT_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        # --- Load A tile [BM, BK]
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # --- Decode B on-the-fly into fp16 tile [BK, BN]
        # packed B: Bp[k, n//2] stores two 4-bit vals (lo for even n, hi for odd n)
        n2 = rn // 2
        is_hi = (rn & 1).to(tl.int1)  # odd columns -> hi nibble

        bp_ptrs = Bp_ptr + rk[:, None] * stride_bpk + n2[None, :] * stride_bpn2
        bp_mask = (rk[:, None] < K) & (rn[None, :] < N)
        bp = tl.load(bp_ptrs, mask=bp_mask, other=0).to(tl.uint8)

        lo = bp & 0x0F
        hi = (bp >> 4) & 0x0F
        q = tl.where(is_hi[None, :], hi, lo).to(tl.int32)  # [BK, BN] 0..15

        # scale/zero per group along K
        g = (rk // GROUP_SIZE).to(tl.int32)  # [BK]
        s_ptrs = S_ptr + g[:, None] * stride_sg + rn[None, :] * stride_sn
        z_ptrs = Z_ptr + g[:, None] * stride_zg + rn[None, :] * stride_zn
        s = tl.load(s_ptrs, mask=bp_mask, other=0.0).to(tl.float16)
        z = tl.load(z_ptrs, mask=bp_mask, other=0).to(tl.uint8).to(tl.int32)

        b = (q - z).to(tl.float16) * s  # fp16 dequantized [BK, BN]

        # --- GEMM accumulate
        # tl.dot will use tensor cores for fp16 inputs where applicable.
        acc += tl.dot(a, b)

    # --- Atomic add partial sums into Cacc (fp32)
    c_ptrs = Cacc_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask)


@triton.jit
def _cast_fp32_to_fp16_kernel(Cacc_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr,
                              stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                              stride_om: tl.constexpr, stride_on: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ptrs = Cacc_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    out_ptrs = C_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, x.to(tl.float16), mask=mask)


def w4a16_triton(
    A: torch.Tensor,            # [M, K] fp16
    packed_w: torch.Tensor,     # [K, ceil(N/2)] uint8
    scales: torch.Tensor,       # [K//gs, N] fp16
    zeros: torch.Tensor,        # [K//gs, N] uint8
    N: int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Decode-fused INT4 weight-only GEMM using Split-K + atomic accumulation (FP32),
    then cast to FP16 output.

    This is tuned for A100-ish shapes where M can be small (decode) and K,N are large.

    Returns:
      C: [M, N] fp16
    """
    assert A.is_cuda and packed_w.is_cuda and scales.is_cuda and zeros.is_cuda
    assert A.ndim == 2
    assert A.dtype == torch.float16, "This kernel expects FP16 activations."
    M, K = A.shape
    assert K % group_size == 0, "K must be divisible by group_size."
    assert scales.shape[1] == N and zeros.shape[1] == N

    # Accum buffer for split-k atomic adds
    Cacc = torch.zeros((M, N), device=A.device, dtype=torch.float32)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # 3D grid: (M-tiles, N-tiles, SPLIT_K)
    # SPLIT_K is chosen by autotuner as a compile-time meta-parameter.
    # Triton will launch pid_sk in [0..SPLIT_K-1].
    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
                meta["SPLIT_K"])

    _w4a16_splitk_atomic_kernel[grid](
        A, packed_w, scales, zeros, Cacc,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bpk=packed_w.stride(0), stride_bpn2=packed_w.stride(1),
        stride_sg=scales.stride(0), stride_sn=scales.stride(1),
        stride_zg=zeros.stride(0), stride_zn=zeros.stride(1),
        stride_cm=Cacc.stride(0), stride_cn=Cacc.stride(1),
        GROUP_SIZE=group_size,
    )

    # Cast fp32 -> fp16 (small overhead, keeps atomic path simple/robust)
    cast_grid = (triton.cdiv(M, 32), triton.cdiv(N, 128))
    _cast_fp32_to_fp16_kernel[cast_grid](
        Cacc, C,
        M=M, N=N,
        stride_cm=Cacc.stride(0), stride_cn=Cacc.stride(1),
        stride_om=C.stride(0), stride_on=C.stride(1),
        BLOCK_M=32, BLOCK_N=128,
        num_warps=4,
    )
    return C
