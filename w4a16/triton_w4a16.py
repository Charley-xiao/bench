# w4a16/triton_w4a16.py
import torch
import triton
import triton.language as tl

# ----------------------------
# Tunable knobs (A100-ish)
# ----------------------------
# We keep a small set of reasonable tilings. We do NOT use atomic-add split-k (broken on your env),
# instead we use "partials + reduce" when split_k > 1.
_DEFAULT_STORE_META = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3)
_DEFAULT_PARTIALS_META = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=4)
_DEFAULT_REDUCE_META = dict(BLOCK_M=32, BLOCK_N=128, num_warps=4)


@triton.jit
def _w4a16_store_kernel(
    A_ptr, Bp_ptr, S_ptr, Z_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, Np: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bpk: tl.constexpr, stride_bpn2: tl.constexpr,
    stride_sg: tl.constexpr, stride_sn: tl.constexpr,
    stride_zg: tl.constexpr, stride_zn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm_in = rm < M
    rn_in = rn < N

    # Safe indices for pointer arithmetic
    rn_safe = tl.where(rn_in, rn, 0)
    n2 = rn_safe // 2
    n2_in = n2 < Np
    n2_safe = tl.where(n2_in, n2, 0)
    is_hi = (rn_safe & 1).to(tl.int1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        rk_in = rk < K
        rk_safe = tl.where(rk_in, rk, 0)

        # A [BM, BK]
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk_safe[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=rm_in[:, None] & rk_in[None, :], other=0.0).to(tl.float16)

        # packed B bytes [BK, BN]
        bp_ptrs = Bp_ptr + rk_safe[:, None] * stride_bpk + n2_safe[None, :] * stride_bpn2
        bp = tl.load(bp_ptrs, mask=rk_in[:, None] & rn_in[None, :] & n2_in[None, :], other=0).to(tl.uint8)

        lo = bp & 0x0F
        hi = (bp >> 4) & 0x0F
        q = tl.where(is_hi[None, :], hi, lo).to(tl.int32)  # 0..15

        # scales/zeros per group
        g = (rk_safe // GROUP_SIZE).to(tl.int32)  # rk_safe is always in-bounds
        s_ptrs = S_ptr + g[:, None] * stride_sg + rn_safe[None, :] * stride_sn
        z_ptrs = Z_ptr + g[:, None] * stride_zg + rn_safe[None, :] * stride_zn
        sz_mask = rk_in[:, None] & rn_in[None, :]
        s = tl.load(s_ptrs, mask=sz_mask, other=0.0).to(tl.float16)
        z = tl.load(z_ptrs, mask=sz_mask, other=0).to(tl.uint8).to(tl.int32)

        b = (q - z).to(tl.float16) * s  # dequantized fp16 [BK, BN]

        acc += tl.dot(a, b)

    # Store fp16 output
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn_safe[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=rm_in[:, None] & rn_in[None, :])


@triton.jit
def _w4a16_splitk_partials_kernel(
    A_ptr, Bp_ptr, S_ptr, Z_ptr, P_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, Np: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bpk: tl.constexpr, stride_bpn2: tl.constexpr,
    stride_sg: tl.constexpr, stride_sn: tl.constexpr,
    stride_zg: tl.constexpr, stride_zn: tl.constexpr,
    # P is [SPLIT_K, M, N] fp32
    stride_ps: tl.constexpr, stride_pm: tl.constexpr, stride_pn: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # 3D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_sk = tl.program_id(2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm_in = rm < M
    rn_in = rn < N

    rn_safe = tl.where(rn_in, rn, 0)
    n2 = rn_safe // 2
    n2_in = n2 < Np
    n2_safe = tl.where(n2_in, n2, 0)
    is_hi = (rn_safe & 1).to(tl.int1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # each split handles k0 = pid_sk*BK + t*(BK*SPLIT_K)
    for k0 in range(pid_sk * BLOCK_K, K, BLOCK_K * SPLIT_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        rk_in = rk < K
        rk_safe = tl.where(rk_in, rk, 0)

        a_ptrs = A_ptr + rm[:, None] * stride_am + rk_safe[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=rm_in[:, None] & rk_in[None, :], other=0.0).to(tl.float16)

        bp_ptrs = Bp_ptr + rk_safe[:, None] * stride_bpk + n2_safe[None, :] * stride_bpn2
        bp = tl.load(bp_ptrs, mask=rk_in[:, None] & rn_in[None, :] & n2_in[None, :], other=0).to(tl.uint8)

        lo = bp & 0x0F
        hi = (bp >> 4) & 0x0F
        q = tl.where(is_hi[None, :], hi, lo).to(tl.int32)

        g = (rk_safe // GROUP_SIZE).to(tl.int32)
        s_ptrs = S_ptr + g[:, None] * stride_sg + rn_safe[None, :] * stride_sn
        z_ptrs = Z_ptr + g[:, None] * stride_zg + rn_safe[None, :] * stride_zn
        sz_mask = rk_in[:, None] & rn_in[None, :]
        s = tl.load(s_ptrs, mask=sz_mask, other=0.0).to(tl.float16)
        z = tl.load(z_ptrs, mask=sz_mask, other=0).to(tl.uint8).to(tl.int32)

        b = (q - z).to(tl.float16) * s
        acc += tl.dot(a, b)

    # write P[pid_sk, rm, rn] (fp32)
    p_ptrs = P_ptr + pid_sk * stride_ps + rm[:, None] * stride_pm + rn_safe[None, :] * stride_pn
    tl.store(p_ptrs, acc, mask=rm_in[:, None] & rn_in[None, :])


@triton.jit
def _reduce_partials_to_fp16_kernel(
    P_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr,
    stride_ps: tl.constexpr, stride_pm: tl.constexpr, stride_pn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm_in = rm < M
    rn_in = rn < N
    rn_safe = tl.where(rn_in, rn, 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduce over split dimension
    for s in range(0, SPLIT_K):
        p_ptrs = P_ptr + s * stride_ps + rm[:, None] * stride_pm + rn_safe[None, :] * stride_pn
        acc += tl.load(p_ptrs, mask=rm_in[:, None] & rn_in[None, :], other=0.0).to(tl.float32)

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn_safe[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=rm_in[:, None] & rn_in[None, :])


def w4a16_triton(
    A: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    N: int,
    group_size: int = 128,
    force_split_k: int | None = None,
) -> torch.Tensor:
    """
    Drop-in API:

      - force_split_k=1: single-pass kernel (no split, no atomics)
      - force_split_k>1: Split-K using partials + reduce (no atomics)
      - force_split_k=None: choose split_k heuristically from M (decode vs prefill)
          * M <= 32  -> 4
          * M <= 128 -> 2
          * else     -> 1

    This avoids tl.atomic_add, which is producing non-finite results in your environment.
    """
    assert A.is_cuda and packed_w.is_cuda and scales.is_cuda and zeros.is_cuda
    assert A.dtype == torch.float16
    assert packed_w.dtype == torch.uint8
    assert scales.dtype == torch.float16
    assert zeros.dtype == torch.uint8

    M, K = A.shape
    assert K % group_size == 0
    Np = packed_w.shape[1]  # should be ceil(N/2)

    if force_split_k is None:
        if M <= 32:
            split_k = 4
        elif M <= 128:
            split_k = 2
        else:
            split_k = 1
    else:
        split_k = int(force_split_k)
        if split_k < 1:
            raise ValueError("force_split_k must be None or >= 1")

    # ------------------------
    # Path: SPLIT_K = 1 store
    # ------------------------
    if split_k == 1:
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        meta = _DEFAULT_STORE_META
        grid = (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _w4a16_store_kernel[grid](
            A, packed_w, scales, zeros, C,
            M=M, N=N, K=K, Np=Np,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_bpk=packed_w.stride(0), stride_bpn2=packed_w.stride(1),
            stride_sg=scales.stride(0), stride_sn=scales.stride(1),
            stride_zg=zeros.stride(0), stride_zn=zeros.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
            GROUP_SIZE=group_size,
            BLOCK_M=meta["BLOCK_M"], BLOCK_N=meta["BLOCK_N"], BLOCK_K=meta["BLOCK_K"],
            num_warps=meta["num_warps"], num_stages=meta["num_stages"],
        )
        return C

    # --------------------------------------------
    # Path: SPLIT_K > 1 partials + reduce (no atomic)
    # --------------------------------------------
    P = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
    meta = _DEFAULT_PARTIALS_META

    grid3 = (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        split_k,
    )
    _w4a16_splitk_partials_kernel[grid3](
        A, packed_w, scales, zeros, P,
        M=M, N=N, K=K, Np=Np,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bpk=packed_w.stride(0), stride_bpn2=packed_w.stride(1),
        stride_sg=scales.stride(0), stride_sn=scales.stride(1),
        stride_zg=zeros.stride(0), stride_zn=zeros.stride(1),
        stride_ps=P.stride(0), stride_pm=P.stride(1), stride_pn=P.stride(2),
        GROUP_SIZE=group_size,
        BLOCK_M=meta["BLOCK_M"], BLOCK_N=meta["BLOCK_N"], BLOCK_K=meta["BLOCK_K"],
        SPLIT_K=split_k,
        num_warps=meta["num_warps"], num_stages=meta["num_stages"],
    )

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    rmeta = _DEFAULT_REDUCE_META
    grid2 = (triton.cdiv(M, rmeta["BLOCK_M"]), triton.cdiv(N, rmeta["BLOCK_N"]))
    _reduce_partials_to_fp16_kernel[grid2](
        P, C,
        M=M, N=N, SPLIT_K=split_k,
        stride_ps=P.stride(0), stride_pm=P.stride(1), stride_pn=P.stride(2),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_M=rmeta["BLOCK_M"], BLOCK_N=rmeta["BLOCK_N"],
        num_warps=rmeta["num_warps"],
    )
    return C
