# triton_kernels.py
import torch
import triton
import triton.language as tl


# -------------------------
# 1) Vector Add
# -------------------------
@triton.jit
def vadd_kernel(X_ptr, Y_ptr, Z_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(Z_ptr + offs, x + y, mask=mask)


def vadd_triton(x: torch.Tensor, y: torch.Tensor, block=1024):
    assert x.is_cuda and y.is_cuda
    assert x.numel() == y.numel()
    z = torch.empty_like(x)
    N = x.numel()
    grid = (triton.cdiv(N, block),)
    vadd_kernel[grid](x, y, z, N=N, BLOCK=block)
    return z


# -------------------------
# 2) Matrix Transpose
# -------------------------
@triton.jit
def transpose_kernel(X_ptr, Y_ptr, M: tl.constexpr, N: tl.constexpr,
                     stride_xm, stride_xn, stride_ym, stride_yn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load tile X[rm, rn]
    x_ptrs = X_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Store transposed to Y[rn, rm]
    y_ptrs = Y_ptr + rn[:, None] * stride_ym + rm[None, :] * stride_yn
    tl.store(y_ptrs, tl.trans(x), mask=mask.T)


def transpose_triton(x: torch.Tensor, block_m=32, block_n=32):
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    transpose_kernel[grid](
        x, y,
        M=M, N=N,
        stride_xm=x.stride(0), stride_xn=x.stride(1),
        stride_ym=y.stride(0), stride_yn=y.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4
    )
    return y


# -------------------------
# 3) Reduction: sum(x) -> one scalar
# -------------------------
@triton.jit
def reduce_sum_stage1(X_ptr, Partial_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.store(Partial_ptr + pid, s)


def sum_triton(x: torch.Tensor, block=1024):
    """
    Two-stage sum:
      stage1: reduce each block to partial sums
      stage2: torch.sum on CPU/GPU for the tiny partial array
    (You can make stage2 Triton too later.)
    """
    assert x.is_cuda and x.ndim == 1
    N = x.numel()
    grid = (triton.cdiv(N, block),)
    partial = torch.empty((grid[0],), device=x.device, dtype=torch.float32)
    reduce_sum_stage1[grid](x, partial, N=N, BLOCK=block)
    return partial.sum()
