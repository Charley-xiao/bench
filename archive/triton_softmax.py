# triton_softmax.py
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd_kernel(
    X_ptr, Y_ptr,
    stride_xm: tl.constexpr, stride_xn: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Row-wise softmax.
    One Triton program computes one row.
    """
    row = tl.program_id(0)

    # Column offsets for this program
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointers
    x_ptrs = X_ptr + row * stride_xm + col_offsets * stride_xn

    # Load (masked)
    x = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    numerator = tl.exp(x)
    denom = tl.sum(numerator, axis=0)
    y = numerator / denom

    # Store
    y_ptrs = Y_ptr + row * stride_ym + col_offsets * stride_yn
    tl.store(y_ptrs, y.to(tl.float16), mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    x: [M, N] CUDA tensor (fp16/bf16 recommended)
    returns: [M, N] fp16 output
    """
    assert x.is_cuda, "x must be on CUDA"
    assert x.ndim == 2, "x must be 2D [M, N]"
    M, N = x.shape

    # Triton tutorial style: handle rows that fit in SRAM via one program per row.
    BLOCK_SIZE = triton.next_power_of_2(N)
    # Practical limit (tune if needed; depends on GPU/resources)
    if BLOCK_SIZE > 65536:
        raise ValueError(f"N too large for this simple one-program-per-row kernel (N={N}).")

    # Heuristic: more warps for larger rows
    if BLOCK_SIZE <= 1024:
        num_warps = 4
    elif BLOCK_SIZE <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = (M,)
    _softmax_fwd_kernel[grid](
        x, y,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        n_cols=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


# -------------------------
# Quick correctness + speed
# -------------------------
def _check_correctness():
    torch.manual_seed(0)
    x = torch.randn((4096, 1024), device="cuda", dtype=torch.float16)
    y_ref = torch.softmax(x.float(), dim=1).half()
    y_tri = softmax_triton(x)
    max_err = (y_ref - y_tri).abs().max().item()
    print("max abs error:", max_err)
    torch.testing.assert_close(y_tri, y_ref, rtol=1e-2, atol=1e-2)
    print("✅ correctness ok")


def _bench():
    import triton.testing

    M, N = 8192, 2048
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)

    def run_triton():
        softmax_triton(x)

    def run_torch():
        torch.softmax(x, dim=1)

    ms_triton = triton.testing.do_bench(run_triton, warmup=25, rep=100)
    ms_torch = triton.testing.do_bench(run_torch, warmup=25, rep=100)

    # Rough effective bandwidth estimate (read X + write Y)
    bytes_moved = 2 * (M * N * x.element_size())  # read + write
    gb_s_triton = bytes_moved / (ms_triton * 1e-3) / 1e9
    gb_s_torch = bytes_moved / (ms_torch * 1e-3) / 1e9

    print(f"Triton: {ms_triton:.3f} ms  (~{gb_s_triton:.1f} GB/s)")
    print(f"Torch : {ms_torch:.3f} ms  (~{gb_s_torch:.1f} GB/s)")


if __name__ == "__main__":
    _check_correctness()
    _bench()
