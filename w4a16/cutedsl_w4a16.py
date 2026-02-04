import torch

def w4a16_cutedsl(A, packed_w, scales, zeros, N: int, group_size: int = 128):
    """
    Experimental CuTe DSL implementation.
    If CuTe DSL isn't installed/configured, raise a helpful error.
    """
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except Exception as e:
        raise ImportError(
            "CuTe DSL (cutlass.cute) not available. "
            "Install/configure CUTLASS Python bindings per your environment."
        ) from e

    # ---- Minimal experimental path ----
    # Strategy (starter):
    # 1) decode packed_w into a small KxN tile inside the kernel
    # 2) multiply-accumulate with A tile
    #
    # You will likely tune/replace this with a CUTLASS GEMM + custom iterator
    # as you get comfortable with CuTe layouts.

    @cute.kernel
    def _kernel(mA: cute.Tensor, mBp: cute.Tensor, mS: cute.Tensor, mZ: cute.Tensor, mC: cute.Tensor,
                M: cutlass.Constexpr, K: cutlass.Constexpr, N: cutlass.Constexpr,
                GROUP: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bmx, bnx, _ = cute.arch.block_idx()

        # Very small tiles for starter correctness (tune later)
        BM = 16
        BN = 64
        BK = 32

        row = bmx * BM + (tidx // 32)  # crude mapping (starter)
        col_base = bnx * BN

        if row >= M:
            return

        # Accumulate BN outputs for this row (split across threads later)
        # For simplicity, each thread computes a single column here (starter).
        col = col_base + (tidx % 64)
        if col >= N:
            return

        acc = 0.0
        for k0 in range(0, K, BK):
            for kk in range(BK):
                k = k0 + kk
                if k >= K:
                    break

                a = mA[(row, k)]

                # decode one int4 weight at (k, col)
                byte = mBp[(k, col // 2)]
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                q = hi if (col & 1) else lo

                g = k // GROUP
                s = mS[(g, col)]
                z = mZ[(g, col)]
                w = (cutlass.Int32(q) - cutlass.Int32(z)) * cutlass.Float32(s)
                acc += cutlass.Float32(a) * w

        mC[(row, col)] = cutlass.Float16(acc)

    M, K = A.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    mA = from_dlpack(A, assumed_align=16)
    mBp = from_dlpack(packed_w, assumed_align=16)
    mS = from_dlpack(scales, assumed_align=16)
    mZ = from_dlpack(zeros, assumed_align=16)
    mC = from_dlpack(C, assumed_align=16)

    threads = 256
    grid = ( (M + 15) // 16, (N + 63) // 64, 1 )
    _kernel(mA, mBp, mS, mZ, mC, M, K, N, group_size).launch(
        grid=grid, block=(threads, 1, 1)
    )
    return C
