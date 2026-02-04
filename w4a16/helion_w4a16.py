import torch

try:
    import helion
    import helion.language as hl
except Exception:
    helion = None
    hl = None


def _require_helion():
    if helion is None:
        raise ImportError("Helion not installed / import failed.")


@helion.kernel()
def w4a16_helion_kernel(A, Bp, S, Z, N: int, group_size: int):
    """
    A:  [M, K] fp16
    Bp: [K, ceil(N/2)] uint8 packed int4
    S:  [K//gs, N] fp16
    Z:  [K//gs, N] uint8
    returns C [M, N] fp16

    This is a *reference* Helion kernel; expect Triton to be faster initially.
    """
    _require_helion()
    M, K = A.size()
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    bm = hl.register_block_size(M)
    bn = hl.register_block_size(N)
    bk = hl.register_block_size(K)

    for tm in hl.tile(M, block_size=bm):
        for tn in hl.tile(N, block_size=bn):
            acc = hl.zeros([tm, tn], dtype=torch.float32)

            for tk0 in range(0, K, bk):
                tk = slice(tk0, tk0 + bk)

                a = A[tm, tk].to(torch.float16)  # [tm, bk]

                # decode B tile [bk, tn]
                n_idx = hl.arange(tn.start, tn.stop)  # conceptual, Helion will lower this
                n2 = n_idx // 2
                is_hi = (n_idx & 1).to(torch.bool)

                bp = Bp[tk, n2]  # [bk, tn] packed bytes
                lo = bp & 0x0F
                hi = (bp >> 4) & 0x0F
                q = torch.where(is_hi[None, :], hi, lo).to(torch.int32)

                g = (torch.arange(tk0, tk0 + bk, device=A.device) // group_size).to(torch.int64)
                s = S[g[:, None], n_idx[None, :]].to(torch.float16)
                z = Z[g[:, None], n_idx[None, :]].to(torch.int32)

                b = (q - z).to(torch.float16) * s
                acc += a @ b

            C[tm, tn] = acc.to(torch.float16)

    return C


def w4a16_helion(A, packed_w, scales, zeros, N: int, group_size: int = 128):
    _require_helion()
    return w4a16_helion_kernel(A, packed_w, scales, zeros, N, group_size)
