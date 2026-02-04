import torch

try:
    import helion
    import helion.language as hl
except Exception:
    helion = None
    hl = None


def _require_helion():
    if helion is None or hl is None:
        raise ImportError("Helion not installed / import failed.")


@helion.kernel()
def w4a16_helion_kernel(A, Bp, S, Z, N: int, group_size: int):
    """
    A:  [M, K] fp16
    Bp: [K, ceil(N/2)] uint8 packed int4
    S:  [K//gs, N] fp16
    Z:  [K//gs, N] uint8
    returns C [M, N] fp16
    """
    _require_helion()

    M, K = A.size()
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Register tunable block sizes so Helion can use them for allocations/arange.
    # register_block_size is the intended way to get block sizes usable in allocations. :contentReference[oaicite:2]{index=2}
    BM = hl.register_block_size(1, M)
    BN = hl.register_block_size(1, N)
    BK = hl.register_block_size(1, K)

    # 2D tile loops -> launch grid
    for tm in hl.tile(M, block_size=BM):
        for tn in hl.tile(N, block_size=BN):
            # Fixed-size indices for this block (no dynamic-sized arange)
            rm = hl.tile_begin(tm) + hl.arange(0, BM)   # [BM]
            rn = hl.tile_begin(tn) + hl.arange(0, BN)   # [BN]

            rm_in = rm < M
            rn_in = rn < N

            # Packed weights indexing
            n2 = rn // 2
            is_hi = (rn & 1).to(torch.bool)

            acc = hl.zeros([BM, BN], dtype=torch.float32)

            # K loop tiles
            for tk in hl.tile(K, block_size=BK):
                rk = hl.tile_begin(tk) + hl.arange(0, BK)  # [BK]
                rk_in = rk < K

                # A block: [BM, BK]
                a = A[rm[:, None], rk[None, :]].to(torch.float16)
                a = torch.where(rm_in[:, None] & rk_in[None, :], a, torch.zeros_like(a))

                # B packed bytes: [BK, BN]
                # (Bp shape is [K, ceil(N/2)], so we index by (rk, n2))
                bp = Bp[rk[:, None], n2[None, :]].to(torch.uint8)
                bp = torch.where(rk_in[:, None] & rn_in[None, :], bp, torch.zeros_like(bp))

                lo = bp & 0x0F
                hi = (bp >> 4) & 0x0F
                q = torch.where(is_hi[None, :], hi, lo).to(torch.int32)  # [BK, BN]

                # group per rk
                g = (rk // group_size).to(torch.int64)  # [BK]

                # scales/zeros: [BK, BN]
                s = S[g[:, None], rn[None, :]].to(torch.float16)
                z = Z[g[:, None], rn[None, :]].to(torch.int32)
                sz_mask = rk_in[:, None] & rn_in[None, :]

                # mask tail
                s = torch.where(sz_mask, s, torch.zeros_like(s))
                z = torch.where(sz_mask, z, torch.zeros_like(z))

                b = (q - z).to(torch.float16) * s  # [BK, BN]

                # acc += A @ B
                acc = torch.addmm(acc, a, b)

            # store output tile
            out = acc.to(torch.float16)
            # write with mask
            # (Helion will lower this; keep explicit masking to avoid OOB)
            C[rm[:, None], rn[None, :]] = torch.where(
                rm_in[:, None] & rn_in[None, :],
                out,
                C[rm[:, None], rn[None, :]],
            )

    return C


def w4a16_helion(A, packed_w, scales, zeros, N: int, group_size: int = 128):
    _require_helion()
    return w4a16_helion_kernel(A, packed_w, scales, zeros, N, group_size)
