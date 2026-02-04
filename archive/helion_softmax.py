# helion_softmax.py
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel()
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable-ish 2-pass softmax (streaming max/sum, then write).
    Input:  [m, n]
    Output: [m, n]
    """
    m, n = x.size()
    out = torch.empty_like(x)

    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)

    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)

        # pass 1: compute row-wise max & denom
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_max = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_max)
            di = di * torch.exp(mi - mi_next) + torch.exp(values - mi_next[:, None]).sum(dim=1)
            mi = mi_next

        # pass 2: write normalized exp
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]

    return out


if __name__ == "__main__":
    x = torch.randn((4096, 2560), device="cuda", dtype=torch.float16)
    y = softmax_two_pass(x)
    y_ref = torch.nn.functional.softmax(x, dim=1)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
    print("Helion softmax OK")
