import torch


def pack_int4(w_int: torch.Tensor) -> torch.Tensor:
    """
    Pack int4 values in [0, 15] into uint8 (2 per byte).
    Input:  w_int [K, N] uint8 in 0..15
    Output: packed [K, ceil(N/2)] uint8
    """
    assert w_int.dtype == torch.uint8
    K, N = w_int.shape
    Np = (N + 1) // 2
    out = torch.empty((K, Np), device=w_int.device, dtype=torch.uint8)

    lo = w_int[:, 0::2]
    hi = w_int[:, 1::2] if (N % 2 == 0) else torch.zeros((K, Np), device=w_int.device, dtype=torch.uint8)
    if N % 2 == 1:
        hi[:, :-1] = w_int[:, 1::2]

    out[:] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return out


def unpack_int4(packed: torch.Tensor, N: int) -> torch.Tensor:
    """
    Unpack packed uint8 into uint8 in [0, 15].
    packed: [K, ceil(N/2)]
    returns: [K, N] uint8
    """
    assert packed.dtype == torch.uint8
    K, Np = packed.shape
    out = torch.empty((K, N), device=packed.device, dtype=torch.uint8)

    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F

    out[:, 0::2] = lo[:, : out[:, 0::2].shape[1]]
    if N % 2 == 0:
        out[:, 1::2] = hi
    else:
        out[:, 1::2] = hi[:, :-1]
    return out


@torch.no_grad()
def quantize_int4_weight_only(
    W: torch.Tensor,
    group_size: int = 128,
    symmetric: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize W [K, N] FP16/FP32 to:
      - packed uint8 weights (2 int4 per byte), representing values in 0..15
      - scales [K//group_size, N] fp16
      - zeros  [K//group_size, N] uint8 (0..15)

    Affine per-group per-column:
      q = clamp(round(x/scale + zero), 0, 15)
      x_hat = (q - zero) * scale

    symmetric=True uses zero=8 (roughly centered) and scale from maxabs.
    """
    assert W.ndim == 2
    K, N = W.shape
    assert K % group_size == 0, "K must be divisible by group_size for this starter implementation."
    device = W.device
    Wf = W.float()

    G = K // group_size
    scales = torch.empty((G, N), device=device, dtype=torch.float16)
    zeros = torch.empty((G, N), device=device, dtype=torch.uint8)
    q_all = torch.empty((K, N), device=device, dtype=torch.uint8)

    for g in range(G):
        k0 = g * group_size
        k1 = (g + 1) * group_size
        block = Wf[k0:k1, :]  # [gs, N]

        if symmetric:
            maxabs = block.abs().amax(dim=0) + 1e-8
            scale = (maxabs / 7.0).clamp_min(1e-8)
            zero = torch.full((N,), 8, device=device, dtype=torch.int32)
        else:
            mn = block.amin(dim=0)
            mx = block.amax(dim=0)
            scale = ((mx - mn) / 15.0).clamp_min(1e-8)
            zero = torch.round((-mn / scale)).to(torch.int32).clamp(0, 15)

        scales[g, :] = scale.half()
        zeros[g, :] = zero.to(torch.uint8)

        q = torch.round(block / scale + zero).to(torch.int32).clamp(0, 15).to(torch.uint8)
        q_all[k0:k1, :] = q

    packed = pack_int4(q_all)
    return packed, scales, zeros
