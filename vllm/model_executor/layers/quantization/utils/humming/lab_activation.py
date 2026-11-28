# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert four-bit activation levels to signed bytes for integer MMA."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _iq4nl_level(x):
    # Midpoints between adjacent IQ4_NL levels. Ties select the lower level.
    neg_low = tl.where(
        x <= -93.5, tl.where(x <= -115.5, -127, -104), tl.where(x <= -74.0, -83, -65)
    )
    neg_high = tl.where(
        x <= -28.5, tl.where(x <= -42.0, -49, -35), tl.where(x <= -16.0, -22, -10)
    )
    pos_low = tl.where(
        x <= 19.0, tl.where(x <= 7.0, 1, 13), tl.where(x <= 31.5, 25, 38)
    )
    pos_high = tl.where(
        x <= 79.0, tl.where(x <= 61.0, 53, 69), tl.where(x <= 101.0, 89, 113)
    )
    return tl.where(
        x <= -4.5,
        tl.where(x <= -57.0, neg_low, neg_high),
        tl.where(x <= 45.5, pos_low, pos_high),
    )


@triton.jit
def _quantize_four_bit(
    inputs,
    codes,
    scales,
    groups,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    IQ4NL: tl.constexpr,
):
    group = tl.program_id(0) * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    col = tl.arange(0, BLOCK_K)
    offset = group[:, None].to(tl.int64) * K + col[None, :]
    valid = (group[:, None] < groups) & (col[None, :] < K)
    x = tl.load(inputs + offset, valid, other=0).to(tl.float32)
    if IQ4NL:
        positive_scale = tl.max(x, axis=1) * (1.0 / 113.0)
        negative_scale = -tl.min(x, axis=1) * (1.0 / 127.0)
        scale = tl.maximum(positive_scale, negative_scale)
        normalized = tl.div_rn(x, tl.where(scale > 0, scale, 1.0)[:, None])
        level = _iq4nl_level(normalized)
    else:
        peak = tl.max(tl.abs(x), axis=1)
        scale = peak * (1.0 / 7.0)
        # An exact division keeps halfway values consistent across layouts.
        scaled = tl.div_rn(x * 7.0, tl.where(peak > 0, peak, 1.0)[:, None])
        level = tl.clamp(tl.extra.cuda.libdevice.rint(scaled), -7.0, 7.0)
    tl.store(codes + offset, level.to(tl.int8), valid)
    tl.store(scales + group, scale, group < groups)


def quantize_for_integer_mma(
    inputs: torch.Tensor, group_size: int, codebook: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit exact selected levels and FP32 scales in Humming's row-major layout."""
    rows, hidden = inputs.shape
    group_size = group_size or hidden
    groups_per_row = hidden // group_size
    codes = torch.empty_like(inputs, dtype=torch.int8)
    # Humming may load a full vector when reading the last per-token scale.
    padded_rows = triton.cdiv(rows, 4) * 4 if groups_per_row == 1 else rows
    scales = torch.empty(
        (padded_rows, groups_per_row), device=inputs.device, dtype=torch.float32
    )[:rows]
    if rows == 0:
        return codes, scales
    groups = rows * groups_per_row
    block_k = triton.next_power_of_2(group_size)
    groups_per_block = max(1, 256 // block_k)
    _quantize_four_bit[(triton.cdiv(groups, groups_per_block),)](
        inputs,
        codes,
        scales,
        groups,
        group_size,
        block_k,
        groups_per_block,
        codebook,
        num_warps=4 if block_k <= 4096 else 8,
    )
    return codes, scales
