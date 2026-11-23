# SPDX-License-Identifier: Apache-2.0
"""Lab (AS1) int8 all-reduce for prefill-sized TP messages (2026-11-23).

Why: on AS1's two RTX 3090s (PCIe 4.0 via the CPU host bridge, no NVLink) the bf16 NCCL ring all-reduce runs at the
P2P ceiling (~8.8 GB/s for every NCCL protocol / P2P level) and is 38-39 % of prefill time (nsys, Qwen3.8-27B TP2).
Fewer bytes is the only lever. Same design as llama.cpp lab patch 0077 (Pascal): int8 payload in 32-element groups
with one fp32 scale (1.125 B/element instead of 2), optionally companded.

How (TP2 or more, any message >= LAB_AR_INT8_MIN elements): each rank quantizes its partial sum, one NCCL all-gather
moves the int8 payloads, every rank dequantizes all payloads and adds them in rank order, so all ranks hold
bit-identical results (TP requires it). Decode-sized messages (cudagraph capture sizes) never reach this path.
Modes (LAB_AR_INT8_MODE): 0 = linear int8, 2 = cube-root companded (dense models needed it in llama.cpp 0077b).
Default on (cube-root, mode 2); opt out with LAB_AR_INT8=0. KLD + speed evidence: lab SHARED-EXPERIMENT-LOG 2026-11-23.
"""

import os

import torch

from vllm.triton_utils import tl, triton

GROUP = 32


def enabled() -> bool:
    return os.environ.get("LAB_AR_INT8", "1") != "0"


MIN_NUMEL = int(os.environ.get("LAB_AR_INT8_MIN", str(1 << 20)))
MODE = int(os.environ.get("LAB_AR_INT8_MODE", "2"))


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, n_groups, MODE: tl.constexpr, R: tl.constexpr, G: tl.constexpr):
    g = tl.program_id(0) * R + tl.arange(0, R)
    gm = g < n_groups
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    x = tl.load(x_ptr + offs, mask=gm[:, None], other=0.0).to(tl.float32)
    if MODE == 2:
        a = tl.abs(x)
        c = tl.where(a > 0, tl.exp(tl.log(tl.maximum(a, 1e-30)) * (1.0 / 3.0)), 0.0)
        x = tl.where(x < 0, -c, c)
    s = tl.max(tl.abs(x), axis=1)
    inv = tl.where(s > 0, 127.0 / s, 0.0)
    q = x * inv[:, None]
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)          # round half away from zero
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=gm[:, None])
    tl.store(s_ptr + g, s, mask=gm)


@triton.jit
def _dequant_sum_kernel(q_ptr, s_ptr, out_ptr, n_groups, q_stride, s_stride, WORLD: tl.constexpr, MODE: tl.constexpr,
                        R: tl.constexpr, G: tl.constexpr):
    g = tl.program_id(0) * R + tl.arange(0, R)
    gm = g < n_groups
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    acc = tl.zeros((R, G), dtype=tl.float32)
    for r in tl.static_range(WORLD):                                # fixed rank order: identical sums on every rank
        q = tl.load(q_ptr + r * q_stride + offs, mask=gm[:, None], other=0).to(tl.float32)
        s = tl.load(s_ptr + r * s_stride + g, mask=gm, other=0.0)
        v = q * (s * (1.0 / 127.0))[:, None]
        if MODE == 2:
            v = v * v * v
        acc += v
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=gm[:, None])


def applies(x: torch.Tensor, world_size: int) -> bool:
    return (enabled() and world_size >= 2 and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
            and x.numel() >= MIN_NUMEL and x.numel() % GROUP == 0 and x.is_contiguous()
            and not torch.cuda.is_current_stream_capturing())


def all_reduce(pynccl_comm, x: torch.Tensor):
    """int8 all-reduce of x over pynccl_comm; returns a new tensor, or None to fall back."""
    if pynccl_comm is None or pynccl_comm.disabled:
        return None
    world = pynccl_comm.world_size
    n = x.numel()
    ng = n // GROUP
    pay = n + 4 * ng                                                # int8 payload + fp32 scales, one buffer
    buf = torch.empty(pay, dtype=torch.uint8, device=x.device)
    q = buf[:n].view(torch.int8)
    s = buf[n:].view(torch.float32)
    R = 16
    grid = (triton.cdiv(ng, R),)
    _quant_kernel[grid](x, q, s, ng, MODE=MODE, R=R, G=GROUP)
    allbuf = torch.empty(world * pay, dtype=torch.uint8, device=x.device)
    pynccl_comm.all_gather(allbuf, buf)
    allq = allbuf.view(world, pay)[:, :n].view(torch.int8)
    alls = allbuf.view(world, pay)[:, n:].view(torch.float32)      # scales start at a 4-byte aligned offset (n % 32 == 0)
    out = torch.empty_like(x)
    _dequant_sum_kernel[grid](allq, alls, out, ng, pay, pay // 4, WORLD=world, MODE=MODE, R=R, G=GROUP)
    return out
