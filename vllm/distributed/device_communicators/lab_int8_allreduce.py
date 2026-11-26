# SPDX-License-Identifier: Apache-2.0
"""Lab (AS1) int8 all-reduce for prefill-sized TP messages (2026-11-23).

Why: on AS1's two RTX 3090s (PCIe 4.0 via the CPU host bridge, no NVLink) the bf16 NCCL ring all-reduce runs at the
P2P ceiling (~8.8 GB/s for every NCCL protocol / P2P level) and was 38-39 % of prefill time (nsys, Qwen3.8-27B TP2).
Fewer bytes is the only lever. Same design as llama.cpp lab patch 0077 (Pascal): int8 payload in small groups with one
scale per group, companded.

How (any TP size, messages >= LAB_AR_INT8_MIN elements): each rank quantizes its partial sum, one NCCL all-gather
moves the int8 payloads, every rank dequantizes all payloads and adds them in rank order, so all ranks hold
bit-identical results (TP requires it). Decode-sized messages (cudagraph capture sizes) never reach this path.
v2 (2026-11-23): the message is processed in fixed slices through one persistent workspace, so a call allocates only
its output (like NCCL) - v1's per-call buffers raised vLLM's profiled peak and cost ~10k KV tokens.
v2 defaults (KLD vs BF16 on Qwen3.8 R1: bf16 0.009471, v1 cube g32 0.009634, v2 sqrt g16 0.009484, cube g16
0.009588): square-root companding, 16-element groups, fp16 scales (same 1.125 B/element as v1).
Knobs: LAB_AR_INT8 (default on, =0 off), LAB_AR_INT8_MODE (0 linear, 1 square-root, 2 cube-root companding),
LAB_AR_INT8_GROUP (16 / 32 elements per scale), LAB_AR_INT8_SCALE (fp16 / fp32), LAB_AR_INT8_SLICE (elements per
slice), LAB_AR_INT8_MIN (smallest message routed here).
"""

import os

import torch

from vllm.triton_utils import tl, triton


def enabled() -> bool:
    return os.environ.get("LAB_AR_INT8", "1") != "0"


MIN_NUMEL = int(os.environ.get("LAB_AR_INT8_MIN", str(1 << 20)))
MODE = int(os.environ.get("LAB_AR_INT8_MODE", "1"))
GROUP = int(os.environ.get("LAB_AR_INT8_GROUP", "16"))
SCALE_DTYPE = torch.float16 if os.environ.get("LAB_AR_INT8_SCALE", "fp16") == "fp16" else torch.float32
SLICE = int(os.environ.get("LAB_AR_INT8_SLICE", str(2 << 20)))


@triton.jit
def _quant_kernel(x_ptr, q_ptr, s_ptr, n_groups, MODE: tl.constexpr, R: tl.constexpr, G: tl.constexpr):
    g = tl.program_id(0) * R + tl.arange(0, R)
    gm = g < n_groups
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    x = tl.load(x_ptr + offs, mask=gm[:, None], other=0.0).to(tl.float32)
    if MODE == 1:
        a = tl.sqrt(tl.abs(x))
        x = tl.where(x < 0, -a, a)
    if MODE == 2:
        a = tl.abs(x)
        c = tl.where(a > 0, tl.exp(tl.log(tl.maximum(a, 1e-30)) * (1.0 / 3.0)), 0.0)
        x = tl.where(x < 0, -c, c)
    s = tl.max(tl.abs(x), axis=1).to(s_ptr.dtype.element_ty).to(tl.float32)   # quantize with the stored scale
    inv = tl.where(s > 0, 127.0 / s, 0.0)
    q = x * inv[:, None]
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)                        # round half away from zero
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=gm[:, None])
    tl.store(s_ptr + g, s.to(s_ptr.dtype.element_ty), mask=gm)


@triton.jit
def _dequant_sum_kernel(q_ptr, s_ptr, out_ptr, n_groups, q_stride, s_stride, WORLD: tl.constexpr, MODE: tl.constexpr,
                        R: tl.constexpr, G: tl.constexpr):
    g = tl.program_id(0) * R + tl.arange(0, R)
    gm = g < n_groups
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    acc = tl.zeros((R, G), dtype=tl.float32)
    for r in tl.static_range(WORLD):                                            # fixed rank order on every rank
        q = tl.load(q_ptr + r * q_stride + offs, mask=gm[:, None], other=0).to(tl.float32)
        s = tl.load(s_ptr + r * s_stride + g, mask=gm, other=0.0).to(tl.float32)
        v = q * (s * (1.0 / 127.0))[:, None]
        if MODE == 1:
            v = v * tl.abs(v)
        if MODE == 2:
            v = v * v * v
        acc += v
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=gm[:, None])


def applies(x: torch.Tensor, world_size: int) -> bool:
    return (enabled() and world_size >= 2 and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
            and x.numel() >= MIN_NUMEL and x.numel() % GROUP == 0 and x.is_contiguous()
            and not torch.cuda.is_current_stream_capturing())


_WS = {}


def _workspace(device, world, slice_n):
    """One persistent workspace per (device, stream): [own payload | all payloads], payload = int8 slice + scales.
    Calls on one stream are ordered, so slices and calls can reuse it; another stream gets its own."""
    sb = torch.tensor([], dtype=SCALE_DTYPE).element_size()
    pay = slice_n + (slice_n // GROUP) * sb
    key = (device, torch.cuda.current_stream(device).cuda_stream, world, pay)
    ws = _WS.get(key)
    if ws is None:
        ws = _WS[key] = torch.empty((world + 1) * pay, dtype=torch.uint8, device=device)
    return ws, pay, sb


def all_reduce(pynccl_comm, x: torch.Tensor):
    """int8 all-reduce of x over pynccl_comm; returns a new tensor, or None to fall back."""
    if pynccl_comm is None or pynccl_comm.disabled:
        return None
    world = pynccl_comm.world_size
    n = x.numel()
    sl = SLICE - SLICE % GROUP                       # workspace always sized for a full slice (one per stream)
    ws, pay, sb = _workspace(x.device, world, sl)
    xf = x.view(-1)
    out = torch.empty_like(x)
    of = out.view(-1)
    R = max(1, 512 // GROUP)
    for start in range(0, n, sl):
        m = min(sl, n - start)
        ng = m // GROUP
        pm = m + ng * sb                             # this slice's payload: int8 values then scales (sends no padding)
        own = ws[:pm]
        q = own[:m].view(torch.int8)
        s = own[m:pm].view(SCALE_DTYPE)              # m % GROUP == 0 keeps the scales 2/4-byte aligned
        grid = (triton.cdiv(ng, R),)
        _quant_kernel[grid](xf[start:start + m], q, s, ng, MODE=MODE, R=R, G=GROUP)
        allbuf = ws[pay:pay + world * pm]
        pynccl_comm.all_gather(allbuf, own)
        a2 = allbuf.view(world, pm)
        allq = a2[:, :m].view(torch.int8)
        alls = a2[:, m:pm].view(SCALE_DTYPE)
        _dequant_sum_kernel[grid](allq, alls, of[start:start + m], ng, pm, pm // sb, WORLD=world, MODE=MODE,
                                  R=R, G=GROUP)
    return out
