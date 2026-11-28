# SPDX-License-Identifier: Apache-2.0
"""Lab INT8 tensor-parallel all-reduce for large, eager messages.

Each rank quantizes its partial sum into persistent slices. Every rank then
dequantizes and adds the payloads in the same order. On AS1, TP2 uses CUDA IPC
to exchange those payloads. Compressed NCCL gathering is also exposed for
operator comparisons and other TP sizes.

The default uses square-root companding, 16-element groups and FP16 scales.
LAB_AR_INT8=0 disables the path. LAB_AR_INT8_MODE selects linear, square-root
or cube-root companding (0, 1 or 2). LAB_AR_INT8_GROUP and LAB_AR_INT8_SCALE
set scale granularity and dtype. LAB_AR_INT8_SLICE and LAB_AR_INT8_MIN set
the slice size and activation threshold in elements.
"""

import os

import torch

from vllm.triton_utils import tl, triton


def enabled() -> bool:
    return os.environ.get("LAB_AR_INT8", "1") != "0"


MIN_NUMEL = int(os.environ.get("LAB_AR_INT8_MIN", str(1 << 20)))
MODE = int(os.environ.get("LAB_AR_INT8_MODE", "1"))
GROUP = int(os.environ.get("LAB_AR_INT8_GROUP", "16"))
SCALE_DTYPE = (
    torch.float16
    if os.environ.get("LAB_AR_INT8_SCALE", "fp16") == "fp16"
    else torch.float32
)
SLICE = int(os.environ.get("LAB_AR_INT8_SLICE", str(2 << 20)))
SCALE_ELEM_SIZE = torch.tensor([], dtype=SCALE_DTYPE).element_size()


@triton.jit
def _quant_kernel(
    x_ptr, q_ptr, s_ptr, n_groups, MODE: tl.constexpr, R: tl.constexpr, G: tl.constexpr
):
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
    s = (
        tl.max(tl.abs(x), axis=1).to(s_ptr.dtype.element_ty).to(tl.float32)
    )  # quantize with the stored scale
    inv = tl.where(s > 0, 127.0 / s, 0.0)
    q = x * inv[:, None]
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)  # round half away from zero
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + offs, q, mask=gm[:, None])
    tl.store(s_ptr + g, s.to(s_ptr.dtype.element_ty), mask=gm)


@triton.jit
def _dequant_sum_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    n_groups,
    q_stride,
    s_stride,
    WORLD: tl.constexpr,
    MODE: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
):
    g = tl.program_id(0) * R + tl.arange(0, R)
    gm = g < n_groups
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    acc = tl.zeros((R, G), dtype=tl.float32)
    for r in tl.static_range(WORLD):  # fixed rank order on every rank
        q = tl.load(q_ptr + r * q_stride + offs, mask=gm[:, None], other=0).to(
            tl.float32
        )
        s = tl.load(s_ptr + r * s_stride + g, mask=gm, other=0.0).to(tl.float32)
        v = q * (s * (1.0 / 127.0))[:, None]
        if MODE == 1:
            v = v * tl.abs(v)
        if MODE == 2:
            v = v * v * v
        acc += v
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=gm[:, None])


@triton.jit
def _dequant_sum2_kernel(
    q0_ptr,
    s0_ptr,
    q1_ptr,
    s1_ptr,
    out_ptr,
    n_groups,
    MODE: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
):
    # TP2 variant for the IPC exchange: payload0 from rank 0, payload1 from rank 1,
    # same op order as _dequant_sum_kernel so results are bit-identical for
    # identical payloads.
    g = tl.program_id(0) * R + tl.arange(0, R)
    gm = g < n_groups
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    q = tl.load(q0_ptr + offs, mask=gm[:, None], other=0).to(tl.float32)
    s = tl.load(s0_ptr + g, mask=gm, other=0.0).to(tl.float32)
    v = q * (s * (1.0 / 127.0))[:, None]
    if MODE == 1:
        v = v * tl.abs(v)
    if MODE == 2:
        v = v * v * v
    acc = v
    q = tl.load(q1_ptr + offs, mask=gm[:, None], other=0).to(tl.float32)
    s = tl.load(s1_ptr + g, mask=gm, other=0.0).to(tl.float32)
    v = q * (s * (1.0 / 127.0))[:, None]
    if MODE == 1:
        v = v * tl.abs(v)
    if MODE == 2:
        v = v * v * v
    acc += v
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=gm[:, None])


def applies(x: torch.Tensor, world_size: int) -> bool:
    return (
        enabled()
        and world_size >= 2
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.numel() >= MIN_NUMEL
        and x.numel() % GROUP == 0
        and x.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    )


_WS = {}


def _workspace(device, world, slice_n):
    """Allocate persistent local and gathered payloads for one device stream."""
    sb = torch.tensor([], dtype=SCALE_DTYPE).element_size()
    pay = slice_n + (slice_n // GROUP) * sb
    key = (device, torch.cuda.current_stream(device).cuda_stream, world, pay)
    ws = _WS.get(key)
    if ws is None:
        ws = _WS[key] = torch.empty((world + 1) * pay, dtype=torch.uint8, device=device)
    return ws, pay, sb


def all_reduce(pynccl_comm, x: torch.Tensor, ipc=None):
    # int8 all-reduce of x. Payload exchange: lab IPC all-gather (ipc, v3) when
    # available, else the NCCL all-gather through pynccl_comm (v1/v2). Returns a
    # new tensor, or None to fall back to the default backend.
    if pynccl_comm is None or pynccl_comm.disabled:
        return None
    world = pynccl_comm.world_size
    n = x.numel()
    xf = x.view(-1)
    out = torch.empty_like(x)
    of = out.view(-1)
    R = max(1, 512 // GROUP)
    sb = SCALE_ELEM_SIZE
    if ipc is not None and not getattr(ipc, "disabled", True) and world == 2:
        sl = min(SLICE // GROUP, ipc.max_payload // (GROUP + sb)) * GROUP
        if sl <= 0:
            return None
        stream = ipc.order_stream()
        for start in range(0, n, sl):
            m = min(sl, n - start)
            ng = m // GROUP
            pm = m + ng * sb
            own, peer = ipc.begin(pm)
            q = own[:m].view(torch.int8)
            sc = own[m:pm].view(SCALE_DTYPE)
            grid = (triton.cdiv(ng, R),)
            _quant_kernel[grid](
                xf[start : start + m], q, sc, ng, MODE=MODE, R=R, G=GROUP
            )
            ipc.publish_and_wait()
            if ipc.rank == 0:
                q0, s0, q1, s1 = (
                    q,
                    sc,
                    peer[:m].view(torch.int8),
                    peer[m:pm].view(SCALE_DTYPE),
                )
            else:
                q0, s0, q1, s1 = (
                    peer[:m].view(torch.int8),
                    peer[m:pm].view(SCALE_DTYPE),
                    q,
                    sc,
                )
            _dequant_sum2_kernel[grid](
                q0, s0, q1, s1, of[start : start + m], ng, MODE=MODE, R=R, G=GROUP
            )
        ipc.record_consumption(stream)
        return out
    sl = SLICE - SLICE % GROUP
    ws, pay, sb = _workspace(x.device, world, sl)
    for start in range(0, n, sl):
        m = min(sl, n - start)
        ng = m // GROUP
        pm = (
            m + ng * sb
        )  # this slice's payload: int8 values then scales (sends no padding)
        own = ws[:pm]
        q = own[:m].view(torch.int8)
        s = own[m:pm].view(
            SCALE_DTYPE
        )  # m % GROUP == 0 keeps the scales 2/4-byte aligned
        grid = (triton.cdiv(ng, R),)
        _quant_kernel[grid](xf[start : start + m], q, s, ng, MODE=MODE, R=R, G=GROUP)
        allbuf = ws[pay : pay + world * pm]
        pynccl_comm.all_gather(allbuf, own)
        a2 = allbuf.view(world, pm)
        allq = a2[:, :m].view(torch.int8)
        alls = a2[:, m:pm].view(SCALE_DTYPE)
        _dequant_sum_kernel[grid](
            allq,
            alls,
            of[start : start + m],
            ng,
            pm,
            pm // sb,
            WORLD=world,
            MODE=MODE,
            R=R,
            G=GROUP,
        )
    return out
