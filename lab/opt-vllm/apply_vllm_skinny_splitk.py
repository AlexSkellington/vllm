#!/usr/bin/env python3
"""apply_vllm_skinny_splitk.py VENV -- lab-local vLLM patch (2026-11-01): split-K GEMV for skinny unquantized weights.

nsys on AS1 (Qwen3.8-27B INT4, TP2, MTP 2, single-stream decode): the GDN layers' bf16 in_proj_ba (48 x 5120 per GPU;
the INT4 recipe keeps it unquantized) runs through cuBLAS as a 4-block, one-warp-per-block tile kernel, 29.6 us per
call, ~48 calls per step = ~1.4 ms of a ~20 ms decode step, for a 0.5 MB weight. With VLLM_LAB_SKINNY_SPLITK=1,
unquantized linears whose weight has <= 128 output rows and K >= 2048 (K % 256 == 0, no bias) go through a registered
custom op: batches of 2..8 tokens use a Triton split-K GEMV (fp32 accumulation, per-split partials summed in fp32,
deterministic: no atomics), other batch sizes call F.linear exactly as before (isolated test, 48 x 5120 bf16:
cuBLAS 27.8-27.9 us at 2-3 tokens, 22.9 us at 4-8; split-K 6.4-7.2 us; cuBLAS wins at 1 and 16 tokens; error vs
fp32 identical). Unset = upstream behaviour (the op is never
reached). The flag is also registered in vllm/envs.py: vLLM keys its torch.compile cache on every KNOWN env var, so
flipping the flag recompiles instead of reusing a graph traced with the other setting. Idempotent per file; refuses
to write if an anchor moved.
"""
import shutil, sys
from pathlib import Path

venv = Path(sys.argv[1])
MARK = "lab patch: VLLM_LAB_SKINNY_SPLITK"
e = next(venv.glob("lib/python3*/site-packages/vllm/envs.py"))
es = e.read_text()
if MARK not in es:
    ea = '    "VLLM_DISABLE_COMPILE_CACHE": disable_compile_cache,\n'
    assert es.count(ea) == 1, "envs.py anchor moved"
    if not e.with_name("envs.py.bak-pre-lab-skinny").exists():
        shutil.copy2(e, e.with_name("envs.py.bak-pre-lab-skinny"))
    e.write_text(es.replace(ea, ea + '    # %s (registered so the compile-cache key includes it)\n'
                                     '    "VLLM_LAB_SKINNY_SPLITK": lambda: os.getenv("VLLM_LAB_SKINNY_SPLITK", "0"),\n' % MARK))
    print("registered in", e)
f = next(venv.glob("lib/python3*/site-packages/vllm/model_executor/layers/utils.py"))
s = f.read_text()
def _cuda_hook(f):
    # 2026-11-09: newer vLLM (dev452+) routes CUDA unquantized linears through cuda_unquantized_gemm, which never calls
    # default_unquantized_gemm, so the split-K op must also be hooked there. The production venv had this hook added by
    # hand; without it a fresh venv (dev1016) silently fell back to cuBLAS (~+1 ms/token on Qwen3.8 decode). Idempotent.
    t = f.read_text()
    head = """def cuda_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
"""
    if t.count(head) != 1:
        print("no cuda_unquantized_gemm in this vLLM: default hook only"); return
    i = t.index(head) + len(head)
    body = t[i:t.find("\ndef ", i)]
    if "torch.ops.vllm.lab_skinny_linear(x, weight)" in body:
        print("cuda hook already applied:", f); return
    hook = """    if (_LAB_SKINNY and bias is None and weight.dim() == 2 and weight.shape[0] <= 128 and weight.shape[1] >= 2048
            and weight.shape[1] % _LAB_SKINNY_BK == 0 and x.dtype in (torch.bfloat16, torch.float16)
            and weight.dtype == x.dtype and x.is_cuda and not getattr(layer, "use_skinny_gemm", False)):
        return torch.ops.vllm.lab_skinny_linear(x, weight)  # lab patch: VLLM_LAB_SKINNY_SPLITK
"""
    f.write_text(t[:i] + hook + t[i:])
    print("cuda hook applied:", f)


if MARK in s:
    print("default hook already applied:", f); _cuda_hook(f); sys.exit(0)
anchor = """def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)
"""
assert s.count(anchor) == 1, "default_unquantized_gemm anchor moved"
new = '''# lab patch: VLLM_LAB_SKINNY_SPLITK (AS1, 2026-11-01) -- see /opt/vllm/apply_vllm_skinny_splitk.py
import os as _lab_os

_LAB_SKINNY = _lab_os.environ.get("VLLM_LAB_SKINNY_SPLITK", "0") == "1"
_LAB_SKINNY_MIN_M, _LAB_SKINNY_MAX_M = 2, 8
_LAB_SKINNY_BK = 256

if _LAB_SKINNY:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _lab_skinny_kernel(x_ptr, w_ptr, p_ptr, M, N, sxm, swn, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        m = tl.program_id(2)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_s * BLOCK_K + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + m * sxm + offs_k).to(tl.float32)
        wv = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :], mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc = tl.sum(wv * xv[None, :], axis=1)
        tl.store(p_ptr + (pid_s * M + m) * N + offs_n, acc, mask=offs_n < N)


def _lab_skinny_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if not (_LAB_SKINNY_MIN_M <= m <= _LAB_SKINNY_MAX_M) or not x2.is_contiguous() or not weight.is_contiguous():
        return torch.nn.functional.linear(x, weight)
    splits = k // _LAB_SKINNY_BK
    part = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(n, 16), splits, m)
    _lab_skinny_kernel[grid](x2, weight, part, m, n, x2.stride(0), weight.stride(0), BLOCK_N=16, BLOCK_K=_LAB_SKINNY_BK)
    return part.sum(0).to(x.dtype).reshape(*x.shape[:-1], n)


def _lab_skinny_linear_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


if _LAB_SKINNY:
    direct_register_custom_op(op_name="lab_skinny_linear", op_func=_lab_skinny_linear, fake_impl=_lab_skinny_linear_fake)


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    if (_LAB_SKINNY and bias is None and weight.dim() == 2 and weight.shape[0] <= 128 and weight.shape[1] >= 2048
            and weight.shape[1] % _LAB_SKINNY_BK == 0 and x.dtype in (torch.bfloat16, torch.float16)
            and weight.dtype == x.dtype and x.is_cuda):
        return torch.ops.vllm.lab_skinny_linear(x, weight)
    return torch.nn.functional.linear(x, weight, bias)
'''
bak = f.with_name(f.name + ".bak-pre-lab-skinny")
if not bak.exists():
    shutil.copy2(f, bak)
f.write_text(s.replace(anchor, new))
print("applied:", f, "(backup", bak.name + ")")
_cuda_hook(f)
