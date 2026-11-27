# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen native INT8 query-key scores on existing integer KV cache layouts."""

import argparse
import json
import math
import os
from functools import partial

import torch

from benchmarks.kernels.bench_alexct_iq4 import time_graphs
from vllm.v1.attention.ops.int4_per_token_head import reshape_and_cache_int4
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


def make_cache(bits, length, block, heads, dim):
    scalars = (length // block, block, heads)
    key = torch.randn((length, heads, dim), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    if bits == 4:
        k_cache = torch.empty((*scalars, dim // 2), device="cuda", dtype=torch.uint8)
        v_cache = torch.empty_like(k_cache)
        k_scale = torch.empty(scalars, device="cuda", dtype=torch.float32)
        v_scale = torch.empty_like(k_scale)
        reshape_and_cache_int4(
            key,
            value,
            k_cache,
            v_cache,
            torch.arange(length, device="cuda"),
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
        )
        return k_cache, v_cache, k_scale, v_scale

    def pack(tensor):
        scale = tensor.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
        codes = (tensor.float() / scale).round().clamp(-127, 127).to(torch.int8)
        return codes.view(*scalars, dim), scale.view(*scalars)

    k_cache, k_scale = pack(key)
    v_cache, v_scale = pack(value)
    return k_cache, v_cache, k_scale, v_scale


def run(length, rows, dim, split_k):
    torch.manual_seed(2026)
    block, kv_heads, query_heads = 32, 2, 12 if split_k else 8
    block_table = torch.arange(length // block, device="cuda", dtype=torch.int32)[
        None, :
    ]
    for bits, mode in (
        (4, KVQuantMode.INT4_PER_TOKEN_HEAD),
        (8, KVQuantMode.INT8_PER_TOKEN_HEAD),
    ):
        if split_k and bits != 4:
            continue
        k, v, k_scale, v_scale = make_cache(bits, length, block, kv_heads, dim)
        for m in rows:
            query = torch.randn(
                (m, query_heads, dim), device="cuda", dtype=torch.bfloat16
            )
            out = torch.empty_like(query)
            split_options = {}
            if split_k:
                segment_shape = (m, query_heads, 16)
                split_options = dict(
                    seq_threshold_3D=64,
                    num_par_softmax_segments=16,
                    softmax_segm_output=torch.empty(
                        (*segment_shape, dim), device="cuda", dtype=torch.float32
                    ),
                    softmax_segm_max=torch.empty(
                        segment_shape, device="cuda", dtype=torch.float32
                    ),
                    softmax_segm_expsum=torch.empty(
                        segment_shape, device="cuda", dtype=torch.float32
                    ),
                )
            call = partial(
                unified_attention,
                query,
                k,
                v,
                out,
                cu_seqlens_q=torch.tensor([0, m], device="cuda", dtype=torch.int32),
                max_seqlen_q=m,
                seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
                max_seqlen_k=length,
                softmax_scale=1 / math.sqrt(dim),
                causal=True,
                window_size=(-1, -1),
                block_table=block_table,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                kv_quant_mode=mode,
                k_scale_cache=k_scale,
                v_scale_cache=v_scale,
                **split_options,
            )

            option = "LAB_INT4_KV_SPLIT_K" if split_k else "LAB_INTEGER_KV_QK"

            def invoke(enabled, function=call, env_var=option):
                os.environ[env_var] = enabled
                function()

            invoke("0")
            reference = out.float()
            invoke("1")
            error = ((out.float() - reference).norm() / reference.norm()).item()
            if (split_k and m == 1) or (not split_k and m < 64):
                assert error == 0
            tolerance = 0.005 if split_k else 0.025
            assert error < tolerance, (bits, m, error)
            control, trial = time_graphs([partial(invoke, "0"), partial(invoke, "1")])
            print(
                json.dumps(
                    dict(
                        bits=bits,
                        q=m,
                        kv=length,
                        head_dim=dim,
                        rms=error,
                        control_us=control,
                        trial_us=trial,
                        option=option,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 128, 512, 2048])
    parser.add_argument("--split-k", action="store_true")
    args = parser.parse_args()
    if args.split_k and not all(1 <= m <= 8 for m in args.rows):
        parser.error("--split-k requires --rows within 1 to 8")
    run(args.length, args.rows, args.dim, args.split_k)
