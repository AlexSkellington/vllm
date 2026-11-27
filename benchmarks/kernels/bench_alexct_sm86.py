# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen integer GEMM schedules on SM86, including activation quantization."""

import argparse
import json
import math
import statistics
from functools import partial
from pathlib import Path

import torch
from humming.forward import humming_forward, may_process_input
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.testing.process_input import decode_quantized
from humming.transform import prepare_and_transform_humming_tensors
from humming.tune import get_heuristics_config
from humming.tune.base import DeviceHeuristics
from humming.tune.sm8x import Sm86Heuristics

from vllm.utils import _lab_humming_cache


def reference(config, x, weight, group_size, activation_bits):
    codes, group_scales, token_scales = may_process_input(config, x)
    values = decode_quantized(codes, f"int{activation_bits}")
    if group_size:
        values = values.view(x.shape[0], -1, group_size)
        scaled = values * group_scales[:, : values.size(1)].unsqueeze(-1)
        values = scaled.view_as(x)
    else:
        values *= token_scales.reshape(-1, 1)
    return values @ weight.T


def timing(functions):
    graphs = []
    pool = torch.cuda.graph_pool_handle()
    for function in functions:
        function()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            for _ in range(16):
                function()
        graph.replay()
        graphs.append(graph)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = [[] for _ in graphs]
    for trial in range(10):
        order = range(len(graphs))
        for i in order if trial % 2 == 0 else reversed(order):
            start.record()
            graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) * 1000 / 16)
    return [statistics.median(values) for values in samples]


def schedules(base, m, n, k, activation_bits, multiprocessors):
    yield "stock", base
    yield "two-stages", base | {"num_stages": 2}
    yield "output-grid", base | {"use_stream_k": False}
    warp_k = 512 // activation_bits
    for block_n, warp_n, k_warps in (
        (64, 32, 2),
        (128, 32, 1),
        (128, 64, 2),
        (256, 64, 1),
    ):
        block_k = warp_k * k_warps
        if n % block_n or k % block_k:
            continue
        block_m = 16 if m <= 16 else 32 if m <= 64 else 64
        for stream_k in (False, True):
            tiles = math.ceil(m / block_m) * (n // block_n)
            ctas_per_sm = 2 if block_m <= 32 else 1
            sms = (
                multiprocessors
                if stream_k
                else min(multiprocessors, math.ceil(tiles / ctas_per_sm))
            )
            tuning = base | {
                "block_shape": (block_m, block_n, block_k),
                "warp_shape": (block_m, warp_n, warp_k),
                "num_sms": sms,
                "num_ctas_per_sm": ctas_per_sm,
                "num_stages": 2,
                "use_stream_k": stream_k,
            }
            yield f"n{block_n}-wn{warp_n}-k{block_k}-stream{int(stream_k)}", tuning


def run(n, k, m_values, activation_groups, modes, stock_only, compare_upstream, output):
    _lab_humming_cache.install()
    assert torch.cuda.get_device_capability() == (8, 6)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(141)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    weight = torch.randn((n, k), dtype=torch.bfloat16)
    checked_cols = min(n, 256)
    for weight_bits, activation_bits in modes:
        schema = HummingWeightSchema(
            b_dtype=f"int{weight_bits}", weight_scale_group_size=128
        )
        chunks = [
            schema.quant_tensor(chunk.cuda(), schema, weight.dtype)
            for chunk in weight.split(512)
        ]
        packed = {
            name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]
        }
        del chunks
        reference_weight = schema.dequant_tensors(
            {name: value[:checked_cols] for name, value in packed.items()}
        ).float()
        for activation_group in activation_groups:
            input_schema = HummingInputSchema()
            if activation_bits < 16:
                input_schema = HummingInputSchema(
                    a_dtype=f"int{activation_bits}",
                    input_scale_group_size=activation_group,
                )
            layer_config, transformed = prepare_and_transform_humming_tensors(
                tensors=packed,
                shape_n=n,
                shape_k=k,
                weight_schema=schema,
                input_schema=input_schema,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                torch_dtype=weight.dtype,
                device=packed["weight"].device,
            )
            if activation_bits < 16:
                assert (
                    layer_config.a_dtype.is_integer_type
                    and layer_config.b_dtype.is_integer_type
                )
            apply = partial(
                humming_forward,
                layer_config,
                locks=torch.zeros(2048, dtype=torch.int32, device="cuda"),
                **transformed,
            )
            for m in m_values:
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                rows = min(m, 16)
                if activation_bits == 16:
                    target = x[:rows].float() @ reference_weight.T
                else:
                    target = reference(
                        layer_config,
                        x[:rows],
                        reference_weight,
                        activation_group,
                        activation_bits,
                    )
                output_buffer = torch.empty((m, n), dtype=x.dtype, device=x.device)
                base = get_heuristics_config(layer_config, shape_m=m)
                if compare_upstream:
                    # At v0.1.18 SM86 inherited its schedule directly from this base.
                    upstream = DeviceHeuristics.get_config.__func__(
                        Sm86Heuristics, layer_config, m
                    )
                    for name in ("mma_type", "raster_group_m"):
                        if name in base:
                            upstream[name] = base[name]
                    variants = [("alexct", base), ("upstream", upstream)]
                elif stock_only or activation_bits == 16:
                    variants = [("default", base)]
                else:
                    variants = schedules(base, m, n, k, activation_bits, sms)
                seen = set()
                timed = []
                records = []
                for name, config in variants:
                    key = json.dumps(config, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    row = {
                        "w": weight_bits,
                        "a": activation_bits,
                        "g": activation_group,
                        "m": m,
                        "n": n,
                        "k": k,
                        "schedule": name,
                        "config": config,
                    }
                    records.append(row)
                    try:
                        result = apply(x, tuning_config=config, outputs=output_buffer)
                        error = (
                            result[:rows, :checked_cols].float() - target
                        ).norm() / target.norm()
                        row["relative_rms"] = error.item()
                        if row["relative_rms"] < 0.008:
                            timed.append(
                                (
                                    row,
                                    partial(
                                        apply,
                                        x,
                                        tuning_config=config,
                                        outputs=output_buffer,
                                    ),
                                )
                            )
                    except (AssertionError, RuntimeError, ValueError) as error:
                        row["error"] = str(error).splitlines()[0]
                latencies = timing([function for _, function in timed])
                for (row, _), latency in zip(timed, latencies, strict=True):
                    row["microseconds"] = latency
                for row in records:
                    print(json.dumps(row), file=output, flush=True)
                del timed
            del apply, transformed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 512, 1600])
    parser.add_argument("--groups", nargs="+", type=int, default=[128])
    parser.add_argument("--weights", nargs="+", type=int, default=[4, 5, 6, 8])
    parser.add_argument("--stock-only", action="store_true")
    parser.add_argument("--compare-upstream", action="store_true")
    parser.add_argument("--a16-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested_modes = [
        (b, a) for b in args.weights for a in ((4, 8) if b == 4 else (8,))
    ]
    if args.a16_control:
        requested_modes.extend((b, 16) for b in args.weights)
    with args.output.open("w") as stream:
        run(
            args.n,
            args.k,
            args.rows,
            args.groups,
            requested_modes,
            args.stock_only,
            args.compare_upstream,
            stream,
        )
