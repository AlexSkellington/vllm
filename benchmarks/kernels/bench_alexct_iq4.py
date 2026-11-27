# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare IQ4 codebook kernels with BF16, token INT8 and group-32 INT8 inputs."""

import argparse
import json
import statistics
from functools import partial

import torch
from humming.forward import humming_forward
from humming.schema.humming import HummingInputSchema, HummingWeightSchema
from humming.transform import prepare_and_transform_humming_tensors

from vllm.utils import _lab_humming_cache


def time_graphs(functions):
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    for function in functions:
        function()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            for _ in range(16):
                function()
        graph.replay()
        graphs.append(graph)
    samples = [[] for _ in graphs]
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for repeat in range(10):
        order = range(len(graphs))
        for index in order if repeat % 2 == 0 else reversed(order):
            start.record()
            graphs[index].replay()
            stop.record()
            stop.synchronize()
            samples[index].append(start.elapsed_time(stop) * 1000 / 16)
    return [statistics.median(item) for item in samples]


def run(n, k, rows):
    _lab_humming_cache.install()
    torch.manual_seed(2026)
    dtype = torch.bfloat16
    schema = HummingWeightSchema(b_dtype="iq4nl", weight_scale_group_size=32)
    packed = {
        "weight": torch.randint(
            -(2**31), 2**31, (n, k // 8), device="cuda", dtype=torch.int32
        ),
        "weight_scale": torch.randn((n, k // 32), device="cuda", dtype=dtype) / 64,
    }
    inputs = (
        ("bf16", HummingInputSchema()),
        ("int8_token", HummingInputSchema(a_dtype="int8")),
        ("int8_group32", HummingInputSchema(a_dtype="int8", input_scale_group_size=32)),
    )
    arms = {}
    for name, input_schema in inputs:
        if not input_schema.is_compatible_with(schema, dtype):
            raise ValueError(f"Unsupported IQ4 benchmarking mode {name}")
        config, tensors = prepare_and_transform_humming_tensors(
            packed, n, k, schema, input_schema, torch_dtype=dtype, device="cuda:0"
        )
        arms[name] = partial(
            humming_forward,
            config,
            **tensors,
            locks=torch.zeros(2048, dtype=torch.int32, device="cuda"),
        )
    for m in rows:
        x = torch.randn((m, k), device="cuda", dtype=dtype)
        output = torch.empty((m, n), device="cuda", dtype=dtype)
        control = arms["bf16"](x).float()
        calls = [partial(function, x, outputs=output) for function in arms.values()]
        errors = []
        for call in calls:
            candidate = call().float()
            errors.append(((candidate - control).norm() / control.norm()).item())
        for name, error, time in zip(arms, errors, time_graphs(calls), strict=True):
            print(
                json.dumps(dict(n=n, k=k, m=m, mode=name, rms=error, us=time)),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 512, 1600, 4096])
    args = parser.parse_args()
    run(args.n, args.k, args.rows)
