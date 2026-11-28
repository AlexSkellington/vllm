# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate AS1 INT8 collectives with torchrun and two CUDA devices."""

import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators import lab_int8_allreduce as ar
from vllm.distributed.device_communicators.lab_ipc_allgather import LabIpcAllGather
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def check(ipc, device, calls):
    comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
    n = 2048 * 5120
    rank = dist.get_rank()
    generator = torch.Generator(device=device).manual_seed(267 + rank)
    x = torch.randn(n, dtype=torch.bfloat16, device=device, generator=generator)
    sizes = (1 << 18, 1 << 20, (2 << 20) + 16, 1 << 22, n)
    expected = {}
    for size in sizes:
        view = x[:size]
        ref = ar.all_reduce(comm, view)
        actual = ar.all_reduce(comm, view, ipc)
        assert torch.equal(actual, ref), (size, (actual - ref).abs().max().item())
        expected[size] = ref[:32].clone()
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    for start in range(0, calls, 8):
        outstanding = []
        for i in range(start, min(calls, start + 8)):
            size = sizes[i % len(sizes)]
            with torch.cuda.stream(streams[i % len(streams)]):
                actual = ar.all_reduce(comm, x[:size], ipc)
                outstanding.append((actual, size))
        for stream in streams:
            main.wait_stream(stream)
        for actual, size in outstanding:
            assert torch.equal(actual[:32], expected[size]), (start, size)
    timings = {}
    for name, backend in (("ipc", ipc), ("nccl_int8", None)):
        for _ in range(8):
            ar.all_reduce(comm, x, backend)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(50):
            ar.all_reduce(comm, x, backend)
        torch.cuda.synchronize(device)
        timings[name] = (time.perf_counter() - start) * 20
    if rank == 0:
        print(
            json.dumps({"verified_calls": calls, "elements": n, "ms": timings}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=600)
    parser.add_argument("--mismatched-slots", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=75))
    assert dist.get_world_size() == 2
    os.environ["LAB_AR_IPC_MAX_MB"] = "4" if args.mismatched_slots and rank else "8"
    ipc = LabIpcAllGather(dist.group.WORLD, device, rank, 2)
    if args.mismatched_slots:
        assert ipc.disabled
        if rank == 0:
            print("Both ranks disabled IPC after the slot mismatch", flush=True)
    else:
        assert not ipc.disabled
        check(ipc, device, args.calls)
    dist.barrier()
    del ipc
    dist.barrier()
    dist.destroy_process_group()
