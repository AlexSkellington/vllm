# SPDX-License-Identifier: Apache-2.0
"""Lab (AS1) CUDA-IPC payload all-gather for the int8 all-reduce (2026-11-25).

Why: the NCCL all-gather used by lab_int8_allreduce v1/v2 deadlocked on
prefill-sized mixed prefill+decode steps (2026-11-23, 600-burst hang at burst
~80) - exactly the cases where vLLM would otherwise use its custom IPC
all-reduce. This module moves the int8 payload over the same plain P2P
machinery (CUDA-IPC-mapped persistent buffers + a device-side release/acquire
flag spin), which the sync-free FlashInfer decode plan already exercised
safely on this PCIe-only TP2 box.

TP2 only (other world sizes keep the NCCL path). Eager-only: the int8 AR path
is never used under graph capture, and setup (one gloo object exchange of
torch CUDA-IPC handles plus a functional canary that doubles as the P2P
check) runs at worker init. Both ranks must pass setup before the
communicator can use the exchange.

Protocol (one in-flight message per rank, round-robin slot = seq & 1):
  1. quantize the slice payload into own_buf[slot]
  2. publish: atomic_xchg(own_flag, seq, release, sys)
  3. wait:    spin on peer_flag until >= seq (acquire, sys)
  4. dequantize-sum payload0 + payload1 in fixed rank order into out - the
     same op order as the NCCL path with the same payloads, so results are
     bit-identical.
Monotonic u64 seq prevents ABA across the two round-robin slots.

Env: LAB_AR_IPC (auto|1|0; 0 never sets up), LAB_AR_IPC_MAX_MB (payload
bytes per slot, default 32; VRAM cost ~= 2 x this per GPU).
"""

import os
import time

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

FLAG_BYTES = 8


@triton.jit
def _publish_kernel(flag_ptr, seq):
    tl.atomic_xchg(flag_ptr, seq, sem="release", scope="sys")


@triton.jit
def _wait_kernel(flag_ptr, seq):
    # Device-side barrier: unbounded by design, like a NCCL collective. Both
    # ranks issue the same sequence of collectives in lockstep; a desync here
    # means the engine is already broken.
    while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="sys") < seq:
        pass


class LabIpcAllGather:
    """Persistent P2P-mapped buffer pair + flag protocol for TP2 payload
    exchange. All methods are no-ops / asserts when disabled."""

    def __init__(self, cpu_group, device, rank: int, world: int):
        self.disabled = False
        self.rank = rank
        self.world = world
        self.device = torch.device(device)
        self.max_payload = 0
        self.own_buf: torch.Tensor | None = None
        self.peer_buf: torch.Tensor | None = None
        self.own_flag: torch.Tensor | None = None
        self.peer_flag: torch.Tensor | None = None
        self._last_stream: torch.cuda.Stream | None = None
        self._consumed = None
        self.seq = 0

        if world != 2:
            logger.info_once(
                "lab IPC all-gather: world=%d != 2; staying disabled (NCCL path)", world
            )
            self.disabled = True
            return
        local_handle = None
        try:
            self.max_payload = int(os.environ.get("LAB_AR_IPC_MAX_MB", "32")) << 20
            if self.device.type == "cuda" and self.max_payload > 0:
                slot = self.max_payload
                self.own_buf = torch.empty(
                    2 * slot + FLAG_BYTES, dtype=torch.uint8, device=self.device
                )
                self.own_flag = self.own_buf[-FLAG_BYTES:].view(torch.uint64)
                self.own_flag.zero_()
                torch.cuda.current_stream(self.device).synchronize()
                _, args = reduce_tensor(self.own_buf)
                local_handle = (slot, args)
        except (TypeError, ValueError, RuntimeError):
            logger.exception("lab IPC all-gather could not export local storage")
        gathered: list = [None] * 2
        dist.all_gather_object(gathered, local_handle, group=cpu_group)
        if any(handle is None or handle[0] != self.max_payload for handle in gathered):
            self._disable()
            return

        mapped = False
        try:
            peer_args = list(gathered[1 - rank][1])
            # CUDA IPC storage must be opened in the receiving device context.
            # Reuse the local handle's encoded device for rebuild_cuda_tensor.
            peer_args[6] = gathered[rank][1][6]
            with torch.cuda.device(self.device):
                self.peer_buf = rebuild_cuda_tensor(*peer_args)
                self.peer_flag = self.peer_buf[-FLAG_BYTES:].view(torch.uint64)
                self._consumed = torch.cuda.Event()
            mapped = True
        except (TypeError, ValueError, RuntimeError):
            logger.exception("lab IPC all-gather could not map peer storage")
        if not self._both_ready(cpu_group, mapped):
            self._disable()
            return
        if not self._both_ready(cpu_group, self._canary()):
            self._disable()
            return
        logger.info_once(
            "lab IPC all-gather ready: TP2, %d MB payload/slot (2 slots per GPU)",
            self.max_payload >> 20,
        )

    @staticmethod
    def _both_ready(cpu_group, local: bool) -> bool:
        ready = torch.tensor(int(local), dtype=torch.int32, device="cpu")
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=cpu_group)
        return bool(ready.item())

    def _disable(self):
        logger.warning_once("lab IPC all-gather disabled on both ranks")
        self.disabled = True
        self.own_buf = self.peer_buf = None
        self.own_flag = self.peer_flag = None

    def _canary(self) -> bool:
        """Symmetric round-trip with a bounded host wait; this IS the P2P
        check. Rank 0 expects the rank-1 marker and vice versa."""
        try:
            mine = 0xA0 | (self.rank & 1)
            expect = 0xA0 | ((1 - self.rank) & 1)
            self.seq = 1
            self.own_buf[0] = mine
            self.own_flag.fill_(1)
            torch.cuda.synchronize(self.device)
            t0 = time.time()
            while self.peer_flag.item() < 1:
                if time.time() - t0 > 30:
                    return False
                time.sleep(0.005)
            torch.cuda.synchronize(self.device)
            return self.peer_buf[0].item() == expect
        except Exception:
            logger.exception("lab IPC all-gather canary raised")
            return False

    def order_stream(self) -> torch.cuda.Stream:
        stream = torch.cuda.current_stream(self.device)
        if self._last_stream is not None and stream != self._last_stream:
            stream.wait_event(self._consumed)
        return stream

    def record_consumption(self, stream: torch.cuda.Stream) -> None:
        self._consumed.record(stream)
        self._last_stream = stream

    def begin(self, payload_bytes: int):
        """Reserve the next round-robin slot. Returns (own_view, peer_view),
        each payload_bytes long. Call publish_and_wait() after the producer
        kernel and before the consumer kernel (all on the current stream)."""
        assert not self.disabled
        assert 0 < payload_bytes <= self.max_payload
        self.seq += 1
        off = ((self.seq - 1) & 1) * self.max_payload
        own = self.own_buf[off : off + payload_bytes]
        peer = self.peer_buf[off : off + payload_bytes]
        return own, peer

    def publish_and_wait(self):
        seq = self.seq
        _publish_kernel[(1,)](self.own_flag, seq)
        _wait_kernel[(1,)](self.peer_flag, seq)
