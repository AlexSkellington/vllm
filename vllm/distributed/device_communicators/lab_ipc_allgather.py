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
check) runs at worker init. Any setup failure disables the module and the
caller falls back to pynccl all_gather.

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
        self.max_payload = int(os.environ.get("LAB_AR_IPC_MAX_MB", "32")) << 20
        self.own_buf: torch.Tensor | None = None
        self.peer_buf: torch.Tensor | None = None
        self.own_flag: torch.Tensor | None = None
        self.peer_flag: torch.Tensor | None = None
        self.seq = 0

        if world != 2:
            logger.info_once("lab IPC all-gather: world=%d != 2; staying disabled (NCCL path)",
                             world)
            self.disabled = True
            return
        if self.device.type != "cuda":
            logger.info_once("lab IPC all-gather: no cuda device; staying disabled")
            self.disabled = True
            return

        slot = self.max_payload
        self.own_buf = torch.empty(2 * slot + FLAG_BYTES, dtype=torch.uint8,
                                   device=self.device)
        _fn, args = reduce_tensor(self.own_buf)
        gathered: list = [None] * 2
        dist.all_gather_object(gathered, args, group=cpu_group)
        peer_args = list(gathered[1 - rank])
        # Rebuild the peer storage in OUR device context. The pickled
        # storage_device (arg 6) is the peer ordinal; taken as-is it opens
        # the IPC handle in the peer device context. Host copies work from
        # there (canary) but triton's context-relative cuPointerGetAttribute
        # pointer check rejects the pointer when the kernel launches on our
        # own device. _share_cuda_ yields torch's C++ device encoding (a
        # Python pickle is rejected by _new_shared_cuda).
        _dev_bytes = torch.empty(8, device=self.device)._typed_storage()._share_cuda_()[0]
        peer_args[6] = _dev_bytes
        self.peer_buf = rebuild_cuda_tensor(*peer_args)
        self.own_flag = self.own_buf[2 * slot:2 * slot + FLAG_BYTES].view(torch.uint64)
        self.peer_flag = self.peer_buf[2 * slot:2 * slot + FLAG_BYTES].view(torch.uint64)

        if not self._canary():
            logger.warning_once("lab IPC all-gather canary failed (P2P mapping?); "
                                "falling back to NCCL all-gather")
            self.disabled = True
            self.own_buf = None
            self.peer_buf = None
            self.own_flag = None
            self.peer_flag = None
            return
        logger.info_once("lab IPC all-gather ready: TP2, %d MB payload/slot "
                         "(2 slots per GPU)", slot >> 20)

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
            logger.exception("lab IPC all-gather canary raised; falling back to "
                             "NCCL all-gather")
            return False

    def begin(self, payload_bytes: int):
        """Reserve the next round-robin slot. Returns (own_view, peer_view),
        each payload_bytes long. Call publish_and_wait() after the producer
        kernel and before the consumer kernel (all on the current stream)."""
        assert not self.disabled
        assert 0 < payload_bytes <= self.max_payload
        self.seq += 1
        off = ((self.seq - 1) & 1) * self.max_payload
        own = self.own_buf[off:off + payload_bytes]
        peer = self.peer_buf[off:off + payload_bytes]
        return own, peer

    def publish_and_wait(self):
        seq = self.seq
        _publish_kernel[(1,)](self.own_flag, seq)
        _wait_kernel[(1,)](self.peer_flag, seq)
