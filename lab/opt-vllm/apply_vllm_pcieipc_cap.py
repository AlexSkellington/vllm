#!/usr/bin/env python3
"""apply_vllm_pcieipc_cap.py VENV -- lab-local vLLM patch (2026-10-31): token cap for the FlashInfer PCIe IPC
all-reduce (VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1).

Measured on AS1 (2x RTX 3090, PCIe, TP2, MTP 2): PCIe IPC wins for the small all-reduces of 1-4 streams
(single-stream decode +2.9%, 2 streams +4.7%) but loses from 8 streams up (-2..-3%). Upstream sends every
graph-captured batch (up to 512 tokens) through it. VLLM_PCIE_IPC_MAX_TOKENS=N limits it to batches of at most
N tokens (and sizes/tunes the workspace only for those); larger batches fall through to CUSTOM -> PYNCCL exactly
as without the backend. Unset = upstream behaviour. Idempotent; refuses to write if an anchor moved.
"""
import shutil, sys
from pathlib import Path

venv = Path(sys.argv[1])
f = next(venv.glob("lib/python3*/site-packages/vllm/distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py"))
s = f.read_text()
MARK = "lab patch: VLLM_PCIE_IPC_MAX_TOKENS"
if MARK in s:
    print("already applied:", f); sys.exit(0)

edits = [
    ("        batches = tuple(sorted({int(size) for size in capture_sizes if size > 0}))\n",
     "        batches = tuple(sorted({int(size) for size in capture_sizes if size > 0}))\n"
     "        # lab patch: VLLM_PCIE_IPC_MAX_TOKENS -- prepare/tune only the batches this backend will serve.\n"
     "        if self.max_tokens:\n"
     "            batches = tuple(b for b in batches if b <= self.max_tokens)\n"),
    ("            and inp.shape[1] == self.hidden_dim\n",
     "            and inp.shape[1] == self.hidden_dim\n"
     "            and (not self.max_tokens or inp.shape[0] <= self.max_tokens)\n"),
    ("        self.workspace: Any | None = None\n",
     "        self.workspace: Any | None = None\n"
     "        # lab patch: VLLM_PCIE_IPC_MAX_TOKENS (0/unset = upstream: every captured batch size).\n"
     "        self.max_tokens = int(os.environ.get(\"VLLM_PCIE_IPC_MAX_TOKENS\", \"0\") or 0)\n"),
    ("from collections.abc import Sequence\n", "import os\nfrom collections.abc import Sequence\n"),
]
for old, new in edits:
    assert s.count(old) == 1, ("anchor moved", old)
for old, new in edits:
    s = s.replace(old, new)
compile(s, str(f), "exec")
bak = f.with_name(f.name + ".bak-pre-lab-cap")
if not bak.exists():
    shutil.copy2(f, bak)
f.write_text(s)
print("applied:", f, "backup:", bak)
