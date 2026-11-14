#!/usr/bin/env python3
"""vllm-kvcache-trim (AS1 CT100, 2026-11-10, Claude): least-recently-used eviction for vLLM's filesystem KV tier.

vLLM's TieringOffloadingSpec "fs" tier writes one content-addressed file per chunk and finds blocks by file existence,
so it survives restarts, but it never deletes anything. This keeps it inside a model-agnostic budget:
  budget = 90% of the filesystem / dataset that holds the cache (a ZFS quota counts as its size), and >= 5% kept free;
  when over, files with the oldest access time go first (the dataset runs atime=on, relatime=off, so every read of a
  chunk refreshes it) until usage is <= 85% (hysteresis: no trimming on every run).
Deleting a chunk is safe: a lookup for it is a miss and vLLM recomputes from there. vLLM's own config files (*.json)
are never touched. Usage: vllm-kvcache-trim [DIR] (default /var/cache/vllm-kvcache); prints one summary line."""
import os, sys, time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/var/cache/vllm-kvcache"
HIGH, LOW, FREE_MIN = 0.90, 0.85, 0.05


def walk(d):
    with os.scandir(d) as it:
        for e in it:
            if e.is_dir(follow_symlinks=False):
                yield from walk(e.path)
            elif e.is_file(follow_symlinks=False) and not e.name.endswith(".json") and not e.name.startswith(".tmp"):
                st = e.stat(follow_symlinks=False)
                yield st.st_atime, st.st_blocks * 512, e.path


t0 = time.time()
vfs = os.statvfs(ROOT)
cap = vfs.f_blocks * vfs.f_frsize
free = vfs.f_bavail * vfs.f_frsize
files = list(walk(ROOT))
used = sum(f[1] for f in files)
removed = freed = 0
if cap and (used > HIGH * cap or free < FREE_MIN * cap):
    files.sort()  # oldest access first
    for atime, size, path in files:
        if used <= LOW * cap and free >= (FREE_MIN + 0.02) * cap:
            break
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        used -= size; free += size; removed += 1; freed += size
print("vllm-kvcache-trim: %d files, %.1f GiB of %.1f GiB budget (cap %.1f GiB); removed %d (%.1f GiB) in %.1f s" % (
    len(files), used / 2**30, HIGH * cap / 2**30, cap / 2**30, removed, freed / 2**30, time.time() - t0))
