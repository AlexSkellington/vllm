# SPDX-License-Identifier: Apache-2.0
"""Lab (AS1, 2026-11-21): make Humming's JIT kernel cache survive venv changes.

Humming keys each compiled kernel (~/.humming/cache/<md5>) on a signature that contains
  * the compiler flags, incl. ``-I<venv>/site-packages/humming/include``, and
  * ``Compiler.cuh_last_update_time()`` = JSON of {absolute .cuh path: mtime}.
Both change with every new virtualenv (the AS1 deploy watcher builds one per fork merge), so every deploy recompiled
every kernel (~130 per model on 2x RTX 3090, part of a 5-min engine start) although nothing in the kernels changed.

Fix (model-agnostic, keeps correctness): hash the header *contents* (relative paths) instead of path+mtime, and
replace the install path of the include dir by a fixed token before hashing. Any change to a header, to the
generated kernel code, to the flags or to NVRTC/NVCC still yields a new key. Opt out: LAB_HUMMING_STABLE_CACHE=0.
"""

import functools
import glob
import hashlib
import os
import types

_installed = False


def install() -> None:
    global _installed
    if _installed or os.environ.get("LAB_HUMMING_STABLE_CACHE", "1") == "0":
        return
    _installed = True
    try:
        import humming.jit.compiler as hc
    except Exception:  # humming missing or layout changed: keep upstream behaviour
        return
    comp = hc.Compiler
    inc = os.path.abspath(comp.humming_include_dir())

    @functools.lru_cache(maxsize=1)
    def _content_key() -> str:
        md5 = hashlib.md5()
        for fn in sorted(glob.glob(os.path.join(inc, "**", "*.cuh"), recursive=True)):
            md5.update(os.path.relpath(fn, inc).encode())
            with open(fn, "rb") as f:
                md5.update(f.read())
        return "cuh-content:" + md5.hexdigest()

    comp.cuh_last_update_time = staticmethod(_content_key)

    real = hc.jit_utils

    def _hash_to_hex(s: str) -> str:
        return real.hash_to_hex(s.replace(inc, "<humming-include>"))

    proxy = types.ModuleType(real.__name__)
    proxy.__dict__.update(real.__dict__)
    proxy.hash_to_hex = _hash_to_hex
    hc.jit_utils = proxy  # only compiler.py's compile() sees the normalising hash
