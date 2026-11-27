# SPDX-License-Identifier: Apache-2.0
"""Lab (AS1) Humming hooks, installed from the vllm.utils.humming facade before Humming's first use.

1. Stable JIT cache keys (2026-11-21). Humming keys each compiled kernel (~/.humming/cache/<md5>) on a signature with
   the compiler flags (incl. ``-I<venv>/.../humming/include``) and {absolute .cuh path: mtime}. Both change with every
   new virtualenv (the AS1 deploy watcher builds one per fork merge), so every deploy recompiled every kernel. Fix: hash
   header *contents* (relative paths) and replace install paths by fixed tokens before hashing. Any change to a header,
   the kernel code, the flags or NVRTC/NVCC still gives a new key. Opt out: LAB_HUMMING_STABLE_CACHE=0.
2. Codebook weights (2026-11-22). vllm/lab_humming_include/ holds override copies of humming/datatype/{dtypes,
   dequant,dequant_single}.cuh adding Iq4nlType (exact llama.cpp IQ4_NL levels) and Iq4nl7Type (levels / 2, 7-bit),
   both UInt4 layout with table dequant (__byte_perm, llama.cpp get_int_from_table_16 pattern); prepended to the NVRTC
   include path; Python dtypes "iq4nl" / "iq4nl7" registered. Compressed-tensors checkpoints mark such layers with
   weights.observer "lab_iq4nl" / "lab_iq4nl7" (tools/as1/q_rtn.py rule bits "lut" / "lut7").
"""

import dataclasses
import functools
import glob
import hashlib
import os
import types

_installed = False
LAB_INC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab_humming_include")


def _register_codebook_dtypes() -> None:
    """Python dtypes for the lab codebook weight types (C++ classes in lab_humming_include/humming/datatype/dtypes.cuh):
    "iq4nl" = exact llama.cpp IQ4_NL levels (Iq4nlType), "iq4nl7" = levels / 2 rounded to 7 bits (Iq4nl7Type)."""
    import humming.dtypes as hd

    if getattr(hd, "iq4nl", None) is not None:
        return

    def make(name, cpp, uid):
        @dataclasses.dataclass(kw_only=True, repr=False)
        class LutType(hd.IntegerType):
            __hash__ = hd.DataType.__hash__

            def to_str(self):
                return name

            def to_cpp_str(self):
                return cpp

            def id(self):
                return uid

        return LutType(is_signed=False, num_bits=4)

    books = {"iq4nl": make("iq4nl", "Iq4nlType", int(3 * 1e7 + 4 * 1e4 + 8)),
             "iq4nl7": make("iq4nl7", "Iq4nl7Type", int(3 * 1e7 + 4 * 1e4 + 7))}
    for k, v in books.items():
        setattr(hd, k, v)
    for cls in (hd.DataType, hd.IntegerType):
        orig = cls.__dict__["from_str"].__func__

        def from_str(c, s, _orig=orig):
            if isinstance(s, str) and s.lower() in books:
                return books[s.lower()]
            return _orig(c, s)

        cls.from_str = classmethod(from_str)


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    try:
        import humming.jit.compiler as hc
    except Exception:  # humming missing or layout changed: keep upstream behaviour
        return
    comp = hc.Compiler
    inc = os.path.abspath(comp.humming_include_dir())
    has_lab = os.path.isdir(LAB_INC)
    if has_lab:
        comp.include_dirs = staticmethod(lambda: [LAB_INC, inc])
        _register_codebook_dtypes()
        from vllm.utils import _lab_humming_sm86

        _lab_humming_sm86.install()
    if os.environ.get("LAB_HUMMING_STABLE_CACHE", "1") == "0":
        return

    @functools.lru_cache(maxsize=1)
    def _content_key() -> str:
        md5 = hashlib.md5()
        for tag, root in (("humming", inc), ("lab", LAB_INC if has_lab else None)):
            if root is None:
                continue
            for fn in sorted(glob.glob(os.path.join(root, "**", "*.cuh"), recursive=True)):
                md5.update((tag + ":" + os.path.relpath(fn, root)).encode())
                with open(fn, "rb") as f:
                    md5.update(f.read())
        return "cuh-content:" + md5.hexdigest()

    comp.cuh_last_update_time = staticmethod(_content_key)

    real = hc.jit_utils

    def _hash_to_hex(s: str) -> str:
        return real.hash_to_hex(s.replace(LAB_INC, "<lab-include>").replace(inc, "<humming-include>"))

    proxy = types.ModuleType(real.__name__)
    proxy.__dict__.update(real.__dict__)
    proxy.hash_to_hex = _hash_to_hex
    hc.jit_utils = proxy  # only compiler.py's compile() sees the normalising hash
