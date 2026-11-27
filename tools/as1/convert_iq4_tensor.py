# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export one GGUF IQ4_NL or IQ4_XS matrix for the lab's Humming loader."""

import argparse
import json
from pathlib import Path

import numpy as np


def convert_iq4(data: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major int32 codes and FP32 scales for every 32 weights."""
    if data.ndim != 2 or data.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional array of GGUF bytes")
    if kind == "IQ4_NL":
        blocks = data.reshape(data.shape[0], -1, 18)
        scales = np.ascontiguousarray(blocks[..., :2]).view("<f2")[..., 0]
        nibbles = blocks[..., 2:]
    elif kind == "IQ4_XS":
        blocks = data.reshape(data.shape[0], -1, 136)
        base = np.ascontiguousarray(blocks[..., :2]).view("<f2")[..., 0]
        high = np.ascontiguousarray(blocks[..., 2:4]).view("<u2")[..., 0]
        group = np.arange(8, dtype=np.uint16)
        low = (blocks[..., 4:8][..., group // 2] >> (4 * (group % 2))) & 15
        high = ((high[..., None] >> (2 * group)) & 3) << 4
        multiplier = (low | high).astype(np.int16) - 32
        scales = (base[..., None].astype(np.float32) * multiplier).reshape(
            data.shape[0], -1
        )
        nibbles = blocks[..., 8:].reshape(data.shape[0], -1, 16)
    else:
        raise ValueError(f"Unsupported GGUF format {kind}")

    # GGUF stores index i and i+16 in one byte. Humming packs consecutive K.
    codes = np.concatenate((nibbles & 15, nibbles >> 4), axis=-1)
    words = codes.reshape(data.shape[0], -1, 8).astype(np.uint32)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    packed = np.bitwise_or.reduce(words << shifts, axis=-1).view(np.int32)
    return packed, np.ascontiguousarray(scales, dtype=np.float32)


def main() -> None:
    from gguf import GGUFReader
    from safetensors.numpy import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("prefix", help="Destination vLLM checkpoint layer name")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tensor = next(
        (t for t in GGUFReader(args.gguf).tensors if t.name == args.tensor), None
    )
    if tensor is None:
        parser.error(f"Tensor {args.tensor!r} does not exist")
    kind = tensor.tensor_type.name
    packed, scales = convert_iq4(tensor.data, kind)
    save_file(
        {
            f"{args.prefix}.weight_packed": packed,
            f"{args.prefix}.weight_scale": scales,
            f"{args.prefix}.weight_shape": np.array(
                [packed.shape[0], packed.shape[1] * 8], dtype=np.int64
            ),
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "targets": [args.prefix],
                "weights": {
                    "type": "int",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "dynamic": False,
                    "observer": f"lab_{kind.replace('_', '').lower()}",
                },
            }
        )
    )


if __name__ == "__main__":
    main()
