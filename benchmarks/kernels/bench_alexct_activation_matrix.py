# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the Alex_C.T packed linear activation modes with a dense reference."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from humming.forward import may_process_input
from humming.ops import pack_weight
from humming.testing.process_input import decode_quantized
from vllm.distributed import init_distributed_environment, initialize_model_parallel

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)

LEVELS = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def reference_activations(x, mode, group, levels):
    if mode == "a16":
        return x.float()
    grouped = x.float().reshape(x.size(0), -1, group)
    if mode == "iq4nl":
        scale = torch.maximum(
            grouped.amax(dim=-1, keepdim=True) * (1 / 113),
            -grouped.amin(dim=-1, keepdim=True) * (1 / 127),
        )
        normalized = grouped / torch.where(scale > 0, scale, 1)
        indices = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        quantized = levels[indices]
    else:
        peak = grouped.abs().amax(dim=-1, keepdim=True)
        scale = peak * (1 / 7)
        normalized = (grouped * 7) / torch.where(peak > 0, peak, 1)
        quantized = normalized.round().clamp(-7, 7)
    return (quantized * scale).view_as(x)


def build_layer(n, k, weight_args, dtype, packed, scales):
    config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "linear": {"targets": ["Linear"], "weights": weight_args},
            },
            "ignore": [],
        }
    )
    with torch.device("cuda"):
        layer = ReplicatedLinear(
            k,
            n,
            bias=False,
            params_dtype=dtype,
            quant_config=config,
            prefix="model.proj",
            disable_tp=True,
            return_bias=False,
        )
    layer.weight_packed.weight_loader(layer.weight_packed, packed)
    layer.weight_scale.weight_loader(layer.weight_scale, scales)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def verify(dtype, rows, mixed_groups, per_token):
    torch.manual_seed(2026)
    n, k = 256, 512
    levels = torch.tensor(LEVELS, dtype=torch.float32, device="cuda")
    for bits, observer in [
        (4, None),
        (5, None),
        (6, None),
        (8, None),
        (4, "lab_iq4nl"),
        (4, "lab_iq4xs"),
    ]:
        group = 32 if observer else 128
        weight_args = dict(
            num_bits=bits,
            type="int",
            strategy="group",
            group_size=group,
            symmetric=True,
            dynamic=False,
        )
        if observer:
            weight_args["observer"] = observer
        codes = torch.randint(0, 1 << bits, (n, k), device="cuda", dtype=torch.int32)
        if observer:
            # A book without index zero still must retain the original encoding.
            codes.clamp_min_(1)
        packed = pack_weight(codes, bits)
        scales = torch.rand((n, k // group), device="cuda", dtype=dtype) / 64
        if observer:
            scales[:, ::2].neg_()
        values = levels[codes.long()] if observer else codes.float() - (1 << (bits - 1))
        weight = (values.view(n, -1, group) * scales.float().unsqueeze(-1)).view(n, k)
        for mode in ("a16", "int4", "iq4nl", "int8"):
            act_group = 32 if (mixed_groups and mode == "iq4nl") else group
            if per_token:
                act_group = 0
            env = "VLLM_HUMMING_INPUT_QUANT_CONFIG"
            if mode == "a16":
                os.environ.pop(env, None)
            else:
                os.environ[env] = json.dumps(dict(dtype=mode, group_size=act_group))
            layer = build_layer(n, k, weight_args, dtype, packed, scales)
            backend = layer.scheme.kernel.layer_config
            if observer:
                assert str(backend.b_dtype) == "iq4nl"
            expected_dtype = (
                "int4" if mode == "int4" and bits == 4 and observer is None else "int8"
            )
            if mode != "a16":
                assert str(backend.a_dtype) == expected_dtype
            for m in rows:
                x = torch.randn((m, k), dtype=dtype, device="cuda")
                if m > 1:
                    x[0].zero_()
                if mode != "a16" and not layer.scheme.widen_activation:
                    q, group_scales, token_scales = may_process_input(backend, x)
                    q = decode_quantized(q, f"int{backend.a_dtype.num_bits}")
                    a_scales = group_scales if act_group else token_scales
                    dequantized = (
                        q.reshape(m, -1, act_group or k) * a_scales.unsqueeze(-1)
                    ).reshape_as(x)
                else:
                    dequantized = reference_activations(x, mode, act_group or k, levels)
                target = dequantized @ weight.T
                actual = layer(x).float()
                rms = ((actual - target).norm() / target.norm()).item()
                assert rms < 0.012, (bits, observer, mode, m, rms)
                if m > 1:
                    assert torch.count_nonzero(actual[0]).item() == 0
                print(
                    json.dumps(
                        dict(
                            bits=bits,
                            weight=observer,
                            mode=mode,
                            dtype=str(dtype),
                            m=m,
                            rms=rms,
                            mma_input=str(backend.a_dtype),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 17, 512])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--mixed-groups", action="store_true")
    parser.add_argument("--per-token", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    config = VllmConfig(
        kernel_config=KernelConfig(
            linear_backend_per_quant={"mixed_precision": "humming"}
        )
    )
    with TemporaryDirectory() as temporary, set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{Path(temporary) / 'rank'}",
            backend="gloo",
        )
        initialize_model_parallel(1, 1, backend="gloo")
        verify(
            torch.float16 if args.fp16 else torch.bfloat16,
            args.rows,
            args.mixed_groups,
            args.per_token,
        )
