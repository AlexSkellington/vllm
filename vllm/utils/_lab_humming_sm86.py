# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alex_C.T SM86 schedules and the scale layouts supported by our CUDA header."""

import dataclasses
import functools


def install() -> None:
    """Extend the pinned Humming 0.1.18 runtime before kernel selection."""
    from humming import dtypes
    from humming.config import GemmType, WeightScaleType
    from humming.device import current_device
    from humming.schema import humming as schemas
    from humming.tune.sm8x import Sm86Heuristics
    from humming.utils.math import ceil_div

    original_config = Sm86Heuristics.get_config.__func__
    original_compatible = schemas.is_humming_schema_compatible

    @functools.wraps(original_compatible)
    def compatible(input_schema, weight_schema, param_dtype=None):
        input_group = input_schema.input_scale_group_size
        weight_group = weight_schema.weight_scale_group_size
        group_sizes = (
            (32, 64, 128) if input_schema.a_dtype == dtypes.int8 else (64, 128)
        )
        if (
            current_device.sm_version == 86
            and input_schema.a_dtype in (dtypes.int4, dtypes.int8)
            and weight_schema.b_dtype.is_integer_type
            and not weight_schema.has_zero_point
            and weight_schema.weight_scale_type == WeightScaleType.GROUP
            and input_group != weight_group
            and input_group in group_sizes
            and weight_group in group_sizes
        ):
            # The lab MMA header handles promotion at either K scale boundary.
            aligned = dataclasses.replace(
                input_schema, input_scale_group_size=weight_group
            )
            return original_compatible(aligned, weight_schema, param_dtype)
        return original_compatible(input_schema, weight_schema, param_dtype)

    @functools.wraps(original_config)
    def get_config(
        cls,
        layer_config,
        shape_m,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=GemmType.DENSE,
        use_m_major_input_scale=False,
    ):
        config = original_config(
            cls,
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            use_m_major_input_scale,
        )
        if (
            cls.sm_version != 86
            or gemm_type != GemmType.DENSE
            or use_batch_invariant
            or use_f16_accum
            or layer_config.c_dtype != dtypes.bfloat16
            or layer_config.has_zero_point
            or current_device.sm_count != 82
        ):
            return config
        shape = (layer_config.shape_n, layer_config.shape_k)
        if shape not in ((3072, 5120), (5120, 8704)):
            return config
        groups = (
            layer_config.input_scale_group_size,
            layer_config.weight_scale_group_size,
        )
        if groups != (128, 128) or not layer_config.b_dtype.is_integer_type:
            return config
        a_dtype = layer_config.a_dtype
        weight_bits = layer_config.b_dtype.num_bits
        if (a_dtype, weight_bits) not in (
            (dtypes.int4, 4),
            (dtypes.int8, 4),
            (dtypes.int8, 5),
            (dtypes.int8, 6),
            (dtypes.int8, 8),
        ):
            return config
        if shape_m <= 16:
            num_tiles = layer_config.shape_n // config["block_shape"][1]
            num_sms = min(
                current_device.sm_count, ceil_div(num_tiles, config["num_ctas_per_sm"])
            )
            return config | {"use_stream_k": False, "num_sms": num_sms}
        if a_dtype != dtypes.int8:
            return config
        is_down_projection = shape == (5120, 8704)
        if weight_bits == 4:
            prefill_min_rows = 1600 if is_down_projection else 384
            if prefill_min_rows <= shape_m <= 4096:
                return config | {"num_stages": 2}
        if weight_bits in (5, 6):
            if shape_m == 4096 or (not is_down_projection and 512 <= shape_m <= 640):
                return config | {"num_stages": 2}
            if is_down_projection and shape_m == 512:
                return config | {"use_stream_k": False}
        if weight_bits == 8 and (
            shape_m == 2048
            or (
                is_down_projection
                and (shape_m in (512, 4096) or 1024 <= shape_m <= 1600)
            )
        ):
            return config | {"use_stream_k": False}
        return config

    schemas.is_humming_schema_compatible = compatible
    Sm86Heuristics.get_config = classmethod(get_config)
