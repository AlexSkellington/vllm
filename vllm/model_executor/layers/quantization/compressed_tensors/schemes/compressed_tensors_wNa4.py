# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed integer weights with dynamic INT4 activations."""

from .compressed_tensors_wNa8 import CompressedTensorsWNA8Int

__all__ = ["CompressedTensorsWNA4Int"]


class CompressedTensorsWNA4Int(CompressedTensorsWNA8Int):
    activation_bits = 4
