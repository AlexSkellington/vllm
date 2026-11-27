# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from tools.as1.convert_iq4_tensor import convert_iq4


def test_iq4_nl_keeps_index_order():
    block = np.empty((1, 18), dtype=np.uint8)
    block[0, :2] = np.array([0.5], dtype="<f2").view(np.uint8)
    i = np.arange(16, dtype=np.uint8)
    block[0, 2:] = i | ((15 - i) << 4)
    packed, scales = convert_iq4(block, "IQ4_NL")
    assert packed.view(np.uint32).tolist() == [
        [0x76543210, 0xFEDCBA98, 0x89ABCDEF, 0x01234567]
    ]
    np.testing.assert_array_equal(scales, [[0.5]])


def test_iq4_xs_reads_all_signed_subblock_scale_bits():
    multipliers = [-32, -17, -9, -1, 0, 1, 16, 31]
    biased = [value + 32 for value in multipliers]
    block = np.zeros((1, 136), dtype=np.uint8)
    block[0, :2] = np.array([0.25], dtype="<f2").view(np.uint8)
    high = sum((value >> 4) << (2 * i) for i, value in enumerate(biased))
    block[0, 2:4] = np.array([high], dtype="<u2").view(np.uint8)
    for i in range(4):
        block[0, 4 + i] = (biased[2 * i] & 15) | ((biased[2 * i + 1] & 15) << 4)
    for i in range(8):
        block[0, 8 + 16 * i : 24 + 16 * i] = i | ((15 - i) << 4)
    packed, scales = convert_iq4(block, "IQ4_XS")
    np.testing.assert_array_equal(scales, [np.array(multipliers) / 4])
    packed = packed.view(np.uint32).reshape(8, 4)
    for i, words in enumerate(packed):
        low = i * 0x11111111
        high = (15 - i) * 0x11111111
        assert words.tolist() == [low, low, high, high]
