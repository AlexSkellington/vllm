#pragma once

#include <humming/datatype/dequant_fused.cuh>
#include <humming/datatype/dequant_native.cuh>
#include <humming/datatype/dequant_prepare.cuh>
#include <humming/datatype/dequant_single.cuh>
#include <humming/utils/all.cuh>


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint>
CUDA_INLINE void dequant_b1248(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  static_assert(SourceType::kBits <= TargetType::kBits);
  constexpr uint32_t kResultPackedNums = 32 / TargetType::kBits;
  constexpr uint32_t kResultPackedNumSourceBits = kResultPackedNums * SourceType::kBits;
  constexpr bool kIsFpToFp = SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType;
  constexpr uint32_t reverse_pattern = TargetType::kBits / SourceType::kBits;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t index = j * 4 + i;
    if constexpr (kIsFpToFp) {
      index = index / reverse_pattern * reverse_pattern + reverse_pattern - 1 - index % reverse_pattern;
    }
    uint32_t zp_val = kHasZeroPoint ? zp_vals[i] : 0;
    uint32_t qb_index = kResultPackedNumSourceBits * index / 32;

    uint32_t shift_count;
    if constexpr (!kIsFpToFp && SourceType::kBits * 4 % TargetType::kBits == 0) {
      shift_count = SourceType::kBits * i % TargetType::kBits;
    } else {
      shift_count = SourceType::kBits * index % TargetType::kBits;
    }

    uint32_t qb_val = qb[qb_index];
    if (kIsFpToFp && shift_count) qb_val = qb_val << shift_count;
    if (!kIsFpToFp && shift_count) qb_val = qb_val >> shift_count;
    res[i] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val, zp_val);
  }
}


template <class SourceType>
CUDA_INLINE void repack_native_mxf8f6f4(const uint32_t *qb, uint32_t *res, uint32_t j) {
  if constexpr (std::is_same<SourceType, Float4E2M1>::value) {
    res[0] = (qb[2 * j] & 0x0F0F0F0Fu) << 2;
    res[1] = (qb[2 * j] & 0xF0F0F0F0u) >> 2;
    res[2] = (qb[2 * j + 1] & 0x0F0F0F0Fu) << 2;
    res[3] = (qb[2 * j + 1] & 0xF0F0F0F0u) >> 2;
  } else {
    static_assert(std::is_same<SourceType, Float6E3M2>::value || std::is_same<SourceType, Float6E2M3>::value);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      uint32_t index = j * 4 + i;
      res[i] = get_quanted_value_group<6, true>(qb + index / 8 * 6, index % 8);
    }
  }
}


template <class SourceType, class TargetType, bool kHasZeroPoint, bool kIsFpZeroPoint, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant_b3567(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  static_assert(SourceType::kBits <= TargetType::kBits);
  constexpr uint32_t kPaddedNumBits = static_next_power_of_2(SourceType::kBits);
  constexpr bool kIsFpToFp = SourceType::kIsFloatingPointType && TargetType::kIsFloatingPointType;
  constexpr uint32_t reverse_pattern = TargetType::kBits / static_next_power_of_2(SourceType::kBits);
  uint32_t qb_val;

  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t zp_val = kHasZeroPoint ? zp_vals[i] : 0;
    uint32_t index;

    if constexpr (kNumWarpShapeNSplits == 1 || SourceType::kBits == 6) {
      index = j * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    } else if (threadIdx.x / 32 % 2 == 0) {
      index = j * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    } else {
      index = (j + TargetType::kBits / 8) * 4 + i;
      if (index * kPaddedNumBits % TargetType::kBits == 0) {
        const uint32_t idx1 = index / TargetType::kBits;
        const uint32_t idx2 = index * kPaddedNumBits / TargetType::kBits % kPaddedNumBits;
        const uint32_t qb_offset = idx1 * SourceType::kBits;
        qb_val = get_quanted_value_group<SourceType::kNumBits, !kIsFpToFp>(qb + qb_offset, idx2);
      }
    }

    uint32_t shift_count;
    if constexpr (kPaddedNumBits * 4 % TargetType::kBits == 0) {
      shift_count = kPaddedNumBits * i % TargetType::kBits;
    } else {
      shift_count = kPaddedNumBits * index % TargetType::kBits;
    }

    uint32_t qb_val2 = qb_val;
    if (kIsFpToFp && shift_count) qb_val2 = qb_val << shift_count;
    if (!kIsFpToFp && shift_count) qb_val2 = qb_val >> shift_count;

    if constexpr (!kIsFpToFp) {
      res[i] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val2, zp_val);
    } else {
      uint32_t reversed_index = i / reverse_pattern * reverse_pattern + reverse_pattern - 1 - i % reverse_pattern;
      res[reversed_index] = dequant_single<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb_val2, zp_val);
    }
  }
}




// lab (AS1 2026-11-22): 8 codebook indices of one word -> 4 f16/bf16 pairs (res[i] = {lut[n_i], lut[n_(4+i)]}).
template <class TargetType>
CUDA_INLINE void lut7_dequant_x8(uint32_t q, uint32_t *res) {
  using scalar_t2 = typename F16Conversion<TargetType>::scalar_t2;
  constexpr uint32_t kT0 = 0x20160C00u, kT1 = 0x3B352E28u, kT2 = 0x534C4640u, kT3 = 0x786C625Au;  // KV_NL7 + 64
  constexpr uint32_t base = std::is_same<TargetType, Float16>::value ? 0x64006400u : 0x43004300u;
  uint32_t offset = base + 0x00400040u;
  const scalar_t2 off2 = *reinterpret_cast<scalar_t2 *>(&offset);
  const uint32_t q0 = q & 0x7777u, q1 = (q >> 16) & 0x7777u;
  const uint32_t a = __byte_perm(__byte_perm(kT0, kT1, q0), __byte_perm(kT2, kT3, q0), 0x3210u | ((q & 0x8888u) >> 1));
  const uint32_t b = __byte_perm(__byte_perm(kT0, kT1, q1), __byte_perm(kT2, kT3, q1), 0x3210u | ((q >> 17) & 0x4444u));
  PRAGMA_UNROLL
  for (uint32_t i = 0; i < 4; i++) {
    uint32_t r = (__byte_perm(a, b, i | ((4u + i) << 8)) & 0x00FF00FFu) | base;
    scalar_t2 v = __hsub2(*reinterpret_cast<scalar_t2 *>(&r), off2);
    res[i] = *reinterpret_cast<uint32_t *>(&v);
  }
}


// lab (AS1 2026-11-22): exact IQ4_NL: 8 indices of one word -> 4 pairs via split low/high byte tables.
template <class TargetType>
CUDA_INLINE void lutnl_dequant_x8(uint32_t q, uint32_t *res) {
  constexpr bool kF16 = std::is_same<TargetType, Float16>::value;
  constexpr uint32_t L0 = kF16 ? 0x103080F0u : 0x82A6D0FEu, L1 = kF16 ? 0x00806020u : 0x20B00C44u;
  constexpr uint32_t L2 = kF16 ? 0xC0408000u : 0x18C85080u, L3 = kF16 ? 0x109050A0u : 0xE2B28A54u;
  constexpr uint32_t H0 = kF16 ? 0xD4D5D6D7u : 0xC2C2C2C2u, H1 = kF16 ? 0xC9CDD0D2u : 0xC1C1C2C2u;
  constexpr uint32_t H2 = kF16 ? 0x504E4A3Cu : 0x4241413Fu, H3 = kF16 ? 0x57555452u : 0x42424242u;
  const uint32_t q0 = q & 0x7777u, q1 = (q >> 16) & 0x7777u;
  const uint32_t s0 = 0x3210u | ((q & 0x8888u) >> 1), s1 = 0x3210u | ((q >> 17) & 0x4444u);
  const uint32_t al = __byte_perm(__byte_perm(L0, L1, q0), __byte_perm(L2, L3, q0), s0);  // low bytes, idx 0..3
  const uint32_t ah = __byte_perm(__byte_perm(H0, H1, q0), __byte_perm(H2, H3, q0), s0);  // high bytes, idx 0..3
  const uint32_t bl = __byte_perm(__byte_perm(L0, L1, q1), __byte_perm(L2, L3, q1), s1);  // idx 4..7
  const uint32_t bh = __byte_perm(__byte_perm(H0, H1, q1), __byte_perm(H2, H3, q1), s1);
  const uint32_t a01 = __byte_perm(al, ah, 0x5140u), a23 = __byte_perm(al, ah, 0x7362u);   // {v0, v1}, {v2, v3}
  const uint32_t b01 = __byte_perm(bl, bh, 0x5140u), b23 = __byte_perm(bl, bh, 0x7362u);   // {v4, v5}, {v6, v7}
  res[0] = __byte_perm(a01, b01, 0x5410u);  // {v0, v4}
  res[1] = __byte_perm(a01, b01, 0x7632u);  // {v1, v5}
  res[2] = __byte_perm(a23, b23, 0x5410u);  // {v2, v6}
  res[3] = __byte_perm(a23, b23, 0x7632u);  // {v3, v7}
}

template <class SourceType, class TargetType, bool kHasZeroPoint = false, bool kIsFpZeroPoint = false, uint32_t kNumWarpShapeNSplits = 1>
CUDA_INLINE void dequant(const uint32_t *qb, uint32_t *res, uint32_t j, uint32_t *zp_vals = nullptr) {
  if constexpr (SourceType::kIsLut) {
    static_assert(SourceType::kBits == 4 && TargetType::kBits == 16 && !kHasZeroPoint);
    if constexpr (SourceType::kLutId == 2) lutnl_dequant_x8<TargetType>(qb[j], res);
    else lut7_dequant_x8<TargetType>(qb[j], res);
  } else
  if constexpr (SourceType::kBits == static_next_power_of_2(SourceType::kBits)) {
    dequant_b1248<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint>(qb, res, j, zp_vals);
  } else {
    dequant_b3567<SourceType, TargetType, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(qb, res, j, zp_vals);
  }
}


template <class SourceType, class TargetType>
constexpr bool kNativeScaleDequantSupported =
    kNativeDequantSupported<SourceType, TargetType> ||
    (std::is_same<TargetType, BFloat16>::value && kNativeDequantSupported<SourceType, Float16>);


// Software conversion may retain (0, 2, 1, 3) order for callers that can absorb the permutation.
template <class SourceType, class TargetType, bool kKeepDequantOrder = false>
CUDA_INLINE void dequant_scale(const uint32_t *src, uint32_t *dst, uint32_t index) {
  if constexpr (kNativeDequantSupported<SourceType, TargetType>) {
    dequant_native<SourceType, TargetType>(src, dst, index);
  } else if constexpr (kNativeScaleDequantSupported<SourceType, TargetType>) {
    uint32_t packed[4];
    dequant_native<SourceType, Float16>(src, packed, index);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      float2 values = __half22float2(*reinterpret_cast<half2 *>(&packed[i]));
      nv_bfloat162 result = __float22bfloat162_rn(values);
      dst[i] = *reinterpret_cast<uint32_t *>(&result);
    }
  } else {
    dequant<SourceType, TargetType>(src, dst, index);
    if constexpr (!kKeepDequantOrder) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        uint32_t even = dst[2 * i];
        uint32_t odd = dst[2 * i + 1];
        dst[2 * i] = __byte_perm(even, odd, 0x5410);
        dst[2 * i + 1] = __byte_perm(even, odd, 0x7632);
      }
    }
  }
}


template <class SourceType>
CUDA_INLINE void dequant_scale_float32(const uint32_t *src, uint32_t *dst, uint32_t index) {
  uint32_t pairs[4];
  if constexpr (kNativeDequantSupported<SourceType, Float16>) {
    dequant_native<SourceType, Float16>(src, pairs, index);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      reinterpret_cast<float2 *>(dst)[i] = __half22float2(*reinterpret_cast<half2 *>(&pairs[i]));
    }
  } else {
    // Expand eight scale bytes into four packed BF16 pairs before widening to FP32.
    dequant_scale<SourceType, BFloat16, true>(src, pairs, index);
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 4; i++) {
      nv_bfloat162 values = *reinterpret_cast<nv_bfloat162 *>(&pairs[i]);
      constexpr uint32_t kExponentOffset = 128 - (1u << (SourceType::kExponentBits - 1));
      if constexpr (kExponentOffset != 0) {
        constexpr uint32_t kFactorBits = ((127 + kExponentOffset) * 0x00010001u) << 7;
        const nv_bfloat162 factor = *reinterpret_cast<const nv_bfloat162 *>(&kFactorBits);
        values = __hmul2(values, factor);
      }
      float2 converted = __bfloat1622float2(values);
      if constexpr (kNativeScaleDequantSupported<SourceType, BFloat16>) {
        reinterpret_cast<float2 *>(dst)[i] = converted;
      } else {
        // Restore byte order through register indices instead of packed permutations.
        constexpr uint32_t kValuesPerGroup = 4;
        uint32_t first = i / 2 * kValuesPerGroup + i % 2;
        reinterpret_cast<float *>(dst)[first] = converted.x;
        reinterpret_cast<float *>(dst)[first + 2] = converted.y;
      }
    }
  }
}
