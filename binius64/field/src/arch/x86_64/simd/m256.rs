// Copyright 2024-2025 Irreducible Inc.

use core::arch::x86_64::*;

use super::simd_arithmetic::TowerSimdType;
use crate::{BinaryField, arch::x86_64::m256::M256};

impl TowerSimdType for M256 {
    #[inline(always)]
    fn xor(a: Self, b: Self) -> Self {
        unsafe { _mm256_xor_si256(a.0, b.0) }.into()
    }

    #[inline(always)]
    fn shuffle_epi8(a: Self, b: Self) -> Self {
        unsafe { _mm256_shuffle_epi8(a.0, b.0) }.into()
    }

    #[inline(always)]
    fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self {
        let mask = Self::even_mask::<Scalar>();
        unsafe { _mm256_blendv_epi8(a.0, b.0, mask.0) }.into()
    }

    #[inline(always)]
    fn set_alpha_even<Scalar: BinaryField>(self) -> Self {
        unsafe {
            let alpha = Self::alpha::<Scalar>();
            let mask = Self::even_mask::<Scalar>();
            // NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not
            // a
            _mm256_blendv_epi8(self.0, alpha.0, mask.0)
        }
        .into()
    }

    #[inline(always)]
    fn set1_epi128(val: __m128i) -> Self {
        unsafe { _mm256_broadcastsi128_si256(val) }.into()
    }

    #[inline(always)]
    fn set_epi_64(val: i64) -> Self {
        unsafe { _mm256_set1_epi64x(val) }.into()
    }

    #[inline(always)]
    fn apply_mask<Scalar: BinaryField>(mut mask: Self, a: Self) -> Self {
        let tower_level = Scalar::N_BITS.ilog2();
        match tower_level {
            0..=2 => {
                for i in 0..tower_level {
                    mask |= mask >> (1 << i);
                }

                unsafe { _mm256_and_si256(mask.0, a.0) }
            }
            3 => unsafe { _mm256_blendv_epi8(_mm256_setzero_si256(), a.0, mask.0) },
            4..=7 => {
                let shuffle = Self::make_epi8_mask_shuffle::<Scalar>();
                unsafe {
                    let mask = _mm256_shuffle_epi8(mask.0, shuffle.0);
                    _mm256_blendv_epi8(_mm256_setzero_si256(), a.0, mask)
                }
            }
            _ => panic!("unsupported bit count"),
        }
        .into()
    }

    #[inline(always)]
    fn bslli_epi128<const IMM8: i32>(self) -> Self {
        unsafe { _mm256_bslli_epi128::<IMM8>(self.0) }.into()
    }

    #[inline(always)]
    fn bsrli_epi128<const IMM8: i32>(self) -> Self {
        unsafe { _mm256_bsrli_epi128::<IMM8>(self.0) }.into()
    }
}
