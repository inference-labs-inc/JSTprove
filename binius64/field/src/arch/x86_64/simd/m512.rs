// Copyright 2024-2025 Irreducible Inc.

use core::arch::x86_64::*;

use super::simd_arithmetic::TowerSimdType;
use crate::{BinaryField, arch::x86_64::m512::M512};

impl TowerSimdType for M512 {
    #[inline(always)]
    fn xor(a: Self, b: Self) -> Self {
        unsafe { _mm512_xor_si512(a.0, b.0) }.into()
    }

    #[inline(always)]
    fn shuffle_epi8(a: Self, b: Self) -> Self {
        unsafe { _mm512_shuffle_epi8(a.0, b.0) }.into()
    }

    #[inline(always)]
    fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self {
        let mask = even_mask::<Scalar>();
        unsafe { _mm512_mask_blend_epi8(mask, b.0, a.0) }.into()
    }

    #[inline(always)]
    fn set_alpha_even<Scalar: BinaryField>(self) -> Self {
        unsafe {
            let alpha = Self::alpha::<Scalar>();
            let mask = even_mask::<Scalar>();
            // NOTE: There appears to be a bug in _mm_blendv_epi8 where the mask bit selects b, not
            // a
            _mm512_mask_blend_epi8(mask, alpha.0, self.0)
        }
        .into()
    }

    #[inline(always)]
    fn set1_epi128(val: __m128i) -> Self {
        unsafe { _mm512_broadcast_i32x4(val) }.into()
    }

    #[inline(always)]
    fn set_epi_64(val: i64) -> Self {
        unsafe { _mm512_set1_epi64(val) }.into()
    }

    #[inline(always)]
    fn bslli_epi128<const IMM8: i32>(self) -> Self {
        unsafe { _mm512_bslli_epi128::<IMM8>(self.0) }.into()
    }

    #[inline(always)]
    fn bsrli_epi128<const IMM8: i32>(self) -> Self {
        unsafe { _mm512_bsrli_epi128::<IMM8>(self.0) }.into()
    }

    #[inline(always)]
    fn apply_mask<Scalar: BinaryField>(mut mask: Self, a: Self) -> Self {
        let tower_level = Scalar::N_BITS.ilog2();
        match tower_level {
            0..=2 => {
                for i in 0..tower_level {
                    mask |= mask >> (1 << i);
                }

                unsafe { _mm512_and_si512(a.0, mask.0) }
            }
            3 => unsafe { _mm512_maskz_mov_epi8(_mm512_movepi8_mask(mask.0), a.0) },
            4 => unsafe { _mm512_maskz_mov_epi16(_mm512_movepi16_mask(mask.0), a.0) },
            5..=7 => {
                let shuffle = Self::make_epi8_mask_shuffle::<Scalar>();
                unsafe {
                    let mask = _mm512_shuffle_epi8(mask.0, shuffle.0);
                    _mm512_maskz_mov_epi8(_mm512_movepi8_mask(mask), a.0)
                }
            }
            _ => panic!("unsupported bit count"),
        }
        .into()
    }
}

fn even_mask<Scalar: BinaryField>() -> u64 {
    match Scalar::N_BITS.ilog2() {
        3 => 0xAAAAAAAAAAAAAAAA,
        4 => 0xCCCCCCCCCCCCCCCC,
        5 => 0xF0F0F0F0F0F0F0F0,
        6 => 0xFF00FF00FF00FF00,
        _ => panic!("unsupported bit count"),
    }
}
