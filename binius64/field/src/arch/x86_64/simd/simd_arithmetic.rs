// Copyright 2024-2025 Irreducible Inc.

use std::{any::TypeId, arch::x86_64::*};

use crate::{
    BinaryField, TowerField,
    aes_field::AESTowerField8b,
    arch::portable::{packed::PackedPrimitiveType, reuse_multiply_arithmetic::Alpha},
    underlier::{UnderlierType, UnderlierWithBitOps},
};

pub trait TowerSimdType: Sized + Copy + UnderlierWithBitOps {
    /// Blend odd and even elements
    fn blend_odd_even<Scalar: BinaryField>(a: Self, b: Self) -> Self;
    /// Set alpha to even elements
    fn set_alpha_even<Scalar: BinaryField>(self) -> Self;
    /// Apply `mask` to `a` (set zeros at positions where high bit of the `mask` is 0).
    fn apply_mask<Scalar: BinaryField>(mask: Self, a: Self) -> Self;

    /// Bit xor operation
    fn xor(a: Self, b: Self) -> Self;

    /// Shuffle 8-bit elements within 128-bit lanes
    fn shuffle_epi8(a: Self, b: Self) -> Self;

    /// Byte shifts within 128-bit lanes
    fn bslli_epi128<const IMM8: i32>(self) -> Self;
    fn bsrli_epi128<const IMM8: i32>(self) -> Self;

    /// Initialize value with a single element
    fn set1_epi128(val: __m128i) -> Self;
    fn set_epi_64(val: i64) -> Self;

    #[inline(always)]
    fn dup_shuffle<Scalar: BinaryField>() -> Self {
        let shuffle_mask_128 = unsafe {
            match Scalar::N_BITS.ilog2() {
                3 => _mm_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0),
                4 => _mm_set_epi8(13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0),
                5 => _mm_set_epi8(11, 10, 9, 8, 11, 10, 9, 8, 3, 2, 1, 0, 3, 2, 1, 0),
                6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0),
                _ => panic!("unsupported bit count"),
            }
        };

        Self::set1_epi128(shuffle_mask_128)
    }

    #[inline(always)]
    fn flip_shuffle<Scalar: BinaryField>() -> Self {
        let flip_mask_128 = unsafe {
            match Scalar::N_BITS.ilog2() {
                3 => _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1),
                4 => _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2),
                5 => _mm_set_epi8(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4),
                6 => _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8),
                _ => panic!("unsupported bit count"),
            }
        };

        Self::set1_epi128(flip_mask_128)
    }

    /// Creates mask to propagate the highest bit form mask to other element bytes
    #[inline(always)]
    fn make_epi8_mask_shuffle<Scalar: BinaryField>() -> Self {
        let epi8_mask_128 = unsafe {
            match Scalar::N_BITS.ilog2() {
                4 => _mm_set_epi8(15, 15, 13, 13, 11, 11, 9, 9, 7, 7, 5, 5, 3, 3, 1, 1),
                5 => _mm_set_epi8(15, 15, 15, 15, 11, 11, 11, 11, 7, 7, 7, 7, 3, 3, 3, 3),
                6 => _mm_set_epi8(15, 15, 15, 15, 15, 15, 15, 15, 7, 7, 7, 7, 7, 7, 7, 7),
                7 => _mm_set1_epi8(15),
                _ => panic!("unsupported bit count"),
            }
        };

        Self::set1_epi128(epi8_mask_128)
    }

    #[inline(always)]
    fn alpha<Scalar: BinaryField>() -> Self {
        let alpha_128 = {
            match Scalar::N_BITS.ilog2() {
                3 => {
                    // Compiler will optimize this if out for each instantiation
                    let type_id = TypeId::of::<Scalar>();
                    let value = if type_id == TypeId::of::<AESTowerField8b>() {
                        0xd3u8 as i8
                    } else {
                        panic!("tower field not supported")
                    };
                    unsafe { _mm_set1_epi8(value) }
                }
                4 => unsafe { _mm_set1_epi16(0x0100) },
                5 => unsafe { _mm_set1_epi32(0x00010000) },
                6 => unsafe { _mm_set1_epi64x(0x0000000100000000) },
                _ => panic!("unsupported bit count"),
            }
        };

        Self::set1_epi128(alpha_128)
    }

    #[inline(always)]
    fn even_mask<Scalar: BinaryField>() -> Self {
        let mask_128 = {
            match Scalar::N_BITS.ilog2() {
                3 => unsafe { _mm_set1_epi16(0x00FF) },
                4 => unsafe { _mm_set1_epi32(0x0000FFFF) },
                5 => unsafe { _mm_set1_epi64x(0x00000000FFFFFFFF) },
                6 => unsafe { _mm_set_epi64x(0, -1) },
                _ => panic!("unsupported bit count"),
            }
        };

        Self::set1_epi128(mask_128)
    }
}

impl<U: UnderlierType + TowerSimdType, Scalar: TowerField> Alpha
    for PackedPrimitiveType<U, Scalar>
{
    #[inline(always)]
    fn alpha() -> Self {
        U::alpha::<Scalar>().into()
    }
}
