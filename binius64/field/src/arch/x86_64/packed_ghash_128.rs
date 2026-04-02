// Copyright 2024-2025 Irreducible Inc.

//! PCLMULQDQ-accelerated implementation of GHASH for x86_64.
//!
//! This module provides optimized GHASH multiplication using the PCLMULQDQ instruction
//! available on modern x86_64 processors. The implementation follows the algorithm
//! described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use cfg_if::cfg_if;

use super::m128::M128;
use crate::{
    BinaryField128bGhash,
    arch::portable::packed_macros::{portable_macros::*, *},
    arithmetic_traits::{
        InvertOrZero, TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
        impl_square_with,
    },
};

#[cfg(target_feature = "pclmulqdq")]
impl crate::arch::shared::ghash::ClMulUnderlier for M128 {
    #[inline]
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_clmulepi64_si128::<IMM8>(a.into(), b.into()) }.into()
    }

    #[inline]
    fn move_64_to_hi(a: Self) -> Self {
        unsafe { std::arch::x86_64::_mm_slli_si128::<8>(a.into()) }.into()
    }
}

/// Strategy for x86_64 GHASH field arithmetic operations.
pub struct GhashStrategy;

// Define PackedBinaryGhash1x128b using the macro
define_packed_binary_field!(
    PackedBinaryGhash1x128b,
    BinaryField128bGhash,
    M128,
    (GhashStrategy),
    (GhashStrategy),
    (GhashStrategy),
    (None),
    (None)
);

// Implement TaggedMul for GhashStrategy
cfg_if! {
    if #[cfg(target_feature = "pclmulqdq")] {
        impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
                    self.to_underlier(),
                    rhs.to_underlier(),
                ))
            }
        }
    } else {
        impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;

                let portable_lhs = PortablePackedBinaryGhash1x128b::from(u128::from(self.to_underlier()));
                let portable_rhs = PortablePackedBinaryGhash1x128b::from(u128::from(rhs.to_underlier()));

                Self::from_underlier(std::ops::Mul::mul(portable_lhs, portable_rhs).to_underlier().into())
            }
        }
    }
}

// Implement TaggedSquare for GhashStrategy
cfg_if! {
    if #[cfg(target_feature = "pclmulqdq")] {
        impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
            #[inline]
            fn square(self) -> Self {
                Self::from_underlier(crate::arch::shared::ghash::square_clmul(
                    self.to_underlier(),
                ))
            }
        }
    } else {
        impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
            #[inline]
            fn square(self) -> Self {
                use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;

                let portable_val = PortablePackedBinaryGhash1x128b::from(u128::from(self.to_underlier()));

                Self::from_underlier(crate::arithmetic_traits::Square::square(portable_val).to_underlier().into())
            }
        }
    }
}

// Implement TaggedInvertOrZero for GhashStrategy (always uses portable fallback)
impl TaggedInvertOrZero<GhashStrategy> for PackedBinaryGhash1x128b {
    fn invert_or_zero(self) -> Self {
        let portable = super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b::from(
            u128::from(self.to_underlier()),
        );

        Self::from_underlier(InvertOrZero::invert_or_zero(portable).to_underlier().into())
    }
}
