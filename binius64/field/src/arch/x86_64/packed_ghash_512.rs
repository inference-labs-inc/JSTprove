// Copyright 2024-2025 Irreducible Inc.

//! VPCLMULQDQ-accelerated implementation of GHASH for x86_64 AVX-512.
//!
//! This module provides optimized GHASH multiplication using the VPCLMULQDQ instruction
//! available on modern x86_64 processors with AVX-512 support. The implementation follows
//! the algorithm described in the GHASH specification with polynomial x^128 + x^7 + x^2 + x + 1.

use cfg_if::cfg_if;

use super::m512::M512;
use crate::{
    BinaryField128bGhash,
    arch::portable::packed_macros::{portable_macros::*, *},
    arithmetic_traits::{
        InvertOrZero, TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
        impl_square_with,
    },
    underlier::UnderlierWithBitOps,
};

#[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))]
impl crate::arch::shared::ghash::ClMulUnderlier for M512 {
    #[inline]
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
        unsafe { std::arch::x86_64::_mm512_clmulepi64_epi128::<IMM8>(a.into(), b.into()) }.into()
    }

    #[inline]
    fn move_64_to_hi(a: Self) -> Self {
        unsafe {
            std::arch::x86_64::_mm512_unpacklo_epi64(
                std::arch::x86_64::_mm512_setzero_si512(),
                a.into(),
            )
        }
        .into()
    }
}

/// Strategy for x86_64 AVX-512 GHASH field arithmetic operations.
pub struct Ghash512Strategy;

// Define PackedBinaryGhash4x128b using the macro
define_packed_binary_field!(
    PackedBinaryGhash4x128b,
    BinaryField128bGhash,
    M512,
    (Ghash512Strategy),
    (Ghash512Strategy),
    (Ghash512Strategy),
    (None),
    (None)
);

// Implement TaggedMul for Ghash512Strategy
cfg_if! {
    if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
        impl TaggedMul<Ghash512Strategy> for PackedBinaryGhash4x128b {
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
                    self.to_underlier(),
                    rhs.to_underlier(),
                ))
            }
        }
    } else {
        impl TaggedMul<Ghash512Strategy> for PackedBinaryGhash4x128b {
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                // Fallback: perform scalar multiplication on each 128-bit element
                let mut result_underlier = self.to_underlier();
                unsafe {
                    let self_0 = self.to_underlier().get_subvalue::<u128>(0);
                    let self_1 = self.to_underlier().get_subvalue::<u128>(1);
                    let self_2 = self.to_underlier().get_subvalue::<u128>(2);
                    let self_3 = self.to_underlier().get_subvalue::<u128>(3);
                    let rhs_0 = rhs.to_underlier().get_subvalue::<u128>(0);
                    let rhs_1 = rhs.to_underlier().get_subvalue::<u128>(1);
                    let rhs_2 = rhs.to_underlier().get_subvalue::<u128>(2);
                    let rhs_3 = rhs.to_underlier().get_subvalue::<u128>(3);

                    // Use the portable scalar multiplication for each element
                    use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;
                    let result_0 = std::ops::Mul::mul(
                        PortablePackedBinaryGhash1x128b::from(self_0),
                        PortablePackedBinaryGhash1x128b::from(rhs_0),
                    );
                    let result_1 = std::ops::Mul::mul(
                        PortablePackedBinaryGhash1x128b::from(self_1),
                        PortablePackedBinaryGhash1x128b::from(rhs_1),
                    );
                    let result_2 = std::ops::Mul::mul(
                        PortablePackedBinaryGhash1x128b::from(self_2),
                        PortablePackedBinaryGhash1x128b::from(rhs_2),
                    );
                    let result_3 = std::ops::Mul::mul(
                        PortablePackedBinaryGhash1x128b::from(self_3),
                        PortablePackedBinaryGhash1x128b::from(rhs_3),
                    );

                    result_underlier.set_subvalue(0, result_0.to_underlier());
                    result_underlier.set_subvalue(1, result_1.to_underlier());
                    result_underlier.set_subvalue(2, result_2.to_underlier());
                    result_underlier.set_subvalue(3, result_3.to_underlier());
                }

                Self::from_underlier(result_underlier)
            }
        }
    }
}

// Implement TaggedSquare for Ghash512Strategy
cfg_if! {
    if #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx512f"))] {
        impl TaggedSquare<Ghash512Strategy> for PackedBinaryGhash4x128b {
            #[inline]
            fn square(self) -> Self {
                Self::from_underlier(crate::arch::shared::ghash::square_clmul(
                    self.to_underlier(),
                ))
            }
        }
    } else {
        // Potentially we could use an optimized square implementation here with a scaled underlier.
        // But this case (an architecture with AVX512 but without VPCLMULQDQ) is pretty rare.
        impl_square_with!(PackedBinaryGhash4x128b @ crate::arch::ReuseMultiplyStrategy);
    }
}

// Implement TaggedInvertOrZero for Ghash512Strategy (always uses element-wise fallback)
impl TaggedInvertOrZero<Ghash512Strategy> for PackedBinaryGhash4x128b {
    fn invert_or_zero(self) -> Self {
        // Fallback: perform scalar invert on each 128-bit element
        let mut result_underlier = self.to_underlier();
        unsafe {
            let self_0 = self.to_underlier().get_subvalue::<u128>(0);
            let self_1 = self.to_underlier().get_subvalue::<u128>(1);
            let self_2 = self.to_underlier().get_subvalue::<u128>(2);
            let self_3 = self.to_underlier().get_subvalue::<u128>(3);

            // Use the portable scalar invert for each element
            use super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b as PortablePackedBinaryGhash1x128b;
            let result_0 =
                InvertOrZero::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_0));
            let result_1 =
                InvertOrZero::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_1));
            let result_2 =
                InvertOrZero::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_2));
            let result_3 =
                InvertOrZero::invert_or_zero(PortablePackedBinaryGhash1x128b::from(self_3));

            result_underlier.set_subvalue(0, result_0.to_underlier());
            result_underlier.set_subvalue(1, result_1.to_underlier());
            result_underlier.set_subvalue(2, result_2.to_underlier());
            result_underlier.set_subvalue(3, result_3.to_underlier());
        }

        Self::from_underlier(result_underlier)
    }
}
