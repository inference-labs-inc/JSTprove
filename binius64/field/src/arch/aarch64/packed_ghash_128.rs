// Copyright 2023-2025 Irreducible Inc.
// Copyright (c) 2019-2023 RustCrypto Developers

//! ARMv8 `PMULL`-accelerated implementation of GHASH.
//!
//! Based on the optimized GHASH implementation using carryless multiplication
//! instructions available on ARMv8 processors with NEON support.

use core::arch::aarch64::*;

use super::m128::M128;
use crate::{
    BinaryField128bGhash,
    arch::{
        portable::packed_macros::{portable_macros::*, *},
        shared::ghash::ClMulUnderlier,
    },
    arithmetic_traits::{
        InvertOrZero, TaggedInvertOrZero, TaggedMul, TaggedSquare, impl_invert_with, impl_mul_with,
        impl_square_with,
    },
};

impl ClMulUnderlier for M128 {
    #[inline]
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
        let a_u64x2: uint64x2_t = a.into();
        let b_u64x2: uint64x2_t = b.into();

        let result = match IMM8 {
            0x00 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 0), vgetq_lane_u64(b_u64x2, 0)) },
            0x11 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 1), vgetq_lane_u64(b_u64x2, 1)) },
            0x10 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 0), vgetq_lane_u64(b_u64x2, 1)) },
            0x01 => unsafe { vmull_p64(vgetq_lane_u64(a_u64x2, 1), vgetq_lane_u64(b_u64x2, 0)) },
            _ => panic!("Unsupported IMM8 value for clmulepi64"),
        };

        unsafe { std::mem::transmute::<u128, uint64x2_t>(result) }.into()
    }

    #[inline]
    fn move_64_to_hi(a: Self) -> Self {
        let a_bytes: uint8x16_t = a.into();
        // Shift left by 8 bytes
        unsafe {
            let zero = vdupq_n_u8(0);
            vextq_u8::<8>(zero, a_bytes).into()
        }
    }
}

/// Strategy for aarch64 GHASH field arithmetic operations.
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
impl TaggedMul<GhashStrategy> for PackedBinaryGhash1x128b {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_underlier(crate::arch::shared::ghash::mul_clmul(
            self.to_underlier(),
            rhs.to_underlier(),
        ))
    }
}

// Implement TaggedSquare for GhashStrategy
impl TaggedSquare<GhashStrategy> for PackedBinaryGhash1x128b {
    #[inline]
    fn square(self) -> Self {
        Self::from_underlier(crate::arch::shared::ghash::square_clmul(
            self.to_underlier(),
        ))
    }
}

// Implement TaggedInvertOrZero for GhashStrategy (uses portable fallback)
impl TaggedInvertOrZero<GhashStrategy> for PackedBinaryGhash1x128b {
    fn invert_or_zero(self) -> Self {
        let portable = super::super::portable::packed_ghash_128::PackedBinaryGhash1x128b::from(
            u128::from(self.to_underlier()),
        );

        Self::from_underlier(InvertOrZero::invert_or_zero(portable).to_underlier().into())
    }
}
