// Copyright 2024-2025 Irreducible Inc.

use crate::{
    AESTowerField8b,
    arch::{
        GfniStrategy, portable::packed::PackedPrimitiveType,
        x86_64::simd::simd_arithmetic::TowerSimdType,
    },
    arithmetic_traits::{TaggedInvertOrZero, TaggedMul},
    underlier::UnderlierType,
};

#[rustfmt::skip]
pub const IDENTITY_MAP: i64 = u64::from_le_bytes([
	0b10000000,
	0b01000000,
	0b00100000,
	0b00010000,
	0b00001000,
	0b00000100,
	0b00000010,
	0b00000001,
]) as i64;

pub(super) trait GfniType: Copy + TowerSimdType {
    #[allow(unused)]
    fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self;
    fn gf2p8mul_epi8(a: Self, b: Self) -> Self;
    fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self;
}

impl<U: GfniType + UnderlierType> TaggedMul<GfniStrategy>
    for PackedPrimitiveType<U, AESTowerField8b>
{
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        U::gf2p8mul_epi8(self.0, rhs.0).into()
    }
}

impl<U: GfniType + UnderlierType> TaggedInvertOrZero<GfniStrategy>
    for PackedPrimitiveType<U, AESTowerField8b>
{
    #[inline(always)]
    fn invert_or_zero(self) -> Self {
        let val_gfni = self.to_underlier();

        // Calculate inversion and linear transformation to the original field with a single
        // instruction
        let transform_after = U::set_epi_64(IDENTITY_MAP);
        let inv_gfni = U::gf2p8affineinv_epi64_epi8(val_gfni, transform_after);

        inv_gfni.into()
    }
}
