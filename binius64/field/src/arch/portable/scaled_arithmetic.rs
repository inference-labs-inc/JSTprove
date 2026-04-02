// Copyright 2025-2026 The Binius Developers

use std::array;

use bytemuck::{Pod, TransparentWrapper};

use super::packed::PackedPrimitiveType;
use crate::{
    BinaryField,
    arch::ScaledStrategy,
    arithmetic_traits::{
        InvertOrZero, MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
    },
    underlier::{ScaledUnderlier, UnderlierType},
};

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> TaggedMul<ScaledStrategy>
    for PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>
where
    PackedPrimitiveType<U, Scalar>: std::ops::Mul<Output = PackedPrimitiveType<U, Scalar>>,
{
    fn mul(self, rhs: Self) -> Self {
        Self::wrap(ScaledUnderlier(array::from_fn(|i| {
            let lhs_i = self.0.0[i];
            let rhs_i = rhs.0.0[i];
            PackedPrimitiveType::peel(
                PackedPrimitiveType::wrap(lhs_i) * PackedPrimitiveType::wrap(rhs_i),
            )
        })))
    }
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> TaggedMulAlpha<ScaledStrategy>
    for PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>
where
    PackedPrimitiveType<U, Scalar>: MulAlpha,
{
    fn mul_alpha(self) -> Self {
        Self::wrap(ScaledUnderlier(self.0.0.map(|sub_underlier| {
            PackedPrimitiveType::peel(PackedPrimitiveType::wrap(sub_underlier).mul_alpha())
        })))
    }
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> TaggedSquare<ScaledStrategy>
    for PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>
where
    PackedPrimitiveType<U, Scalar>: Square,
{
    fn square(self) -> Self {
        Self::wrap(ScaledUnderlier(self.0.0.map(|sub_underlier| {
            PackedPrimitiveType::peel(PackedPrimitiveType::wrap(sub_underlier).square())
        })))
    }
}

impl<U: UnderlierType + Pod, Scalar: BinaryField, const N: usize> TaggedInvertOrZero<ScaledStrategy>
    for PackedPrimitiveType<ScaledUnderlier<U, N>, Scalar>
where
    PackedPrimitiveType<U, Scalar>: InvertOrZero,
{
    fn invert_or_zero(self) -> Self {
        Self::wrap(ScaledUnderlier(self.0.0.map(|sub_underlier| {
            PackedPrimitiveType::peel(PackedPrimitiveType::wrap(sub_underlier).invert_or_zero())
        })))
    }
}
