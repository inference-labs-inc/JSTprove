// Copyright 2024-2025 Irreducible Inc.

use crate::{
    arch::PairwiseStrategy,
    arithmetic_traits::{
        InvertOrZero, MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
    },
    linear_transformation::Transformation,
    packed::PackedField,
};

impl<PT: PackedField> TaggedMul<PairwiseStrategy> for PT {
    #[inline]
    fn mul(self, b: Self) -> Self {
        if PT::WIDTH == 1 {
            // fallback to be able to benchmark this strategy
            self * b
        } else {
            Self::from_fn(|i| self.get(i) * b.get(i))
        }
    }
}

impl<PT: PackedField> TaggedSquare<PairwiseStrategy> for PT
where
    PT::Scalar: Square,
{
    #[inline]
    fn square(self) -> Self {
        if PT::WIDTH == 1 {
            // fallback to be able to benchmark this strategy
            Square::square(self)
        } else {
            Self::from_fn(|i| Square::square(self.get(i)))
        }
    }
}

impl<PT: PackedField> TaggedInvertOrZero<PairwiseStrategy> for PT
where
    PT::Scalar: InvertOrZero,
{
    #[inline]
    fn invert_or_zero(self) -> Self {
        if PT::WIDTH == 1 {
            // fallback to be able to benchmark this strategy
            InvertOrZero::invert_or_zero(self)
        } else {
            Self::from_fn(|i| InvertOrZero::invert_or_zero(self.get(i)))
        }
    }
}

impl<PT: PackedField + MulAlpha> TaggedMulAlpha<PairwiseStrategy> for PT
where
    PT::Scalar: MulAlpha,
{
    #[inline]
    fn mul_alpha(self) -> Self {
        if PT::WIDTH == 1 {
            // fallback to be able to benchmark this strategy
            MulAlpha::mul_alpha(self)
        } else {
            Self::from_fn(|i| MulAlpha::mul_alpha(self.get(i)))
        }
    }
}

/// Per element transformation
pub struct PairwiseTransformation<I> {
    inner: I,
}

impl<I> PairwiseTransformation<I> {
    pub const fn new(inner: I) -> Self {
        Self { inner }
    }
}

impl<IP, OP, IF, OF, I> Transformation<IP, OP> for PairwiseTransformation<I>
where
    IP: PackedField<Scalar = IF>,
    OP: PackedField<Scalar = OF>,
    I: Transformation<IF, OF>,
{
    fn transform(&self, data: &IP) -> OP {
        OP::from_fn(|i| self.inner.transform(&data.get(i)))
    }
}
