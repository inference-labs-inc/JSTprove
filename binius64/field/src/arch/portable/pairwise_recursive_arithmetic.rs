// Copyright 2024-2025 Irreducible Inc.

use crate::{
    TowerExtensionField,
    arch::PairwiseRecursiveStrategy,
    arithmetic_traits::{
        InvertOrZero, MulAlpha, Square, TaggedInvertOrZero, TaggedMul, TaggedMulAlpha, TaggedSquare,
    },
    packed::PackedField,
};

impl<P> TaggedMul<PairwiseRecursiveStrategy> for P
where
    P: PackedField,
    P::Scalar: TowerExtensionField<DirectSubfield: MulAlpha>,
{
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        P::from_fn(|i| {
            let (a0, a1) = self.get(i).into();
            let (b0, b1) = rhs.get(i).into();
            let (z0, z2) = (a0 * b0, a1 * b1);
            let z0z2 = z0 + z2;
            let z1 = (a0 + a1) * (b0 + b1) - z0z2;
            let z2a = MulAlpha::mul_alpha(z2);

            (z0z2, z1 + z2a).into()
        })
    }
}

impl<P> TaggedSquare<PairwiseRecursiveStrategy> for P
where
    P: PackedField,
    P::Scalar: TowerExtensionField<DirectSubfield: MulAlpha>,
{
    #[inline]
    fn square(self) -> Self {
        P::from_fn(|i| {
            let (a0, a1) = self.get(i).into();
            let (z0, z2) = (Square::square(a0), Square::square(a1));
            let z2a = MulAlpha::mul_alpha(z2);
            (z0 + z2, z2a).into()
        })
    }
}

impl<P> TaggedMulAlpha<PairwiseRecursiveStrategy> for P
where
    P: PackedField,
    P::Scalar: TowerExtensionField<DirectSubfield: MulAlpha>,
{
    #[inline]
    fn mul_alpha(self) -> Self {
        P::from_fn(|i| {
            let (a0, a1) = self.get(i).into();
            let z1 = MulAlpha::mul_alpha(a1);

            (a1, a0 + z1).into()
        })
    }
}

impl<P> TaggedInvertOrZero<PairwiseRecursiveStrategy> for P
where
    P: PackedField,
    P::Scalar: TowerExtensionField<DirectSubfield: MulAlpha + InvertOrZero>,
{
    #[inline]
    fn invert_or_zero(self) -> Self {
        P::from_fn(|i| {
            let (a0, a1) = self.get(i).into();
            let a0z1 = a0 + MulAlpha::mul_alpha(a1);
            let delta = a0 * a0z1 + Square::square(a1);
            let delta_inv = InvertOrZero::invert_or_zero(delta);
            let inv0 = delta_inv * a0z1;
            let inv1 = delta_inv * a1;
            (inv0, inv1).into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{define_invert_tests, define_multiply_tests, define_square_tests};

    define_multiply_tests!(
        TaggedMul<PairwiseRecursiveStrategy>::mul,
        TaggedMul<PairwiseRecursiveStrategy>
    );

    define_square_tests!(
        TaggedSquare<PairwiseRecursiveStrategy>::square,
        TaggedSquare<PairwiseRecursiveStrategy>
    );

    define_invert_tests!(
        TaggedInvertOrZero<PairwiseRecursiveStrategy>::invert_or_zero,
        TaggedInvertOrZero<PairwiseRecursiveStrategy>
    );
}
