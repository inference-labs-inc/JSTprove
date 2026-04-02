// Copyright 2023-2025 Irreducible Inc.

use super::{arithmetic_traits::InvertOrZero, binary_field::*};
use crate::{PackedField, arithmetic_traits::MulAlpha};

pub(crate) trait TowerFieldArithmetic: TowerField {
    fn multiply(self, rhs: Self) -> Self;

    fn multiply_alpha(self) -> Self;

    fn square(self) -> Self;
}

macro_rules! impl_arithmetic_using_packed {
	($name:ident) => {
		impl InvertOrZero for $name {
			#[inline]
			fn invert_or_zero(self) -> Self {
				use $crate::packed_extension::PackedSubfield;

				$crate::binary_field_arithmetic::invert_or_zero_using_packed::<
					PackedSubfield<Self, Self>,
				>(self)
			}
		}

		impl TowerFieldArithmetic for $name {
			#[inline]
			fn multiply(self, rhs: Self) -> Self {
				use $crate::packed_extension::PackedSubfield;

				$crate::binary_field_arithmetic::multiple_using_packed::<PackedSubfield<Self, Self>>(
					self, rhs,
				)
			}

			#[inline]
			fn multiply_alpha(self) -> Self {
				use $crate::packed_extension::PackedSubfield;

				$crate::binary_field_arithmetic::mul_alpha_using_packed::<PackedSubfield<Self, Self>>(
					self,
				)
			}

			#[inline]
			fn square(self) -> Self {
				use $crate::packed_extension::PackedSubfield;

				$crate::binary_field_arithmetic::square_using_packed::<PackedSubfield<Self, Self>>(
					self,
				)
			}
		}
	};
}

pub(crate) use impl_arithmetic_using_packed;

// TODO: try to get rid of `TowerFieldArithmetic` and use `impl_arithmetic_using_packed` here
impl TowerField for BinaryField1b {
    fn min_tower_level(self) -> usize {
        0
    }

    #[inline]
    fn mul_primitive(self, iota: usize) -> Self {
        panic!("mul_primitive called on BinaryField1b with iota={iota}, but tower level is 0");
    }
}

impl InvertOrZero for BinaryField1b {
    #[inline]
    fn invert_or_zero(self) -> Self {
        self
    }
}

impl TowerFieldArithmetic for BinaryField1b {
    #[inline]
    fn multiply(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }

    #[inline]
    fn multiply_alpha(self) -> Self {
        self
    }

    #[inline]
    fn square(self) -> Self {
        self
    }
}

/// For some architectures it may be faster to used SIM versions for packed fields than to use
/// portable single-element arithmetics. That's why we need these functions
#[inline]
pub(super) fn multiple_using_packed<P: PackedField>(lhs: P::Scalar, rhs: P::Scalar) -> P::Scalar {
    (P::set_single(lhs) * P::set_single(rhs)).get(0)
}

#[inline]
pub(super) fn square_using_packed<P: PackedField>(value: P::Scalar) -> P::Scalar {
    P::set_single(value).square().get(0)
}

#[inline]
pub(super) fn invert_or_zero_using_packed<P: PackedField>(value: P::Scalar) -> P::Scalar {
    P::set_single(value).invert_or_zero().get(0)
}

#[inline]
pub(super) fn mul_alpha_using_packed<P: PackedField + MulAlpha>(value: P::Scalar) -> P::Scalar {
    P::set_single(value).mul_alpha().get(0)
}
