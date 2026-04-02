// Copyright 2024-2025 Irreducible Inc.

use std::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{DeserializeBytes, SerializeBytes};
use bytemuck::Zeroable;

use super::extension::ExtensionField;
use crate::{
    Random,
    arithmetic_traits::{InvertOrZero, Square},
};

/// This trait is based on `ff::Field` with some unused functionality removed.
pub trait Field:
    Sized
    + Eq
    + Copy
    + Clone
    + Default
    + Send
    + Sync
    + Debug
    + Display
    + Hash
    + 'static
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + Sum
    + Product
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Square
    + InvertOrZero
    + Random
    + Zeroable
    + SerializeBytes
    + DeserializeBytes
{
    /// The zero element of the field, the additive identity.
    const ZERO: Self;

    /// The one element of the field, the multiplicative identity.
    const ONE: Self;

    /// The characteristic of the field.
    const CHARACTERISTIC: usize;

    /// Fixed generator of the multiplicative group.
    const MULTIPLICATIVE_GENERATOR: Self;

    /// Returns true iff this element is zero.
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    /// Doubles this element.
    #[must_use]
    fn double(&self) -> Self;

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> Option<Self> {
        let inv = self.invert_or_zero();
        (!inv.is_zero()).then_some(inv)
    }

    /// Exponentiates `self` by `exp`, where `exp` is a little-endian order integer
    /// exponent.
    ///
    /// # Guarantees
    ///
    /// This operation is constant time with respect to `self`, for all exponents with the
    /// same number of digits (`exp.as_ref().len()`). It is variable time with respect to
    /// the number of digits in the exponent.
    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                res = res.square();
                let mut tmp = res;
                tmp *= self;
                if ((*e >> i) & 1) != 0 {
                    res = tmp;
                }
            }
        }
        res
    }

    /// Exponentiates `self` by `exp`, where `exp` is a little-endian order integer
    /// exponent.
    ///
    /// # Guarantees
    ///
    /// **This operation is variable time with respect to `self`, for all exponent.** If
    /// the exponent is fixed, this operation is effectively constant time. However, for
    /// stronger constant-time guarantees, [`Field::pow`] should be used.
    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                res = res.square();

                if ((*e >> i) & 1) == 1 {
                    res.mul_assign(self);
                }
            }
        }

        res
    }
}

/// Operations for types that represent vectors of field elements.
///
/// This trait abstracts over:
/// - [`Field`] types (single field elements, which are trivially vectors of length 1)
/// - [`PackedField`](crate::PackedField) types (SIMD-accelerated vectors of field elements)
/// - Symbolic field types (for constraint system representations)
///
/// Mathematically, instances of this trait represent vectors of field elements where
/// arithmetic operations like addition, subtraction, multiplication, squaring, and
/// inversion are defined element-wise. For a packed field with width N, multiplying
/// two values performs N independent field multiplications in parallel.
///
/// # Required Methods
///
/// - [`zero()`](Self::zero) - Returns the additive identity (all elements are zero)
/// - [`one()`](Self::one) - Returns the multiplicative identity (all elements are one)
pub trait FieldOps:
    Clone
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Sum
    + Product
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Square
    + InvertOrZero
{
    type Scalar: Field;

    /// Returns the zero element (additive identity).
    fn zero() -> Self;

    /// Returns the one element (multiplicative identity).
    fn one() -> Self;

    /// Transpose the subfield elements in a slice of field elements.
    ///
    /// ## Arguments
    ///
    /// * `elems` - a slice of $n$ elements, where $n$ is the degee of the extension of
    ///   `Self::Scalar` over `FSub`. They are overwritten with the result elements.
    ///
    /// ## Preconditions
    ///
    /// * `elems.len()` must equal `<Self::Scalar as ExtensionField<FSub>>::DEGREE`
    fn square_transpose<FSub: Field>(elems: &mut [Self])
    where
        Self::Scalar: ExtensionField<FSub>;
}

impl<F: Field> FieldOps for F {
    type Scalar = F;

    fn zero() -> Self {
        Self::ZERO
    }

    fn one() -> Self {
        Self::ONE
    }

    fn square_transpose<FSub: Field>(elems: &mut [Self])
    where
        F: ExtensionField<FSub>,
    {
        <F as ExtensionField<FSub>>::square_transpose(elems)
    }
}
