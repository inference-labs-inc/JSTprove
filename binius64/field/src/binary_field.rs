// Copyright 2023-2025 Irreducible Inc.

use std::{
    fmt::{Debug, Display, Formatter},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
    DeserializeBytes, SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
};
use bytemuck::Zeroable;

use super::{
    UnderlierWithBitOps, WithUnderlier, binary_field_arithmetic::TowerFieldArithmetic,
    extension::ExtensionField,
};
use crate::{Field, underlier::U1};

/// A finite field with characteristic 2.
pub trait BinaryField:
    ExtensionField<BinaryField1b> + WithUnderlier<Underlier: UnderlierWithBitOps>
{
    const N_BITS: usize = Self::DEGREE;
}

/// A binary field *isomorphic* to a binary tower field.
///
/// The canonical binary field tower construction is specified in [DP23], section 2.3. This is a
/// family of binary fields with extension degree $2^{\iota}$ for any tower height $\iota$. This
/// trait can be implemented on any binary field *isomorphic* to the canonical tower field.
///
/// [DP23]: https://eprint.iacr.org/2023/1784
pub trait TowerField: BinaryField {
    /// The level $\iota$ in the tower, where this field is isomorphic to $T_{\iota}$.
    const TOWER_LEVEL: usize = Self::N_BITS.ilog2() as usize;

    /// Returns the smallest valid `TOWER_LEVEL` in the tower that can fit the same value.
    ///
    /// Since which `TOWER_LEVEL` values are valid depends on the tower,
    /// `F::Canonical::from(elem).min_tower_level()` can return a different result
    /// from `elem.min_tower_level()`.
    fn min_tower_level(self) -> usize;

    /// Returns the i'th basis element of this field as an extension over the tower subfield with
    /// level $\iota$.
    ///
    /// # Preconditions
    ///
    /// * `iota` must be at most `TOWER_LEVEL`.
    /// * `i` must be less than `2^(TOWER_LEVEL - iota)`.
    fn basis(iota: usize, i: usize) -> Self {
        assert!(
            iota <= Self::TOWER_LEVEL,
            "iota {iota} exceeds tower level {}",
            Self::TOWER_LEVEL
        );
        let n_basis_elts = 1 << (Self::TOWER_LEVEL - iota);
        assert!(
            i < n_basis_elts,
            "index {i} out of range for {n_basis_elts} basis elements"
        );
        <Self as ExtensionField<BinaryField1b>>::basis(i << iota)
    }

    /// Multiplies a field element by the canonical primitive element of the extension $T_{\iota +
    /// 1} / T_{iota}$.
    ///
    /// We represent the tower field $T_{\iota + 1}$ as a vector space over $T_{\iota}$ with the
    /// basis $\{1, \beta^{(\iota)}_1\}$. This operation multiplies the element by
    /// $\beta^{(\iota)}_1$.
    ///
    /// # Preconditions
    ///
    /// * `iota` must be less than `TOWER_LEVEL`.
    fn mul_primitive(self, iota: usize) -> Self {
        assert!(
            iota < Self::TOWER_LEVEL,
            "iota {iota} must be less than tower level {}",
            Self::TOWER_LEVEL
        );
        self * <Self as ExtensionField<BinaryField1b>>::basis(1 << iota)
    }
}

/// Returns the i'th basis element of `FExt` as a field extension of `FSub`.
///
/// This is an alias function for [`ExtensionField::basis`].
///
/// ## Pre-conditions
///
/// * `i` must be in the range $[0, d)$, where $d$ is the field extension degree.
#[inline]
pub fn ext_basis<FExt, FSub>(i: usize) -> FExt
where
    FSub: Field,
    FExt: ExtensionField<FSub>,
{
    <FExt as ExtensionField<FSub>>::basis(i)
}

pub(super) trait TowerExtensionField:
    TowerField
    + ExtensionField<Self::DirectSubfield>
    + From<(Self::DirectSubfield, Self::DirectSubfield)>
    + Into<(Self::DirectSubfield, Self::DirectSubfield)>
{
    type DirectSubfield: TowerField;
}

/// Macro to generate an implementation of a BinaryField.
macro_rules! binary_field {
	($vis:vis $name:ident($typ:ty), $gen:expr) => {
		#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, bytemuck::TransparentWrapper)]
		#[repr(transparent)]
		$vis struct $name(pub(crate) $typ);

		impl $name {
			pub const fn new(value: $typ) -> Self {
				Self(value)
			}

			pub const fn val(self) -> $typ {
				self.0
			}
		}

		unsafe impl $crate::underlier::WithUnderlier for $name {
			type Underlier = $typ;
		}

		impl Neg for $name {
			type Output = Self;

			fn neg(self) -> Self::Output {
				self
			}
		}

		impl Add<Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn add(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Add<&Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn add(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn sub(self, rhs: Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Sub<&Self> for $name {
			type Output = Self;

			#[allow(clippy::suspicious_arithmetic_impl)]
			fn sub(self, rhs: &Self) -> Self::Output {
				$name(self.0 ^ rhs.0)
			}
		}

		impl Mul<Self> for $name {
			type Output = Self;

			fn mul(self, rhs: Self) -> Self::Output {
				$crate::tracing::trace_multiplication!($name);

				TowerFieldArithmetic::multiply(self, rhs)
			}
		}

		impl Mul<&Self> for $name {
			type Output = Self;

			fn mul(self, rhs: &Self) -> Self::Output {
				self * *rhs
			}
		}

		impl AddAssign<Self> for $name {
			fn add_assign(&mut self, rhs: Self) {
				*self = *self + rhs;
			}
		}

		impl AddAssign<&Self> for $name {
			fn add_assign(&mut self, rhs: &Self) {
				*self = *self + *rhs;
			}
		}

		impl SubAssign<Self> for $name {
			fn sub_assign(&mut self, rhs: Self) {
				*self = *self - rhs;
			}
		}

		impl SubAssign<&Self> for $name {
			fn sub_assign(&mut self, rhs: &Self) {
				*self = *self - *rhs;
			}
		}

		impl MulAssign<Self> for $name {
			fn mul_assign(&mut self, rhs: Self) {
				*self = *self * rhs;
			}
		}

		impl MulAssign<&Self> for $name {
			fn mul_assign(&mut self, rhs: &Self) {
				*self = *self * rhs;
			}
		}

		impl Sum<Self> for $name {
			fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::ZERO, |acc, x| acc + x)
			}
		}

		impl<'a> Sum<&'a Self> for $name {
			fn sum<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
				iter.fold(Self::ZERO, |acc, x| acc + x)
			}
		}

		impl Product<Self> for $name {
			fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
				iter.fold(Self::ONE, |acc, x| acc * x)
			}
		}

		impl<'a> Product<&'a Self> for $name {
			fn product<I: Iterator<Item=&'a Self>>(iter: I) -> Self {
				iter.fold(Self::ONE, |acc, x| acc * x)
			}
		}


		impl crate::arithmetic_traits::Square for $name {
			fn square(self) -> Self {
				TowerFieldArithmetic::square(self)
			}
		}

		impl Field for $name {
			const ZERO: Self = $name::new(<$typ as $crate::underlier::UnderlierWithBitOps>::ZERO);
			const ONE: Self = $name::new(<$typ as $crate::underlier::UnderlierWithBitOps>::ONE);
			const CHARACTERISTIC: usize = 2;
			const MULTIPLICATIVE_GENERATOR: $name = $name($gen);

			fn double(&self) -> Self {
				Self::ZERO
			}
		}

		impl ::rand::distr::Distribution<$name> for ::rand::distr::StandardUniform {
			fn sample<R: ::rand::Rng + ?Sized>(&self, rng: &mut R) -> $name {
				$name(rng.random())
			}
		}

		impl Display for $name {
			fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
				write!(f, "0x{repr:0>width$x}", repr=self.val(), width=Self::N_BITS.max(4) / 4)
			}
		}

		impl Debug for $name {
			fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
				let structure_name = std::any::type_name::<$name>().split("::").last().expect("exist");

				write!(f, "{}({})",structure_name, self)
			}
		}

		impl BinaryField for $name {}

		impl From<$typ> for $name {
			fn from(val: $typ) -> Self {
				return Self(val)
			}
		}

		impl From<$name> for $typ {
			fn from(val: $name) -> Self {
				return val.0
			}
		}
	}
}

pub(crate) use binary_field;

macro_rules! mul_by_binary_field_1b {
    ($name:ident) => {
        impl Mul<BinaryField1b> for $name {
            type Output = Self;

            #[inline]
            #[allow(clippy::suspicious_arithmetic_impl)]
            fn mul(self, rhs: BinaryField1b) -> Self::Output {
                use $crate::underlier::{UnderlierWithBitOps, WithUnderlier};

                $crate::tracing::trace_multiplication!(BinaryField128b, BinaryField1b);

                Self(self.0 & <$name as WithUnderlier>::Underlier::fill_with_bit(u8::from(rhs.0)))
            }
        }
    };
}

pub(crate) use mul_by_binary_field_1b;

macro_rules! impl_field_extension {
    ($subfield_name:ident($subfield_typ:ty) < @$log_degree:expr => $name:ident($typ:ty)) => {
        impl TryFrom<$name> for $subfield_name {
            type Error = ();

            #[inline]
            fn try_from(elem: $name) -> Result<Self, Self::Error> {
                use $crate::underlier::NumCast;

                if elem.0 >> $subfield_name::N_BITS
                    == <$typ as $crate::underlier::UnderlierWithBitOps>::ZERO
                {
                    Ok($subfield_name::new(<$subfield_typ>::num_cast_from(
                        elem.val(),
                    )))
                } else {
                    Err(())
                }
            }
        }

        impl From<$subfield_name> for $name {
            #[inline]
            fn from(elem: $subfield_name) -> Self {
                $name::new(<$typ>::from(elem.val()))
            }
        }

        impl Add<$subfield_name> for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: $subfield_name) -> Self::Output {
                self + Self::from(rhs)
            }
        }

        impl Sub<$subfield_name> for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: $subfield_name) -> Self::Output {
                self - Self::from(rhs)
            }
        }

        impl AddAssign<$subfield_name> for $name {
            #[inline]
            fn add_assign(&mut self, rhs: $subfield_name) {
                *self = *self + rhs;
            }
        }

        impl SubAssign<$subfield_name> for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: $subfield_name) {
                *self = *self - rhs;
            }
        }

        impl MulAssign<$subfield_name> for $name {
            #[inline]
            fn mul_assign(&mut self, rhs: $subfield_name) {
                *self = *self * rhs;
            }
        }

        impl Add<$name> for $subfield_name {
            type Output = $name;

            #[inline]
            fn add(self, rhs: $name) -> Self::Output {
                rhs + self
            }
        }

        impl Sub<$name> for $subfield_name {
            type Output = $name;

            #[allow(clippy::suspicious_arithmetic_impl)]
            #[inline]
            fn sub(self, rhs: $name) -> Self::Output {
                rhs + self
            }
        }

        impl Mul<$name> for $subfield_name {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: $name) -> Self::Output {
                rhs * self
            }
        }

        impl ExtensionField<$subfield_name> for $name {
            const LOG_DEGREE: usize = $log_degree;

            #[inline]
            fn basis(i: usize) -> Self {
                use $crate::underlier::UnderlierWithBitOps;

                assert!(
                    i < 1 << $log_degree,
                    "index {} out of range for degree {}",
                    i,
                    1 << $log_degree
                );
                Self::new(<$typ>::ONE << (i * $subfield_name::N_BITS))
            }

            #[inline]
            fn from_bases_sparse(
                base_elems: impl IntoIterator<Item = $subfield_name>,
                log_stride: usize,
            ) -> Self {
                use $crate::underlier::UnderlierWithBitOps;

                debug_assert!($name::N_BITS.is_power_of_two());
                let shift_step = ($subfield_name::N_BITS << log_stride) & ($name::N_BITS - 1);
                let mut value = <$typ>::ZERO;
                let mut shift = 0;

                for elem in base_elems.into_iter() {
                    assert!(
                        shift < $name::N_BITS,
                        "too many base elements for extension degree"
                    );
                    value |= <$typ>::from(elem.val()) << shift;
                    shift += shift_step;
                }

                Self::new(value)
            }

            #[inline]
            fn iter_bases(&self) -> impl Iterator<Item = $subfield_name> {
                use binius_utils::iter::IterExtensions;
                use $crate::underlier::{Divisible, WithUnderlier};

                Divisible::<<$subfield_name as WithUnderlier>::Underlier>::ref_iter(&self.0)
                    .map_skippable($subfield_name::from)
            }

            #[inline]
            fn into_iter_bases(self) -> impl Iterator<Item = $subfield_name> {
                use binius_utils::iter::IterExtensions;
                use $crate::underlier::{Divisible, WithUnderlier};

                Divisible::<<$subfield_name as WithUnderlier>::Underlier>::value_iter(self.0)
                    .map_skippable($subfield_name::from)
            }

            #[inline]
            unsafe fn get_base_unchecked(&self, i: usize) -> $subfield_name {
                use $crate::underlier::{UnderlierWithBitOps, WithUnderlier};
                unsafe { $subfield_name::from_underlier(self.to_underlier().get_subvalue(i)) }
            }

            #[inline]
            fn square_transpose(values: &mut [Self]) {
                crate::transpose::square_transforms_extension_field::<$subfield_name, Self>(values)
            }
        }
    };
}

pub(crate) use impl_field_extension;

binary_field!(pub BinaryField1b(U1), U1::new(0x1));

macro_rules! serialize_deserialize {
    ($bin_type:ty) => {
        impl SerializeBytes for $bin_type {
            fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
                self.0.serialize(write_buf)
            }
        }

        impl DeserializeBytes for $bin_type {
            fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError> {
                Ok(Self(DeserializeBytes::deserialize(read_buf)?))
            }
        }
    };
}

serialize_deserialize!(BinaryField1b);

impl BinaryField1b {
    /// Creates value without checking that it is within valid range (0 or 1)
    ///
    /// # Safety
    /// Value should not exceed 1
    #[inline]
    pub unsafe fn new_unchecked(val: u8) -> Self {
        debug_assert!(val < 2, "val has to be less than 2, but it's {val}");

        Self::new(U1::new_unchecked(val))
    }
}

impl From<u8> for BinaryField1b {
    #[inline]
    fn from(val: u8) -> Self {
        Self::new(U1::new(val))
    }
}

impl From<BinaryField1b> for u8 {
    #[inline]
    fn from(value: BinaryField1b) -> Self {
        value.val().into()
    }
}

impl From<bool> for BinaryField1b {
    #[inline]
    fn from(value: bool) -> Self {
        Self::from(U1::new_unchecked(value.into()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use binius_utils::{DeserializeBytes, SerializeBytes, bytes::BytesMut};
    use proptest::prelude::*;

    use super::BinaryField1b as BF1;
    use crate::{AESTowerField8b, BinaryField, BinaryField1b, BinaryField128bGhash, Field};

    #[test]
    fn test_gf2_add() {
        assert_eq!(BF1::from(0) + BF1::from(0), BF1::from(0));
        assert_eq!(BF1::from(0) + BF1::from(1), BF1::from(1));
        assert_eq!(BF1::from(1) + BF1::from(0), BF1::from(1));
        assert_eq!(BF1::from(1) + BF1::from(1), BF1::from(0));
    }

    #[test]
    fn test_gf2_sub() {
        assert_eq!(BF1::from(0) - BF1::from(0), BF1::from(0));
        assert_eq!(BF1::from(0) - BF1::from(1), BF1::from(1));
        assert_eq!(BF1::from(1) - BF1::from(0), BF1::from(1));
        assert_eq!(BF1::from(1) - BF1::from(1), BF1::from(0));
    }

    #[test]
    fn test_gf2_mul() {
        assert_eq!(BF1::from(0) * BF1::from(0), BF1::from(0));
        assert_eq!(BF1::from(0) * BF1::from(1), BF1::from(0));
        assert_eq!(BF1::from(1) * BF1::from(0), BF1::from(0));
        assert_eq!(BF1::from(1) * BF1::from(1), BF1::from(1));
    }

    pub(crate) fn is_binary_field_valid_generator<F: BinaryField>() -> bool {
        // Binary fields should contain a multiplicative subgroup of size 2^n - 1
        let mut order = if F::N_BITS == 128 {
            u128::MAX
        } else {
            (1 << F::N_BITS) - 1
        };

        // Naive factorization of group order - represented as a multiset of prime factors
        let mut factorization = Vec::new();

        let mut prime = 2;
        while prime * prime <= order {
            while order.is_multiple_of(prime) {
                order /= prime;
                factorization.push(prime);
            }

            prime += if prime > 2 { 2 } else { 1 };
        }

        if order > 1 {
            factorization.push(order);
        }

        // Iterate over all divisors (some may be tested several times if order is non-square-free)
        for mask in 0..(1 << factorization.len()) {
            let mut divisor = 1;

            for (bit_index, &prime) in factorization.iter().enumerate() {
                if (1 << bit_index) & mask != 0 {
                    divisor *= prime;
                }
            }

            // Compute pow(generator, divisor) in log time
            divisor = divisor.reverse_bits();

            let mut pow_divisor = F::ONE;
            while divisor > 0 {
                pow_divisor *= pow_divisor;

                if divisor & 1 != 0 {
                    pow_divisor *= F::MULTIPLICATIVE_GENERATOR;
                }

                divisor >>= 1;
            }

            // Generator invariant
            let is_root_of_unity = pow_divisor == F::ONE;
            let is_full_group = mask + 1 == 1 << factorization.len();

            if is_root_of_unity && !is_full_group || !is_root_of_unity && is_full_group {
                return false;
            }
        }

        true
    }

    #[test]
    fn test_multiplicative_generators() {
        assert!(is_binary_field_valid_generator::<BinaryField1b>());
        assert!(is_binary_field_valid_generator::<AESTowerField8b>());
        assert!(is_binary_field_valid_generator::<BinaryField128bGhash>());
    }

    #[test]
    fn test_field_degrees() {
        assert_eq!(BinaryField1b::N_BITS, 1);
        assert_eq!(AESTowerField8b::N_BITS, 8);
        assert_eq!(BinaryField128bGhash::N_BITS, 128);
    }

    #[test]
    fn test_field_formatting() {
        assert_eq!(format!("{}", BinaryField1b::from(1)), "0x1");
        assert_eq!(format!("{}", AESTowerField8b::from(3)), "0x03");
        assert_eq!(
            format!("{}", BinaryField128bGhash::from(5)),
            "0x00000000000000000000000000000005"
        );
    }

    #[test]
    fn test_inverse_on_zero() {
        assert!(BinaryField1b::ZERO.invert().is_none());
        assert!(AESTowerField8b::ZERO.invert().is_none());
        assert!(BinaryField128bGhash::ZERO.invert().is_none());
    }

    proptest! {
        #[test]
        fn test_inverse_8b(val in 1u8..) {
            let x = AESTowerField8b::new(val);
            let x_inverse = x.invert().unwrap();
            assert_eq!(x * x_inverse, AESTowerField8b::ONE);
        }

        #[test]
        fn test_inverse_128b(val in 1u128..) {
            let x = BinaryField128bGhash::new(val);
            let x_inverse = x.invert().unwrap();
            assert_eq!(x * x_inverse, BinaryField128bGhash::ONE);
        }
    }

    #[test]
    fn test_serialization() {
        let mut buffer = BytesMut::new();
        let b1 = BinaryField1b::from(0x1);
        let b8 = AESTowerField8b::new(0x12);
        let b128 = BinaryField128bGhash::new(0x147AD0369CF258BE8899AABBCCDDEEFF);

        b1.serialize(&mut buffer).unwrap();
        b8.serialize(&mut buffer).unwrap();
        b128.serialize(&mut buffer).unwrap();

        let mut read_buffer = buffer.freeze();

        assert_eq!(BinaryField1b::deserialize(&mut read_buffer).unwrap(), b1);
        assert_eq!(AESTowerField8b::deserialize(&mut read_buffer).unwrap(), b8);
        assert_eq!(
            BinaryField128bGhash::deserialize(&mut read_buffer).unwrap(),
            b128
        );
    }

    #[test]
    fn test_gf2_new_unchecked() {
        for i in 0..2 {
            assert_eq!(unsafe { BF1::new_unchecked(i) }, BF1::from(i));
        }
    }
}
