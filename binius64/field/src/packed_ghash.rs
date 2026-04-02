// Copyright 2024-2025 Irreducible Inc.

pub use crate::arch::{
    packed_ghash_128::PackedBinaryGhash1x128b, packed_ghash_256::PackedBinaryGhash2x128b,
    packed_ghash_512::PackedBinaryGhash4x128b,
};

#[cfg(test)]
mod test_utils {
    /// Test if `mult_func` operation is a valid multiply operation on the given values for
    /// all possible packed fields defined on 8-512 bits.
    macro_rules! define_multiply_tests {
        ($mult_func:path, $constraint:ty) => {
            $crate::packed_binary_field::test_utils::define_check_packed_mul!(
                $mult_func,
                $constraint
            );

            proptest! {
                #[test]
                fn test_mul_packed_128(a_val in any::<u128>(), b_val in any::<u128>()) {
                    TestMult::<$crate::arch::packed_ghash_128::PackedBinaryGhash1x128b>::test_mul(
                        a_val.into(),
                        b_val.into(),
                    );
                }

                #[test]
                fn test_mul_packed_256(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
                    TestMult::<$crate::arch::packed_ghash_256::PackedBinaryGhash2x128b>::test_mul(
                        a_val.into(),
                        b_val.into(),
                    );
                }

                #[test]
                fn test_mul_packed_512(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
                    TestMult::<$crate::arch::packed_ghash_512::PackedBinaryGhash4x128b>::test_mul(
                        a_val.into(),
                        b_val.into(),
                    );
                }
            }
        };
    }

    /// Test if `square_func` operation is a valid square operation on the given value for
    /// all possible packed fields.
    macro_rules! define_square_tests {
		($square_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_square!(
				$square_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_square_packed_128(a_val in any::<u128>()) {
					TestSquare::<$crate::arch::packed_ghash_128::PackedBinaryGhash1x128b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_256(a_val in any::<[u128; 2]>()) {
					TestSquare::<$crate::arch::packed_ghash_256::PackedBinaryGhash2x128b>::test_square(a_val.into());
				}

				#[test]
				fn test_square_packed_512(a_val in any::<[u128; 4]>()) {
					TestSquare::<$crate::arch::packed_ghash_512::PackedBinaryGhash4x128b>::test_square(a_val.into());
				}
			}
		};
	}

    /// Test if `invert_func` operation is a valid invert operation on the given value for
    /// all possible packed fields.
    macro_rules! define_invert_tests {
		($invert_func:path, $constraint:ident) => {
			$crate::packed_binary_field::test_utils::define_check_packed_inverse!(
				$invert_func,
				$constraint
			);

			proptest! {
				#[test]
				fn test_invert_packed_128(a_val in any::<u128>()) {
					TestInvert::<$crate::arch::packed_ghash_128::PackedBinaryGhash1x128b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_256(a_val in any::<[u128; 2]>()) {
					TestInvert::<$crate::arch::packed_ghash_256::PackedBinaryGhash2x128b>::test_invert(a_val.into());
				}

				#[test]
				fn test_invert_packed_512(a_val in any::<[u128; 4]>()) {
					TestInvert::<$crate::arch::packed_ghash_512::PackedBinaryGhash4x128b>::test_invert(a_val.into());
				}
			}
		};
	}

    pub(crate) use define_invert_tests;
    pub(crate) use define_multiply_tests;
    pub(crate) use define_square_tests;
}

#[cfg(test)]
mod tests {
    use std::ops::Mul;

    use proptest::{arbitrary::any, proptest};

    use super::test_utils::{define_invert_tests, define_multiply_tests, define_square_tests};
    use crate::{
        BinaryField128bGhash, PackedField,
        arch::{
            packed_ghash_256::PackedBinaryGhash2x128b, packed_ghash_512::PackedBinaryGhash4x128b,
        },
        arithmetic_traits::{InvertOrZero, Square},
        underlier::WithUnderlier,
    };

    fn check_get_set<const WIDTH: usize, PT>(a: [u128; WIDTH], b: [u128; WIDTH])
    where
        PT: PackedField<Scalar = BinaryField128bGhash>
            + WithUnderlier<Underlier: From<[u128; WIDTH]>>,
    {
        let mut val = PT::from_underlier(a.into());
        for i in 0..WIDTH {
            assert_eq!(val.get(i), BinaryField128bGhash::from(a[i]));
            val.set(i, BinaryField128bGhash::from(b[i]));
            assert_eq!(val.get(i), BinaryField128bGhash::from(b[i]));
        }
    }

    proptest! {
        #[test]
        fn test_get_set_256(a in any::<[u128; 2]>(), b in any::<[u128; 2]>()) {
            check_get_set::<2, PackedBinaryGhash2x128b>(a, b);
        }

        #[test]
        fn test_get_set_512(a in any::<[u128; 4]>(), b in any::<[u128; 4]>()) {
            check_get_set::<4, PackedBinaryGhash4x128b>(a, b);
        }
    }

    define_multiply_tests!(Mul::mul, PackedField);

    define_square_tests!(Square::square, PackedField);

    define_invert_tests!(InvertOrZero::invert_or_zero, PackedField);
}
