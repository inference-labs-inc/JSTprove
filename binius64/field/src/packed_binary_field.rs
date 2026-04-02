// Copyright 2023-2025 Irreducible Inc.

pub use crate::arch::{
    packed_1::*, packed_2::*, packed_4::*, packed_8::*, packed_16::*, packed_32::*, packed_64::*,
    packed_128::*, packed_256::*, packed_512::*, packed_aes_8::*, packed_aes_16::*,
    packed_aes_32::*, packed_aes_64::*, packed_aes_128::*, packed_aes_256::*, packed_aes_512::*,
};

/// Common code to test different multiply, square and invert implementations
#[cfg(test)]
pub mod test_utils {
    use crate::{
        PackedField,
        underlier::{U1, U2, U4, WithUnderlier},
    };

    pub struct Unit;

    impl From<U1> for Unit {
        fn from(_: U1) -> Self {
            Self
        }
    }

    impl From<U2> for Unit {
        fn from(_: U2) -> Self {
            Self
        }
    }

    impl From<U4> for Unit {
        fn from(_: U4) -> Self {
            Self
        }
    }

    impl From<u8> for Unit {
        fn from(_: u8) -> Self {
            Self
        }
    }

    impl From<u16> for Unit {
        fn from(_: u16) -> Self {
            Self
        }
    }

    impl From<u32> for Unit {
        fn from(_: u32) -> Self {
            Self
        }
    }

    impl From<u64> for Unit {
        fn from(_: u64) -> Self {
            Self
        }
    }

    impl From<u128> for Unit {
        fn from(_: u128) -> Self {
            Self
        }
    }

    impl From<[u128; 2]> for Unit {
        fn from(_: [u128; 2]) -> Self {
            Self
        }
    }

    impl From<[u128; 4]> for Unit {
        fn from(_: [u128; 4]) -> Self {
            Self
        }
    }

    /// We use such helper macros to run tests only for the
    /// types that implement the `constraint` trait.
    /// The idea is inspired by `impls` trait.
    macro_rules! define_check_packed_mul {
        ($mult_func:path, $constraint:path) => {
            #[allow(unused)]
            trait TestMulTrait<T> {
                fn test_mul(_a: T, _b: T) {}
            }

            impl<T> TestMulTrait<$crate::packed_binary_field::test_utils::Unit> for T {}

            struct TestMult<T>(std::marker::PhantomData<T>);

            impl<T: $constraint + PackedField + $crate::underlier::WithUnderlier> TestMult<T> {
                #[allow(unused)]
                fn test_mul(
                    a: <T as $crate::underlier::WithUnderlier>::Underlier,
                    b: <T as $crate::underlier::WithUnderlier>::Underlier,
                ) {
                    let a = T::from_underlier(a);
                    let b = T::from_underlier(b);

                    let c = $mult_func(a, b);
                    for i in 0..T::WIDTH {
                        assert_eq!(c.get(i), a.get(i) * b.get(i));
                    }
                }
            }
        };
    }

    pub(crate) use define_check_packed_mul;

    macro_rules! define_check_packed_square {
        ($square_func:path, $constraint:path) => {
            #[allow(unused)]
            trait TestSquareTrait<T> {
                fn test_square(_a: T) {}
            }

            impl<T> TestSquareTrait<$crate::packed_binary_field::test_utils::Unit> for T {}

            struct TestSquare<T>(std::marker::PhantomData<T>);

            impl<T: $constraint + PackedField + $crate::underlier::WithUnderlier> TestSquare<T> {
                #[allow(unused)]
                fn test_square(a: <T as $crate::underlier::WithUnderlier>::Underlier) {
                    let a = T::from_underlier(a);

                    let c = $square_func(a);
                    for i in 0..T::WIDTH {
                        assert_eq!(c.get(i), a.get(i) * a.get(i));
                    }
                }
            }
        };
    }

    pub(crate) use define_check_packed_square;

    macro_rules! define_check_packed_inverse {
        ($invert_func:path, $constraint:path) => {
            #[allow(unused)]
            trait TestInvertTrait<T> {
                fn test_invert(_a: T) {}
            }

            impl<T> TestInvertTrait<$crate::packed_binary_field::test_utils::Unit> for T {}

            struct TestInvert<T>(std::marker::PhantomData<T>);

            #[allow(unused)]
            impl<T: $constraint + PackedField + $crate::underlier::WithUnderlier> TestInvert<T> {
                fn test_invert(a: <T as $crate::underlier::WithUnderlier>::Underlier) {
                    use crate::Field;

                    let a = T::from_underlier(a);

                    let c = $invert_func(a);
                    for i in 0..T::WIDTH {
                        assert!(
                            (c.get(i).is_zero().into()
                                && a.get(i).is_zero().into()
                                && c.get(i).is_zero().into())
                                || T::Scalar::ONE == a.get(i) * c.get(i)
                        );
                    }
                }
            }
        };
    }

    pub(crate) use define_check_packed_inverse;

    /// Test if `mult_func` operation is a valid multiply operation on the given values for
    /// all possible packed fields defined on u128.
    macro_rules! define_multiply_tests {
		($mult_func:path, $constraint:path) => {
			$crate::packed_binary_field::test_utils::define_check_packed_mul!(
				$mult_func,
				$constraint
			);

			proptest::proptest! {
				#[test]
				fn test_mul_packed_8(a_val in proptest::prelude::any::<u8>(), b_val in proptest::prelude::any::<u8>()) {
					use $crate::arch::packed_8::*;
					use $crate::arch::packed_aes_8::*;

					TestMult::<PackedBinaryField8x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField1x8b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_16(a_val in proptest::prelude::any::<u16>(), b_val in proptest::prelude::any::<u16>()) {
					use $crate::arch::packed_16::*;
					use $crate::arch::packed_aes_16::*;

					TestMult::<PackedBinaryField16x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField2x8b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_32(a_val in proptest::prelude::any::<u32>(), b_val in proptest::prelude::any::<u32>()) {
					use $crate::arch::packed_32::*;
					use $crate::arch::packed_aes_32::*;

					TestMult::<PackedBinaryField32x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField4x8b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_64(a_val in proptest::prelude::any::<u64>(), b_val in proptest::prelude::any::<u64>()) {
					use $crate::arch::packed_64::*;
					use $crate::arch::packed_aes_64::*;

					TestMult::<PackedBinaryField64x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField8x8b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_128(a_val in proptest::prelude::any::<u128>(), b_val in proptest::prelude::any::<u128>()) {
					use $crate::arch::packed_128::*;
					use $crate::arch::packed_aes_128::*;
					use $crate::arch::packed_ghash_128::*;

					TestMult::<PackedBinaryField128x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField16x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryGhash1x128b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_256(a_val in proptest::prelude::any::<[u128; 2]>(), b_val in proptest::prelude::any::<[u128; 2]>()) {
					use $crate::arch::packed_256::*;
					use $crate::arch::packed_aes_256::*;
					use $crate::arch::packed_ghash_256::*;

					TestMult::<PackedBinaryField256x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField32x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryGhash2x128b>::test_mul(a_val.into(), b_val.into());
				}

				#[test]
				fn test_mul_packed_512(a_val in proptest::prelude::any::<[u128; 4]>(), b_val in proptest::prelude::any::<[u128; 4]>()) {
					use $crate::arch::packed_512::*;
					use $crate::arch::packed_aes_512::*;
					use $crate::arch::packed_ghash_512::*;

					TestMult::<PackedBinaryField512x1b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedAESBinaryField64x8b>::test_mul(a_val.into(), b_val.into());
					TestMult::<PackedBinaryGhash4x128b>::test_mul(a_val.into(), b_val.into());
				}
			}
		};
	}

    /// Test if `square_func` operation is a valid square operation on the given value for
    /// all possible packed fields.
    macro_rules! define_square_tests {
        ($square_func:path, $constraint:path) => {
            $crate::packed_binary_field::test_utils::define_check_packed_square!(
                $square_func,
                $constraint
            );

            proptest::proptest! {
                #[test]
                fn test_square_packed_8(a_val in proptest::prelude::any::<u8>()) {
                    use $crate::arch::packed_8::*;
                    use $crate::arch::packed_aes_8::*;

                    TestSquare::<PackedBinaryField8x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField1x8b>::test_square(a_val.into());
                }

                #[test]
                fn test_square_packed_16(a_val in proptest::prelude::any::<u16>()) {
                    use $crate::arch::packed_16::*;
                    use $crate::arch::packed_aes_16::*;

                    TestSquare::<PackedBinaryField16x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField2x8b>::test_square(a_val.into());
                }

                #[test]
                fn test_square_packed_32(a_val in proptest::prelude::any::<u32>()) {
                    use $crate::arch::packed_32::*;
                    use $crate::arch::packed_aes_32::*;

                    TestSquare::<PackedBinaryField32x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField4x8b>::test_square(a_val.into());
                }

                #[test]
                fn test_square_packed_64(a_val in proptest::prelude::any::<u64>()) {
                    use $crate::arch::packed_64::*;
                    use $crate::arch::packed_aes_64::*;

                    TestSquare::<PackedBinaryField64x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField8x8b>::test_square(a_val.into());
                }

                #[test]
                fn test_square_packed_128(a_val in proptest::prelude::any::<u128>()) {
                    use $crate::arch::packed_128::*;
                    use $crate::arch::packed_aes_128::*;
                    use $crate::arch::packed_ghash_128::*;

                    TestSquare::<PackedBinaryField128x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField16x8b>::test_square(a_val.into());
                    TestSquare::<PackedBinaryGhash1x128b>::test_square(a_val.into());
                }

                #[test]
                fn test_square_packed_256(a_val in proptest::prelude::any::<[u128; 2]>()) {
                    use $crate::arch::packed_256::*;
                    use $crate::arch::packed_aes_256::*;
                    use $crate::arch::packed_ghash_256::*;

                    TestSquare::<PackedBinaryField256x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField32x8b>::test_square(a_val.into());
                    TestSquare::<PackedBinaryGhash2x128b>::test_square(a_val.into());
                }

                #[test]
                fn test_square_packed_512(a_val in proptest::prelude::any::<[u128; 4]>()) {
                    use $crate::arch::packed_512::*;
                    use $crate::arch::packed_aes_512::*;
                    use $crate::arch::packed_ghash_512::*;

                    TestSquare::<PackedBinaryField512x1b>::test_square(a_val.into());
                    TestSquare::<PackedAESBinaryField64x8b>::test_square(a_val.into());
                    TestSquare::<PackedBinaryGhash4x128b>::test_square(a_val.into());
                }
            }
        };
    }

    /// Test if `invert_func` operation is a valid invert operation on the given value for
    /// all possible packed fields.
    macro_rules! define_invert_tests {
        ($invert_func:path, $constraint:path) => {
            $crate::packed_binary_field::test_utils::define_check_packed_inverse!(
                $invert_func,
                $constraint
            );

            proptest::proptest! {
                #[test]
                fn test_invert_packed_8(a_val in proptest::prelude::any::<u8>()) {
                    use $crate::arch::packed_8::*;
                    use $crate::arch::packed_aes_8::*;

                    TestInvert::<PackedBinaryField8x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField1x8b>::test_invert(a_val.into());
                }

                #[test]
                fn test_invert_packed_16(a_val in proptest::prelude::any::<u16>()) {
                    use $crate::arch::packed_16::*;
                    use $crate::arch::packed_aes_16::*;

                    TestInvert::<PackedBinaryField16x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField2x8b>::test_invert(a_val.into());
                }

                #[test]
                fn test_invert_packed_32(a_val in proptest::prelude::any::<u32>()) {
                    use $crate::arch::packed_32::*;
                    use $crate::arch::packed_aes_32::*;

                    TestInvert::<PackedBinaryField32x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField4x8b>::test_invert(a_val.into());
                }

                #[test]
                fn test_invert_packed_64(a_val in proptest::prelude::any::<u64>()) {
                    use $crate::arch::packed_64::*;
                    use $crate::arch::packed_aes_64::*;

                    TestInvert::<PackedBinaryField64x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField8x8b>::test_invert(a_val.into());
                }

                #[test]
                fn test_invert_packed_128(a_val in proptest::prelude::any::<u128>()) {
                    use $crate::arch::packed_128::*;
                    use $crate::arch::packed_aes_128::*;
                    use $crate::arch::packed_ghash_128::*;

                    TestInvert::<PackedBinaryField128x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField16x8b>::test_invert(a_val.into());
                    TestInvert::<PackedBinaryGhash1x128b>::test_invert(a_val.into());
                }

                #[test]
                fn test_invert_packed_256(a_val in proptest::prelude::any::<[u128; 2]>()) {
                    use $crate::arch::packed_256::*;
                    use $crate::arch::packed_aes_256::*;
                    use $crate::arch::packed_ghash_256::*;

                    TestInvert::<PackedBinaryField256x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField32x8b>::test_invert(a_val.into());
                    TestInvert::<PackedBinaryGhash2x128b>::test_invert(a_val.into());
                }

                #[test]
                fn test_invert_packed_512(a_val in proptest::prelude::any::<[u128; 4]>()) {
                    use $crate::arch::packed_512::*;
                    use $crate::arch::packed_aes_512::*;
                    use $crate::arch::packed_ghash_512::*;

                    TestInvert::<PackedBinaryField512x1b>::test_invert(a_val.into());
                    TestInvert::<PackedAESBinaryField64x8b>::test_invert(a_val.into());
                    TestInvert::<PackedBinaryGhash4x128b>::test_invert(a_val.into());
                }
            }
        };
    }

    pub(crate) use define_invert_tests;
    pub(crate) use define_multiply_tests;
    pub(crate) use define_square_tests;

    pub fn check_interleave<P: PackedField + WithUnderlier>(
        lhs: P::Underlier,
        rhs: P::Underlier,
        log_block_len: usize,
    ) {
        let lhs = P::from_underlier(lhs);
        let rhs = P::from_underlier(rhs);
        let (a, b) = lhs.interleave(rhs, log_block_len);
        let block_len = 1 << log_block_len;
        for i in (0..P::WIDTH).step_by(block_len * 2) {
            for j in 0..block_len {
                assert_eq!(a.get(i + j), lhs.get(i + j));
                assert_eq!(a.get(i + j + block_len), rhs.get(i + j));

                assert_eq!(b.get(i + j), lhs.get(i + j + block_len));
                assert_eq!(b.get(i + j + block_len), rhs.get(i + j + block_len));
            }
        }
    }

    pub fn check_interleave_all_heights<P: PackedField + WithUnderlier>(
        lhs: P::Underlier,
        rhs: P::Underlier,
    ) {
        for log_block_len in 0..P::LOG_WIDTH {
            check_interleave::<P>(lhs, rhs, log_block_len);
        }
    }

    pub fn check_unzip<P: PackedField + WithUnderlier>(
        lhs: P::Underlier,
        rhs: P::Underlier,
        log_block_len: usize,
    ) {
        let lhs = P::from_underlier(lhs);
        let rhs = P::from_underlier(rhs);
        let block_len = 1 << log_block_len;
        let (a, b) = lhs.unzip(rhs, log_block_len);
        for i in (0..P::WIDTH / 2).step_by(block_len) {
            for j in 0..block_len {
                assert_eq!(
                    a.get(i + j),
                    lhs.get(2 * i + j),
                    "i: {}, j: {}, log_block_len: {}, P: {:?}",
                    i,
                    j,
                    log_block_len,
                    P::zero()
                );
                assert_eq!(
                    b.get(i + j),
                    lhs.get(2 * i + j + block_len),
                    "i: {}, j: {}, log_block_len: {}, P: {:?}",
                    i,
                    j,
                    log_block_len,
                    P::zero()
                );
            }
        }

        for i in (0..P::WIDTH / 2).step_by(block_len) {
            for j in 0..block_len {
                assert_eq!(
                    a.get(i + j + P::WIDTH / 2),
                    rhs.get(2 * i + j),
                    "i: {}, j: {}, log_block_len: {}, P: {:?}",
                    i,
                    j,
                    log_block_len,
                    P::zero()
                );
                assert_eq!(b.get(i + j + P::WIDTH / 2), rhs.get(2 * i + j + block_len));
            }
        }
    }

    pub fn check_transpose_all_heights<P: PackedField + WithUnderlier>(
        lhs: P::Underlier,
        rhs: P::Underlier,
    ) {
        for log_block_len in 0..P::LOG_WIDTH {
            check_unzip::<P>(lhs, rhs, log_block_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::repeat_with, ops::Mul};

    use binius_utils::{DeserializeBytes, SerializeBytes, bytes::BytesMut};
    use proptest::prelude::*;
    use rand::prelude::*;
    use test_utils::check_interleave_all_heights;

    use super::{
        test_utils::{define_invert_tests, define_multiply_tests, define_square_tests},
        *,
    };
    use crate::{
        PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
        Random,
        arithmetic_traits::{InvertOrZero, Square},
        test_utils::check_transpose_all_heights,
        underlier::{U2, U4},
    };

    fn test_add_packed<P: PackedField + From<u128>>(a_val: u128, b_val: u128) {
        let a = P::from(a_val);
        let b = P::from(b_val);
        let c = a + b;
        for i in 0..P::WIDTH {
            assert_eq!(c.get(i), a.get(i) + b.get(i));
        }
    }

    fn test_mul_packed<P: PackedField>(a: P, b: P) {
        let c = a * b;
        for i in 0..P::WIDTH {
            assert_eq!(c.get(i), a.get(i) * b.get(i));
        }
    }

    fn test_mul_packed_random<P: PackedField>() {
        let mut rng = StdRng::seed_from_u64(0);
        test_mul_packed(P::random(&mut rng), P::random(&mut rng))
    }

    fn test_set_then_get<P: PackedField>() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut elem = P::random(&mut rng);

        let scalars = repeat_with(|| P::Scalar::random(&mut rng))
            .take(P::WIDTH)
            .collect::<Vec<_>>();

        for (i, val) in scalars.iter().enumerate() {
            elem.set(i, *val);
        }
        for (i, val) in scalars.iter().enumerate() {
            assert_eq!(elem.get(i), *val);
        }
    }

    fn test_serialize_then_deserialize<P: PackedField + DeserializeBytes + SerializeBytes>() {
        let mut buffer = BytesMut::new();
        let mut rng = StdRng::seed_from_u64(0);
        let packed = P::random(&mut rng);
        packed.serialize(&mut buffer).unwrap();

        let mut read_buffer = buffer.freeze();

        assert_eq!(P::deserialize(&mut read_buffer).unwrap(), packed);
    }

    #[test]
    fn test_set_then_get_128b() {
        test_set_then_get::<PackedBinaryGhash1x128b>();
        test_set_then_get::<PackedBinaryGhash2x128b>();
        test_set_then_get::<PackedBinaryGhash4x128b>();
    }

    #[test]
    fn test_serialize_then_deserialize_128b() {
        test_serialize_then_deserialize::<PackedBinaryGhash1x128b>();
        test_serialize_then_deserialize::<PackedBinaryGhash2x128b>();
        test_serialize_then_deserialize::<PackedBinaryGhash4x128b>();
    }

    #[test]
    fn test_serialize_deserialize_different_packing_width() {
        let mut rng = StdRng::seed_from_u64(0);

        let packed0 = PackedBinaryGhash1x128b::random(&mut rng);
        let packed1 = PackedBinaryGhash1x128b::random(&mut rng);

        let mut buffer = BytesMut::new();
        packed0.serialize(&mut buffer).unwrap();
        packed1.serialize(&mut buffer).unwrap();

        let mut read_buffer = buffer.freeze();
        let packed01 = PackedBinaryGhash2x128b::deserialize(&mut read_buffer).unwrap();

        assert!(
            packed01
                .iter()
                .zip([packed0, packed1])
                .all(|(x, y)| x == y.get(0))
        );
    }

    // TODO: Generate lots more proptests using macros
    proptest! {
        #[test]
        fn test_add_packed_128x1b(a_val in any::<u128>(), b_val in any::<u128>()) {
            test_add_packed::<PackedBinaryField128x1b>(a_val, b_val)
        }

        #[test]
        fn test_add_packed_16x8b(a_val in any::<u128>(), b_val in any::<u128>()) {
            test_add_packed::<PackedAESBinaryField16x8b>(a_val, b_val)
        }

        #[test]
        fn test_add_packed_1x128b(a_val in any::<u128>(), b_val in any::<u128>()) {
            test_add_packed::<PackedBinaryGhash1x128b>(a_val, b_val)
        }
    }

    #[test]
    fn test_mul_packed_256x1b() {
        test_mul_packed_random::<PackedBinaryField256x1b>()
    }

    #[test]
    fn test_mul_packed_32x8b() {
        test_mul_packed_random::<PackedAESBinaryField32x8b>()
    }

    #[test]
    fn test_mul_packed_2x128b() {
        test_mul_packed_random::<PackedBinaryGhash2x128b>()
    }

    #[test]
    fn test_iter_size_hint() {
        assert_valid_iterator_with_exact_size_hint::<PackedBinaryField128x1b>();
    }

    fn assert_valid_iterator_with_exact_size_hint<P: PackedField>() {
        assert_eq!(P::default().iter().size_hint(), (P::WIDTH, Some(P::WIDTH)));
        assert_eq!(
            P::default().into_iter().size_hint(),
            (P::WIDTH, Some(P::WIDTH))
        );
        assert_eq!(P::default().iter().count(), P::WIDTH);
        assert_eq!(P::default().into_iter().count(), P::WIDTH);
    }

    define_multiply_tests!(Mul::mul, PackedField);

    define_square_tests!(Square::square, PackedField);

    define_invert_tests!(InvertOrZero::invert_or_zero, PackedField);

    proptest! {
        #[test]
        fn test_interleave_2b(a_val in 0u8..3, b_val in 0u8..3) {
            check_interleave_all_heights::<PackedBinaryField2x1b>(U2::new(a_val), U2::new(b_val));
        }

        #[test]
        fn test_interleave_4b(a_val in 0u8..16, b_val in 0u8..16) {
            check_interleave_all_heights::<PackedBinaryField4x1b>(U4::new(a_val), U4::new(b_val));
        }

        #[test]
        fn test_interleave_8b(a_val in 0u8.., b_val in 0u8..) {
            check_interleave_all_heights::<PackedBinaryField8x1b>(a_val, b_val);
            check_interleave_all_heights::<PackedAESBinaryField1x8b>(a_val, b_val);
        }

        #[test]
        fn test_interleave_16b(a_val in 0u16.., b_val in 0u16..) {
            check_interleave_all_heights::<PackedBinaryField16x1b>(a_val, b_val);
            check_interleave_all_heights::<PackedAESBinaryField2x8b>(a_val, b_val);
        }

        #[test]
        fn test_interleave_32b(a_val in 0u32.., b_val in 0u32..) {
            check_interleave_all_heights::<PackedBinaryField32x1b>(a_val, b_val);
            check_interleave_all_heights::<PackedAESBinaryField4x8b>(a_val, b_val);
        }

        #[test]
        fn test_interleave_64b(a_val in 0u64.., b_val in 0u64..) {
            check_interleave_all_heights::<PackedBinaryField64x1b>(a_val, b_val);
            check_interleave_all_heights::<PackedAESBinaryField8x8b>(a_val, b_val);
        }

        #[test]
        #[allow(clippy::useless_conversion)] // this warning depends on the target platform
        fn test_interleave_128b(a_val in 0u128.., b_val in 0u128..) {
            check_interleave_all_heights::<PackedBinaryField128x1b>(a_val.into(), b_val.into());
            check_interleave_all_heights::<PackedAESBinaryField16x8b>(a_val.into(), b_val.into());
            check_interleave_all_heights::<PackedBinaryGhash1x128b>(a_val.into(), b_val.into());
        }

        #[test]
        fn test_interleave_256b(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
            check_interleave_all_heights::<PackedBinaryField256x1b>(a_val.into(), b_val.into());
            check_interleave_all_heights::<PackedAESBinaryField32x8b>(a_val.into(), b_val.into());
            check_interleave_all_heights::<PackedBinaryGhash2x128b>(a_val.into(), b_val.into());
        }

        #[test]
        fn test_interleave_512b(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
            check_interleave_all_heights::<PackedBinaryField512x1b>(a_val.into(), b_val.into());
            check_interleave_all_heights::<PackedAESBinaryField64x8b>(a_val.into(), b_val.into());
            check_interleave_all_heights::<PackedBinaryGhash4x128b>(a_val.into(), b_val.into());
        }

        #[test]
        fn check_transpose_2b(a_val in 0u8..3, b_val in 0u8..3) {
            check_transpose_all_heights::<PackedBinaryField2x1b>(U2::new(a_val), U2::new(b_val));
        }

        #[test]
        fn check_transpose_4b(a_val in 0u8..16, b_val in 0u8..16) {
            check_transpose_all_heights::<PackedBinaryField4x1b>(U4::new(a_val), U4::new(b_val));
        }

        #[test]
        fn check_transpose_8b(a_val in 0u8.., b_val in 0u8..) {
            check_transpose_all_heights::<PackedBinaryField8x1b>(a_val, b_val);
            check_transpose_all_heights::<PackedAESBinaryField1x8b>(a_val, b_val);
        }

        #[test]
        fn check_transpose_16b(a_val in 0u16.., b_val in 0u16..) {
            check_transpose_all_heights::<PackedBinaryField16x1b>(a_val, b_val);
            check_transpose_all_heights::<PackedAESBinaryField2x8b>(a_val, b_val);
        }

        #[test]
        fn check_transpose_32b(a_val in 0u32.., b_val in 0u32..) {
            check_transpose_all_heights::<PackedBinaryField32x1b>(a_val, b_val);
            check_transpose_all_heights::<PackedAESBinaryField4x8b>(a_val, b_val);
        }

        #[test]
        fn check_transpose_64b(a_val in 0u64.., b_val in 0u64..) {
            check_transpose_all_heights::<PackedBinaryField64x1b>(a_val, b_val);
            check_transpose_all_heights::<PackedAESBinaryField8x8b>(a_val, b_val);
        }

        #[test]
        #[allow(clippy::useless_conversion)] // this warning depends on the target platform
        fn check_transpose_128b(a_val in 0u128.., b_val in 0u128..) {
            check_transpose_all_heights::<PackedBinaryField128x1b>(a_val.into(), b_val.into());
            check_transpose_all_heights::<PackedAESBinaryField16x8b>(a_val.into(), b_val.into());
            check_transpose_all_heights::<PackedBinaryGhash1x128b>(a_val.into(), b_val.into());
        }

        #[test]
        fn check_transpose_256b(a_val in any::<[u128; 2]>(), b_val in any::<[u128; 2]>()) {
            check_transpose_all_heights::<PackedBinaryField256x1b>(a_val.into(), b_val.into());
            check_transpose_all_heights::<PackedAESBinaryField32x8b>(a_val.into(), b_val.into());
            check_transpose_all_heights::<PackedBinaryGhash2x128b>(a_val.into(), b_val.into());
        }

        #[test]
        fn check_transpose_512b(a_val in any::<[u128; 4]>(), b_val in any::<[u128; 4]>()) {
            check_transpose_all_heights::<PackedBinaryField512x1b>(a_val.into(), b_val.into());
            check_transpose_all_heights::<PackedAESBinaryField64x8b>(a_val.into(), b_val.into());
            check_transpose_all_heights::<PackedBinaryGhash4x128b>(a_val.into(), b_val.into());
        }
    }
}
