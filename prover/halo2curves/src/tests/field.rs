use ff::{FromUniformBytes, PrimeField};
use rand::RngCore;

#[macro_export]
macro_rules! field_testing_suite {
    ($field: ident, "field_arithmetic") => {
        fn random_multiplication_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                let a = F::random(&mut rng);
                let b = F::random(&mut rng);
                let c = F::random(&mut rng);

                let mut t0 = a; // (a * b) * c
                t0.mul_assign(&b);
                t0.mul_assign(&c);

                let mut t1 = a; // (a * c) * b
                t1.mul_assign(&c);
                t1.mul_assign(&b);

                let mut t2 = b; // (b * c) * a
                t2.mul_assign(&c);
                t2.mul_assign(&a);

                assert_eq!(t0, t1);
                assert_eq!(t1, t2);
            }
        }

        fn random_addition_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                let a = F::random(&mut rng);
                let b = F::random(&mut rng);
                let c = F::random(&mut rng);

                let mut t0 = a; // (a + b) + c
                t0.add_assign(&b);
                t0.add_assign(&c);

                let mut t1 = a; // (a + c) + b
                t1.add_assign(&c);
                t1.add_assign(&b);

                let mut t2 = b; // (b + c) + a
                t2.add_assign(&c);
                t2.add_assign(&a);

                assert_eq!(t0, t1);
                assert_eq!(t1, t2);
            }
        }

        fn random_subtraction_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                let a = F::random(&mut rng);
                let b = F::random(&mut rng);

                let mut t0 = a; // (a - b)
                t0.sub_assign(&b);

                let mut t1 = b; // (b - a)
                t1.sub_assign(&a);

                let mut t2 = t0; // (a - b) + (b - a) = 0
                t2.add_assign(&t1);

                assert_eq!(t2.is_zero().unwrap_u8(), 1);
            }
        }

        fn random_negation_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                let a = F::random(&mut rng);
                let mut b = a;
                b = b.neg();
                b.add_assign(&a);

                assert_eq!(b.is_zero().unwrap_u8(), 1);
            }
        }

        fn random_doubling_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                let mut a = F::random(&mut rng);
                let mut b = a;
                a.add_assign(&b);
                b = b.double();

                assert_eq!(a, b);
            }
        }

        fn random_squaring_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                let mut a = F::random(&mut rng);
                let mut b = a;
                a.mul_assign(&b);
                b = b.square();

                assert_eq!(a, b);
            }
        }

        fn random_inversion_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            assert!(bool::from(F::ZERO.invert().is_none()));

            for _ in 0..n {
                let mut a = F::random(&mut rng);
                let b = a.invert().unwrap(); // probabilistically nonzero
                a.mul_assign(&b);

                assert_eq!(a, F::ONE);
            }
        }

        fn random_expansion_tests<F: Field, R: RngCore>(mut rng: R, n: usize) {
            for _ in 0..n {
                // Compare (a + b)(c + d) and (a*c + b*c + a*d + b*d)

                let a = F::random(&mut rng);
                let b = F::random(&mut rng);
                let c = F::random(&mut rng);
                let d = F::random(&mut rng);

                let mut t0 = a;
                t0.add_assign(&b);
                let mut t1 = c;
                t1.add_assign(&d);
                t0.mul_assign(&t1);

                let mut t2 = a;
                t2.mul_assign(&c);
                let mut t3 = b;
                t3.mul_assign(&c);
                let mut t4 = a;
                t4.mul_assign(&d);
                let mut t5 = b;
                t5.mul_assign(&d);

                t2.add_assign(&t3);
                t2.add_assign(&t4);
                t2.add_assign(&t5);

                assert_eq!(t0, t2);
            }
        }

        fn zero_tests<F: Field, R: RngCore>(mut rng: R) {
            assert_eq!(F::ZERO.is_zero().unwrap_u8(), 1);
            {
                let mut z = F::ZERO;
                z = z.neg();
                assert_eq!(z.is_zero().unwrap_u8(), 1);
            }

            assert!(bool::from(F::ZERO.invert().is_none()));

            // Multiplication by zero
            {
                let mut a = F::random(&mut rng);
                a.mul_assign(&F::ZERO);
                assert_eq!(a.is_zero().unwrap_u8(), 1);
            }

            // Addition by zero
            {
                let mut a = F::random(&mut rng);
                let copy = a;
                a.add_assign(&F::ZERO);
                assert_eq!(a, copy);
            }
        }

        fn one_tests<F: Field, R: RngCore>(mut rng: R)  {
            assert!(bool::from(F::ONE.invert().is_some()));

            // Multiplication by one
            {
                let mut a = F::random(&mut rng);
                let copy = a;
                a.mul_assign(&F::ONE);
                assert_eq!(a, copy);
            }

            // Addition by one
            {
                let mut a = F::random(&mut rng);
                let copy = a;
                a.add_assign(&F::ONE);
                assert_eq!(a, copy + F::ONE);
            }
        }

        use ff::Field;
        use rand::SeedableRng;
        use rand_xorshift::XorShiftRng;

        #[test]
        fn test_field() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
                0xbc, 0xe5,
            ]);

            // reduce the number of tests for high-degree extension fields since TOO long
            let n = if impls::impls!($field: ff::PrimeField) { 1000000 } else { 100000 };

            // normal cases
            random_multiplication_tests::<$field, _>(&mut rng, n);
            random_addition_tests::<$field, _>(&mut rng, n);
            random_subtraction_tests::<$field, _>(&mut rng, n);
            random_negation_tests::<$field, _>(&mut rng, n);
            random_doubling_tests::<$field, _>(&mut rng, n);
            random_squaring_tests::<$field, _>(&mut rng, n);
            random_inversion_tests::<$field, _>(&mut rng, n);
            random_expansion_tests::<$field, _>(&mut rng, n);

            // edge cases
            zero_tests::<$field, _>(&mut rng);
            one_tests::<$field, _>(&mut rng);
        }
    };

    ($field: ident, "conversion") => {
        #[test]
        fn test_conversion() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54,
                0x06, 0xbc, 0xe5,
            ]);
            for _ in 0..1000000 {
                let a = $field::random(&mut rng);
                let bytes = a.to_repr();
                let b = $field::from_repr(bytes).unwrap();
                assert_eq!(a, b);
            }
        }
    };

    ($field: ident, "serialization") => {
        macro_rules! random_serialization_test {
            ($f: ident) => {
                let mut rng = XorShiftRng::from_seed([
                    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54,
                    0x06, 0xbc, 0xe5,
                ]);
                for _ in 0..1000000 {
                    let a = $f::random(&mut rng);
                    let bytes = a.to_raw_bytes();
                    let b = $f::from_raw_bytes(&bytes).unwrap();
                    assert_eq!(a, b);
                    let mut buf = Vec::new();
                    a.write_raw(&mut buf).unwrap();
                    let b = $f::read_raw(&mut &buf[..]).unwrap();
                    assert_eq!(a, b);
                }
            };
        }

        #[cfg(feature = "derive_serde")]
        macro_rules! random_serde_test {
            ($f: ident) => {
                let mut rng = XorShiftRng::from_seed([
                    0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54,
                    0x06, 0xbc, 0xe5,
                ]);
                for _ in 0..1000000 {
                    // byte serialization
                    let a = $f::random(&mut rng);
                    let bytes = bincode::serialize(&a).unwrap();
                    let reader = std::io::Cursor::new(bytes);
                    let b: $f = bincode::deserialize_from(reader).unwrap();
                    assert_eq!(a, b);

                    // json serialization
                    let json = serde_json::to_string(&a).unwrap();
                    let reader = std::io::Cursor::new(json);
                    let b: $f = serde_json::from_reader(reader).unwrap();
                    assert_eq!(a, b);
                }
            };
        }

        #[test]
        fn test_serialization() {
            use $crate::serde::SerdeObject;
            random_serialization_test!($field);
            #[cfg(feature = "derive_serde")]
            random_serde_test!($field);
        }
    };

    ($field: ident, "quadratic_residue") => {
        #[test]
        fn test_quadratic_residue() {
            use $crate::ff_ext::Legendre;
            use ff::Field;
            use rand_core::SeedableRng;
            use rand_xorshift::XorShiftRng;

            // random quadratic residue test
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54,
                0x06, 0xbc, 0xe5,
            ]);
            for _ in 0..100000 {
                let elem = $field::random(&mut rng);
                let is_quad_res_or_zero: bool = elem.sqrt().is_some().into();
                let is_quad_non_res: bool = elem.ct_quadratic_non_residue().into();
                assert_eq!(!is_quad_non_res, is_quad_res_or_zero)
            }
        }
    };

    ($field: ident, "bits") => {
        #[test]
        #[cfg(feature = "bits")]
        fn test_bits() {
            use ff::PrimeFieldBits;
            // random bit test
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54,
                0x06, 0xbc, 0xe5,
            ]);
            for _ in 0..1000000 {
                let a = $field::random(&mut rng);
                let bytes = a.to_repr();
                let bits = a.to_le_bits();
                for idx in 0..bits.len() {
                    assert_eq!(bits[idx], ((bytes.as_ref()[idx / 8] >> (idx % 8)) & 1) == 1);
                }
            }
        }
    };

    ($field: ident, "serialization_check") => {
        fn is_less_than<const N: usize>(x: &[u64; N], y: &[u64; N]) -> bool {
            for i in (1..N).rev() {
                match x[i].cmp(&y[i]) {
                    core::cmp::Ordering::Less => return true,
                    core::cmp::Ordering::Greater => return false,
                    _ => {}
                }
            }
            x[0].lt(&y[0])
        }

        #[test]
        fn test_serialization_check() {
            use $crate::serde::SerdeObject;
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
                0xbc, 0xe5,
            ]);
            const LIMBS: usize = $field::size() / 8;
            // failure check
            for _ in 0..1000000 {
                let rand_word = [(); LIMBS].map(|_| rng.next_u64());
                let a = $field(rand_word);
                let rand_bytes = a.to_raw_bytes();
                match is_less_than::<LIMBS>(&rand_word, &MODULUS.0) {
                    false => {
                        assert!($field::from_raw_bytes(&rand_bytes).is_none());
                    }
                    _ => {
                        assert_eq!($field::from_raw_bytes(&rand_bytes), Some(a));
                    }
                }
            }
        }
    };

    ($field: ident, "constants", $modulus_str: expr) => {
        #[test]
        fn test_primefield_constants() {
            assert_eq!($field::MODULUS, $modulus_str);
            assert_eq!(
                $field::ROOT_OF_UNITY_INV,
                $field::ROOT_OF_UNITY.invert().unwrap()
            );
            assert_eq!($field::from(2u64) * $field::TWO_INV, $field::ONE);
            if $field::S != 0 {
                assert_eq!(
                    $field::ROOT_OF_UNITY.pow_vartime([1 << $field::S]),
                    $field::one()
                );
                assert_eq!(
                    $field::DELTA,
                    $field::MULTIPLICATIVE_GENERATOR.pow([1u64 << $field::S])
                );
            }
        }
    };

    ($field: ident, "sqrt") => {
        #[test]
        fn test_sqrt() {
            use $crate::ff_ext::Legendre;
            use rand_core::OsRng;

            let v = ($field::TWO_INV).square().sqrt().unwrap();
            assert!(v == $field::TWO_INV || (-v) == $field::TWO_INV);

            for _ in 0..10000 {
                let a = $field::random(OsRng);
                if a.legendre() == -1 {
                    assert!(bool::from(a.sqrt().is_none()));
                }
            }

            for _ in 0..10000 {
                let a = $field::random(OsRng);
                let mut b = a;
                b = b.square();
                assert_eq!(b.legendre(), 1);

                let b = b.sqrt().unwrap();
                let mut negb = b;
                negb = negb.neg();

                assert!(a == b || a == negb);
            }

            let mut c = $field::one();
            for _ in 0..10000 {
                let mut b = c;
                b = b.square();
                assert_eq!(b.legendre(), 1);

                b = b.sqrt().unwrap();

                if b != c {
                    b = b.neg();
                }

                assert_eq!(b, c);

                c += &$field::one();
            }
        }
    };

    ($field: ident, "zeta" $(, $base_field: ident)*) => {
        #[test]
        fn test_zeta() {
            assert_eq!($field::ZETA * $field::ZETA * $field::ZETA, $field::ONE);
            assert_ne!($field::ZETA * $field::ZETA, $field::ONE);
            $(
                let zeta = $field::new($base_field::ZETA.square(), $base_field::zero());
                assert_eq!(zeta, $field::ZETA);
            )*
        }
    };

    ($field: ident, "from_uniform_bytes", $($L:expr),* $(,)?) => {

        #[test]
        fn test_from_uniform_bytes() {
            $(
                $crate::tests::field::run_test_from_uniform_bytes::<$field, $L>();
            )*
        }
    };

    ($ext_field: ident, "f2_tests", $base_field: ident) => {
        #[test]
        fn test_ser() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            let a0 = $ext_field::random(&mut rng);
            let a_bytes = a0.to_bytes();
            let a1 = $ext_field::from_bytes(&a_bytes).unwrap();
            assert_eq!(a0, a1);
        }

        #[test]
        fn test_f2_ordering() {
            let mut a = $ext_field {
                c0: $base_field::zero(),
                c1: $base_field::zero(),
            };

            let mut b = a;

            assert!(a.cmp(&b) == Ordering::Equal);
            b.c0 += &$base_field::one();
            assert!(a.cmp(&b) == Ordering::Less);
            a.c0 += &$base_field::one();
            assert!(a.cmp(&b) == Ordering::Equal);
            b.c1 += &$base_field::one();
            assert!(a.cmp(&b) == Ordering::Less);
            a.c0 += &$base_field::one();
            assert!(a.cmp(&b) == Ordering::Less);
            a.c1 += &$base_field::one();
            assert!(a.cmp(&b) == Ordering::Greater);
            b.c0 += &$base_field::one();
            assert!(a.cmp(&b) == Ordering::Equal);
        }

        #[test]
        fn test_f2_basics() {
            assert_eq!(
                $ext_field {
                    c0: $base_field::zero(),
                    c1: $base_field::zero(),
                },
                $ext_field::ZERO
            );
            assert_eq!(
                $ext_field {
                    c0: $base_field::one(),
                    c1: $base_field::zero(),
                },
                $ext_field::ONE
            );
            assert_eq!($ext_field::ZERO.is_zero().unwrap_u8(), 1);
            assert_eq!($ext_field::ONE.is_zero().unwrap_u8(), 0);
            assert_eq!(
                $ext_field {
                    c0: $base_field::zero(),
                    c1: $base_field::one(),
                }
                .is_zero()
                .unwrap_u8(),
                0
            );
        }
    };

    ($ext_field: ident, "f6_tests", $base_field: ident) => {
        #[test]
        fn test_f6_mul_nonresidue() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            let nqr = $ext_field {
                c0: $base_field::zero(),
                c1: $base_field::one(),
                c2: $base_field::zero(),
            };

            for _ in 0..1000 {
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;
                a.mul_by_nonresidue();
                b.mul_assign(&nqr);

                assert_eq!(a, b);
            }
        }

        #[test]
        fn test_f6_mul_by_1() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..1000 {
                let c1 = $base_field::random(&mut rng);
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;

                a.mul_by_1(&c1);
                b.mul_assign(&$ext_field {
                    c0: $base_field::zero(),
                    c1,
                    c2: $base_field::zero(),
                });

                assert_eq!(a, b);
            }
        }

        #[test]
        fn test_f6_mul_by_01() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..1000 {
                let c0 = $base_field::random(&mut rng);
                let c1 = $base_field::random(&mut rng);
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;

                a.mul_by_01(&c0, &c1);
                b.mul_assign(&$ext_field {
                    c0,
                    c1,
                    c2: $base_field::zero(),
                });

                assert_eq!(a, b);
            }
        }

        #[test]
        fn test_squaring() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..1000 {
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;
                b.mul_assign(&a);
                a.square_assign();
                assert_eq!(a, b);
            }
        }
    };

    ($ext_field: ident, "f12_tests", $base_field_1: ident, $base_field_2: ident) => {
        #[test]
        fn test_f12_mul_by_014() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..1000 {
                let c0 = $base_field_2::random(&mut rng);
                let c1 = $base_field_2::random(&mut rng);
                let c5 = $base_field_2::random(&mut rng);
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;

                a.mul_by_014(&c0, &c1, &c5);
                b.mul_assign(&$ext_field {
                    c0: $base_field_1 {
                        c0,
                        c1,
                        c2: $base_field_2::zero(),
                    },
                    c1: $base_field_1 {
                        c0: $base_field_2::zero(),
                        c1: c5,
                        c2: $base_field_2::zero(),
                    },
                });

                assert_eq!(a, b);
            }
        }

        #[test]
        fn test_f12_mul_by_034() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..1000 {
                let c0 = $base_field_2::random(&mut rng);
                let c3 = $base_field_2::random(&mut rng);
                let c4 = $base_field_2::random(&mut rng);
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;

                a.mul_by_034(&c0, &c3, &c4);
                b.mul_assign(&$ext_field {
                    c0: $base_field_1 {
                        c0,
                        c1: $base_field_2::zero(),
                        c2: $base_field_2::zero(),
                    },
                    c1: $base_field_1 {
                        c0: c3,
                        c1: c4,
                        c2: $base_field_2::zero(),
                    },
                });

                assert_eq!(a, b);
            }
        }

        #[test]
        fn test_squaring() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..1000 {
                let mut a = $ext_field::random(&mut rng);
                let mut b = a;
                b.mul_assign(&a);
                a.square_assign();
                assert_eq!(a, b);
            }
        }
    };

    ($ext_field: ident, "frobenius", $frobenius_param: expr) => {
        #[test]
        fn test_frobenius() {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
                0xe5,
            ]);

            for _ in 0..50 {
                for i in 0..8 {
                    let mut a = $ext_field::random(&mut rng);
                    let mut b = a;

                    for _ in 0..i {
                        a = a.pow($frobenius_param);
                    }
                    b.frobenius_map(i);

                    assert_eq!(a, b);
                }
            }
        }
    };
}

pub(crate) fn run_test_from_uniform_bytes<F: PrimeField, const L: usize>()
where
    F: FromUniformBytes<L>,
{
    use num_bigint::BigUint;
    use rand_core::OsRng;

    let mut uniform_bytes = [0u8; L];
    OsRng.fill_bytes(&mut uniform_bytes[..]);

    let e0 = {
        let e0 = BigUint::from_bytes_le(&uniform_bytes);
        let e0 = e0 % crate::tests::modulus::<F>();
        let bytes = e0.to_bytes_le();
        let mut e0 = F::Repr::default();
        e0.as_mut()[..bytes.len()].copy_from_slice(&bytes);
        F::from_repr(e0).unwrap()
    };

    let e1 = F::from_uniform_bytes(&uniform_bytes);
    assert_eq!(e0, e1);
}
