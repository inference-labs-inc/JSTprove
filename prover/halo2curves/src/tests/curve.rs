#[macro_export]
macro_rules! curve_testing_suite {
    ($($curve: ident),*) => {
        macro_rules! is_on_curve {
            ($c: ident) => {
                assert!(bool::from($c::identity().is_on_curve()));
                assert!(bool::from($c::generator().is_on_curve()));

                for _ in 0..100 {
                    let point = $c::random(OsRng);
                    assert!(bool::from(point.is_on_curve()));
                    let affine_point: <$c as CurveExt>::AffineExt = point.into();
                    assert!(bool::from(affine_point.is_on_curve()));
                }
            }
        }

        macro_rules! equality {
            ($c: ident) => {
                let a = $c::generator();
                let b = $c::identity();

                assert!(a == a);
                assert!(b == b);
                assert!(a != b);
                assert!(b != a);

                for _ in 0..100 {
                    let a = $c::random(OsRng);
                    let b = $c::random(OsRng);

                    assert!(a == a);
                    assert!(b == b);
                    assert!(a != b);
                    assert!(b != a);

                    let a: <$c as CurveExt>::AffineExt = a.into();
                    let b: <$c as CurveExt>::AffineExt = b.into();

                    assert!(a == a);
                    assert!(b == b);
                    assert!(a != b);
                    assert!(b != a);
                }
            }
        }

        macro_rules! projective_affine_roundtrip {
            ($c: ident) => {
                let a = $c::generator();
                let b = $c::identity();

                assert!(bool::from(<$c as CurveExt>::AffineExt::from(a).is_on_curve()));
                assert!(!bool::from(<$c as CurveExt>::AffineExt::from(a).is_identity()));
                assert!(bool::from(<$c as CurveExt>::AffineExt::from(b).is_on_curve()));
                assert!(bool::from(<$c as CurveExt>::AffineExt::from(b).is_identity()));

                let a = <$c as CurveExt>::AffineExt::generator();
                let b = <$c as CurveExt>::AffineExt::identity();

                assert!(bool::from($c::from(a).is_on_curve()));
                assert!(!bool::from($c::from(a).is_identity()));
                assert!(bool::from($c::from(b).is_on_curve()));
                assert!(bool::from($c::from(b).is_identity()));
            }
        }

        macro_rules! projective_addition {
            ($c: ident) => {
                let a = $c::identity();
                let b = $c::identity();
                let c = a + b;
                assert!(bool::from(c.is_identity()));
                assert!(bool::from(c.is_on_curve()));
                let c = a - b;
                assert!(bool::from(c.is_identity()));
                assert!(bool::from(c.is_on_curve()));

                let a = $c::identity();
                let a = -a;
                assert!(bool::from(a.is_on_curve()));
                assert!(bool::from(a.is_identity()));

                let a = $c::random(OsRng);
                assert!(a == a + $c::identity());
                assert!(a == $c::identity() + a);
                assert!(-a == $c::identity() - a);

                let a = $c::identity();
                let a = a.double();
                assert!(bool::from(c.is_on_curve()));
                assert!(bool::from(a.is_identity()));

                let a = $c::generator();
                let a = a.double();
                assert!(bool::from(c.is_on_curve()));
                assert_eq!(a, $c::generator() + $c::generator());

                let a = $c::random(OsRng);
                assert!(a.double() - a == a);

                let a = $c::random(OsRng);
                let b = $c::random(OsRng);
                let c = $c::random(OsRng);
                assert!(a + b == b + a);
                assert!(a - b == -(b - a));
                assert!(c + (a + b) == a + (c + b));
                assert!((a - b) - c == (a - c) - b);

                let a = $c::generator().double().double(); // 4P
                let b = $c::generator().double(); // 2P
                let c = a + b;

                let mut d = $c::generator();
                for _ in 0..5 {
                    d += $c::generator();
                }

                assert!(c == d);
                assert!(!bool::from(c.is_identity()));
                assert!(bool::from(c.is_on_curve()));
                assert!(!bool::from(d.is_identity()));
                assert!(bool::from(d.is_on_curve()));
            }
        }

        macro_rules! mixed_addition {
            ($c: ident) => {
                let a = $c::identity();
                let b = <$c as group::Curve>::AffineRepr::identity();
                let c = a + b;
                assert!(bool::from(c.is_identity()));
                assert!(bool::from(c.is_on_curve()));
                let c = a - b;
                assert!(bool::from(c.is_identity()));
                assert!(bool::from(c.is_on_curve()));

                let a = $c::identity();
                let a = -a;
                assert!(bool::from(a.is_on_curve()));
                assert!(bool::from(a.is_identity()));
                let a = <$c as CurveExt>::AffineExt::identity();
                let a = -a;
                assert!(bool::from(a.is_on_curve()));
                assert!(bool::from(a.is_identity()));

                let a: <$c as CurveExt>::AffineExt = $c::random(OsRng).into();
                assert!(a.to_curve() == a + <$c as CurveExt>::AffineExt::identity());

                let a = $c::random(OsRng);
                assert!(a.double() - a == a);

                let a = $c::random(OsRng);
                let b: <$c as CurveExt>::AffineExt = $c::random(OsRng).into();
                let c0 = a + b;
                let c1 = a + $c::from(b);
                assert_eq!(c0, c1);
            }
        }

        macro_rules! multiplication {
            ($c: ident) => {
                for _ in 1..1000 {
                    let s1 = <$c as CurveExt>::ScalarExt::random(OsRng);
                    let s2 = <$c as CurveExt>::ScalarExt::random(OsRng);

                    let t0 = $c::identity() * s1;
                    assert!(bool::from(t0.is_identity()));

                    let a = $c::random(OsRng);
                    let t0 = a * <$c as CurveExt>::ScalarExt::ONE;
                    assert_eq!(a, t0);

                    let t0 = a * <$c as CurveExt>::ScalarExt::ZERO;
                    assert!(bool::from(t0.is_identity()));

                    let t0 = a * s1 + a * s2;

                    let s3 = s1 + s2;
                    let t1 = a * s3;

                    assert_eq!(t0, t1);

                    let mut t0 = a * s1;
                    let mut t1 = a * s2;
                    t0 += t1;
                    let s3 = s1 + s2;
                    t1 = a * s3;
                    assert_eq!(t0, t1);
                }
            }
        }

        macro_rules! batch_normalize {
            ($c: ident) => {
                let a = $c::generator().double();
                let b = a.double();
                let c = b.double();

                for a_identity in (0..1).map(|n| n == 1) {
                    for b_identity in (0..1).map(|n| n == 1) {
                        for c_identity in (0..1).map(|n| n == 1) {
                            let mut v = [a, b, c];
                            if a_identity {
                                v[0] = $c::identity()
                            }
                            if b_identity {
                                v[1] = $c::identity()
                            }
                            if c_identity {
                                v[2] = $c::identity()
                            }

                            let mut t = [
                                <$c as CurveExt>::AffineExt::identity(),
                                <$c as CurveExt>::AffineExt::identity(),
                                <$c as CurveExt>::AffineExt::identity(),
                            ];
                            let expected = [
                                <$c as CurveExt>::AffineExt::from(v[0]),
                                <$c as CurveExt>::AffineExt::from(v[1]),
                                <$c as CurveExt>::AffineExt::from(v[2]),
                            ];

                            $c::batch_normalize(&v[..], &mut t[..]);

                            assert_eq!(&t[..], &expected[..]);
                        }
                    }
                }
            }
        }

        macro_rules! serdes {
            ($c: ident) => {
                assert!(bool::from(
                    $c::from_bytes(&$c::identity().to_bytes())
                        .unwrap()
                        .is_identity()
                ));

                assert!(bool::from(
                    <$c as CurveExt>::AffineExt::from_uncompressed(&<$c as CurveExt>::AffineExt::identity().to_uncompressed())
                        .unwrap()
                        .is_identity()
                ));

                assert!(bool::from(
                    <$c as CurveExt>::AffineExt::from_bytes(&<$c as CurveExt>::AffineExt::identity().to_bytes())
                        .unwrap()
                        .is_identity()
                ));

                for _ in 0..100 {
                    let projective_point = $c::random(OsRng);
                    let affine_point: <$c as CurveExt>::AffineExt = projective_point.into();

                    // Compressed format tests
                    let projective_repr = projective_point.to_bytes();
                    let affine_repr = affine_point.to_bytes();

                    let projective_point_rec = $c::from_bytes(&projective_repr).unwrap();
                    let projective_point_rec_unchecked = $c::from_bytes(&projective_repr).unwrap();

                    let affine_point_rec = <$c as CurveExt>::AffineExt::from_bytes(&affine_repr).unwrap();
                    let affine_point_rec_unchecked = <$c as CurveExt>::AffineExt::from_bytes(&affine_repr).unwrap();

                    assert_eq!(projective_point, projective_point_rec);
                    assert_eq!(projective_point, projective_point_rec_unchecked);
                    assert_eq!(affine_point, affine_point_rec);
                    assert_eq!(affine_point, affine_point_rec_unchecked);

                    // Uncompressed format
                    let affine_repr = affine_point.to_uncompressed();

                    let affine_point_rec = <$c as CurveExt>::AffineExt::from_uncompressed_unchecked(&affine_repr).unwrap();
                    let affine_point_rec_unchecked = <$c as CurveExt>::AffineExt::from_uncompressed_unchecked(&affine_repr).unwrap();

                    assert_eq!(affine_point, affine_point_rec);
                    assert_eq!(affine_point, affine_point_rec_unchecked);
                }
            }
        }

        // TODO Change name
        macro_rules! random_serialization_test {
            ($c: ident) => {
                for _ in 0..100 {
                    let projective_point = $c::random(OsRng);
                    let affine_point: <$c as CurveExt>::AffineExt = projective_point.into();

                    let projective_bytes = projective_point.to_raw_bytes();
                    let projective_point_rec = $c::from_raw_bytes(&projective_bytes).unwrap();
                    assert_eq!(projective_point, projective_point_rec);
                    let mut buf = Vec::new();
                    projective_point.write_raw(&mut buf).unwrap();
                    let projective_point_rec = $c::read_raw(&mut &buf[..]).unwrap();
                    assert_eq!(projective_point, projective_point_rec);

                    let affine_bytes = affine_point.to_raw_bytes();
                    let affine_point_rec = <$c as CurveExt>::AffineExt::from_raw_bytes(&affine_bytes).unwrap();
                    assert_eq!(affine_point, affine_point_rec);
                    let mut buf = Vec::new();
                    affine_point.write_raw(&mut buf).unwrap();
                    let affine_point_rec = <$c as CurveExt>::AffineExt::read_raw(&mut &buf[..]).unwrap();
                    assert_eq!(affine_point, affine_point_rec);
                }
            }
        }

        #[cfg(feature = "derive_serde")]
        macro_rules! random_serde_test {
            ($c: ident) => {
                for _ in 0..100 {
                    let projective_point = $c::random(OsRng);
                    let affine_point: <$c as CurveExt>::AffineExt = projective_point.into();
                    {
                        let affine_bytes = bincode::serialize(&affine_point).unwrap();
                        let reader = std::io::Cursor::new(affine_bytes);
                        let affine_point_rec: <$c as CurveExt>::AffineExt = bincode::deserialize_from(reader).unwrap();
                        assert_eq!(projective_point.to_affine(), affine_point_rec);
                        assert_eq!(affine_point, affine_point_rec);
                    }
                    {
                        let affine_json = serde_json::to_string(&affine_point).unwrap();
                        let reader = std::io::Cursor::new(affine_json);
                        let affine_point_rec: <$c as CurveExt>::AffineExt = serde_json::from_reader(reader).unwrap();
                        assert_eq!(affine_point, affine_point_rec);
                    }
                    {
                        let projective_bytes = bincode::serialize(&projective_point).unwrap();
                        let reader = std::io::Cursor::new(projective_bytes);
                        let projective_point_rec: $c = bincode::deserialize_from(reader).unwrap();
                        assert_eq!(projective_point, projective_point_rec);
                    }
                    {
                        let projective_json = serde_json::to_string(&projective_point).unwrap();
                        let reader = std::io::Cursor::new(projective_json);
                        let projective_point_rec: $c = serde_json::from_reader(reader).unwrap();
                        assert_eq!(projective_point, projective_point_rec);
                    }
                }
            }
        }

        use $crate::ff::Field;
        use $crate::group::prime::PrimeCurveAffine;
        use $crate::{group::GroupEncoding, serde::SerdeObject};
        use $crate::{CurveAffine, CurveExt};
        use rand_core::OsRng;


        #[test]
        fn test_curve() {
            $(
                is_on_curve!($curve);
                equality!($curve);
                projective_affine_roundtrip!($curve);
                projective_addition!($curve);
                mixed_addition!($curve);
                multiplication!($curve);
                batch_normalize!($curve);
                serdes!($curve);
            )*
        }

        #[test]
        fn test_serialization() {
            $(
                random_serialization_test!($curve);
                #[cfg(feature = "derive_serde")]
                random_serde_test!($curve);
            )*
        }
    };

    ($($curve: ident),*, "hash_to_curve") => {
        macro_rules! hash_to_curve_test {
            ($c: ident) => {
                let hasher = $c::hash_to_curve("test");
                let mut rng = OsRng;
                for _ in 0..1000 {
                    let message = iter::repeat_with(|| rng.next_u32().to_be_bytes())
                        .take(32)
                        .flatten()
                        .collect::<Vec<_>>();
                    assert!(bool::from(hasher(&message).is_on_curve()));
                }
            }
        }

        #[test]
        fn test_hash_to_curve() {
            use rand_core::{OsRng, RngCore};
            use std::iter;
            $(
                hash_to_curve_test!($curve);
            )*
        }
    };

    ($($curve: ident),*, "clear_cofactor") => {
        #[test]
        fn test_cofactor_clearing() {
            use rand_core::OsRng;
            $(
                for _ in 0..50 {
                let point = $curve::random(OsRng);
                assert!(bool::from(point.is_on_curve()));
                use group::cofactor::CofactorGroup;
                assert!(bool::from(point.is_torsion_free()));
                }

            )*
        }
    };

    ($($curve: ident),*, "endo_consistency") => {
        #[test]
        fn test_endo_consistency() {
            use rand_core::OsRng;
            $(
                let g = $curve::generator();
                assert_eq!(g * <$curve as CurveExt>::ScalarExt::ZETA, g.endo());

                for _ in 0..100 {
                    let g = $curve::random(OsRng);
                    assert_eq!(g * <$curve as CurveExt>::ScalarExt::ZETA, g.endo());
                }
            )*
        }
    };

    ($curve: ident, "endo" $(, $z_other_raw: expr)*) => {
        #[test]
        fn test_endo() {
            use rand_core::OsRng;

            let z_impl = <$curve as CurveExt>::ScalarExt::ZETA;
            assert_eq!(z_impl * z_impl + z_impl, -<$curve as CurveExt>::ScalarExt::ONE);
            $(
                let z_other = <$curve as CurveExt>::ScalarExt::from_raw($z_other_raw as [u64; 4]);
                assert_eq!(z_other * z_other + z_other, -<$curve as CurveExt>::ScalarExt::ONE);
            )*

            for _ in 0..100000 {
                let k = <$curve as CurveExt>::ScalarExt::random(OsRng);
                let (k1, k1_neg, k2, k2_neg) = $curve::decompose_scalar(&k);
                if k1_neg & k2_neg {
                    assert_eq!(k, -<$curve as CurveExt>::ScalarExt::from_u128(k1) + <$curve as CurveExt>::ScalarExt::ZETA * <$curve as CurveExt>::ScalarExt::from_u128(k2))
                } else if k1_neg {
                    assert_eq!(k, -<$curve as CurveExt>::ScalarExt::from_u128(k1) - <$curve as CurveExt>::ScalarExt::ZETA * <$curve as CurveExt>::ScalarExt::from_u128(k2))
                } else if k2_neg {
                    assert_eq!(k, <$curve as CurveExt>::ScalarExt::from_u128(k1) + <$curve as CurveExt>::ScalarExt::ZETA * <$curve as CurveExt>::ScalarExt::from_u128(k2))
                } else {
                    assert_eq!(k, <$curve as CurveExt>::ScalarExt::from_u128(k1) - <$curve as CurveExt>::ScalarExt::ZETA * <$curve as CurveExt>::ScalarExt::from_u128(k2))
                }
            }
        }
    };

    ($curve: ident, "ecdsa_example") => {
        #[test]
        fn ecdsa_example() {
            use ff::FromUniformBytes;
            use rand_core::OsRng;

            fn mod_n(x: <$curve as CurveExt>::Base) -> <$curve as CurveExt>::ScalarExt {
                let mut x_repr = [0u8; 32];
                x_repr.copy_from_slice(x.to_repr().as_ref());
                let mut x_bytes = [0u8; 64];
                x_bytes[..32].copy_from_slice(&x_repr[..]);
                <$curve as CurveExt>::ScalarExt::from_uniform_bytes(&x_bytes)
            }

            let g = $curve::generator();

            for _ in 0..1000 {
                // Generate a key pair
                let sk = <$curve as CurveExt>::ScalarExt::random(OsRng);
                let pk = (g * sk).to_affine();

                // Generate a valid signature
                // Suppose `m_hash` is the message hash
                let msg_hash = <$curve as CurveExt>::ScalarExt::random(OsRng);

                let (r, s) = {
                    // Draw arandomness
                    let k = <$curve as CurveExt>::ScalarExt::random(OsRng);
                    let k_inv = k.invert().unwrap();

                    // Calculate `r`
                    let r_point = (g * k).to_affine().coordinates().unwrap();
                    let x = r_point.x();
                    let r = mod_n(*x);

                    // Calculate `s`
                    let s = k_inv * (msg_hash + (r * sk));

                    (r, s)
                };

                {
                    // Verify
                    let s_inv = s.invert().unwrap();
                    let u_1 = msg_hash * s_inv;
                    let u_2 = r * s_inv;

                    let v_1 = g * u_1;
                    let v_2 = pk * u_2;

                    let r_point = (v_1 + v_2).to_affine().coordinates().unwrap();
                    let x_candidate = r_point.x();
                    let r_candidate = mod_n(*x_candidate);

                    assert_eq!(r, r_candidate);
                }
            }
        }
    };

    ($curve: ident, "constants", $p: expr, $a: expr, $b: expr, $gen_x: expr, $gen_y: expr, $order: expr) => {
        #[test]
        #[allow(non_snake_case)]
        fn $curve() {
            assert!($p == <$curve as CurveExt>::Base::MODULUS);

            let a = $curve::a();
            let b = $curve::b();
            assert!(a == $a);
            assert!(b == $b);

            let generator = $curve::generator();
            let generator_affine: <$curve as CurveExt>::AffineExt = generator.into();
            assert!(generator_affine.x == $gen_x);
            assert!(generator_affine.y == $gen_y);

            assert!($order == <$curve as CurveExt>::ScalarExt::MODULUS);
        }
    };
}

use crate::{CurveAffine, CurveExt};
use group::Curve;

pub(crate) struct TestH2C<C: CurveAffine> {
    msg: &'static [u8],
    expect: C,
}

impl<C: CurveAffine> TestH2C<C> {
    pub(crate) fn new(msg: &'static [u8], expect: C) -> Self {
        Self { msg, expect }
    }

    pub(crate) fn run(&self, domain_prefix: &str) {
        let r0 = C::CurveExt::hash_to_curve(domain_prefix)(self.msg);
        assert_eq!(r0.to_affine(), self.expect);
    }
}
