use crate::arithmetic::mul_512;
use crate::arithmetic::sbb;
use crate::{
    arithmetic::{CurveEndo, EndoParameters},
    endo,
};
use ff::PrimeField;
use ff::WithSmallOrderMulGroup;
pub use pasta_curves::{pallas, vesta, Ep, EpAffine, Eq, EqAffine, Fp, Fq};
use std::convert::TryInto;

// Generated using https://github.com/ConsenSys/gnark-crypto/blob/master/ecc/utils.go
// with `pasta_curves::Fp::ZETA`
// See https://github.com/demining/Endomorphism-Secp256k1/blob/main/README.md
// to have more details about the endomorphism.
const ENDO_PARAMS_EQ: EndoParameters = EndoParameters {
    // round(b2/n)
    gamma1: [0x32c49e4c00000003, 0x279a745902a2654e, 0x1, 0x0],
    // round(-b1/n)
    gamma2: [0x31f0256800000002, 0x4f34e8b2066389a4, 0x2, 0x0],
    b1: [0x8cb1279300000001, 0x49e69d1640a89953, 0x0, 0x0],
    b2: [0x0c7c095a00000001, 0x93cd3a2c8198e269, 0x0, 0x0],
};

// Generated using https://github.com/ConsenSys/gnark-crypto/blob/master/ecc/utils.go
// with `pasta_curves::Fq::ZETA`
// See https://github.com/demining/Endomorphism-Secp256k1/blob/main/README.md
// to have more details about the endomorphism.
const ENDO_PARAMS_EP: EndoParameters = EndoParameters {
    // round(b2/n)
    gamma1: [0x32c49e4bffffffff, 0x279a745902a2654e, 0x1, 0x0],
    // round(-b1/n)
    gamma2: [0x31f0256800000002, 0x4f34e8b2066389a4, 0x2, 0x0],
    b1: [0x8cb1279300000000, 0x49e69d1640a89953, 0x0, 0x0],
    b2: [0x0c7c095a00000001, 0x93cd3a2c8198e269, 0x0, 0x0],
};

endo!(Eq, Fp, ENDO_PARAMS_EQ);
endo!(Ep, Fq, ENDO_PARAMS_EP);

#[test]
fn test_endo() {
    use ff::Field;
    use rand_core::OsRng;

    for _ in 0..100000 {
        let k = Fp::random(OsRng);
        let (k1, k1_neg, k2, k2_neg) = Eq::decompose_scalar(&k);
        if k1_neg & k2_neg {
            assert_eq!(k, -Fp::from_u128(k1) + Fp::ZETA * Fp::from_u128(k2))
        } else if k1_neg {
            assert_eq!(k, -Fp::from_u128(k1) - Fp::ZETA * Fp::from_u128(k2))
        } else if k2_neg {
            assert_eq!(k, Fp::from_u128(k1) + Fp::ZETA * Fp::from_u128(k2))
        } else {
            assert_eq!(k, Fp::from_u128(k1) - Fp::ZETA * Fp::from_u128(k2))
        }
    }

    for _ in 0..100000 {
        let k = Fp::random(OsRng);
        let (k1, k1_neg, k2, k2_neg) = Eq::decompose_scalar(&k);
        if k1_neg & k2_neg {
            assert_eq!(k, -Fp::from_u128(k1) + Fp::ZETA * Fp::from_u128(k2))
        } else if k1_neg {
            assert_eq!(k, -Fp::from_u128(k1) - Fp::ZETA * Fp::from_u128(k2))
        } else if k2_neg {
            assert_eq!(k, Fp::from_u128(k1) + Fp::ZETA * Fp::from_u128(k2))
        } else {
            assert_eq!(k, Fp::from_u128(k1) - Fp::ZETA * Fp::from_u128(k2))
        }
    }
}
