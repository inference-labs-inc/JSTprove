use crate::arithmetic::mul_512;
use crate::arithmetic::sbb;
use crate::arithmetic::CurveEndo;
use crate::arithmetic::EndoParameters;
use crate::derive::curve::{IDENTITY_MASK, IDENTITY_SHIFT, SIGN_MASK, SIGN_SHIFT};
use crate::ff::WithSmallOrderMulGroup;
use crate::ff::{Field, PrimeField};
use crate::group::Curve;
use crate::group::{prime::PrimeCurveAffine, Group, GroupEncoding};
use crate::grumpkin::Fq;
use crate::grumpkin::Fr;
use crate::{
    endo, impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    new_curve_impl,
};
use crate::{Coordinates, CurveAffine, CurveExt};
use core::cmp;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "derive_serde")]
use serde::{Deserialize, Serialize};

new_curve_impl!(
    (pub),
    G1,
    G1Affine,
    Fq,
    Fr,
    (G1_GENERATOR_X, G1_GENERATOR_Y),
    G1_A,
    G1_B,
    "grumpkin_g1",
    |domain_prefix| crate::hash_to_curve::hash_to_curve(domain_prefix, G1::default_hash_to_curve_suite()),
);

// Parameters in montgomery form taken from
// https://github.com/AztecProtocol/barretenberg/blob/97ccf76c42db581a8b8f8bfbcffe8ca015a3dd22/cpp/src/barretenberg/ecc/curves/grumpkin/grumpkin.hpp#L14
const G1_GENERATOR_X: Fq = Fq::one();
const G1_GENERATOR_Y: Fq = Fq([
    0x11b2dff1448c41d8,
    0x23d3446f21c77dc3,
    0xaa7b8cf435dfafbb,
    0x14b34cf69dc25d68,
]);
const G1_A: Fq = Fq::zero();
const G1_B: Fq = Fq([
    0xdd7056026000005a,
    0x223fa97acb319311,
    0xcc388229877910c0,
    0x034394632b724eaa,
]);

// Generated using https://github.com/ConsenSys/gnark-crypto/blob/master/ecc/utils.go
// with `bn256::Fq::ZETA`
// See https://github.com/demining/Endomorphism-Secp256k1/blob/main/README.md
// to have more details about the endomorphism.
const ENDO_PARAMS_GRUMPKIN: EndoParameters = EndoParameters {
    gamma1: [0xd91d232ec7e0b3d2, 0x2, 0, 0],
    gamma2: [0x5398fd0300ff655f, 0x4ccef014a773d2d2, 0x02, 0],
    b1: [0x89d3256894d213e2, 0, 0, 0],
    b2: [0x0be4e1541221250b, 0x6f4d8248eeb859fd, 0, 0],
};

endo!(G1, Fr, ENDO_PARAMS_GRUMPKIN);

impl group::cofactor::CofactorGroup for G1 {
    type Subgroup = G1;

    fn clear_cofactor(&self) -> Self {
        *self
    }

    fn into_subgroup(self) -> CtOption<Self::Subgroup> {
        CtOption::new(self, 1.into())
    }

    fn is_torsion_free(&self) -> Choice {
        1.into()
    }
}

impl G1 {
    const SVDW_Z: Fq = Fq::ONE;

    fn default_hash_to_curve_suite() -> crate::hash_to_curve::Suite<Self, sha2::Sha256, 48> {
        crate::hash_to_curve::Suite::<G1, sha2::Sha256, 48>::new(
            b"GRUMPKIN_XMD:SHA-256_SVDW_RO_",
            Self::SVDW_Z,
            crate::hash_to_curve::Method::SVDW,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use group::UncompressedEncoding;
    crate::curve_testing_suite!(G1);
    crate::curve_testing_suite!(G1, "endo_consistency");
    crate::curve_testing_suite!(G1, "endo");
    crate::curve_testing_suite!(
        G1,
        "constants",
        Fq::MODULUS,
        G1_A,
        G1_B,
        G1_GENERATOR_X,
        G1_GENERATOR_Y,
        Fr::MODULUS
    );
}
