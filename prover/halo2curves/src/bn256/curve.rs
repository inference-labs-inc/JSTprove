use crate::arithmetic::mul_512;
use crate::arithmetic::sbb;
use crate::arithmetic::CurveEndo;
use crate::arithmetic::EndoParameters;
use crate::bn256::Fq;
use crate::bn256::Fq2;
use crate::bn256::Fr;
use crate::derive::curve::{IDENTITY_MASK, IDENTITY_SHIFT, SIGN_MASK, SIGN_SHIFT};
use crate::endo;
use crate::ff::WithSmallOrderMulGroup;
use crate::ff::{Field, PrimeField};
use crate::group::Curve;
use crate::group::{cofactor::CofactorGroup, prime::PrimeCurveAffine, Group, GroupEncoding};
use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    new_curve_impl,
};
use crate::{Coordinates, CurveAffine, CurveExt};
use core::cmp;
use core::fmt::Debug;
use core::iter::Sum;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use std::convert::TryInto;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "derive_serde")]
use serde::{Deserialize, Serialize};

new_curve_impl!(
    (pub),
    G1,
    G1Affine,
    Fq,
    Fr,
    (G1_GENERATOR_X,G1_GENERATOR_Y),
    G1_A,
    G1_B,
    "bn256_g1",
    |domain_prefix| crate::hash_to_curve::hash_to_curve(domain_prefix, G1::default_hash_to_curve_suite()),
);

new_curve_impl!(
    (pub),
    G2,
    G2Affine,
    Fq2,
    Fr,
    (G2_GENERATOR_X, G2_GENERATOR_Y),
    G2_A,
    G2_B,
    "bn256_g2",
    |domain_prefix| hash_to_curve_g2(domain_prefix),
);

#[allow(clippy::type_complexity)]
pub(crate) fn hash_to_curve_g2<'a>(domain_prefix: &'a str) -> Box<dyn Fn(&[u8]) -> G2 + 'a> {
    let suite = G2::default_hash_to_curve_suite();
    Box::new(move |message| {
        let r0 = suite.hash_to_curve(domain_prefix, message);
        r0.clear_cofactor()
    })
}

const G1_GENERATOR_X: Fq = Fq::one();
const G1_GENERATOR_Y: Fq = Fq::from_raw([2, 0, 0, 0]);
const G1_A: Fq = Fq::from_raw([0, 0, 0, 0]);
const G1_B: Fq = Fq::from_raw([3, 0, 0, 0]);

const G2_A: Fq2 = Fq2 {
    c0: Fq::from_raw([0, 0, 0, 0]),
    c1: Fq::from_raw([0, 0, 0, 0]),
};

const G2_B: Fq2 = Fq2 {
    c0: Fq::from_raw([
        0x3267e6dc24a138e5,
        0xb5b4c5e559dbefa3,
        0x81be18991be06ac3,
        0x2b149d40ceb8aaae,
    ]),
    c1: Fq::from_raw([
        0xe4a2bd0685c315d2,
        0xa74fa084e52d1852,
        0xcd2cafadeed8fdf4,
        0x009713b03af0fed4,
    ]),
};

const G2_GENERATOR_X: Fq2 = Fq2 {
    c0: Fq::from_raw([
        0x46debd5cd992f6ed,
        0x674322d4f75edadd,
        0x426a00665e5c4479,
        0x1800deef121f1e76,
    ]),
    c1: Fq::from_raw([
        0x97e485b7aef312c2,
        0xf1aa493335a9e712,
        0x7260bfb731fb5d25,
        0x198e9393920d483a,
    ]),
};

const G2_GENERATOR_Y: Fq2 = Fq2 {
    c0: Fq::from_raw([
        0x4ce6cc0166fa7daa,
        0xe3d1e7690c43d37b,
        0x4aab71808dcb408f,
        0x12c85ea5db8c6deb,
    ]),

    c1: Fq::from_raw([
        0x55acdadcd122975b,
        0xbc4b313370b38ef3,
        0xec9e99ad690c3395,
        0x090689d0585ff075,
    ]),
};

// Generated using https://github.com/ConsenSys/gnark-crypto/blob/master/ecc/utils.go
// with `bn256::Fr::ZETA`
// See https://github.com/demining/Endomorphism-Secp256k1/blob/main/README.md
// to have more details about the endomorphism.
const ENDO_PARAMS_BN: EndoParameters = EndoParameters {
    // round(b2/n)
    gamma1: [0xd91d232ec7e0b3d7, 0x2, 0, 0],
    // round(-b1/n)
    gamma2: [0x5398fd0300ff6565, 0x4ccef014a773d2d2, 0x02, 0],
    b1: [0x89d3256894d213e3, 0, 0, 0],
    b2: [0x0be4e1541221250b, 0x6f4d8248eeb859fd, 0, 0],
};

endo!(G1, Fr, ENDO_PARAMS_BN);

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

impl CofactorGroup for G2 {
    type Subgroup = G2;

    fn clear_cofactor(&self) -> Self {
        fn exp_by_x(g2: &G2) -> G2 {
            let x = super::BN_X;
            let mut res = G2::identity();
            for i in (0..64).rev() {
                res = res.double();
                if ((x >> i) & 1) == 1 {
                    res += g2;
                }
            }
            res
        }

        fn psi(mut g2: G2) -> G2 {
            const U0: Fq = Fq::from_raw([
                0x99e39557176f553d,
                0xb78cc310c2c3330c,
                0x4c0bec3cf559b143,
                0x2fb347984f7911f7,
            ]);

            const U1: Fq = Fq::from_raw([
                0x1665d51c640fcba2,
                0x32ae2a1d0b7c9dce,
                0x4ba4cc8bd75a0794,
                0x16c9e55061ebae20,
            ]);
            let u = Fq2::new(U0, U1);

            const V0: Fq = Fq::from_raw([
                0xdc54014671a0135a,
                0xdbaae0eda9c95998,
                0xdc5ec698b6e2f9b9,
                0x063cf305489af5dc,
            ]);

            const V1: Fq = Fq::from_raw([
                0x82d37f632623b0e3,
                0x21807dc98fa25bd2,
                0x0704b5a7ec796f2b,
                0x07c03cbcac41049a,
            ]);
            let v = Fq2::new(V0, V1);

            g2.x.conjugate();
            g2.y.conjugate();
            g2.z.conjugate();

            g2.x *= u;
            g2.y *= v;

            g2
        }

        let u0 = exp_by_x(self);
        let u1 = psi(u0.double() + u0);
        let u2 = psi(psi(u0));
        let u3 = psi(psi(psi(*self)));

        u0 + u1 + u2 + u3
    }

    fn into_subgroup(self) -> CtOption<Self::Subgroup> {
        unimplemented!();
    }

    fn is_torsion_free(&self) -> Choice {
        // "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
        let e: [u8; 32] = [
            0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81,
            0x58, 0x5d, 0x28, 0x33, 0xe8, 0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93,
            0xf0, 0x00, 0x00, 0x01,
        ];

        // self * GROUP_ORDER;
        let mut acc = G2::identity();
        for bit in e
            .iter()
            .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
            .skip(1)
        {
            acc = acc.double();
            acc = G2::conditional_select(&acc, &(acc + self), bit);
        }
        acc.is_identity()
    }
}

impl G1 {
    const SVDW_Z: Fq = Fq::ONE;

    fn default_hash_to_curve_suite() -> crate::hash_to_curve::Suite<Self, sha2::Sha256, 48> {
        crate::hash_to_curve::Suite::<G1, sha2::Sha256, 48>::new(
            b"BN254G1_XMD:SHA-256_SVDW_RO_",
            Self::SVDW_Z,
            crate::hash_to_curve::Method::SVDW,
        )
    }
}

impl G2 {
    const SVDW_Z: Fq2 = Fq2::ONE;

    fn default_hash_to_curve_suite() -> crate::hash_to_curve::Suite<Self, sha2::Sha256, 96> {
        crate::hash_to_curve::Suite::<G2, sha2::Sha256, 96>::new(
            b"BN254G2_XMD:SHA-256_SVDW_RO_",
            Self::SVDW_Z,
            crate::hash_to_curve::Method::SVDW,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::tests::curve::TestH2C;

    use super::*;
    use group::UncompressedEncoding;

    crate::curve_testing_suite!(G2, "clear_cofactor");
    crate::curve_testing_suite!(G1, G2);
    crate::curve_testing_suite!(G1, "endo_consistency");
    crate::curve_testing_suite!(
        G1,
        "endo",
        // Optional `z_other` param. `z_other` is 3-roots of unity, similar to `ZETA`.
        // Reference: https://github.com/privacy-scaling-explorations/halo2curves/blob/main/src/bn256/fr.rs#L145-L151
        [
            0x8b17ea66b99c90dd,
            0x5bfc41088d8daaa7,
            0xb3c4d79d41a91758,
            0x00,
        ]
    );

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

    crate::curve_testing_suite!(
        G2,
        "constants",
        Fq2::MODULUS,
        G2_A,
        G2_B,
        G2_GENERATOR_X,
        G2_GENERATOR_Y,
        Fr::MODULUS
    );

    #[test]
    fn test_hash_to_curve_g1() {
        // Test vectors are taken from gnark-crypto/ecc/bn254/hash_vectors_test.go
        [
            TestH2C::<G1Affine>::new(
                b"",
                crate::tests::point_from_hex(
                    "0a976ab906170db1f9638d376514dbf8c42aef256a54bbd48521f20749e59e86",
                    "02925ead66b9e68bfc309b014398640ab55f6619ab59bc1fab2210ad4c4d53d5",
                ),
            ),
            TestH2C::<G1Affine>::new(
                b"abc",
                crate::tests::point_from_hex(
                    "23f717bee89b1003957139f193e6be7da1df5f1374b26a4643b0378b5baf53d1",
                    "04142f826b71ee574452dbc47e05bc3e1a647478403a7ba38b7b93948f4e151d",
                ),
            ),
            TestH2C::<G1Affine>::new(
                b"abcdef0123456789",
                crate::tests::point_from_hex(
                    "187dbf1c3c89aceceef254d6548d7163fdfa43084145f92c4c91c85c21442d4a",
                    "0abd99d5b0000910b56058f9cc3b0ab0a22d47cf27615f588924fac1e5c63b4d",
                ),
            ),
            TestH2C::<G1Affine>::new(
                b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
                crate::tests::point_from_hex(
                    "00fe2b0743575324fc452d590d217390ad48e5a16cf051bee5c40a2eba233f5c",
                    "0794211e0cc72d3cbbdf8e4e5cd6e7d7e78d101ff94862caae8acbe63e9fdc78",
                ),
            ),
            TestH2C::<G1Affine>::new(
                b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                crate::tests::point_from_hex(
                    "01b05dc540bd79fd0fea4fbb07de08e94fc2e7bd171fe025c479dc212a2173ce",
                    "1bf028afc00c0f843d113758968f580640541728cfc6d32ced9779aa613cd9b0",
                ),
            ),
        ].iter().for_each(|test| {
            test.run("QUUX-V01-CS02-with-");
        });
    }

    #[test]
    fn test_hash_to_curve_g2() {
        pub(crate) fn point_from_hex(x0: &str, x1: &str, y0: &str, y1: &str) -> G2Affine {
            let x0: Fq = crate::tests::hex_to_field(x0);
            let x1: Fq = crate::tests::hex_to_field(x1);
            let x = Fq2 { c0: x0, c1: x1 };
            let y0: Fq = crate::tests::hex_to_field(y0);
            let y1: Fq = crate::tests::hex_to_field(y1);
            let y = Fq2 { c0: y0, c1: y1 };
            G2Affine::from_xy(x, y).unwrap()
        }

        // Test vectors are taken from gnark-crypto/ecc/bn254/hash_vectors_test.go
        [
            TestH2C::<G2Affine>::new(
                b"",
                point_from_hex(
                    "1192005a0f121921a6d5629946199e4b27ff8ee4d6dd4f9581dc550ade851300",
                    "1747d950a6f23c16156e2171bce95d1189b04148ad12628869ed21c96a8c9335",
                    "0498f6bb5ac309a07d9a8b88e6ff4b8de0d5f27a075830e1eb0e68ea318201d8",
                    "2c9755350ca363ef2cf541005437221c5740086c2e909b71d075152484e845f4",
                ),
            ),
            TestH2C::<G2Affine>::new(
                b"abc",
                point_from_hex(
                    "16c88b54eec9af86a41569608cd0f60aab43464e52ce7e6e298bf584b94fccd2",
                    "0b5db3ca7e8ef5edf3a33dfc3242357fbccead98099c3eb564b3d9d13cba4efd",
                    "1c42ba524cb74db8e2c680449746c028f7bea923f245e69f89256af2d6c5f3ac",
                    "22d02d2da7f288545ff8789e789902245ab08c6b1d253561eec789ec2c1bd630",
                ),
            ),
            TestH2C::<G2Affine>::new(
                b"abcdef0123456789",
                point_from_hex(
                    "1435fd84aa43c699230e371f6fea3545ce7e053cbbb06a320296a2b81efddc70",
                    "2a8a360585b6b05996ef69c3c09b2c6fb17afe2b1e944f07559c53178eabf171",
                    "2820188dcdc13ffdca31694942418afa1d6dfaaf259d012fab4da52b0f592e38",
                    "142f08e2441ec431defc24621b73cfe0252d19b243cb55b84bdeb85de039207a",
                ),
            ),
            TestH2C::<G2Affine>::new(
                b"q128_qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",
                point_from_hex(
                    "2cffc213fb63d00d923cb22cda5a2904837bb93a2fe6e875c532c51744388341",
                    "2718ef38d1bc4347f0266c774c8ef4ee5fa7056cc27a4bd7ecf7a888efb95b26",
                    "232553f728341afa64ce66d00535764557a052e38657594e10074ad28728c584",
                    "2206ec0a9288f31ed78531c37295df3b56c42a1284443ee9893adb1521779001",
                ),
            ),
            TestH2C::<G2Affine>::new(
                b"a512_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                point_from_hex(
                    "242a0a159f36f87065e7c5170426012087023165ce47a486e53d6e2845ca625a",
                    "17f9f6292998cf18ccc155903c1fe6b6465d40c794a3e1ed644a4182ad639f4a",
                    "2dc5b7b65c9c79e6ef4afab8fbe3083c66d4ce31c78f6621ece17ecc892cf4b3",
                    "18ef4886c818f01fdf309bc9a46dd904273917f85e74ecd0de62460a68122037",
                ),
            ),
        ].iter().for_each(|test| {
            test.run("QUUX-V01-CS02-with-");
        });
    }
}
