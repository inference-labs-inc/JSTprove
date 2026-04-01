#![allow(clippy::suspicious_arithmetic_impl)]
use crate::ff::{Field, PrimeField};
use crate::group::cofactor::CofactorCurveAffine;
use crate::group::Group;
use crate::pairing::{Engine, MillerLoopResult, MultiMillerLoop, PairingCurveAffine};
use crate::pluto_eris::curve::*;
use crate::pluto_eris::fields::fp::*;
use crate::pluto_eris::fields::fp12::*;
use crate::pluto_eris::fields::fp2::*;
use crate::pluto_eris::fields::fp6::FROBENIUS_COEFF_FP6_C1;
use crate::pluto_eris::fields::fq::*;
use core::borrow::Borrow;
use core::iter::Sum;
use core::ops::{Add, Mul, MulAssign, Neg, Sub};
use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// Adaptation of Algorithm 1, https://eprint.iacr.org/2013/722.pdf
/// the parameter for the curve Pluto: u = -0x4000000000001000008780000000
const NEG_PLUTO_U: u128 = 0x4000000000001000008780000000;

const NEG_SIX_U_PLUS_2_NAF: [i8; 114] = [
    0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, -1, 0, -1, 0, 1, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1,
];

/// Value of (57/(u + 3))^((p - 1)/2) where u^2 + 5 = 0 in Fp2.
const XI_TO_P_MINUS_1_OVER_2: Fp2 = Fp2 {
    c0: Fp::from_raw([
        0x54cf5ad1c0926216,
        0x186c1f3ce4a46d4e,
        0x9c23800ce9c9452f,
        0x50e0d09ff6d6c08b,
        0x7cf421e4d46f6666,
        0x678664ba4b6d8343,
        0x21cc26d5de0f80f4,
    ]),

    c1: Fp::from_raw([
        0xc0505f4c260e91f4,
        0xe7bbd15f10723657,
        0xb4b3e0c35358097e,
        0x87c56f42a558750d,
        0x4b7211d23f34f0ae,
        0xf6839d29e2f0d250,
        0x16ebe8b2e12a1106,
    ]),
};

impl PairingCurveAffine for G1Affine {
    type Pair = G2Affine;
    type PairingResult = Gt;

    fn pairing_with(&self, other: &Self::Pair) -> Self::PairingResult {
        Pluto::pairing(self, other)
    }
}

impl PairingCurveAffine for G2Affine {
    type Pair = G1Affine;
    type PairingResult = Gt;

    fn pairing_with(&self, other: &Self::Pair) -> Self::PairingResult {
        Pluto::pairing(other, self)
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Gt(pub(crate) Fp12);

impl std::fmt::Display for Gt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ConstantTimeEq for Gt {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl ConditionallySelectable for Gt {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Gt(Fp12::conditional_select(&a.0, &b.0, choice))
    }
}

impl Eq for Gt {}
impl PartialEq for Gt {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        bool::from(self.ct_eq(other))
    }
}

impl Gt {
    /// Returns the group identity, which is $1$.
    pub const fn identity() -> Gt {
        Gt(Fp12::ONE)
    }

    /// Doubles this group element.
    pub fn double(&self) -> Gt {
        Gt(self.0.square())
    }
}

impl<'a> Neg for &'a Gt {
    type Output = Gt;

    #[inline]
    fn neg(self) -> Gt {
        // The element is unitary, so we just conjugate.
        let mut u = self.0;
        u.conjugate();
        Gt(u)
    }
}

impl Neg for Gt {
    type Output = Gt;

    #[inline]
    fn neg(self) -> Gt {
        -&self
    }
}

impl<'a, 'b> Add<&'b Gt> for &'a Gt {
    type Output = Gt;

    #[inline]
    fn add(self, rhs: &'b Gt) -> Gt {
        Gt(self.0 * rhs.0)
    }
}

impl<'a, 'b> Sub<&'b Gt> for &'a Gt {
    type Output = Gt;

    #[inline]
    fn sub(self, rhs: &'b Gt) -> Gt {
        self + (-rhs)
    }
}

impl<'a, 'b> Mul<&'b Fq> for &'a Gt {
    type Output = Gt;

    fn mul(self, other: &'b Fq) -> Self::Output {
        let mut acc = Gt::identity();

        for bit in other
            .to_repr()
            .iter()
            .rev()
            .flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
            .skip(1)
        {
            acc = acc.double();
            acc = Gt::conditional_select(&acc, &(acc + self), bit);
        }

        acc
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
};
impl_binops_additive!(Gt, Gt);
impl_binops_multiplicative!(Gt, Fq);

impl<T> Sum<T> for Gt
where
    T: Borrow<Gt>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.fold(Self::identity(), |acc, item| acc + item.borrow())
    }
}

impl Group for Gt {
    type Scalar = Fq;

    fn random(_: impl RngCore) -> Self {
        unimplemented!();
    }

    fn identity() -> Self {
        Self::identity()
    }

    fn generator() -> Self {
        unimplemented!();
    }

    fn is_identity(&self) -> Choice {
        self.ct_eq(&Self::identity())
    }

    #[must_use]
    fn double(&self) -> Self {
        self.double()
    }
}

/// Points of G2 in Jacobian coordinates.
/// These are points lie in the twisted curve E'(Fp2).
#[derive(Clone, Debug)]
pub struct G2Prepared {
    pub(crate) coeffs: Vec<(Fp2, Fp2, Fp2)>,
    pub(crate) infinity: bool,
}

impl G2Prepared {
    /// Returns true if `self` is the infinity point.
    pub fn is_zero(&self) -> bool {
        self.infinity
    }

    /// Prepares a G2 point in affine coordinates.
    pub fn from_affine(q: G2Affine) -> Self {
        if bool::from(q.is_identity()) {
            return G2Prepared {
                coeffs: vec![],
                infinity: true,
            };
        }

        /// Adaptation of Algorithm 26, https://eprint.iacr.org/2010/354.pdf
        fn doubling_step(r: &mut G2) -> (Fp2, Fp2, Fp2) {
            let mut tmp0 = r.x;
            tmp0.square_assign();

            let mut tmp1 = r.y;
            tmp1.square_assign();

            let mut tmp2 = tmp1;
            tmp2.square_assign();

            let mut tmp3 = tmp1;
            tmp3 += &r.x;
            tmp3.square_assign();
            tmp3 -= &tmp0;
            tmp3 -= &tmp2;
            tmp3.double_assign();

            let mut tmp4 = tmp0;
            tmp4.double_assign();
            tmp4 += &tmp0;

            let mut tmp6 = r.x;
            tmp6 += &tmp4;

            let mut tmp5 = tmp4;
            tmp5.square_assign();

            let mut zsquared = r.z;
            zsquared.square_assign();

            r.x = tmp5;
            r.x -= &tmp3;
            r.x -= &tmp3;

            r.z += &r.y;
            r.z.square_assign();
            r.z -= &tmp1;
            r.z -= &zsquared;

            r.y = tmp3;
            r.y -= &r.x;
            r.y.mul_assign(&tmp4);

            tmp2.double_assign();
            tmp2.double_assign();
            tmp2.double_assign();

            r.y -= &tmp2;

            // up to here everything was by algorithm, line 11
            // use R instead of new T

            // tmp3 is the first part of line 12
            tmp3 = tmp4;
            tmp3.mul_assign(&zsquared);
            tmp3.double_assign();
            tmp3 = tmp3.neg();

            // tmp6 is from line 14
            tmp6.square_assign();
            tmp6 -= &tmp0;
            tmp6 -= &tmp5;

            tmp1.double_assign();
            tmp1.double_assign();

            tmp6 -= &tmp1;

            // tmp0 is the first part of line 16
            tmp0 = r.z;
            tmp0.mul_assign(&zsquared);
            tmp0.double_assign();

            (tmp0, tmp3, tmp6)
        }

        // Adaptation of Algorithm 27, https://eprint.iacr.org/2010/354.pdf
        fn addition_step(r: &mut G2, q: &G2Affine) -> (Fp2, Fp2, Fp2) {
            let mut zsquared = r.z;
            zsquared.square_assign();

            let mut ysquared = q.y;
            ysquared.square_assign();

            // t0 corresponds to line 1
            let mut t0 = zsquared;
            t0.mul_assign(&q.x);

            // t1 corresponds to lines 2 and 3
            let mut t1 = q.y;
            t1 += &r.z;
            t1.square_assign();
            t1 -= &ysquared;
            t1 -= &zsquared;
            t1.mul_assign(&zsquared);

            // t2 corresponds to line 4
            let mut t2 = t0;
            t2 -= &r.x;

            // t3 corresponds to line 5
            let mut t3 = t2;
            t3.square_assign();

            // t4 corresponds to line 6
            let mut t4 = t3;
            t4.double_assign();
            t4.double_assign();

            // t5 corresponds to line 7
            let mut t5 = t4;
            t5.mul_assign(&t2);

            // t6 corresponds to line 8
            let mut t6 = t1;
            t6 -= &r.y;
            t6 -= &r.y;

            // t9 corresponds to line 9
            let mut t9 = t6;
            t9.mul_assign(&q.x);

            // corresponds to line 10
            let mut t7 = t4;
            t7.mul_assign(&r.x);

            // corresponds to line 11, but assigns to r.x instead of T.x
            r.x = t6;
            r.x.square_assign();
            r.x -= &t5;
            r.x -= &t7;
            r.x -= &t7;

            // corresponds to line 12, but assigns to r.z instead of T.z
            r.z += &t2;
            r.z.square_assign();
            r.z -= &zsquared;
            r.z -= &t3;

            // corresponds to line 13
            let mut t10 = q.y;
            t10 += &r.z;

            // corresponds to line 14
            let mut t8 = t7;
            t8 -= &r.x;
            t8.mul_assign(&t6);

            // corresponds to line 15
            t0 = r.y;
            t0.mul_assign(&t5);
            t0.double_assign();

            // corresponds to line 12, but assigns to r.y instead of T.y
            r.y = t8;
            r.y -= &t0;

            // corresponds to line 17
            t10.square_assign();
            t10 -= &ysquared;

            let mut ztsquared = r.z;
            ztsquared.square_assign();

            t10 -= &ztsquared;

            // corresponds to line 18
            t9.double_assign();
            t9 -= &t10;

            // t10 = 2*Zt from Algo 27, line 19
            t10 = r.z;
            t10.double_assign();

            // t1 = first multiplicator of line 21
            t6 = t6.neg();

            t1 = t6;
            t1.double_assign();

            // t9 corresponds to t9 from Algo 27
            (t10, t1, t9)
        }

        let mut coeffs = vec![];
        let mut r: G2 = q.into();

        let mut negq = q;
        negq = -negq;

        coeffs.push(doubling_step(&mut r));

        let last_position = NEG_SIX_U_PLUS_2_NAF.len() - 2;
        match NEG_SIX_U_PLUS_2_NAF[last_position] {
            1 => {
                coeffs.push(addition_step(&mut r, &q));
            }
            -1 => {
                coeffs.push(addition_step(&mut r, &negq));
            }
            _ => (),
        }

        for i in (0..last_position).rev() {
            coeffs.push(doubling_step(&mut r));

            match NEG_SIX_U_PLUS_2_NAF[i] {
                1 => {
                    coeffs.push(addition_step(&mut r, &q));
                }
                -1 => {
                    coeffs.push(addition_step(&mut r, &negq));
                }
                _ => continue,
            }
        }

        let mut neg_r = r;
        neg_r = -neg_r;

        let mut q1 = q;

        q1.x.c1 = q1.x.c1.neg();
        q1.x.mul_assign(&FROBENIUS_COEFF_FP6_C1[1]);

        q1.y.c1 = q1.y.c1.neg();
        q1.y.mul_assign(&XI_TO_P_MINUS_1_OVER_2);

        coeffs.push(addition_step(&mut neg_r, &q1));

        let mut minusq2 = q;
        minusq2.x.mul_assign(&FROBENIUS_COEFF_FP6_C1[2]);

        coeffs.push(addition_step(&mut neg_r, &minusq2));

        G2Prepared {
            coeffs,
            infinity: false,
        }
    }
}

impl From<G2Affine> for G2Prepared {
    fn from(q: G2Affine) -> G2Prepared {
        G2Prepared::from_affine(q)
    }
}

impl MillerLoopResult for Gt {
    type Gt = Self;
    fn final_exponentiation(&self) -> Gt {
        fn exp_by_x(f: &mut Fp12) {
            let x = NEG_PLUTO_U;
            let mut res = Fp12::ONE;
            for i in (0..111).rev() {
                res.cyclotomic_square();
                if ((x >> i) & 1) == 1 {
                    res.mul_assign(f);
                }
            }
            res.conjugate();
            *f = res;
        }

        let r = self.0;
        let mut f1 = self.0;
        f1.conjugate();

        Gt(r.invert()
            .map(|mut f2| {
                let mut r = f1;
                r.mul_assign(&f2);
                f2 = r;
                r.frobenius_map(2);
                r.mul_assign(&f2);

                let mut fp = r;
                fp.frobenius_map(1);

                let mut fp2 = r;
                fp2.frobenius_map(2);
                let mut fp3 = fp2;
                fp3.frobenius_map(1);

                let mut fu = r;
                exp_by_x(&mut fu);

                let mut fu2 = fu;
                exp_by_x(&mut fu2);

                let mut fu3 = fu2;
                exp_by_x(&mut fu3);

                let mut y3 = fu;
                y3.frobenius_map(1);

                let mut fu2p = fu2;
                fu2p.frobenius_map(1);

                let mut fu3p = fu3;
                fu3p.frobenius_map(1);

                let mut y2 = fu2;
                y2.frobenius_map(2);

                let mut y0 = fp;
                y0.mul_assign(&fp2);
                y0.mul_assign(&fp3);

                let mut y1 = r;
                y1.conjugate();

                let mut y5 = fu2;
                y5.conjugate();

                y3.conjugate();

                let mut y4 = fu;
                y4.mul_assign(&fu2p);
                y4.conjugate();

                let mut y6 = fu3;
                y6.mul_assign(&fu3p);
                y6.conjugate();

                y6.cyclotomic_square();
                y6.mul_assign(&y4);
                y6.mul_assign(&y5);

                let mut t1 = y3;
                t1.mul_assign(&y5);
                t1.mul_assign(&y6);

                y6.mul_assign(&y2);

                t1.cyclotomic_square();
                t1.mul_assign(&y6);
                t1.cyclotomic_square();

                let mut t0 = t1;
                t0.mul_assign(&y1);

                t1.mul_assign(&y0);

                t0.cyclotomic_square();
                t0.mul_assign(&t1);

                t0
            })
            .unwrap())
    }
}
impl MultiMillerLoop for Pluto {
    /// The prepared form of `Self::G2Affine`.
    type G2Prepared = G2Prepared;

    /// The type returned by `Engine::miller_loop`.
    type Result = Gt;
    fn multi_miller_loop(terms: &[(&G1Affine, &G2Prepared)]) -> Self::Result {
        let mut pairs = vec![];
        for &(p, q) in terms {
            if !bool::from(p.is_identity()) && !q.is_zero() {
                pairs.push((p, q.coeffs.iter()));
            }
        }

        // Final steps of the line function on prepared coefficients
        fn ell(f: &mut Fp12, coeffs: &(Fp2, Fp2, Fp2), p: &G1Affine) {
            let mut c0 = coeffs.0;
            let mut c1 = coeffs.1;

            c0.c0.mul_assign(&p.y);
            c0.c1.mul_assign(&p.y);

            c1.c0.mul_assign(&p.x);
            c1.c1.mul_assign(&p.x);

            // Sparse multiplication in Fq12
            f.mul_by_034(&c0, &c1, &coeffs.2);
        }

        let mut f = Fp12::ONE;

        for &mut (p, ref mut coeffs) in &mut pairs {
            ell(&mut f, coeffs.next().unwrap(), p);
        }

        // length - 2
        let len_min2 = NEG_SIX_U_PLUS_2_NAF.len() - 2;

        if NEG_SIX_U_PLUS_2_NAF[len_min2] != 0 {
            for &mut (p, ref mut coeffs) in &mut pairs {
                ell(&mut f, coeffs.next().unwrap(), p);
            }
        }

        for x in NEG_SIX_U_PLUS_2_NAF[..len_min2].iter().rev() {
            f.square_assign();

            for &mut (p, ref mut coeffs) in &mut pairs {
                ell(&mut f, coeffs.next().unwrap(), p);
            }
            if *x != 0 {
                for &mut (p, ref mut coeffs) in &mut pairs {
                    ell(&mut f, coeffs.next().unwrap(), p);
                }
            }
        }

        f.conjugate();

        for &mut (p, ref mut coeffs) in &mut pairs {
            ell(&mut f, coeffs.next().unwrap(), p);
        }

        for &mut (p, ref mut coeffs) in &mut pairs {
            ell(&mut f, coeffs.next().unwrap(), p);
        }

        for &mut (_p, ref mut coeffs) in &mut pairs {
            assert_eq!(coeffs.next(), None);
        }

        Gt(f)
    }
}

/// Pluto pairing-friendly curve. See: https://github.com/daira/pluto-eris
#[derive(Clone, Debug)]
pub struct Pluto;

impl Engine for Pluto {
    type Fr = Fq;
    type G1 = G1;
    type G1Affine = G1Affine;
    type G2 = G2;
    type G2Affine = G2Affine;
    type Gt = Gt;

    fn pairing(p: &Self::G1Affine, q: &Self::G2Affine) -> Self::Gt {
        let q = G2Prepared::from_affine(*q);
        let terms: &[(&G1Affine, &G2Prepared)] = &[(p, &q)];
        let u = Self::multi_miller_loop(terms);
        u.final_exponentiation()
    }
}

#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand_xorshift::XorShiftRng;

#[test]
fn test_pairing() {
    let g1 = G1::generator();
    let mut g2 = G2::generator();
    g2 = g2.double();
    let pair12 = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(g2));

    let mut g1 = G1::generator();
    let g2 = G2::generator();
    g1 = g1.double();
    let pair21 = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(g2));

    assert_eq!(pair12, pair21);

    let g1 = G1::generator();
    let mut g2 = G2::generator();
    g2 = g2.double().double();
    let pair12 = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(g2));

    let mut g1 = G1::generator();
    let mut g2 = G2::generator();
    g1 = g1.double();
    g2 = g2.double();
    let pair21 = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(g2));

    assert_eq!(pair12, pair21);

    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);
    for _ in 0..100 {
        let a = Fq::random(&mut rng);
        let b = Fq::random(&mut rng);

        let mut g1 = G1::generator();
        g1.mul_assign(a);

        let mut g2 = G2::generator();
        g1.mul_assign(b);

        let pair_ab = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(g2));

        g1 = G1::generator();
        g1.mul_assign(b);

        g2 = G2::generator();
        g1.mul_assign(a);

        let pair_ba = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(g2));

        assert_eq!(pair_ab, pair_ba);
    }
}

#[test]
fn tricking_miller_loop_result() {
    assert_eq!(
        Pluto::multi_miller_loop(&[(&G1Affine::identity(), &G2Affine::generator().into())]).0,
        Fp12::one()
    );
    assert_eq!(
        Pluto::multi_miller_loop(&[(&G1Affine::generator(), &G2Affine::identity().into())]).0,
        Fp12::one()
    );
    assert_ne!(
        Pluto::multi_miller_loop(&[
            (&G1Affine::generator(), &G2Affine::generator().into()),
            (&-G1Affine::generator(), &G2Affine::generator().into())
        ])
        .0,
        Fp12::one()
    );
    assert_eq!(
        Pluto::multi_miller_loop(&[
            (&G1Affine::generator(), &G2Affine::generator().into()),
            (&-G1Affine::generator(), &G2Affine::generator().into())
        ])
        .final_exponentiation(),
        Gt::identity()
    );
}

#[test]
fn random_bilinearity_tests() {
    let mut rng = XorShiftRng::from_seed([
        0x55, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    for _ in 0..10 {
        let mut a = G1::generator();
        let ka = Fq::random(&mut rng);
        a.mul_assign(ka);

        let mut b = G2::generator();
        let kb = Fq::random(&mut rng);
        b.mul_assign(kb);

        let c = Fq::random(&mut rng);
        let d = Fq::random(&mut rng);

        let mut ac = a;
        ac.mul_assign(c);

        let mut ad = a;
        ad.mul_assign(d);

        let mut bc = b;
        bc.mul_assign(c);

        let mut bd = b;
        bd.mul_assign(d);

        let acbd = Pluto::pairing(&G1Affine::from(ac), &G2Affine::from(bd));
        let adbc = Pluto::pairing(&G1Affine::from(ad), &G2Affine::from(bc));

        let mut cd = c;
        cd.mul_assign(&d);

        cd *= Fq([1, 0, 0, 0, 0, 0, 0]);

        let abcd = Gt(Pluto::pairing(&G1Affine::from(a), &G2Affine::from(b))
            .0
            .pow_vartime(cd.0));

        assert_eq!(acbd, adbc);
        assert_eq!(acbd, abcd);
    }
}

#[test]
pub fn engine_tests() {
    let mut rng = XorShiftRng::from_seed([
        0x56, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    for _ in 0..10 {
        let a = G1Affine::from(G1::random(&mut rng));
        let b = G2Affine::from(G2::random(&mut rng));

        assert!(a.pairing_with(&b) == b.pairing_with(&a));
        assert!(a.pairing_with(&b) == Pluto::pairing(&a, &b));
    }

    for _ in 0..10 {
        let z1 = G1Affine::identity();
        let z2 = G2Prepared::from(G2Affine::identity());

        let a = G1Affine::from(G1::random(&mut rng));
        let b = G2Prepared::from(G2Affine::from(G2::random(&mut rng)));
        let c = G1Affine::from(G1::random(&mut rng));
        let d = G2Prepared::from(G2Affine::from(G2::random(&mut rng)));

        assert_eq!(
            Fp12::ONE,
            Pluto::multi_miller_loop(&[(&z1, &b)])
                .final_exponentiation()
                .0,
        );

        assert_eq!(
            Fp12::ONE,
            Pluto::multi_miller_loop(&[(&a, &z2)])
                .final_exponentiation()
                .0,
        );

        assert_eq!(
            Pluto::multi_miller_loop(&[(&z1, &b), (&c, &d)]).final_exponentiation(),
            Pluto::multi_miller_loop(&[(&a, &z2), (&c, &d)]).final_exponentiation(),
        );

        assert_eq!(
            Pluto::multi_miller_loop(&[(&a, &b), (&z1, &d)]).final_exponentiation(),
            Pluto::multi_miller_loop(&[(&a, &b), (&c, &z2)]).final_exponentiation(),
        );
    }
}

#[test]
fn random_miller_loop_tests() {
    let mut rng = XorShiftRng::from_seed([
        0x58, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    // Exercise a double miller loop
    for _ in 0..10 {
        let a = G1Affine::from(G1::random(&mut rng));
        let b = G2Affine::from(G2::random(&mut rng));
        let c = G1Affine::from(G1::random(&mut rng));
        let d = G2Affine::from(G2::random(&mut rng));

        let ab = Pluto::pairing(&a, &b);
        let cd = Pluto::pairing(&c, &d);

        let mut abcd = ab;
        abcd = Gt(abcd.0 * cd.0);

        let b = G2Prepared::from(b);
        let d = G2Prepared::from(d);

        let abcd_with_double_loop =
            Pluto::multi_miller_loop(&[(&a, &b), (&c, &d)]).final_exponentiation();

        assert_eq!(abcd, abcd_with_double_loop);
    }
}

#[test]
pub fn multi_miller_final_exp_tests() {
    let g1 = G1::generator();
    let g2 = G2::generator();

    let mut rng = XorShiftRng::from_seed([
        0x56, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    for _ in 0..10 {
        let s = Fq::random(&mut rng);

        let mut s_g1 = g1;
        s_g1.mul_assign(s);

        let mut s_g2 = g2;
        s_g2.mul_assign(s);

        let s_g2_prepared = G2Prepared::from(G2Affine::from(s_g2));
        let g2_prepared = G2Prepared::from(G2Affine::from(g2));

        let (term_1, term_2) = (
            (&G1Affine::from(g1), &s_g2_prepared),
            (&-G1Affine::from(s_g1), &g2_prepared),
        );

        let terms = &[term_1, term_2];

        assert!(
            bool::from(
                Pluto::multi_miller_loop(&terms[..])
                    .final_exponentiation()
                    .is_identity(),
            ),
            "trivial pairing check failed"
        );

        let lhs = Pluto::pairing(&G1Affine::from(g1), &G2Affine::from(s_g2));
        let rhs = Pluto::pairing(&G1Affine::from(s_g1), &G2Affine::from(g2));

        assert_eq!(lhs, rhs, "failed trivial check");
    }
}
