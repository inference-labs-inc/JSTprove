use super::fq::Fq;
use super::fq2::Fq2;
use super::fq6::Fq6;
use crate::ff::Field;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// -GAMMA is a quadratic non-residue in Fp6. Fp12 = Fp6[X]/(X^2 + GAMMA)
/// We introduce the variable w such that w^2 = -GAMMA
// GAMMA = - v

/// An element of Fq12, represented by c0 + c1 * w.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct Fq12 {
    pub c0: Fq6,
    pub c1: Fq6,
}

impl ConditionallySelectable for Fq12 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fq12 {
            c0: Fq6::conditional_select(&a.c0, &b.c0, choice),
            c1: Fq6::conditional_select(&a.c1, &b.c1, choice),
        }
    }
}

impl ConstantTimeEq for Fq12 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1)
    }
}

impl Neg for Fq12 {
    type Output = Fq12;

    #[inline]
    fn neg(self) -> Fq12 {
        -&self
    }
}

impl<'a> Neg for &'a Fq12 {
    type Output = Fq12;

    #[inline]
    fn neg(self) -> Fq12 {
        self.neg()
    }
}

impl<'a, 'b> Sub<&'b Fq12> for &'a Fq12 {
    type Output = Fq12;

    #[inline]
    fn sub(self, rhs: &'b Fq12) -> Fq12 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fq12> for &'a Fq12 {
    type Output = Fq12;

    #[inline]
    fn add(self, rhs: &'b Fq12) -> Fq12 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fq12> for &'a Fq12 {
    type Output = Fq12;

    #[inline]
    fn mul(self, rhs: &'b Fq12) -> Fq12 {
        self.mul(rhs)
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    impl_sum_prod,
};
impl_binops_additive!(Fq12, Fq12);
impl_binops_multiplicative!(Fq12, Fq12);
impl_sum_prod!(Fq12);

impl Fq12 {
    #[inline]
    pub const fn zero() -> Self {
        Fq12 {
            c0: Fq6::ZERO,
            c1: Fq6::ZERO,
        }
    }

    #[inline]
    pub const fn one() -> Self {
        Fq12 {
            c0: Fq6::ONE,
            c1: Fq6::ZERO,
        }
    }

    pub fn mul_assign(&mut self, other: &Self) {
        let t0 = self.c0 * other.c0;
        let mut t1 = self.c1 * other.c1;
        let t2 = other.c0 + other.c1;

        self.c1 += &self.c0;
        self.c1 *= &t2;
        self.c1 -= &t0;
        self.c1 -= &t1;

        t1.mul_by_nonresidue();
        self.c0 = t0 + t1;
    }

    pub fn square_assign(&mut self) {
        let mut ab = self.c0 * self.c1;

        let c0c1 = self.c0 + self.c1;

        let mut c0 = self.c1;
        c0.mul_by_nonresidue();
        c0 += &self.c0;
        c0 *= &c0c1;
        c0 -= &ab;
        self.c1 = ab;
        self.c1 += &ab;
        ab.mul_by_nonresidue();
        c0 -= &ab;
        self.c0 = c0;
    }

    pub fn double(&self) -> Self {
        Self {
            c0: self.c0.double(),
            c1: self.c1.double(),
        }
    }

    pub fn double_assign(&mut self) {
        self.c0 = self.c0.double();
        self.c1 = self.c1.double();
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 + other.c0,
            c1: self.c1 + other.c1,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut t = *other;
        t.mul_assign(self);
        t
    }

    pub fn square(&self) -> Self {
        let mut t = *self;
        t.square_assign();
        t
    }

    #[inline(always)]
    pub fn neg(&self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }

    #[inline(always)]
    pub fn conjugate(&mut self) {
        self.c1 = -self.c1;
    }

    // pub fn conjugate(&self) -> Self {
    //     Self {
    //         c0: self.c0,
    //         c1: -self.c1,
    //     }
    // }

    pub fn frobenius_map(&mut self, power: usize) {
        self.c0.frobenius_map(power);
        self.c1.frobenius_map(power);

        self.c1.c0.mul_assign(&FROBENIUS_COEFF_FQ12_C1[power % 12]);
        self.c1.c1.mul_assign(&FROBENIUS_COEFF_FQ12_C1[power % 12]);
        self.c1.c2.mul_assign(&FROBENIUS_COEFF_FQ12_C1[power % 12]);
    }

    pub fn mul_by_014(&mut self, c0: &Fq2, c1: &Fq2, c4: &Fq2) {
        let mut aa = self.c0;
        aa.mul_by_01(c0, c1);
        let mut bb = self.c1;
        bb.mul_by_1(c4);
        let o = c1 + c4;
        self.c1 += &self.c0;
        self.c1.mul_by_01(c0, &o);
        self.c1 -= &aa;
        self.c1 -= &bb;
        self.c0 = bb;
        self.c0.mul_by_nonresidue();
        self.c0 += &aa;
    }

    pub fn mul_by_034(&mut self, c0: &Fq2, c3: &Fq2, c4: &Fq2) {
        let t0 = Fq6 {
            c0: self.c0.c0 * c0,
            c1: self.c0.c1 * c0,
            c2: self.c0.c2 * c0,
        };
        let mut t1 = self.c1;
        t1.mul_by_01(c3, c4);
        let o = c0 + c3;
        let mut t2 = self.c0 + self.c1;
        t2.mul_by_01(&o, c4);
        t2 -= t0;
        self.c1 = t2 - t1;
        t1.mul_by_nonresidue();
        self.c0 = t0 + t1;
    }

    pub fn invert(&self) -> CtOption<Self> {
        let mut c0s = self.c0;
        c0s.square_assign();
        let mut c1s = self.c1;
        c1s.square_assign();
        c1s.mul_by_nonresidue();
        c0s -= &c1s;

        c0s.invert().map(|t| {
            let mut tmp = Fq12 { c0: t, c1: t };
            tmp.c0.mul_assign(&self.c0);
            tmp.c1.mul_assign(&self.c1);
            tmp.c1 = tmp.c1.neg();

            tmp
        })
    }

    pub fn cyclotomic_square(&mut self) {
        fn fp4_square(c0: &mut Fq2, c1: &mut Fq2, a0: &Fq2, a1: &Fq2) {
            let t0 = a0.square();
            let t1 = a1.square();
            let mut t2 = t1;
            t2.mul_by_nonresidue();
            *c0 = t2 + t0;
            t2 = a0 + a1;
            t2.square_assign();
            t2 -= t0;
            *c1 = t2 - t1;
        }

        let mut t3 = Fq2::zero();
        let mut t4 = Fq2::zero();
        let mut t5 = Fq2::zero();
        let mut t6 = Fq2::zero();

        fp4_square(&mut t3, &mut t4, &self.c0.c0, &self.c1.c1);
        let mut t2 = t3 - self.c0.c0;
        t2.double_assign();
        self.c0.c0 = t2 + t3;

        t2 = t4 + self.c1.c1;
        t2.double_assign();
        self.c1.c1 = t2 + t4;

        fp4_square(&mut t3, &mut t4, &self.c1.c0, &self.c0.c2);
        fp4_square(&mut t5, &mut t6, &self.c0.c1, &self.c1.c2);

        t2 = t3 - self.c0.c1;
        t2.double_assign();
        self.c0.c1 = t2 + t3;
        t2 = t4 + self.c1.c2;
        t2.double_assign();
        self.c1.c2 = t2 + t4;
        t3 = t6;
        t3.mul_by_nonresidue();
        t2 = t3 + self.c1.c0;
        t2.double_assign();
        self.c1.c0 = t2 + t3;
        t2 = t5 - self.c0.c2;
        t2.double_assign();
        self.c0.c2 = t2 + t5;
    }
}

impl Field for Fq12 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fq12 {
            c0: Fq6::random(&mut rng),
            c1: Fq6::random(&mut rng),
        }
    }

    fn is_zero(&self) -> Choice {
        self.c0.is_zero() & self.c1.is_zero()
    }

    fn square(&self) -> Self {
        self.square()
    }

    fn double(&self) -> Self {
        self.double()
    }

    fn sqrt(&self) -> CtOption<Self> {
        unimplemented!()
    }

    fn sqrt_ratio(_num: &Self, _div: &Self) -> (Choice, Self) {
        unimplemented!()
    }

    fn invert(&self) -> CtOption<Self> {
        self.invert()
    }
}

// non_residue^((modulus^i-1)/6) for i=0,...,11
pub const FROBENIUS_COEFF_FQ12_C1: [Fq2; 12] = [
    // Fq2(u + 9)**(((q^0) - 1) / 6)
    // Fq points are represented in Montgomery form with R = 2^256
    Fq2 {
        c0: Fq([
            0xd35d438dc58f0d9d,
            0x0a78eb28f5c70b3d,
            0x666ea36f7879462c,
            0x0e0a77c19a07df2f,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^1) - 1) / 6)
    Fq2 {
        c0: Fq([
            0xaf9ba69633144907,
            0xca6b1d7387afb78a,
            0x11bded5ef08a2087,
            0x02f34d751a1f3a7c,
        ]),
        c1: Fq([
            0xa222ae234c492d72,
            0xd00f02a4565de15b,
            0xdc2ff3a253dfc926,
            0x10a75716b3899551,
        ]),
    },
    // Fq2(u + 9)**(((q^2) - 1) / 6)
    Fq2 {
        c0: Fq([
            0xca8d800500fa1bf2,
            0xf0c5d61468b39769,
            0x0e201271ad0d4418,
            0x04290f65bad856e6,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^3) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x365316184e46d97d,
            0x0af7129ed4c96d9f,
            0x659da72fca1009b5,
            0x08116d8983a20d23,
        ]),
        c1: Fq([
            0xb1df4af7c39c1939,
            0x3d9f02878a73bf7f,
            0x9b2220928caf0ae0,
            0x26684515eff054a6,
        ]),
    },
    // Fq2(u + 9)**(((q^4) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x3350c88e13e80b9c,
            0x7dce557cdb5e56b9,
            0x6001b4b8b615564a,
            0x2682e617020217e0,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^5) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x86b76f821b329076,
            0x408bf52b4d19b614,
            0x53dfb9d0d985e92d,
            0x051e20146982d2a7,
        ]),
        c1: Fq([
            0x0fbc9cd47752ebc7,
            0x6d8fffe33415de24,
            0xbef22cf038cf41b9,
            0x15c0edff3c66bf54,
        ]),
    },
    // Fq2(u + 9)**(((q^6) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x68c3488912edefaa,
            0x8d087f6872aabf4f,
            0x51e1a24709081231,
            0x2259d6b14729c0fa,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^7) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x8c84e580a568b440,
            0xcd164d1de0c21302,
            0xa692585790f737d5,
            0x2d7100fdc71265ad,
        ]),
        c1: Fq([
            0x99fdddf38c33cfd5,
            0xc77267ed1213e931,
            0xdc2052142da18f36,
            0x1fbcf75c2da80ad7,
        ]),
    },
    // Fq2(u + 9)**(((q^8) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x71930c11d782e155,
            0xa6bb947cffbe3323,
            0xaa303344d4741444,
            0x2c3b3f0d26594943,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^9) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x05cd75fe8a3623ca,
            0x8c8a57f293a85cee,
            0x52b29e86b7714ea8,
            0x2852e0e95d8f9306,
        ]),
        c1: Fq([
            0x8a41411f14e0e40e,
            0x59e26809ddfe0b0d,
            0x1d2e2523f4d24d7d,
            0x09fc095cf1414b83,
        ]),
    },
    // Fq2(u + 9)**(((q^10) - 1) / 6)
    Fq2 {
        c0: Fq([
            0x08cfc388c494f1ab,
            0x19b315148d1373d4,
            0x584e90fdcb6c0213,
            0x09e1685bdf2f8849,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^11) - 1) / 6)
    Fq2 {
        c0: Fq([
            0xb5691c94bd4a6cd1,
            0x56f575661b581478,
            0x64708be5a7fb6f30,
            0x2b462e5e77aecd82,
        ]),
        c1: Fq([
            0x2c63ef42612a1180,
            0x29f16aae345bec69,
            0xf95e18c648b216a4,
            0x1aa36073a4cae0d4,
        ]),
    },
];

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fq12, "field_arithmetic");
    // extension field-specific
    crate::field_testing_suite!(Fq12, "f12_tests", Fq6, Fq2);
    crate::field_testing_suite!(
        Fq12,
        "frobenius",
        // Frobenius endomorphism power parameter for extension field
        //  ϕ: E → E
        //  (x, y) ↦ (x^p, y^p)
        // p: modulus of base field (Here, Fq::MODULUS)
        [
            0x3c208c16d87cfd47,
            0x97816a916871ca8d,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]
    );
}
