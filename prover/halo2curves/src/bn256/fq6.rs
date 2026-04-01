use super::fq::Fq;
use super::fq2::Fq2;
use crate::ff::Field;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// -BETA is a cubic non-residue in Fp2. Fp6 = Fp2[X]/(X^3 + BETA)
/// We introduce the variable v such that v^3 = -BETA
// BETA = - (u + 9)

/// An element of Fq6, represented by c0 + c1 * v + c2 * v^2.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct Fq6 {
    pub c0: Fq2,
    pub c1: Fq2,
    pub c2: Fq2,
}

impl ConditionallySelectable for Fq6 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fq6 {
            c0: Fq2::conditional_select(&a.c0, &b.c0, choice),
            c1: Fq2::conditional_select(&a.c1, &b.c1, choice),
            c2: Fq2::conditional_select(&a.c2, &b.c2, choice),
        }
    }
}

impl ConstantTimeEq for Fq6 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1) & self.c2.ct_eq(&other.c2)
    }
}

impl Neg for Fq6 {
    type Output = Fq6;

    #[inline]
    fn neg(self) -> Fq6 {
        -&self
    }
}

impl<'a> Neg for &'a Fq6 {
    type Output = Fq6;

    #[inline]
    fn neg(self) -> Fq6 {
        self.neg()
    }
}

impl<'a, 'b> Sub<&'b Fq6> for &'a Fq6 {
    type Output = Fq6;

    #[inline]
    fn sub(self, rhs: &'b Fq6) -> Fq6 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fq6> for &'a Fq6 {
    type Output = Fq6;

    #[inline]
    fn add(self, rhs: &'b Fq6) -> Fq6 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fq6> for &'a Fq6 {
    type Output = Fq6;

    #[inline]
    fn mul(self, rhs: &'b Fq6) -> Fq6 {
        self.mul(rhs)
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    impl_sum_prod,
};
impl_binops_additive!(Fq6, Fq6);
impl_binops_multiplicative!(Fq6, Fq6);
impl_sum_prod!(Fq6);

impl Fq6 {
    #[inline]
    pub const fn zero() -> Self {
        Fq6 {
            c0: Fq2::ZERO,
            c1: Fq2::ZERO,
            c2: Fq2::ZERO,
        }
    }

    #[inline]
    pub const fn one() -> Self {
        Fq6 {
            c0: Fq2::ONE,
            c1: Fq2::ZERO,
            c2: Fq2::ZERO,
        }
    }

    pub fn mul_assign(&mut self, other: &Self) {
        let mut a_a = self.c0;
        let mut b_b = self.c1;
        let mut c_c = self.c2;
        a_a *= &other.c0;
        b_b *= &other.c1;
        c_c *= &other.c2;

        let mut t1 = other.c1;
        t1 += &other.c2;
        {
            let mut tmp = self.c1;
            tmp += &self.c2;

            t1 *= &tmp;
            t1 -= &b_b;
            t1 -= &c_c;
            t1.mul_by_nonresidue();
            t1 += &a_a;
        }

        let mut t3 = other.c0;
        t3 += &other.c2;
        {
            let mut tmp = self.c0;
            tmp += &self.c2;

            t3 *= &tmp;
            t3 -= &a_a;
            t3 += &b_b;
            t3 -= &c_c;
        }

        let mut t2 = other.c0;
        t2 += &other.c1;
        {
            let mut tmp = self.c0;
            tmp += &self.c1;

            t2 *= &tmp;
            t2 -= &a_a;
            t2 -= &b_b;
            c_c.mul_by_nonresidue();
            t2 += &c_c;
        }

        self.c0 = t1;
        self.c1 = t2;
        self.c2 = t3;
    }

    pub fn square_assign(&mut self) {
        // s0 = a^2
        let mut s0 = self.c0;
        s0.square_assign();
        // s1 = 2ab
        let mut ab = self.c0;
        ab *= &self.c1;
        let mut s1 = ab;
        s1.double_assign();
        // s2 = (a - b + c)^2
        let mut s2 = self.c0;
        s2 -= &self.c1;
        s2 += &self.c2;
        s2.square_assign();
        // bc
        let mut bc = self.c1;
        bc *= &self.c2;
        // s3 = 2bc
        let mut s3 = bc;
        s3.double_assign();
        // s4 = c^2
        let mut s4 = self.c2;
        s4.square_assign();

        // new c0 = 2bc.mul_by_xi + a^2
        self.c0 = s3;
        self.c0.mul_by_nonresidue();
        // self.c0.mul_by_xi();
        self.c0 += &s0;

        // new c1 = (c^2).mul_by_xi + 2ab
        self.c1 = s4;
        self.c1.mul_by_nonresidue();
        self.c1 += &s1;

        // new c2 = 2ab + (a - b + c)^2 + 2bc - a^2 - c^2 = b^2 + 2ac
        self.c2 = s1;
        self.c2 += &s2;
        self.c2 += &s3;
        self.c2 -= &s0;
        self.c2 -= &s4;
    }

    pub fn double(&self) -> Self {
        Self {
            c0: self.c0.double(),
            c1: self.c1.double(),
            c2: self.c2.double(),
        }
    }

    pub fn double_assign(&mut self) {
        self.c0 = self.c0.double();
        self.c1 = self.c1.double();
        self.c2 = self.c2.double();
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 + other.c0,
            c1: self.c1 + other.c1,
            c2: self.c2 + other.c2,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
            c2: self.c2 - other.c2,
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

    pub fn neg(&self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
        }
    }

    pub fn frobenius_map(&mut self, power: usize) {
        self.c0.frobenius_map(power);
        self.c1.frobenius_map(power);
        self.c2.frobenius_map(power);

        self.c1.mul_assign(&FROBENIUS_COEFF_FQ6_C1[power % 6]);
        self.c2.mul_assign(&FROBENIUS_COEFF_FQ6_C2[power % 6]);
    }

    /// Multiply by cubic nonresidue v.
    pub fn mul_by_nonresidue(&mut self) {
        use std::mem::swap;
        swap(&mut self.c0, &mut self.c1);
        swap(&mut self.c0, &mut self.c2);
        // c0, c1, c2 -> c2, c0, c1
        self.c0.mul_by_nonresidue();
    }

    pub fn mul_by_1(&mut self, c1: &Fq2) {
        let mut b_b = self.c1;
        b_b *= c1;

        let mut t1 = *c1;
        {
            let mut tmp = self.c1;
            tmp += &self.c2;

            t1 *= &tmp;
            t1 -= &b_b;
            t1.mul_by_nonresidue();
        }

        let mut t2 = *c1;
        {
            let mut tmp = self.c0;
            tmp += &self.c1;

            t2 *= &tmp;
            t2 -= &b_b;
        }

        self.c0 = t1;
        self.c1 = t2;
        self.c2 = b_b;
    }

    pub fn mul_by_01(&mut self, c0: &Fq2, c1: &Fq2) {
        let mut a_a = self.c0;
        let mut b_b = self.c1;
        a_a *= c0;
        b_b *= c1;

        let mut t1 = *c1;
        {
            let mut tmp = self.c1;
            tmp += &self.c2;

            t1 *= &tmp;
            t1 -= &b_b;
            t1.mul_by_nonresidue();
            t1 += &a_a;
        }

        let mut t3 = *c0;
        {
            let mut tmp = self.c0;
            tmp += &self.c2;

            t3 *= &tmp;
            t3 -= &a_a;
            t3 += &b_b;
        }

        let mut t2 = *c0;
        t2 += c1;
        {
            let mut tmp = self.c0;
            tmp += &self.c1;

            t2 *= &tmp;
            t2 -= &a_a;
            t2 -= &b_b;
        }

        self.c0 = t1;
        self.c1 = t2;
        self.c2 = t3;
    }

    fn invert(&self) -> CtOption<Self> {
        let mut c0 = self.c2;
        c0.mul_by_nonresidue();
        c0 *= &self.c1;
        c0 = -c0;
        {
            let mut c0s = self.c0;
            c0s.square_assign();
            c0 += &c0s;
        }
        let mut c1 = self.c2;
        c1.square_assign();
        c1.mul_by_nonresidue();
        {
            let mut c01 = self.c0;
            c01 *= &self.c1;
            c1 -= &c01;
        }
        let mut c2 = self.c1;
        c2.square_assign();
        {
            let mut c02 = self.c0;
            c02 *= &self.c2;
            c2 -= &c02;
        }

        let mut tmp1 = self.c2;
        tmp1 *= &c1;
        let mut tmp2 = self.c1;
        tmp2 *= &c2;
        tmp1 += &tmp2;
        tmp1.mul_by_nonresidue();
        tmp2 = self.c0;
        tmp2 *= &c0;
        tmp1 += &tmp2;

        tmp1.invert().map(|t| {
            let mut tmp = Fq6 {
                c0: t,
                c1: t,
                c2: t,
            };
            tmp.c0 *= &c0;
            tmp.c1 *= &c1;
            tmp.c2 *= &c2;

            tmp
        })
    }
}

impl Field for Fq6 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fq6 {
            c0: Fq2::random(&mut rng),
            c1: Fq2::random(&mut rng),
            c2: Fq2::random(&mut rng),
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

pub const FROBENIUS_COEFF_FQ6_C1: [Fq2; 6] = [
    // Fq2(u + 9)**(((q^0) - 1) / 3)
    Fq2 {
        c0: Fq([
            0xd35d438dc58f0d9d,
            0x0a78eb28f5c70b3d,
            0x666ea36f7879462c,
            0x0e0a77c19a07df2f,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^1) - 1) / 3)
    // taken from go-ethereum and also re-calculated manually
    Fq2 {
        c0: Fq([
            0xb5773b104563ab30,
            0x347f91c8a9aa6454,
            0x7a007127242e0991,
            0x1956bcd8118214ec,
        ]),
        c1: Fq([
            0x6e849f1ea0aa4757,
            0xaa1c7b6d89f89141,
            0xb6e713cdfae0ca3a,
            0x26694fbb4e82ebc3,
        ]),
    },
    // Fq2(u + 9)**(((q^2) - 1) / 3)
    // this one and other below are recalculated manually
    Fq2 {
        c0: Fq([
            0x3350c88e13e80b9c,
            0x7dce557cdb5e56b9,
            0x6001b4b8b615564a,
            0x2682e617020217e0,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^3) - 1) / 3)
    Fq2 {
        c0: Fq([
            0xc9af22f716ad6bad,
            0xb311782a4aa662b2,
            0x19eeaf64e248c7f4,
            0x20273e77e3439f82,
        ]),
        c1: Fq([
            0xacc02860f7ce93ac,
            0x3933d5817ba76b4c,
            0x69e6188b446c8467,
            0x0a46036d4417cc55,
        ]),
    },
    // Fq2(u + 9)**(((q^4) - 1) / 3)
    Fq2 {
        c0: Fq([
            0x71930c11d782e155,
            0xa6bb947cffbe3323,
            0xaa303344d4741444,
            0x2c3b3f0d26594943,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((q^5) - 1) / 3)
    Fq2 {
        c0: Fq([
            0xf91aba2654e8e3b1,
            0x4771cb2fdc92ce12,
            0xdcb16ae0fc8bdf35,
            0x274aa195cd9d8be4,
        ]),
        c1: Fq([
            0x5cfc50ae18811f8b,
            0x4bb28433cb43988c,
            0x4fd35f13c3b56219,
            0x301949bd2fc8883a,
        ]),
    },
];

pub const FROBENIUS_COEFF_FQ6_C2: [Fq2; 6] = [
    // Fq2(u + 9)**(((2q^0) - 2) / 3)
    Fq2 {
        c0: Fq([
            0xd35d438dc58f0d9d,
            0x0a78eb28f5c70b3d,
            0x666ea36f7879462c,
            0x0e0a77c19a07df2f,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((2q^1) - 2) / 3)
    Fq2 {
        c0: Fq([
            0x7361d77f843abe92,
            0xa5bb2bd3273411fb,
            0x9c941f314b3e2399,
            0x15df9cddbb9fd3ec,
        ]),
        c1: Fq([
            0x5dddfd154bd8c949,
            0x62cb29a5a4445b60,
            0x37bc870a0c7dd2b9,
            0x24830a9d3171f0fd,
        ]),
    },
    // Fq2(u + 9)**(((2q^2) - 2) / 3)
    Fq2 {
        c0: Fq([
            0x71930c11d782e155,
            0xa6bb947cffbe3323,
            0xaa303344d4741444,
            0x2c3b3f0d26594943,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((2q^3) - 2) / 3)
    Fq2 {
        c0: Fq([
            0x448a93a57b6762df,
            0xbfd62df528fdeadf,
            0xd858f5d00e9bd47a,
            0x06b03d4d3476ec58,
        ]),
        c1: Fq([
            0x2b19daf4bcc936d1,
            0xa1a54e7a56f4299f,
            0xb533eee05adeaef1,
            0x170c812b84dda0b2,
        ]),
    },
    // Fq2(u + 9)**(((2q^4) - 2) / 3)
    Fq2 {
        c0: Fq([
            0x3350c88e13e80b9c,
            0x7dce557cdb5e56b9,
            0x6001b4b8b615564a,
            0x2682e617020217e0,
        ]),
        c1: Fq([0x0, 0x0, 0x0, 0x0]),
    },
    // Fq2(u + 9)**(((2q^5) - 2) / 3)
    Fq2 {
        c0: Fq([
            0x843420f1d8dadbd6,
            0x31f010c9183fcdb2,
            0x436330b527a76049,
            0x13d47447f11adfe4,
        ]),
        c1: Fq([
            0xef494023a857fa74,
            0x2a925d02d5ab101a,
            0x83b015829ba62f10,
            0x2539111d0c13aea3,
        ]),
    },
];

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fq6, "field_arithmetic");
    // extension field-specific
    crate::field_testing_suite!(Fq6, "f6_tests", Fq2);
    crate::field_testing_suite!(
        Fq6,
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
