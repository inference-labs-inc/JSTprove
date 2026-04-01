use super::fp::Fp;
use super::fp2::Fp2;
use super::fp6::Fp6;
use crate::ff::Field;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// -GAMMA is a quadratic non-residue in Fp6. Fp12 = Fp6[X]/(X^2 + GAMMA)
/// We introduce the variable w such that w^2 = -GAMMA
/// GAMMA = - v
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct Fp12 {
    c0: Fp6,
    c1: Fp6,
}

impl ConditionallySelectable for Fp12 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fp12 {
            c0: Fp6::conditional_select(&a.c0, &b.c0, choice),
            c1: Fp6::conditional_select(&a.c1, &b.c1, choice),
        }
    }
}

impl ConstantTimeEq for Fp12 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1)
    }
}

impl Neg for Fp12 {
    type Output = Fp12;

    #[inline]
    fn neg(self) -> Fp12 {
        -&self
    }
}

impl<'a> Neg for &'a Fp12 {
    type Output = Fp12;

    #[inline]
    fn neg(self) -> Fp12 {
        self.neg()
    }
}

impl<'a, 'b> Sub<&'b Fp12> for &'a Fp12 {
    type Output = Fp12;

    #[inline]
    fn sub(self, rhs: &'b Fp12) -> Fp12 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fp12> for &'a Fp12 {
    type Output = Fp12;

    #[inline]
    fn add(self, rhs: &'b Fp12) -> Fp12 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fp12> for &'a Fp12 {
    type Output = Fp12;

    #[inline]
    fn mul(self, rhs: &'b Fp12) -> Fp12 {
        self.mul(rhs)
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    impl_sum_prod,
};
impl_binops_additive!(Fp12, Fp12);
impl_binops_multiplicative!(Fp12, Fp12);
impl_sum_prod!(Fp12);

impl Fp12 {
    #[inline]
    pub const fn zero() -> Self {
        Fp12 {
            c0: Fp6::ZERO,
            c1: Fp6::ZERO,
        }
    }

    #[inline]
    pub const fn one() -> Self {
        Fp12 {
            c0: Fp6::ONE,
            c1: Fp6::ZERO,
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

    pub fn frobenius_map(&mut self, power: usize) {
        self.c0.frobenius_map(power);
        self.c1.frobenius_map(power);

        self.c1.c0.mul_assign(&FROBENIUS_COEFF_FP12_C1[power % 12]);
        self.c1.c1.mul_assign(&FROBENIUS_COEFF_FP12_C1[power % 12]);
        self.c1.c2.mul_assign(&FROBENIUS_COEFF_FP12_C1[power % 12]);
    }

    pub fn mul_by_014(&mut self, c0: &Fp2, c1: &Fp2, c4: &Fp2) {
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

    pub fn mul_by_034(&mut self, c0: &Fp2, c3: &Fp2, c4: &Fp2) {
        let t0 = Fp6 {
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
            let mut tmp = Fp12 { c0: t, c1: t };
            tmp.c0.mul_assign(&self.c0);
            tmp.c1.mul_assign(&self.c1);
            tmp.c1 = tmp.c1.neg();

            tmp
        })
    }

    pub fn cyclotomic_square(&mut self) {
        fn fp4_square(c0: &mut Fp2, c1: &mut Fp2, a0: &Fp2, a1: &Fp2) {
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

        let mut t3 = Fp2::zero();
        let mut t4 = Fp2::zero();
        let mut t5 = Fp2::zero();
        let mut t6 = Fp2::zero();

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

impl Field for Fp12 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fp12 {
            c0: Fp6::random(&mut rng),
            c1: Fp6::random(&mut rng),
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
        // The square root method is typically only required for finding y-coordinate
        // given the x-coordinate of an EC point. Fields over which we have not
        // defined a curve do not need this method.
        unimplemented!()
    }

    fn sqrt_ratio(_num: &Self, _div: &Self) -> (Choice, Self) {
        // The square root method is typically only required for finding y-coordinate
        // given the x-coordinate of an EC point. Fields over which we have not
        // defined a curve do not need this method.
        unimplemented!()
    }

    fn invert(&self) -> CtOption<Self> {
        self.invert()
    }
}

/// Fp2(v)^((p^i-1)/6) for i=0,...,11
pub const FROBENIUS_COEFF_FP12_C1: [Fp2; 12] = [
    // Fp2(v)**(((p^0) - 1) / 6)
    Fp2::ONE,
    // Fp2(v)**(((p^1) - 1) / 6)
    Fp2 {
        // 0x3c3ad3da8b99cb1df0709dc343113ccd9892dedd51f30695d89c647b90de8f41df055384b9e6cfd4e70648622c750f32ee965dfef2303d3
        c0: Fp::from_raw([
            0x2ee965dfef2303d3,
            0x4e70648622c750f3,
            0x1df055384b9e6cfd,
            0x5d89c647b90de8f4,
            0xd9892dedd51f3069,
            0xdf0709dc343113cc,
            0x03c3ad3da8b99cb1,
        ]),
        // 0x149fd9ed2c7affe7aaa3b912182da22dccb29838628f04b6f333d052540294889f03876b2ddb143559f9373f4cf44e6afa0be24ad758a5ff
        c1: Fp::from_raw([
            0xfa0be24ad758a5ff,
            0x59f9373f4cf44e6a,
            0x9f03876b2ddb1435,
            0xf333d05254029488,
            0xccb29838628f04b6,
            0xaaa3b912182da22d,
            0x149fd9ed2c7affe7,
        ]),
    },
    // Fp2(v)**(((p^2) - 1) / 6)
    Fp2 {
        // 0x480000000000360001c950000d7ee0e4a803c956d01c903d720dc8ad8b38dffaf50c100004c37fffffff
        c0: Fp::from_raw([
            0x100004c37fffffff,
            0xc8ad8b38dffaf50c,
            0xc956d01c903d720d,
            0x50000d7ee0e4a803,
            0x00000000360001c9,
            0x0000000000004800,
            0x0000000000000000,
        ]),
        c1: Fp::ZERO,
    },
    // Fp2(v)**(((p^3) - 1) / 6)
    Fp2 {
        // 0x1baee9e044d94d205764b80089c40010af5ca1e56a2a81e6a5d8739325984fc889d390efef216fe4f4af912a897f60a128a3be71be4995ca
        c0: Fp::from_raw([
            0x28a3be71be4995ca,
            0xf4af912a897f60a1,
            0x89d390efef216fe4,
            0xa5d8739325984fc8,
            0xaf5ca1e56a2a81e6,
            0x5764b80089c40010,
            0x1baee9e044d94d20,
        ]),
        // 0x20d4c11700e832829b26f1795339413be65e47a7716bc8bc07cd6b44b03ef1130b3c35a77291b29d6f45d28e4ef1ecb9678f4479a1151232
        c1: Fp::from_raw([
            0x678f4479a1151232,
            0x6f45d28e4ef1ecb9,
            0x0b3c35a77291b29d,
            0x07cd6b44b03ef113,
            0xe65e47a7716bc8bc,
            0x9b26f1795339413b,
            0x20d4c11700e83282,
        ]),
    },
    // Fp2(v)**(((p^4) - 1) / 6)
    Fp2 {
        // 0x480000000000360001c950000d7ee0e4a803c956d01c903d720dc8ad8b38dffaf50c100004c37ffffffe
        c0: Fp::from_raw([
            0x100004c37ffffffe,
            0xc8ad8b38dffaf50c,
            0xc956d01c903d720d,
            0x50000d7ee0e4a803,
            0x00000000360001c9,
            0x0000000000004800,
            0x0000000000000000,
        ]),
        c1: Fp::ZERO,
    },
    // Fp2(v)**(((p^5) - 1) / 6)
    Fp2 {
        // 0x17eb3ca29c1fb06e785dae245592ec43d5d373f7950b517d484ead4b6c8a66d46be33bb7a38302e7a63f2ca466b80fadf9ba5891cf2691f7
        c0: Fp::from_raw([
            0xf9ba5891cf2691f7,
            0xa63f2ca466b80fad,
            0x6be33bb7a38302e7,
            0x484ead4b6c8a66d4,
            0xd5d373f7950b517d,
            0x785dae245592ec43,
            0x17eb3ca29c1fb06e,
        ]),
        // 0xc34e729d46d329af08338673b0b9f0e19abaf6f0edcc40514999af25c3c5c8a6c38ae3c44b69e68154c9b4f01fd9e4e6d83622ec9bc6c33
        c1: Fp::from_raw([
            0x6d83622ec9bc6c33,
            0x154c9b4f01fd9e4e,
            0x6c38ae3c44b69e68,
            0x14999af25c3c5c8a,
            0x19abaf6f0edcc405,
            0xf08338673b0b9f0e,
            0x0c34e729d46d329a,
        ]),
    },
    // Fp2(v)**(((p^6) - 1) / 6)
    Fp2 {
        // 0x24000000000024000130e0000d7f70e4a803ca76f439266f443f9a5cda8a6c7be4a7a5fe8fadffd6a2a7e8c30006b9459ffffcd300000000
        c0: Fp::from_raw([
            0x9ffffcd300000000,
            0xa2a7e8c30006b945,
            0xe4a7a5fe8fadffd6,
            0x443f9a5cda8a6c7b,
            0xa803ca76f439266f,
            0x0130e0000d7f70e4,
            0x2400000000002400,
        ]),
        c1: Fp::ZERO,
    },
    // Fp2(v)**(((p^7) - 1) / 6)
    Fp2 {
        // 0x203c52c25746874e2229d623d94e5d17ce7a9c891f19f605e6b5d415217c8387c6b750c6440f92d95437843cdd3f6852711696f310dcfc2e
        c0: Fp::from_raw([
            0x711696f310dcfc2e,
            0x5437843cdd3f6852,
            0xc6b750c6440f92d9,
            0xe6b5d415217c8387,
            0xce7a9c891f19f605,
            0x2229d623d94e5d17,
            0x203c52c25746874e,
        ]),
        // 0xf602612d3852418568d26edf551ceb6db51323e91aa21b8510bca0a8687d7f345a41e9361d2eba148aeb183b3126adaa5f41a8828a75a02
        c1: Fp::from_raw([
            0xa5f41a8828a75a02,
            0x48aeb183b3126ada,
            0x45a41e9361d2eba1,
            0x510bca0a8687d7f3,
            0xdb51323e91aa21b8,
            0x568d26edf551ceb6,
            0x0f602612d3852418,
        ]),
    },
    // Fp2(v)**(((p^8) - 1) / 6)
    Fp2 {
        // 0x24000000000024000130e0000d7f28e4a803ca76be3924a5f43f8cddf9a5c4781b50d5e1ff708dc8d9fa5d8a200bc4398ffff80f80000002
        c0: Fp::from_raw([
            0x8ffff80f80000002,
            0xd9fa5d8a200bc439,
            0x1b50d5e1ff708dc8,
            0xf43f8cddf9a5c478,
            0xa803ca76be3924a5,
            0x0130e0000d7f28e4,
            0x2400000000002400,
        ]),
        c1: Fp::ZERO,
    },
    // Fp2(v)**(((p^9) - 1) / 6)
    Fp2 {
        // 0x851161fbb26d6dfa9cc27ff83bb70d3f8a728918a0ea4889e6726c9b4f21cb35ad4150ea08c8ff1adf85798768758a4775c3e6141b66a37
        c0: Fp::from_raw([
            0x775c3e6141b66a37,
            0xadf85798768758a4,
            0x5ad4150ea08c8ff1,
            0x9e6726c9b4f21cb3,
            0xf8a728918a0ea488,
            0xa9cc27ff83bb70d3,
            0x0851161fbb26d6df,
        ]),
        // 0x32b3ee8ff17f17d6609ee86ba462fa8c1a582cf82cd5db33c722f182a4b7b68d96b70571d1c4d3933621634b114cc8c3870b8595eeaedcf
        c1: Fp::from_raw([
            0x3870b8595eeaedcf,
            0x33621634b114cc8c,
            0xd96b70571d1c4d39,
            0x3c722f182a4b7b68,
            0xc1a582cf82cd5db3,
            0x6609ee86ba462fa8,
            0x032b3ee8ff17f17d,
        ]),
    },
    // Fp2(v)**(((p^10) - 1) / 6)
    Fp2 {
        // 0x24000000000024000130e0000d7f28e4a803ca76be3924a5f43f8cddf9a5c4781b50d5e1ff708dc8d9fa5d8a200bc4398ffff80f80000003
        c0: Fp::from_raw([
            0x8ffff80f80000003,
            0xd9fa5d8a200bc439,
            0x1b50d5e1ff708dc8,
            0xf43f8cddf9a5c478,
            0xa803ca76be3924a5,
            0x0130e0000d7f28e4,
            0x2400000000002400,
        ]),
        c1: Fp::ZERO,
    },
    // Fp2(v)**(((p^11) - 1) / 6)
    Fp2 {
        // 0xc14c35d63e0739188d331dbb7ec84a0d230567f5f2dd4f1fbf0ed116e0005a778c46a46ec2afceefc68bc1e994ea997a645a44130d96e0a
        c0: Fp::from_raw([
            0xa645a44130d96e0a,
            0xfc68bc1e994ea997,
            0x78c46a46ec2afcee,
            0xfbf0ed116e0005a7,
            0xd230567f5f2dd4f1,
            0x88d331dbb7ec84a0,
            0x0c14c35d63e07391,
        ]),
        // 0x17cb18d62b92f16510ada798d273d1d68e581b07e55c626a2fa5ff6a7e4e0ff1786ef7c24af7616e8d5b4d73fe091af7327c9aa4364393ce
        c1: Fp::from_raw([
            0x327c9aa4364393ce,
            0x8d5b4d73fe091af7,
            0x786ef7c24af7616e,
            0x2fa5ff6a7e4e0ff1,
            0x8e581b07e55c626a,
            0x10ada798d273d1d6,
            0x17cb18d62b92f165,
        ]),
    },
];

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fp12, "field_arithmetic");
    // extension field-specific
    crate::field_testing_suite!(Fp12, "f12_tests", Fp6, Fp2);
    crate::field_testing_suite!(
        Fp12,
        "frobenius",
        // Frobenius endomorphism power parameter for extension field
        //  ϕ: E → E
        //  (x, y) ↦ (x^p, y^p)
        // p: modulus of base field (Here, Fp::MODULUS)
        [
            0x9ffffcd300000001,
            0xa2a7e8c30006b945,
            0xe4a7a5fe8fadffd6,
            0x443f9a5cda8a6c7b,
            0xa803ca76f439266f,
            0x0130e0000d7f70e4,
            0x2400000000002400,
        ]
    );
}
