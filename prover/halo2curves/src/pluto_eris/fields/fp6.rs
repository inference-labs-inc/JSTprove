use super::fp::Fp;
use super::fp2::Fp2;
use crate::ff::Field;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

/// -BETA is a cubic non-residue in Fp2. Fp6 = Fp2[X]/(X^3 + BETA)
/// We introduce the variable v such that v^3 = -BETA
/// BETA = - 57/(z+3)

/// V_CUBE = 57/(u+3)
pub(crate) const V_CUBE: Fp2 = Fp2 {
    // 0xcdb6db6db6dc3b6dbda9924971b3a9ace4a7f2a7bcb449573cd928ee056022c3f6072240ebe2483833bf7b35b701d98ddb6da4b5b6db6e8
    c0: Fp::from_raw([
        0xddb6da4b5b6db6e8,
        0x833bf7b35b701d98,
        0x3f6072240ebe2483,
        0x73cd928ee056022c,
        0xce4a7f2a7bcb4495,
        0xdbda9924971b3a9a,
        0x0cdb6db6db6dc3b6,
    ]),
    // 0x7b6db6db6db756db71cc2492776bcc3489319197d79f5f3457b57ef5366ce1a8c6d1148d5a5491bb523fb0536dcde8eeb6db62d36db6db3
    c1: Fp::from_raw([
        0xeb6db62d36db6db3,
        0xb523fb0536dcde8e,
        0x8c6d1148d5a5491b,
        0x457b57ef5366ce1a,
        0x489319197d79f5f3,
        0xb71cc2492776bcc3,
        0x07b6db6db6db756d,
    ]),
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
/// The `Fp6` element c0 + c1 * v + c2 * v3
pub struct Fp6 {
    pub c0: Fp2,
    pub c1: Fp2,
    pub c2: Fp2,
}

impl ConditionallySelectable for Fp6 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fp6 {
            c0: Fp2::conditional_select(&a.c0, &b.c0, choice),
            c1: Fp2::conditional_select(&a.c1, &b.c1, choice),
            c2: Fp2::conditional_select(&a.c2, &b.c2, choice),
        }
    }
}

impl ConstantTimeEq for Fp6 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1) & self.c2.ct_eq(&other.c2)
    }
}

impl Neg for Fp6 {
    type Output = Fp6;

    #[inline]
    fn neg(self) -> Fp6 {
        -&self
    }
}

impl<'a> Neg for &'a Fp6 {
    type Output = Fp6;

    #[inline]
    fn neg(self) -> Fp6 {
        self.neg()
    }
}

impl<'a, 'b> Sub<&'b Fp6> for &'a Fp6 {
    type Output = Fp6;

    #[inline]
    fn sub(self, rhs: &'b Fp6) -> Fp6 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fp6> for &'a Fp6 {
    type Output = Fp6;

    #[inline]
    fn add(self, rhs: &'b Fp6) -> Fp6 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fp6> for &'a Fp6 {
    type Output = Fp6;

    #[inline]
    fn mul(self, rhs: &'b Fp6) -> Fp6 {
        self.mul(rhs)
    }
}

use crate::{
    impl_add_binop_specify_output, impl_binops_additive, impl_binops_additive_specify_output,
    impl_binops_multiplicative, impl_binops_multiplicative_mixed, impl_sub_binop_specify_output,
    impl_sum_prod,
};
impl_binops_additive!(Fp6, Fp6);
impl_binops_multiplicative!(Fp6, Fp6);
impl_sum_prod!(Fp6);

impl Fp6 {
    #[inline]
    pub const fn zero() -> Self {
        Fp6 {
            c0: Fp2::ZERO,
            c1: Fp2::ZERO,
            c2: Fp2::ZERO,
        }
    }

    #[inline]
    pub const fn one() -> Self {
        Fp6 {
            c0: Fp2::ONE,
            c1: Fp2::ZERO,
            c2: Fp2::ZERO,
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
        // self.c1.mul_by_xi();
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

        self.c1.mul_assign(&FROBENIUS_COEFF_FP6_C1[power % 6]);
        self.c2.mul_assign(&FROBENIUS_COEFF_FP6_C2[power % 6]);
    }

    /// Multiply by cubic nonresidue v.
    pub fn mul_by_nonresidue(&mut self) {
        use std::mem::swap;
        swap(&mut self.c0, &mut self.c1);
        swap(&mut self.c0, &mut self.c2);
        // c0, c1, c2 -> c2, c0, c1
        self.c0.mul_by_nonresidue();
    }

    pub fn mul_by_1(&mut self, c1: &Fp2) {
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

    pub fn mul_by_01(&mut self, c0: &Fp2, c1: &Fp2) {
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
            let mut tmp = Fp6 {
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

impl Field for Fp6 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fp6 {
            c0: Fp2::random(&mut rng),
            c1: Fp2::random(&mut rng),
            c2: Fp2::random(&mut rng),
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

/// Fp2 coefficients for the efficient computation of Frobenius Endomorphism in Fp6.
pub(crate) const FROBENIUS_COEFF_FP6_C1: [Fp2; 6] = [
    // Fp2(v^3)**(((p^0) - 1) / 3)
    Fp2::ONE,
    // Fp2(v^3)**(((p^1) - 1) / 3)
    Fp2 {
        // 0x120de97f024c55bc3bc0d351f4c70da1e3886170077a50986f93678bc921dcd5041bc4bb14cc42dc52e787634eccc335a001825382850d03
        c0: Fp::from_raw([
            0xa001825382850d03,
            0x52e787634eccc335,
            0x041bc4bb14cc42dc,
            0x6f93678bc921dcd5,
            0xe3886170077a5098,
            0x3bc0d351f4c70da1,
            0x120de97f024c55bc,
        ]),
        // 0x2096f3f804d973afd82becc2ef081b76132461908eadbe3da1a7f5502b7091965efa1ddf4658080413be1b7cd3c9ea0e2772fea378a9b322
        c1: Fp::from_raw([
            0x2772fea378a9b322,
            0x13be1b7cd3c9ea0e,
            0x5efa1ddf46580804,
            0xa1a7f5502b709196,
            0x132461908eadbe3d,
            0xd82becc2ef081b76,
            0x2096f3f804d973af,
        ]),
    },
    // Fp2(v^3)**(((p^2) - 1) / 3)
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
    // Fp2(v^3)**(((p^3) - 1) / 3)
    Fp2 {
        // 0x1f9cd069c59f50a72511749de232911d833b798e78bd98c02913e38315a71c287cd52ae30d09b78a8b43b17b4c3ea938a04518fa783eb497
        c0: Fp::from_raw([
            0xa04518fa783eb497,
            0x8b43b17b4c3ea938,
            0x7cd52ae30d09b78a,
            0x2913e38315a71c28,
            0x833b798e78bd98c0,
            0x2511749de232911d,
            0x1f9cd069c59f50a7,
        ]),
        // 0x23affd628747cbaec26943f93dc9eab63f4af36699fe6d74c0aa2122aa7cb689e8faacb3479a973a4a728fcb77b150ee77240d4066e42ac5
        c1: Fp::from_raw([
            0x77240d4066e42ac5,
            0x4a728fcb77b150ee,
            0xe8faacb3479a973a,
            0xc0aa2122aa7cb689,
            0x3f4af36699fe6d74,
            0xc26943f93dc9eab6,
            0x23affd628747cbae,
        ]),
    },
    // Fp2(v^3)**(((p^4) - 1) / 3)
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
    // Fp2(v^3)**(((p^5) - 1) / 3)
    Fp2 {
        // 0x165546173814a19ca18f781044054309e943b9ef683a6385efd7e9aad64bdffa485e5c5efd860546672498a76502061cffb95e58053c3e68
        c0: Fp::from_raw([
            0xffb95e58053c3e68,
            0x672498a76502061c,
            0x485e5c5efd860546,
            0xefd7e9aad64bdffa,
            0xe943b9ef683a6385,
            0xa18f781044054309,
            0x165546173814a19c,
        ]),
        // 0x3b90ea573df08a167cc8f43ee2cdb9cfd983ff6bfc6212c262d1e46df2790d7815a816a9169606ee71f263db492378ea168edc22072221b
        c1: Fp::from_raw([
            0xa168edc22072221b,
            0xe71f263db492378e,
            0x815a816a9169606e,
            0x262d1e46df2790d7,
            0xfd983ff6bfc6212c,
            0x67cc8f43ee2cdb9c,
            0x03b90ea573df08a1,
        ]),
    },
];

/// Fp2 coefficients for the efficient computation of Frobenius Endomorphism in Fp6.
pub(crate) const FROBENIUS_COEFF_FP6_C2: [Fp2; 6] = [
    // Fp2(v^3)**(((2p^0) - 2) / 3)
    Fp2::ONE,
    // Fp2(v^3)**(((2p^1) - 2) / 3)
    Fp2 {
        // 0x93733692ce3cdcfc34610bac6bd22c4dc590efb038c82998c9549048e7b424cc00e17ffb4a61950d0ec132a7b38f09db0a818e422737f7c
        c0: Fp::from_raw([
            0xb0a818e422737f7c,
            0xd0ec132a7b38f09d,
            0xc00e17ffb4a61950,
            0x8c9549048e7b424c,
            0xdc590efb038c8299,
            0xc34610bac6bd22c4,
            0x093733692ce3cdcf,
        ]),
        // 0x12cb19daadc92882ba3593aa6f3e6bf426f29bd46039e3036f61d0bd35f39ebecdac3209d9df546061c90b4940d9031c240ce398421dc7dc
        c1: Fp::from_raw([
            0x240ce398421dc7dc,
            0x61c90b4940d9031c,
            0xcdac3209d9df5460,
            0x6f61d0bd35f39ebe,
            0x26f29bd46039e303,
            0xba3593aa6f3e6bf4,
            0x12cb19daadc92882,
        ]),
    },
    // Fp2(v^3)**(((2p^2) - 2) / 3)
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
    // Fp2(v^3)**(((2p^3) - 2) / 3)
    Fp2 {
        // 0x85cc83a7eeba2ef5f7dd2f9f1405312b2ce0cbc85b8561e1657aaf1e85b82299aa5ace8b26b78d88f57e1c7a87f75556885980d6c8d2186
        c0: Fp::from_raw([
            0x6885980d6c8d2186,
            0x8f57e1c7a87f7555,
            0x9aa5ace8b26b78d8,
            0x1657aaf1e85b8229,
            0xb2ce0cbc85b8561e,
            0x5f7dd2f9f1405312,
            0x085cc83a7eeba2ef,
        ]),
        // 0xda3357ee4e6a9836af75e8ec0dbd23e7abc03d404620899ee0ea8b684b9400d58d5ebe487e523680bbe8a0dd9ea1d312bca2a953ab51c9b
        c1: Fp::from_raw([
            0x2bca2a953ab51c9b,
            0x0bbe8a0dd9ea1d31,
            0x58d5ebe487e52368,
            0xee0ea8b684b9400d,
            0x7abc03d404620899,
            0x6af75e8ec0dbd23e,
            0x0da3357ee4e6a983,
        ]),
    },
    // Fp2(v^3)**(((2p^4) - 2) / 3)
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
    // Fp2(v^3)**(((2p^5) - 2) / 3)
    Fp2 {
        // 0x126c045c5430b340de6cfc4b5581fb0d18dcaebf6af44db7a152a66663b3a80589f3e116289c6dad4263f3d0dc4e535286d24be170ff5eff
        c0: Fp::from_raw([
            0x86d24be170ff5eff,
            0x4263f3d0dc4e5352,
            0x89f3e116289c6dad,
            0xa152a66663b3a805,
            0x18dcaebf6af44db7,
            0xde6cfc4b5581fb0d,
            0x126c045c5430b340,
        ]),
        // 0x391b0a66d5051f9dc03edc6dd6532b206552ace8f9d3ad1e6cf20e91fdd8dafbe2588102de9880e3520536be54398f85028eea5832d1b8a
        c1: Fp::from_raw([
            0x5028eea5832d1b8a,
            0x3520536be54398f8,
            0xbe2588102de9880e,
            0xe6cf20e91fdd8daf,
            0x06552ace8f9d3ad1,
            0xdc03edc6dd6532b2,
            0x0391b0a66d5051f9,
        ]),
    },
];

#[cfg(test)]
mod test {
    use super::*;
    crate::field_testing_suite!(Fp6, "field_arithmetic");
    // extension field-specific
    crate::field_testing_suite!(Fp6, "f6_tests", Fp2);
    crate::field_testing_suite!(
        Fp6,
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
