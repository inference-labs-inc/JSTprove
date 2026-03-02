use std::arch::aarch64::uint32x4_t;
use std::iter::{Product, Sum};
use std::mem::transmute;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use arith::{field_common, ExtensionField, Field, SimdField};
use ethnum::U256;
use gf2::{GF2x8, GF2};
use serdes::{ExpSerde, SerdeResult};

use crate::gf2_ext128::neon::{gfadd, gfmul, mul_by_x_internal, NeonGF2_128};
use crate::GF2_128;

#[derive(Clone, Copy, Debug)]
pub struct NeonGF2_128x8 {
    v: [uint32x4_t; 8],
}

field_common!(NeonGF2_128x8);

impl Default for NeonGF2_128x8 {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for NeonGF2_128x8 {
    fn eq(&self, other: &Self) -> bool {
        self.v.iter().zip(other.v.iter()).all(|(a, b)| unsafe {
            transmute::<uint32x4_t, [u8; 16]>(*a) == transmute::<uint32x4_t, [u8; 16]>(*b)
        })
    }
}

impl Eq for NeonGF2_128x8 {}

impl ExpSerde for NeonGF2_128x8 {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        unsafe {
            writer.write_all(transmute::<[uint32x4_t; 8], [u8; 128]>(self.v).as_ref())?;
        }
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let mut res = [0u8; 128];
        reader.read_exact(&mut res)?;

        let res = unsafe { transmute::<[u8; 128], [uint32x4_t; 8]>(res) };
        Ok(Self { v: res })
    }
}

impl Field for NeonGF2_128x8 {
    const NAME: &'static str = "Neon Galois Field 2^128 SIMD 8";

    const SIZE: usize = 16 * 8;

    const FIELD_SIZE: usize = 128; // in bits

    const ZERO: Self = NeonGF2_128x8 {
        v: [unsafe { transmute::<[u32; 4], uint32x4_t>([0, 0, 0, 0]) }; 8],
    };

    const ONE: Self = NeonGF2_128x8 {
        v: [unsafe { transmute::<[u32; 4], uint32x4_t>([1, 0, 0, 0]) }; 8],
    };

    const INV_2: Self = NeonGF2_128x8 {
        v: [unsafe { transmute::<[u32; 4], uint32x4_t>([0, 0, 0, 0]) }; 8],
    };

    const MODULUS: U256 = unimplemented!(); // should not be used

    #[inline(always)]
    fn zero() -> Self {
        NeonGF2_128x8 {
            v: [unsafe { transmute::<[u32; 4], uint32x4_t>([0, 0, 0, 0]) }; 8],
        }
    }

    #[inline(always)]
    fn one() -> Self {
        NeonGF2_128x8 {
            v: [unsafe { transmute::<[u32; 4], uint32x4_t>([1, 0, 0, 0]) }; 8],
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v
            .iter()
            .all(|vv| unsafe { transmute::<uint32x4_t, [u8; 16]>(*vv) == [0; 16] })
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        NeonGF2_128x8 {
            v: [
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
                unsafe { transmute::<[u64; 2], uint32x4_t>([rng.next_u64(), rng.next_u64()]) },
            ],
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        NeonGF2_128x8 {
            v: [
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([rng.next_u32() % 2, 0, 0, 0]) },
            ],
        }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let p_m2 = !(0u128) - 1;
        Some(Self::exp(self, p_m2))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self * self
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("u32 for GF128 doesn't make sense")
    }

    #[inline(always)]
    fn from_uniform_bytes(_bytes: &[u8]) -> Self {
        unimplemented!("from_uniform_bytes for GF128 doesn't make sense")
    }
}

impl SimdField for NeonGF2_128x8 {
    type Scalar = NeonGF2_128;

    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        NeonGF2_128x8 {
            v: [
                unsafe { gfmul(self.v[0], challenge.v) },
                unsafe { gfmul(self.v[1], challenge.v) },
                unsafe { gfmul(self.v[2], challenge.v) },
                unsafe { gfmul(self.v[3], challenge.v) },
                unsafe { gfmul(self.v[4], challenge.v) },
                unsafe { gfmul(self.v[5], challenge.v) },
                unsafe { gfmul(self.v[6], challenge.v) },
                unsafe { gfmul(self.v[7], challenge.v) },
            ],
        }
    }

    const PACK_SIZE: usize = 8;

    #[inline]
    fn pack_full(base: &Self::Scalar) -> Self {
        Self { v: [base.v; 8] }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == 8);
        let base_vec_array: [Self::Scalar; 8] = base_vec.try_into().unwrap();
        unsafe { transmute(base_vec_array) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let array: [Self::Scalar; 8] = unsafe { transmute(self.v) };
        array.to_vec()
    }
}

impl From<NeonGF2_128> for NeonGF2_128x8 {
    fn from(v: NeonGF2_128) -> Self {
        NeonGF2_128x8 { v: [v.v; 8] }
    }
}

impl Neg for NeonGF2_128x8 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        self
    }
}

impl From<u32> for NeonGF2_128x8 {
    fn from(v: u32) -> Self {
        NeonGF2_128x8 {
            v: [
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v, 0, 0, 0]) },
            ],
        }
    }
}

#[inline(always)]
fn add_internal(a: &NeonGF2_128x8, b: &NeonGF2_128x8) -> NeonGF2_128x8 {
    NeonGF2_128x8 {
        v: [
            unsafe { gfadd(a.v[0], b.v[0]) },
            unsafe { gfadd(a.v[1], b.v[1]) },
            unsafe { gfadd(a.v[2], b.v[2]) },
            unsafe { gfadd(a.v[3], b.v[3]) },
            unsafe { gfadd(a.v[4], b.v[4]) },
            unsafe { gfadd(a.v[5], b.v[5]) },
            unsafe { gfadd(a.v[6], b.v[6]) },
            unsafe { gfadd(a.v[7], b.v[7]) },
        ],
    }
}

#[inline(always)]
fn sub_internal(a: &NeonGF2_128x8, b: &NeonGF2_128x8) -> NeonGF2_128x8 {
    add_internal(a, b)
}

#[inline(always)]
fn mul_internal(a: &NeonGF2_128x8, b: &NeonGF2_128x8) -> NeonGF2_128x8 {
    NeonGF2_128x8 {
        v: [
            unsafe { gfmul(a.v[0], b.v[0]) },
            unsafe { gfmul(a.v[1], b.v[1]) },
            unsafe { gfmul(a.v[2], b.v[2]) },
            unsafe { gfmul(a.v[3], b.v[3]) },
            unsafe { gfmul(a.v[4], b.v[4]) },
            unsafe { gfmul(a.v[5], b.v[5]) },
            unsafe { gfmul(a.v[6], b.v[6]) },
            unsafe { gfmul(a.v[7], b.v[7]) },
        ],
    }
}

impl ExtensionField for NeonGF2_128x8 {
    const DEGREE: usize = NeonGF2_128::DEGREE;
    const W: u32 = NeonGF2_128::W;
    const X: Self = Self {
        v: unsafe {
            transmute::<[u64; 16], [uint32x4_t; 8]>([
                2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0,
            ])
        },
    };
    type BaseField = GF2x8;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        let v0 = (base.v & 1u8) as u32;
        let v1 = ((base.v >> 1) & 1u8) as u32;
        let v2 = ((base.v >> 2) & 1u8) as u32;
        let v3 = ((base.v >> 3) & 1u8) as u32;
        let v4 = ((base.v >> 4) & 1u8) as u32;
        let v5 = ((base.v >> 5) & 1u8) as u32;
        let v6 = ((base.v >> 6) & 1u8) as u32;
        let v7 = ((base.v >> 7) & 1u8) as u32;

        Self {
            v: [
                unsafe { gfmul(self.v[0], transmute::<[u32; 4], uint32x4_t>([v0, 0, 0, 0])) },
                unsafe { gfmul(self.v[1], transmute::<[u32; 4], uint32x4_t>([v1, 0, 0, 0])) },
                unsafe { gfmul(self.v[2], transmute::<[u32; 4], uint32x4_t>([v2, 0, 0, 0])) },
                unsafe { gfmul(self.v[3], transmute::<[u32; 4], uint32x4_t>([v3, 0, 0, 0])) },
                unsafe { gfmul(self.v[4], transmute::<[u32; 4], uint32x4_t>([v4, 0, 0, 0])) },
                unsafe { gfmul(self.v[5], transmute::<[u32; 4], uint32x4_t>([v5, 0, 0, 0])) },
                unsafe { gfmul(self.v[6], transmute::<[u32; 4], uint32x4_t>([v6, 0, 0, 0])) },
                unsafe { gfmul(self.v[7], transmute::<[u32; 4], uint32x4_t>([v7, 0, 0, 0])) },
            ],
        }
    }

    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        let v0 = (base.v & 1u8) as u32;
        let v1 = ((base.v >> 1) & 1u8) as u32;
        let v2 = ((base.v >> 2) & 1u8) as u32;
        let v3 = ((base.v >> 3) & 1u8) as u32;
        let v4 = ((base.v >> 4) & 1u8) as u32;
        let v5 = ((base.v >> 5) & 1u8) as u32;
        let v6 = ((base.v >> 6) & 1u8) as u32;
        let v7 = ((base.v >> 7) & 1u8) as u32;

        Self {
            v: [
                unsafe { gfadd(self.v[0], transmute::<[u32; 4], uint32x4_t>([v0, 0, 0, 0])) },
                unsafe { gfadd(self.v[1], transmute::<[u32; 4], uint32x4_t>([v1, 0, 0, 0])) },
                unsafe { gfadd(self.v[2], transmute::<[u32; 4], uint32x4_t>([v2, 0, 0, 0])) },
                unsafe { gfadd(self.v[3], transmute::<[u32; 4], uint32x4_t>([v3, 0, 0, 0])) },
                unsafe { gfadd(self.v[4], transmute::<[u32; 4], uint32x4_t>([v4, 0, 0, 0])) },
                unsafe { gfadd(self.v[5], transmute::<[u32; 4], uint32x4_t>([v5, 0, 0, 0])) },
                unsafe { gfadd(self.v[6], transmute::<[u32; 4], uint32x4_t>([v6, 0, 0, 0])) },
                unsafe { gfadd(self.v[7], transmute::<[u32; 4], uint32x4_t>([v7, 0, 0, 0])) },
            ],
        }
    }

    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        let mut res = Self::default();
        res.v[0] = mul_by_x_internal(&self.v[0]);
        res.v[1] = mul_by_x_internal(&self.v[1]);
        res.v[2] = mul_by_x_internal(&self.v[2]);
        res.v[3] = mul_by_x_internal(&self.v[3]);
        res.v[4] = mul_by_x_internal(&self.v[4]);
        res.v[5] = mul_by_x_internal(&self.v[5]);
        res.v[6] = mul_by_x_internal(&self.v[6]);
        res.v[7] = mul_by_x_internal(&self.v[7]);
        res
    }

    #[inline(always)]
    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        let mut local_limbs = limbs.to_vec();
        local_limbs.resize(Self::DEGREE, Self::BaseField::ZERO);

        let mut buffer = vec![GF2::ZERO; Self::DEGREE * Self::PACK_SIZE];

        local_limbs.iter().enumerate().for_each(|(ith_limb, limb)| {
            let unpacked = limb.unpack();
            unpacked.iter().enumerate().for_each(|(ith_gf2, gf2_val)| {
                buffer[ith_gf2 * Self::DEGREE + ith_limb] = *gf2_val;
            });
        });

        let gf2_128s: Vec<_> = buffer
            .chunks(Self::DEGREE)
            .map(GF2_128::from_limbs)
            .collect();

        Self::pack(&gf2_128s)
    }

    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        let gf2_128s = self.unpack();

        let mut buffer = vec![GF2::ZERO; Self::DEGREE * Self::PACK_SIZE];
        gf2_128s
            .iter()
            .enumerate()
            .for_each(|(ith_gf2_128, gf2_128_val)| {
                let limbs = gf2_128_val.to_limbs();
                limbs.iter().enumerate().for_each(|(ith_limb, limb)| {
                    buffer[ith_limb * Self::PACK_SIZE + ith_gf2_128] = *limb;
                })
            });

        buffer.chunks(Self::PACK_SIZE).map(GF2x8::pack).collect()
    }
}

impl From<GF2x8> for NeonGF2_128x8 {
    #[inline(always)]
    fn from(v: GF2x8) -> Self {
        let v0 = (v.v & 1u8) as u32;
        let v1 = ((v.v >> 1) & 1u8) as u32;
        let v2 = ((v.v >> 2) & 1u8) as u32;
        let v3 = ((v.v >> 3) & 1u8) as u32;
        let v4 = ((v.v >> 4) & 1u8) as u32;
        let v5 = ((v.v >> 5) & 1u8) as u32;
        let v6 = ((v.v >> 6) & 1u8) as u32;
        let v7 = ((v.v >> 7) & 1u8) as u32;

        NeonGF2_128x8 {
            v: [
                unsafe { transmute::<[u32; 4], uint32x4_t>([v0, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v1, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v2, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v3, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v4, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v5, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v6, 0, 0, 0]) },
                unsafe { transmute::<[u32; 4], uint32x4_t>([v7, 0, 0, 0]) },
            ],
        }
    }
}

impl Mul<GF2x8> for NeonGF2_128x8 {
    type Output = NeonGF2_128x8;

    #[inline]
    fn mul(self, rhs: GF2x8) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl Mul<GF2> for NeonGF2_128x8 {
    type Output = NeonGF2_128x8;

    #[inline(always)]
    fn mul(self, rhs: GF2) -> Self::Output {
        if rhs.is_zero() {
            Self::zero()
        } else {
            self
        }
    }
}

impl Add<GF2> for NeonGF2_128x8 {
    type Output = NeonGF2_128x8;
    #[inline(always)]
    fn add(self, rhs: GF2) -> Self::Output {
        let rhs_extended = unsafe { transmute::<[u32; 4], uint32x4_t>([rhs.v as u32, 0, 0, 0]) };
        NeonGF2_128x8 {
            v: [
                unsafe { gfadd(self.v[0], rhs_extended) },
                unsafe { gfadd(self.v[1], rhs_extended) },
                unsafe { gfadd(self.v[2], rhs_extended) },
                unsafe { gfadd(self.v[3], rhs_extended) },
                unsafe { gfadd(self.v[4], rhs_extended) },
                unsafe { gfadd(self.v[5], rhs_extended) },
                unsafe { gfadd(self.v[6], rhs_extended) },
                unsafe { gfadd(self.v[7], rhs_extended) },
            ],
        }
    }
}

impl std::hash::Hash for NeonGF2_128x8 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<[uint32x4_t; 8], [u8; 128]>(self.v).as_ref());
        }
    }
}
