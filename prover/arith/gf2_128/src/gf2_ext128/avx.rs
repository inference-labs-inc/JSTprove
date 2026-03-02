use std::hash::Hash;
use std::iter::{Product, Sum};
use std::{
    arch::x86_64::*,
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, Field};
use ethnum::U256;
use gf2::GF2;
use serdes::{ExpSerde, SerdeResult};

#[derive(Debug, Clone, Copy)]
pub struct AVXGF2_128 {
    pub v: __m128i,
}

field_common!(AVXGF2_128);

impl ExpSerde for AVXGF2_128 {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        unsafe { writer.write_all(transmute::<__m128i, [u8; 16]>(self.v).as_ref())? };
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 16];
        reader.read_exact(&mut u)?;
        unsafe {
            Ok(AVXGF2_128 {
                v: transmute::<[u8; 16], __m128i>(u),
            })
        }
    }
}

impl Field for AVXGF2_128 {
    const NAME: &'static str = "AVX Galois Field 2^128";

    const SIZE: usize = 128 / 8;

    const FIELD_SIZE: usize = 128; // in bits

    const ZERO: Self = AVXGF2_128 {
        v: unsafe { std::mem::zeroed() },
    };

    const ONE: Self = AVXGF2_128 {
        v: unsafe { std::mem::transmute::<[i32; 4], __m128i>([1, 0, 0, 0]) },
    };

    const INV_2: Self = AVXGF2_128 {
        v: unsafe { std::mem::zeroed() },
    };

    const MODULUS: U256 = unimplemented!(); // should not be used

    #[inline(always)]
    fn zero() -> Self {
        AVXGF2_128 {
            v: unsafe { std::mem::zeroed() },
        }
    }

    #[inline(always)]
    fn one() -> Self {
        AVXGF2_128 {
            // 1 in the first bit
            // TODO check bit order
            v: unsafe { std::mem::transmute::<[i32; 4], __m128i>([1, 0, 0, 0]) },
        }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        let mut u = [0u8; 16];
        rng.fill_bytes(&mut u);
        unsafe {
            AVXGF2_128 {
                v: *(u.as_ptr() as *const __m128i),
            }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        AVXGF2_128 {
            v: unsafe { std::mem::transmute::<[u32; 4], __m128i>([rng.next_u32() % 2, 0, 0, 0]) },
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        unsafe { std::mem::transmute::<__m128i, [u8; 16]>(self.v) == [0; 16] }
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
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        unsafe {
            AVXGF2_128 {
                v: transmute::<[u8; 16], __m128i>(bytes[..16].try_into().unwrap()),
            }
        }
    }
}

impl ExtensionField for AVXGF2_128 {
    const DEGREE: usize = 128;

    const W: u32 = 0x87;

    const X: Self = AVXGF2_128 {
        v: unsafe { std::mem::transmute::<[i32; 4], __m128i>([2, 0, 0, 0]) },
    };

    type BaseField = GF2;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        if base.v == 0 {
            Self::zero()
        } else {
            *self
        }
    }

    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mut res = *self;
        res.v = unsafe { _mm_xor_si128(res.v, _mm_set_epi64x(0, base.v as i64)) };
        res
    }

    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        unsafe {
            // Shift left by 1 bit
            let shifted = _mm_slli_epi64(self.v, 1);

            // Get the most significant bit and move it
            let msb = _mm_srli_epi64(self.v, 63);
            let msb_moved = _mm_slli_si128(msb, 8);

            // Combine the shifted value with the moved msb
            let shifted_consolidated = _mm_or_si128(shifted, msb_moved);

            // Create the reduction value (0x87) and the comparison value (1)
            let reduction = {
                let multiplier = _mm_set_epi64x(0, 0x87);
                let one = _mm_set_epi64x(0, 1);

                // Check if the MSB was 1 and create a mask
                let mask = _mm_cmpeq_epi64(_mm_srli_si128(msb, 8), one);

                _mm_and_si128(mask, multiplier)
            };

            // Apply the reduction conditionally
            let res = _mm_xor_si128(shifted_consolidated, reduction);

            Self { v: res }
        }
    }

    #[inline(always)]
    fn from_limbs(limbs: &[Self::BaseField]) -> Self {
        let mut local_limbs = limbs.to_vec();
        local_limbs.resize(Self::DEGREE, Self::BaseField::ZERO);

        let mut u32_lanes = [0u32; 4];
        local_limbs
            .chunks(32)
            .zip(u32_lanes.iter_mut())
            .for_each(|(limbs_by_32, u32_lane)| {
                limbs_by_32.iter().enumerate().for_each(|(ith_limb, limb)| {
                    *u32_lane |= (limb.v as u32) << ith_limb;
                });
            });

        Self {
            v: unsafe { transmute::<[u32; 4], __m128i>(u32_lanes) },
        }
    }

    #[inline(always)]
    fn to_limbs(&self) -> Vec<Self::BaseField> {
        let mut u32_extracted: [u32; 4] = unsafe { transmute(self.v) };

        let mut res = vec![Self::BaseField::ZERO; Self::DEGREE];
        u32_extracted
            .iter_mut()
            .enumerate()
            .for_each(|(ith_u32, u32_lane)| {
                (0..32).for_each(|ith_bit| {
                    let res_index = ith_bit + ith_u32 * 32;
                    res[res_index] = From::from(*u32_lane);
                    *u32_lane >>= 1;
                })
            });

        res
    }
}

impl Mul<GF2> for AVXGF2_128 {
    type Output = AVXGF2_128;

    #[inline(always)]
    fn mul(self, rhs: GF2) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl From<GF2> for AVXGF2_128 {
    #[inline(always)]
    fn from(v: GF2) -> Self {
        AVXGF2_128 {
            v: unsafe { _mm_set_epi64x(0, v.v as i64) },
        }
    }
}

#[inline]
unsafe fn gfmul(a: __m128i, b: __m128i) -> __m128i {
    let xmm_mask = _mm_setr_epi32((0xffffffff_u32) as i32, 0x0, 0x0, 0x0);

    // a = a0|a1, b = b0|b1

    let mut tmp3 = _mm_clmulepi64_si128(a, b, 0x00); // tmp3 = a0 * b0
    let mut tmp6 = _mm_clmulepi64_si128(a, b, 0x11); // tmp6 = a1 * b1

    let mut tmp4 = _mm_shuffle_epi32(a, 78); // tmp4 = a1|a0
    let mut tmp5 = _mm_shuffle_epi32(b, 78); // tmp5 = b1|b0
    tmp4 = _mm_xor_si128(tmp4, a); // tmp4 = (a0 + a1) | (a0 + a1)
    tmp5 = _mm_xor_si128(tmp5, b); // tmp5 = (b0 + b1) | (b0 + b1)

    tmp4 = _mm_clmulepi64_si128(tmp4, tmp5, 0x00); // tmp4 = (a0 + a1) * (b0 + b1)
    tmp4 = _mm_xor_si128(tmp4, tmp3); // tmp4 = (a0 + a1) * (b0 + b1) - a0 * b0
    tmp4 = _mm_xor_si128(tmp4, tmp6); // tmp4 = (a0 + a1) * (b0 + b1) - a0 * b0 - a1 * b1 = a0 * b1 + a1 * b0

    let tmp5_shifted_left = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp3 = _mm_xor_si128(tmp3, tmp5_shifted_left);
    tmp6 = _mm_xor_si128(tmp6, tmp4);

    let mut tmp7 = _mm_srli_epi32(tmp6, 31);
    let mut tmp8 = _mm_srli_epi32(tmp6, 30);
    let tmp9 = _mm_srli_epi32(tmp6, 25);

    tmp7 = _mm_xor_si128(tmp7, tmp8);
    tmp7 = _mm_xor_si128(tmp7, tmp9);

    tmp8 = _mm_shuffle_epi32(tmp7, 147);
    tmp7 = _mm_and_si128(xmm_mask, tmp8);
    tmp8 = _mm_andnot_si128(xmm_mask, tmp8);

    tmp3 = _mm_xor_si128(tmp3, tmp8);
    tmp6 = _mm_xor_si128(tmp6, tmp7);

    let tmp10 = _mm_slli_epi32(tmp6, 1);
    tmp3 = _mm_xor_si128(tmp3, tmp10);

    let tmp11 = _mm_slli_epi32(tmp6, 2);
    tmp3 = _mm_xor_si128(tmp3, tmp11);

    let tmp12 = _mm_slli_epi32(tmp6, 7);
    tmp3 = _mm_xor_si128(tmp3, tmp12);

    _mm_xor_si128(tmp3, tmp6)
}

impl Default for AVXGF2_128 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for AVXGF2_128 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe { _mm_test_all_ones(_mm_cmpeq_epi8(self.v, other.v)) == 1 }
    }
}

impl Eq for AVXGF2_128 {}

impl Neg for AVXGF2_128 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        self
    }
}

impl From<u32> for AVXGF2_128 {
    #[inline(always)]
    fn from(v: u32) -> Self {
        AVXGF2_128 {
            v: unsafe { std::mem::transmute::<[u32; 4], __m128i>([v, 0, 0, 0]) },
        }
    }
}

#[inline(always)]
fn add_internal(a: &AVXGF2_128, b: &AVXGF2_128) -> AVXGF2_128 {
    AVXGF2_128 {
        v: unsafe { _mm_xor_si128(a.v, b.v) },
    }
}

#[inline(always)]
fn sub_internal(a: &AVXGF2_128, b: &AVXGF2_128) -> AVXGF2_128 {
    AVXGF2_128 {
        v: unsafe { _mm_xor_si128(a.v, b.v) },
    }
}

#[inline(always)]
fn mul_internal(a: &AVXGF2_128, b: &AVXGF2_128) -> AVXGF2_128 {
    AVXGF2_128 {
        v: unsafe { gfmul(a.v, b.v) },
    }
}

impl Hash for AVXGF2_128 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<__m128i, [u8; 16]>(self.v).as_ref());
        }
    }
}
