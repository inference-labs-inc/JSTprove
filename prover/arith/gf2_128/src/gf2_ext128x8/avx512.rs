use std::fmt::Debug;
use std::{
    arch::x86_64::*,
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, ExtensionField, Field, SimdField};
use ethnum::U256;
use gf2::{GF2x8, GF2};
use serdes::{ExpSerde, SerdeError, SerdeResult};

use crate::GF2_128;

#[derive(Clone, Copy)]
pub struct AVX512GF2_128x8 {
    data: [__m512i; 2],
}

field_common!(AVX512GF2_128x8);

impl ExpSerde for AVX512GF2_128x8 {
    #[inline(always)]
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        unsafe {
            writer.write_all(transmute::<[__m512i; 2], [u8; 128]>(self.data).as_ref())?;
        }
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> Result<AVX512GF2_128x8, SerdeError> {
        let mut data = [0u8; 128];
        reader.read_exact(&mut data).unwrap();
        unsafe {
            Ok(Self {
                data: transmute::<[u8; 128], [__m512i; 2]>(data),
            })
        }
    }
}

const PACKED_0: [__m512i; 2] = [
    unsafe { transmute::<[i32; 16], std::arch::x86_64::__m512i>([0; 16]) },
    unsafe { transmute::<[i32; 16], std::arch::x86_64::__m512i>([0; 16]) },
];
const _M512_INV_2: __m512i = unsafe {
    transmute([
        67_u64,
        (1_u64) << 63,
        67_u64,
        (1_u64) << 63,
        67_u64,
        (1_u64) << 63,
        67_u64,
        (1_u64) << 63,
    ])
};
const PACKED_INV_2: [__m512i; 2] = [_M512_INV_2, _M512_INV_2]; // Should not be used?

// p(x) = x^128 + x^7 + x^2 + x + 1
impl Field for AVX512GF2_128x8 {
    const NAME: &'static str = "AVX512 Galois Field 2^128 SIMD 8";

    // size in bytes
    const SIZE: usize = 512 * 2 / 8;

    const ZERO: Self = Self { data: PACKED_0 };

    const ONE: Self = Self {
        data: unsafe {
            [
                transmute::<[u64; 8], __m512i>([1, 0, 1, 0, 1, 0, 1, 0]),
                transmute::<[u64; 8], __m512i>([1, 0, 1, 0, 1, 0, 1, 0]),
            ]
        },
    };

    const INV_2: Self = Self { data: PACKED_INV_2 };

    const FIELD_SIZE: usize = 128;

    const MODULUS: U256 = unimplemented!(); // should not be used

    #[inline(always)]
    fn zero() -> Self {
        unsafe {
            let zero = _mm512_setzero_si512();
            Self { data: [zero, zero] }
        }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        unsafe {
            let zero = _mm512_setzero_si512();
            let cmp_0 = _mm512_cmpeq_epi64_mask(self.data[0], zero);
            let cmp_1 = _mm512_cmpeq_epi64_mask(self.data[1], zero);
            (cmp_0 & cmp_1) == 0xFF // All 16 64-bit integers are equal (zero)
        }
    }

    #[inline(always)]
    fn one() -> Self {
        unsafe {
            let one = _mm512_set_epi64(0, 1, 0, 1, 0, 1, 0, 1);
            Self { data: [one, one] }
        }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        let data = [
            unsafe {
                _mm512_set_epi64(
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                )
            },
            unsafe {
                _mm512_set_epi64(
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                    rng.next_u64() as i64,
                )
            },
        ];
        Self { data }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        let data = [
            unsafe {
                _mm512_set_epi64(
                    0,
                    (rng.next_u64() % 2) as i64,
                    0,
                    (rng.next_u64() % 2) as i64,
                    0,
                    (rng.next_u64() % 2) as i64,
                    0,
                    (rng.next_u64() % 2) as i64,
                )
            },
            unsafe {
                _mm512_set_epi64(
                    0,
                    (rng.next_u64() % 2) as i64,
                    0,
                    (rng.next_u64() % 2) as i64,
                    0,
                    (rng.next_u64() % 2) as i64,
                    0,
                    (rng.next_u64() % 2) as i64,
                )
            },
        ];
        Self { data }
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
    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline(always)]
    fn from_uniform_bytes(_bytes: &[u8]) -> Self {
        todo!()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        *self * *self
    }

    #[inline(always)]
    fn double(&self) -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn mul_by_2(&self) -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn mul_by_3(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn mul_by_5(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn mul_by_6(&self) -> Self {
        Self::ZERO
    }
}
/*
credit to intel for the original implementation
void gfmul(__m128i a, __m128i b, __m128i *res) {
    __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    __m128i tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    __m128i XMMMASK = _mm_setr_epi32(0xffffffff, 0x0, 0x0, 0x0);

    // a = a0|a1, b = b0|b1

    tmp3 = _mm_clmulepi64_si128(a, b, 0x00); // tmp3 = a0 * b0
    tmp6 = _mm_clmulepi64_si128(a, b, 0x11); // tmp6 = a1 * b1

    tmp4 = _mm_shuffle_epi32(a, 78); // tmp4 = a1|a0
    tmp5 = _mm_shuffle_epi32(b, 78); // tmp5 = b1|b0
    tmp4 = _mm_xor_si128(tmp4, a); // tmp4 = (a0 + a1) | (a0 + a1)
    tmp5 = _mm_xor_si128(tmp5, b); // tmp5 = (b0 + b1) | (b0 + b1)

    tmp4 = _mm_clmulepi64_si128(tmp4, tmp5, 0x00); // tmp4 = (a0 + a1) * (b0 + b1)
    tmp4 = _mm_xor_si128(tmp4, tmp3); // tmp4 = (a0 + a1) * (b0 + b1) - a0 * b0
    tmp4 = _mm_xor_si128(tmp4, tmp6); // tmp4 = (a0 + a1) * (b0 + b1) - a0 * b0 - a1 * b1 = a0 * b1 + a1 * b0

    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp3 = _mm_xor_si128(tmp3, tmp5);
    tmp6 = _mm_xor_si128(tmp6, tmp4);

    tmp7 = _mm_srli_epi32(tmp6, 31);
    tmp8 = _mm_srli_epi32(tmp6, 30);
    tmp9 = _mm_srli_epi32(tmp6, 25);

    tmp7 = _mm_xor_si128(tmp7, tmp8);
    tmp7 = _mm_xor_si128(tmp7, tmp9);

    tmp8 = _mm_shuffle_epi32(tmp7, 147);
    tmp7 = _mm_and_si128(XMMMASK, tmp8);
    tmp8 = _mm_andnot_si128(XMMMASK, tmp8);

    tmp3 = _mm_xor_si128(tmp3, tmp8);
    tmp6 = _mm_xor_si128(tmp6, tmp7);

    tmp10 = _mm_slli_epi32(tmp6, 1);
    tmp3 = _mm_xor_si128(tmp3, tmp10);

    tmp11 = _mm_slli_epi32(tmp6, 2);
    tmp3 = _mm_xor_si128(tmp3, tmp11);

    tmp12 = _mm_slli_epi32(tmp6, 7);
    tmp3 = _mm_xor_si128(tmp3, tmp12);

    *res = _mm_xor_si128(tmp3, tmp6);
}

*/

/*
AVX 512 version
void gfmul_avx512(__m512i a, __m512i b, __m512i *res) {
    __m512i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    __m512i tmp7, tmp8, tmp9, tmp10, tmp11, tmp12;
    __m512i XMMMASK = _mm512_set_epi32(
        0, 0, 0, 0xffffffff,
        0, 0, 0, 0xffffffff,
        0, 0, 0, 0xffffffff,
        0, 0, 0, 0xffffffff
    );

    tmp3 = _mm512_clmulepi64_epi128(a, b, 0x00);
    tmp6 = _mm512_clmulepi64_epi128(a, b, 0x11);

    tmp4 = _mm512_shuffle_epi32(a, _MM_PERM_BADC);
    tmp5 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    tmp4 = _mm512_xor_si512(tmp4, a);
    tmp5 = _mm512_xor_si512(tmp5, b);

    tmp4 = _mm512_clmulepi64_epi128(tmp4, tmp5, 0x00);
    tmp4 = _mm512_xor_si512(tmp4, tmp3);
    tmp4 = _mm512_xor_si512(tmp4, tmp6);

    tmp5 = _mm512_bslli_epi128(tmp4, 8);
    tmp4 = _mm512_bsrli_epi128(tmp4, 8);
    tmp3 = _mm512_xor_si512(tmp3, tmp5);
    tmp6 = _mm512_xor_si512(tmp6, tmp4);

    tmp7 = _mm512_srli_epi32(tmp6, 31);
    tmp8 = _mm512_srli_epi32(tmp6, 30);
    tmp9 = _mm512_srli_epi32(tmp6, 25);

    tmp7 = _mm512_xor_si512(tmp7, tmp8);
    tmp7 = _mm512_xor_si512(tmp7, tmp9);

    tmp8 = _mm512_shuffle_epi32(tmp7, _MM_PERM_ABCD);
    tmp7 = _mm512_and_si512(XMMMASK, tmp8);
    tmp8 = _mm512_andnot_si512(XMMMASK, tmp8);

    tmp3 = _mm512_xor_si512(tmp3, tmp8);
    tmp6 = _mm512_xor_si512(tmp6, tmp7);

    tmp10 = _mm512_slli_epi32(tmp6, 1);
    tmp3 = _mm512_xor_si512(tmp3, tmp10);

    tmp11 = _mm512_slli_epi32(tmp6, 2);
    tmp3 = _mm512_xor_si512(tmp3, tmp11);

    tmp12 = _mm512_slli_epi32(tmp6, 7);
    tmp3 = _mm512_xor_si512(tmp3, tmp12);

    *res = _mm512_xor_si512(tmp3, tmp6);
}
 */

impl From<u32> for AVX512GF2_128x8 {
    #[inline(always)]
    fn from(v: u32) -> AVX512GF2_128x8 {
        assert!(v < 2); // only 0 and 1 are allowed
        let data = unsafe { _mm512_set_epi64(0, v as i64, 0, v as i64, 0, v as i64, 0, v as i64) };
        AVX512GF2_128x8 { data: [data, data] }
    }
}

impl Neg for AVX512GF2_128x8 {
    type Output = AVX512GF2_128x8;

    #[inline(always)]
    fn neg(self) -> AVX512GF2_128x8 {
        self
    }
}

impl Debug for AVX512GF2_128x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = [0u8; 128];
        unsafe {
            _mm512_storeu_si512(data.as_mut_ptr() as *mut __m512i, self.data[0]);
            _mm512_storeu_si512((data.as_mut_ptr() as *mut __m512i).offset(1), self.data[1]);
        }
        f.debug_struct("AVX512GF2_128x8")
            .field("data", &data)
            .finish()
    }
}

impl PartialEq for AVX512GF2_128x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp_0 = _mm512_cmpeq_epi64_mask(self.data[0], other.data[0]);
            let cmp_1 = _mm512_cmpeq_epi64_mask(self.data[1], other.data[1]);
            (cmp_0 & cmp_1) == 0xFF // All 16 64-bit integers are equal
        }
    }
}

impl Eq for AVX512GF2_128x8 {}

impl Default for AVX512GF2_128x8 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl From<GF2_128> for AVX512GF2_128x8 {
    #[inline(always)]
    fn from(v: GF2_128) -> AVX512GF2_128x8 {
        Self::pack_full(&v)
    }
}

// The following two must agree:
// 1. [GF2; 8] -> GF2x8 -> GF2_128x8
// 2. [GF2; 8] -> [GF2x128; 8] -> GF2_128x8

// For all simd field:
// [GF2x128; 8] -> GF2_128x8 -> [GF2x128; 8] should not change the order

impl SimdField for AVX512GF2_128x8 {
    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        let simd_challenge = AVX512GF2_128x8::from(*challenge);
        *self * simd_challenge
    }
    type Scalar = GF2_128;

    const PACK_SIZE: usize = 8;

    #[inline(always)]
    fn pack_full(scalar: &GF2_128) -> Self {
        Self {
            data: [unsafe { _mm512_broadcast_i32x4(scalar.v) }, unsafe {
                _mm512_broadcast_i32x4(scalar.v)
            }],
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), Self::PACK_SIZE);
        let base_vec_array: [Self::Scalar; 8] = base_vec.try_into().unwrap();
        unsafe { transmute(base_vec_array) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let array: [Self::Scalar; 8] = unsafe { transmute(self.data) };
        array.to_vec()
    }
}

#[inline(always)]
fn add_internal(a: &AVX512GF2_128x8, b: &AVX512GF2_128x8) -> AVX512GF2_128x8 {
    unsafe {
        AVX512GF2_128x8 {
            data: [
                _mm512_xor_si512(a.data[0], b.data[0]),
                _mm512_xor_si512(a.data[1], b.data[1]),
            ],
        }
    }
}

#[inline(always)]
fn sub_internal(a: &AVX512GF2_128x8, b: &AVX512GF2_128x8) -> AVX512GF2_128x8 {
    unsafe {
        AVX512GF2_128x8 {
            data: [
                _mm512_xor_si512(a.data[0], b.data[0]),
                _mm512_xor_si512(a.data[1], b.data[1]),
            ],
        }
    }
}

#[inline]
fn _m512_mul_internal(a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        let xmmmask = _mm512_set_epi32(
            0,
            0,
            0,
            0xffffffffu32 as i32,
            0,
            0,
            0,
            0xffffffffu32 as i32,
            0,
            0,
            0,
            0xffffffffu32 as i32,
            0,
            0,
            0,
            0xffffffffu32 as i32,
        );

        let mut tmp3 = _mm512_clmulepi64_epi128(a, b, 0x00);
        let mut tmp6 = _mm512_clmulepi64_epi128(a, b, 0x11);

        let mut tmp4 = _mm512_shuffle_epi32(a, _MM_PERM_BADC);
        let mut tmp5 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
        tmp4 = _mm512_xor_si512(tmp4, a);
        tmp5 = _mm512_xor_si512(tmp5, b);

        tmp4 = _mm512_clmulepi64_epi128(tmp4, tmp5, 0x00);
        tmp4 = _mm512_xor_si512(tmp4, tmp3);
        tmp4 = _mm512_xor_si512(tmp4, tmp6);

        tmp5 = _mm512_bslli_epi128(tmp4, 8);
        tmp4 = _mm512_bsrli_epi128(tmp4, 8);
        tmp3 = _mm512_xor_si512(tmp3, tmp5);
        tmp6 = _mm512_xor_si512(tmp6, tmp4);

        let tmp7 = _mm512_srli_epi32(tmp6, 31);
        let tmp8 = _mm512_srli_epi32(tmp6, 30);
        let tmp9 = _mm512_srli_epi32(tmp6, 25);

        let mut tmp7 = _mm512_xor_si512(tmp7, tmp8);
        tmp7 = _mm512_xor_si512(tmp7, tmp9);

        let mut tmp8 = _mm512_shuffle_epi32(tmp7, _MM_PERM_CBAD);
        tmp7 = _mm512_and_si512(xmmmask, tmp8);
        tmp8 = _mm512_andnot_si512(xmmmask, tmp8);

        tmp3 = _mm512_xor_si512(tmp3, tmp8);
        tmp6 = _mm512_xor_si512(tmp6, tmp7);

        let tmp10 = _mm512_slli_epi32(tmp6, 1);
        tmp3 = _mm512_xor_si512(tmp3, tmp10);

        let tmp11 = _mm512_slli_epi32(tmp6, 2);
        tmp3 = _mm512_xor_si512(tmp3, tmp11);

        let tmp12 = _mm512_slli_epi32(tmp6, 7);
        tmp3 = _mm512_xor_si512(tmp3, tmp12);

        _mm512_xor_si512(tmp3, tmp6)
    }
}

#[inline(always)]
fn mul_internal(a: &AVX512GF2_128x8, b: &AVX512GF2_128x8) -> AVX512GF2_128x8 {
    AVX512GF2_128x8 {
        data: [
            _m512_mul_internal(a.data[0], b.data[0]),
            _m512_mul_internal(a.data[1], b.data[1]),
        ],
    }
}

#[inline(always)]
// abcd**** -> aabbccdd
pub fn duplicate_higher_4bits(x: u8) -> u8 {
    let shifted = (x & 0b1000_0000)
        | (x & 0b0100_0000) >> 1
        | (x & 0b0010_0000) >> 2
        | (x & 0b0001_0000) >> 3;

    shifted | (shifted >> 1)
}

// ****abcd -> aabbccdd
#[inline(always)]
pub fn duplicate_lower_4bits(x: u8) -> u8 {
    let shifted = (x & 0b0001) | (x & 0b0010) << 1 | (x & 0b0100) << 2 | (x & 0b1000) << 3;
    shifted | (shifted << 1)
}

impl ExtensionField for AVX512GF2_128x8 {
    const DEGREE: usize = GF2_128::DEGREE;

    const W: u32 = GF2_128::W;

    const X: Self = Self {
        data: unsafe {
            [
                transmute::<[u64; 8], __m512i>([2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0]),
                transmute::<[u64; 8], __m512i>([2u64, 0, 2u64, 0, 2u64, 0, 2u64, 0]),
            ]
        },
    };

    type BaseField = GF2x8;

    #[inline(always)]
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self {
        let mask_high = duplicate_higher_4bits(base.v);
        let mask_low = duplicate_lower_4bits(base.v);

        Self {
            data: [
                unsafe { _mm512_maskz_mov_epi64(mask_low, self.data[0]) },
                unsafe { _mm512_maskz_mov_epi64(mask_high, self.data[1]) },
            ],
        }
    }

    #[inline(always)]
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self {
        let v0 = (base.v & 1u8) as i64;
        let v1 = ((base.v >> 1) & 1u8) as i64;
        let v2 = ((base.v >> 2) & 1u8) as i64;
        let v3 = ((base.v >> 3) & 1u8) as i64;
        let v4 = ((base.v >> 4) & 1u8) as i64;
        let v5 = ((base.v >> 5) & 1u8) as i64;
        let v6 = ((base.v >> 6) & 1u8) as i64;
        let v7 = ((base.v >> 7) & 1u8) as i64;

        let mut res = *self;
        res.data[0] = unsafe {
            _mm512_xor_si512(
                res.data[0],
                transmute::<[i64; 8], __m512i>([v0, 0, v1, 0, v2, 0, v3, 0]),
            )
        };
        res.data[1] = unsafe {
            _mm512_xor_si512(
                res.data[1],
                transmute::<[i64; 8], __m512i>([v4, 0, v5, 0, v6, 0, v7, 0]),
            )
        };

        res
    }

    #[inline(always)]
    fn mul_by_x(&self) -> Self {
        #[inline]
        fn mul_by_x_internal(data: __m512i) -> __m512i {
            unsafe {
                // Shift left by 1 bit
                let shifted = _mm512_slli_epi64(data, 1);

                // Get the most significant bit of each 64-bit part
                let msb = _mm512_srli_epi64(data, 63);

                // Move the MSB from the high 64 bits to the LSB of the low 64 bits
                // for each 128-bit element
                let msb_moved = _mm512_bslli_epi128(msb, 8);

                // Combine the shifted value with the moved msb
                let shifted_consolidated = _mm512_or_si512(shifted, msb_moved);

                // compute the reduced polynomial
                let reduction = {
                    let odd_elements = _mm512_maskz_compress_epi64(0b10101010, msb);
                    let mask = _mm512_maskz_expand_epi64(0b01010101, odd_elements);
                    let multiplier = _mm512_set1_epi64(0x87);
                    _mm512_mul_epu32(multiplier, mask)
                };

                // Apply the reduction conditionally
                _mm512_xor_si512(shifted_consolidated, reduction)
            }
        }

        Self {
            data: [
                mul_by_x_internal(self.data[0]),
                mul_by_x_internal(self.data[1]),
            ],
        }
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

impl From<GF2x8> for AVX512GF2_128x8 {
    #[inline(always)]
    fn from(v: GF2x8) -> Self {
        let v0 = (v.v & 1u8) as i64;
        let v1 = ((v.v >> 1) & 1u8) as i64;
        let v2 = ((v.v >> 2) & 1u8) as i64;
        let v3 = ((v.v >> 3) & 1u8) as i64;
        let v4 = ((v.v >> 4) & 1u8) as i64;
        let v5 = ((v.v >> 5) & 1u8) as i64;
        let v6 = ((v.v >> 6) & 1u8) as i64;
        let v7 = ((v.v >> 7) & 1u8) as i64;

        AVX512GF2_128x8 {
            data: [
                unsafe { transmute::<[i64; 8], __m512i>([v0, 0, v1, 0, v2, 0, v3, 0]) },
                unsafe { transmute::<[i64; 8], __m512i>([v4, 0, v5, 0, v6, 0, v7, 0]) },
            ],
        }
    }
}

impl Mul<GF2> for AVX512GF2_128x8 {
    type Output = AVX512GF2_128x8;

    #[inline(always)]
    fn mul(self, rhs: GF2) -> Self::Output {
        if rhs.is_zero() {
            Self::zero()
        } else {
            self
        }
    }
}

impl Mul<GF2x8> for AVX512GF2_128x8 {
    type Output = AVX512GF2_128x8;

    #[inline(always)]
    fn mul(self, rhs: GF2x8) -> Self::Output {
        self.mul_by_base_field(&rhs)
    }
}

impl Add<GF2> for AVX512GF2_128x8 {
    type Output = AVX512GF2_128x8;
    #[inline(always)]
    fn add(self, rhs: GF2) -> Self::Output {
        let rhs_extended = unsafe { _mm512_maskz_set1_epi64(0b01010101, rhs.v as i64) };
        AVX512GF2_128x8 {
            data: [
                unsafe { _mm512_xor_si512(self.data[0], rhs_extended) },
                unsafe { _mm512_xor_si512(self.data[1], rhs_extended) },
            ],
        }
    }
}

impl std::hash::Hash for AVX512GF2_128x8 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<[__m512i; 2], [u8; 128]>(self.data).as_ref());
        }
    }
}
