use std::{
    arch::aarch64::*,
    fmt::Debug,
    io::{Read, Write},
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, Field, SimdField};
use ethnum::U256;
use rand::{Rng, RngCore};
use serdes::{ExpSerde, SerdeResult};

use crate::{m31::M31_MOD, M31};

const M31_PACK_SIZE: usize = 16;
const PACKED_MOD: uint32x4_t = unsafe { transmute([M31_MOD; 4]) };
const PACKED_0: uint32x4_t = unsafe { transmute([0; 4]) };
const PACKED_INV_2: uint32x4_t = unsafe { transmute([1 << 30; 4]) };

#[inline(always)]
fn reduce_sum(x: uint32x4_t) -> uint32x4_t {
    //aarch64 only
    unsafe { vminq_u32(x, vsubq_u32(x, PACKED_MOD)) }
}

/// NeonM31 packs 16 M31 elements and operates on them in parallel
#[derive(Clone, Copy)]
pub struct NeonM31 {
    pub v: [uint32x4_t; 4],
}

field_common!(NeonM31);

impl NeonM31 {
    pub fn printavxtype() {
        println!("Not avx");
    }
}

impl ExpSerde for NeonM31 {
    #[inline(always)]
    /// serialize self into bytes
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        let data = unsafe { transmute::<[uint32x4_t; 4], [u8; 64]>(self.v) };
        writer.write_all(&data)?;
        Ok(())
    }

    /// deserialize bytes into field
    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut data = [0; 64];
        reader.read_exact(&mut data)?;
        unsafe {
            Ok(NeonM31 {
                v: transmute::<[u8; 64], [uint32x4_t; 4]>(data),
            })
        }
    }
}

impl Field for NeonM31 {
    const NAME: &'static str = "Neon Packed Mersenne 31";

    // size in bytes
    const SIZE: usize = 128 / 8 * 4;

    const FIELD_SIZE: usize = 32;

    const ZERO: Self = Self { v: [PACKED_0; 4] };

    const ONE: Self = Self {
        v: [unsafe { transmute::<[u32; 4], uint32x4_t>([1; 4]) }; 4],
    };

    const INV_2: Self = Self {
        v: [PACKED_INV_2; 4],
    };

    const MODULUS: U256 = M31::MODULUS;

    #[inline(always)]
    fn zero() -> Self {
        Self { v: [PACKED_0; 4] }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        unsafe {
            transmute::<[uint32x4_t; 4], [u32; 16]>(self.v)
                .iter()
                .all(|&x| x == 0)
        }
    }

    #[inline(always)]
    fn one() -> Self {
        NeonM31 {
            v: unsafe { [vdupq_n_u32(1); 4] },
        }
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            NeonM31 {
                v: [
                    vld1q_u32(
                        [
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                        ]
                        .as_ptr(),
                    ),
                    vld1q_u32(
                        [
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                        ]
                        .as_ptr(),
                    ),
                    vld1q_u32(
                        [
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                        ]
                        .as_ptr(),
                    ),
                    vld1q_u32(
                        [
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                            rng.gen::<u32>() % M31_MOD,
                        ]
                        .as_ptr(),
                    ),
                ],
            }
        }
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        unsafe {
            NeonM31 {
                v: [
                    vld1q_u32(
                        [
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                        ]
                        .as_ptr(),
                    ),
                    vld1q_u32(
                        [
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                        ]
                        .as_ptr(),
                    ),
                    vld1q_u32(
                        [
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                        ]
                        .as_ptr(),
                    ),
                    vld1q_u32(
                        [
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                            rng.gen::<bool>() as u32,
                        ]
                        .as_ptr(),
                    ),
                ],
            }
        }
    }

    #[inline]
    fn double(&self) -> Self {
        self.mul_by_2()
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let mut m31_vec = unsafe { transmute::<[uint32x4_t; 4], [M31; 16]>(self.v) };
        let is_non_zero = m31_vec.iter().all(|x| !x.is_zero());
        if !is_non_zero {
            return None;
        }

        m31_vec.iter_mut().for_each(|x| *x = x.inv().unwrap()); // safe unwrap
        Some(Self {
            v: unsafe { transmute::<[M31; 16], [uint32x4_t; 4]>(m31_vec) },
        })
    }

    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    #[inline]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let m = M31::from_uniform_bytes(bytes);
        Self {
            v: unsafe {
                [
                    vdupq_n_u32(m.v),
                    vdupq_n_u32(m.v),
                    vdupq_n_u32(m.v),
                    vdupq_n_u32(m.v),
                ]
            },
        }
    }

    #[inline(always)]
    fn mul_by_2(&self) -> NeonM31 {
        let mut res = NeonM31::zero();
        for i in 0..4 {
            res.v[i] = unsafe {
                let double = vshlq_n_u32(self.v[i], 1);
                reduce_sum(double)
            };
        }
        res
    }

    #[inline(always)]
    fn mul_by_5(&self) -> NeonM31 {
        let mut res = NeonM31::zero();
        for i in 0..4 {
            res.v[i] = unsafe {
                let double = reduce_sum(vshlq_n_u32(self.v[i], 1));
                let quad = reduce_sum(vshlq_n_u32(double, 1));
                reduce_sum(vaddq_u32(quad, self.v[i]))
            };
        }
        res
    }
}

impl SimdField for NeonM31 {
    type Scalar = M31;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        let packed_challenge = NeonM31::pack_full(challenge);
        *self * packed_challenge
    }

    const PACK_SIZE: usize = M31_PACK_SIZE;

    #[inline(always)]
    fn pack_full(x: &M31) -> NeonM31 {
        NeonM31 {
            v: unsafe {
                [
                    vdupq_n_u32(x.v),
                    vdupq_n_u32(x.v),
                    vdupq_n_u32(x.v),
                    vdupq_n_u32(x.v),
                ]
            },
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert!(base_vec.len() == M31_PACK_SIZE);
        let ret: [Self::Scalar; M31_PACK_SIZE] = base_vec.try_into().unwrap();
        unsafe { transmute(ret) }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        let ret = unsafe { transmute::<[uint32x4_t; 4], [Self::Scalar; M31_PACK_SIZE]>(self.v) };
        ret.to_vec()
    }

    #[inline(always)]
    fn horizontal_sum(&self) -> Self::Scalar {
        let ret = unsafe { transmute::<[uint32x4_t; 4], [Self::Scalar; M31_PACK_SIZE]>(self.v) };

        // NOTE(HS): Intentionally manual unrolling
        let mut buffer: u64 = ret[0].v as u64;
        buffer += ret[1].v as u64;
        buffer += ret[2].v as u64;
        buffer += ret[3].v as u64;
        buffer += ret[4].v as u64;
        buffer += ret[5].v as u64;
        buffer += ret[6].v as u64;
        buffer += ret[7].v as u64;
        buffer += ret[8].v as u64;
        buffer += ret[9].v as u64;
        buffer += ret[10].v as u64;
        buffer += ret[11].v as u64;
        buffer += ret[12].v as u64;
        buffer += ret[13].v as u64;
        buffer += ret[14].v as u64;
        buffer += ret[15].v as u64;

        buffer = (buffer & M31_MOD as u64) + (buffer >> 31);
        if buffer == M31_MOD as u64 {
            Self::Scalar::ZERO
        } else {
            Self::Scalar { v: buffer as u32 }
        }
    }
}

impl From<M31> for NeonM31 {
    #[inline(always)]
    fn from(x: M31) -> Self {
        NeonM31::pack_full(&x)
    }
}

impl Debug for NeonM31 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &v in self.v.iter() {
            unsafe {
                let data = [
                    vgetq_lane_u32(v, 0),
                    vgetq_lane_u32(v, 1),
                    vgetq_lane_u32(v, 2),
                    vgetq_lane_u32(v, 3),
                ];
                // if all data is the same, print only one
                if data.iter().all(|&x| x == data[0]) {
                    write!(
                        f,
                        "uint32x4_t<8 x {}>",
                        if M31_MOD - data[0] > 1024 {
                            format!("{}", data[0])
                        } else {
                            format!("-{}", M31_MOD - data[0])
                        }
                    )?;
                } else {
                    write!(f, "uint32x4_t<{data:?}>")?;
                }
            }
        }
        Ok(())
    }
}

impl Default for NeonM31 {
    fn default() -> Self {
        NeonM31::zero()
    }
}

impl PartialEq for NeonM31 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            transmute::<[uint32x4_t; 4], [u32; 16]>([
                reduce_sum(self.v[0]),
                reduce_sum(self.v[1]),
                reduce_sum(self.v[2]),
                reduce_sum(self.v[3]),
            ]) == transmute::<[uint32x4_t; 4], [u32; 16]>([
                reduce_sum(other.v[0]),
                reduce_sum(other.v[1]),
                reduce_sum(other.v[2]),
                reduce_sum(other.v[3]),
            ])
        }
    }
}

impl Eq for NeonM31 {}

impl Mul<&M31> for NeonM31 {
    type Output = NeonM31;
    #[inline(always)]
    fn mul(self, rhs: &M31) -> Self::Output {
        let rhs_p = NeonM31::pack_full(rhs);
        self * rhs_p
    }
}

impl Mul<M31> for NeonM31 {
    type Output = NeonM31;
    #[inline(always)]
    fn mul(self, rhs: M31) -> Self::Output {
        self * &rhs
    }
}

impl Add<M31> for NeonM31 {
    type Output = NeonM31;
    #[inline(always)]
    #[allow(clippy::op_ref)]
    fn add(self, rhs: M31) -> Self::Output {
        self + NeonM31::pack_full(&rhs)
    }
}

impl From<u32> for NeonM31 {
    #[inline(always)]
    fn from(x: u32) -> Self {
        NeonM31::pack_full(&M31::from(x))
    }
}

impl Neg for NeonM31 {
    type Output = NeonM31;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        NeonM31::zero() - self
    }
}

#[inline(always)]
fn add_internal(a: &NeonM31, b: &NeonM31) -> NeonM31 {
    NeonM31 {
        v: [
            unsafe { reduce_sum(vaddq_u32(a.v[0], b.v[0])) },
            unsafe { reduce_sum(vaddq_u32(a.v[1], b.v[1])) },
            unsafe { reduce_sum(vaddq_u32(a.v[2], b.v[2])) },
            unsafe { reduce_sum(vaddq_u32(a.v[3], b.v[3])) },
        ],
    }
}

#[inline(always)]
fn sub_internal(a: &NeonM31, b: &NeonM31) -> NeonM31 {
    NeonM31 {
        v: [
            unsafe {
                let diff = vsubq_u32(a.v[0], b.v[0]);
                let u = vaddq_u32(diff, PACKED_MOD);
                vminq_u32(diff, u)
            },
            unsafe {
                let diff = vsubq_u32(a.v[1], b.v[1]);
                let u = vaddq_u32(diff, PACKED_MOD);
                vminq_u32(diff, u)
            },
            unsafe {
                let diff = vsubq_u32(a.v[2], b.v[2]);
                let u = vaddq_u32(diff, PACKED_MOD);
                vminq_u32(diff, u)
            },
            unsafe {
                let diff = vsubq_u32(a.v[3], b.v[3]);
                let u = vaddq_u32(diff, PACKED_MOD);
                vminq_u32(diff, u)
            },
        ],
    }
}

#[inline]
fn mul_internal(a: &NeonM31, b: &NeonM31) -> NeonM31 {
    let mut res = NeonM31::zero();
    for i in 0..4 {
        res.v[i] = unsafe {
            let prod_hi = vreinterpretq_u32_s32(vqdmulhq_s32(
                vreinterpretq_s32_u32(a.v[i]),
                vreinterpretq_s32_u32(b.v[i]),
            ));
            let prod_lo = vmulq_u32(a.v[i], b.v[i]);
            let t = vmlsq_u32(prod_lo, prod_hi, PACKED_MOD);
            reduce_sum(t)
        };
    }
    res
}

impl std::hash::Hash for NeonM31 {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<[uint32x4_t; 4], [u8; 64]>(self.v).as_ref());
        }
    }
}
