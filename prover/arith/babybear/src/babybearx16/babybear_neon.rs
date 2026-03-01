use std::{
    arch::aarch64::*,
    fmt::Debug,
    hash::{Hash, Hasher},
    io::{Read, Write},
    iter::{Product, Sum},
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::{ExpSerde, SerdeResult};

use crate::{babybear::BABY_BEAR_MOD, BabyBear};

const BABY_BEAR_PACK_SIZE: usize = 16;

#[derive(Clone, Copy)]
pub struct NeonBabyBear {
    pub v: [uint32x4_t; 4],
}

const PACKED_MOD: uint32x4_t = unsafe { transmute::<[u32; 4], uint32x4_t>([BABY_BEAR_MOD; 4]) };

#[inline]
unsafe fn mod_reduce_epi32(x: uint32x4_t) -> uint32x4_t {
    let mask = vcgeq_u32(x, PACKED_MOD);
    vsubq_u32(x, vandq_u32(mask, PACKED_MOD))
}

#[inline]
unsafe fn mod_reduce_epi32x4_twice(x: &[uint32x4_t; 4]) -> [uint32x4_t; 4] {
    x.iter()
        .map(|x| mod_reduce_epi32(mod_reduce_epi32(*x)))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

field_common!(NeonBabyBear);

impl NeonBabyBear {
    #[inline(always)]
    pub fn is_canonical(&self) -> bool {
        self.unpack().iter().all(|x| x.value < BABY_BEAR_MOD)
    }
}

impl ExpSerde for NeonBabyBear {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        unsafe {
            let data = mod_reduce_epi32x4_twice(&self.v);
            let data = transmute::<[uint32x4_t; 4], [u8; 64]>(data);

            writer.write_all(&data)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut data = [0; 64];
        reader.read_exact(&mut data)?;
        unsafe {
            Ok(NeonBabyBear {
                v: transmute::<[u8; 64], [uint32x4_t; 4]>(data),
            })
        }
    }
}

impl Field for NeonBabyBear {
    const NAME: &'static str = "Neon Packed BabyBear";

    const SIZE: usize = 128 / 8 * 4;

    const FIELD_SIZE: usize = 32;

    const ZERO: Self = Self {
        v: unsafe {
            transmute::<[BabyBear; BABY_BEAR_PACK_SIZE], [uint32x4_t; 4]>(
                [BabyBear::ZERO; BABY_BEAR_PACK_SIZE],
            )
        },
    };

    const ONE: Self = Self {
        v: unsafe {
            transmute::<[BabyBear; BABY_BEAR_PACK_SIZE], [uint32x4_t; 4]>(
                [BabyBear::ONE; BABY_BEAR_PACK_SIZE],
            )
        },
    };

    const INV_2: Self = Self {
        v: unsafe {
            transmute::<[BabyBear; BABY_BEAR_PACK_SIZE], [uint32x4_t; 4]>(
                [BabyBear::INV_2; BABY_BEAR_PACK_SIZE],
            )
        },
    };

    const MODULUS: U256 = BabyBear::MODULUS;

    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn one() -> Self {
        Self::ONE
    }

    fn random_unsafe(mut rng: impl RngCore) -> Self {
        // Caution: this may not produce uniformly random elements
        unsafe {
            let mut data = [0u8; 64];
            rng.fill_bytes(&mut data);
            let mut v = transmute::<[u8; 64], [uint32x4_t; 4]>(data);
            v = mod_reduce_epi32x4_twice(&v);
            Self { v }
        }
    }

    fn random_bool(mut rng: impl RngCore) -> Self {
        let sample = (0..BABY_BEAR_PACK_SIZE)
            .map(|_| BabyBear::random_bool(&mut rng))
            .collect::<Vec<_>>();
        Self::pack(&sample)
    }

    fn inv(&self) -> Option<Self> {
        // slow, should not be used in production
        let mut babybear_vec =
            unsafe { transmute::<[uint32x4_t; 4], [BabyBear; BABY_BEAR_PACK_SIZE]>(self.v) };
        let is_nonzero = babybear_vec.iter().all(|x| !x.is_zero());
        if !is_nonzero {
            return None;
        };
        babybear_vec.iter_mut().for_each(|x| *x = x.inv().unwrap());
        Some(Self::pack(&babybear_vec))
    }

    fn as_u32_unchecked(&self) -> u32 {
        unimplemented!("self is a vector, cannot convert to u32")
    }

    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        Self::pack_full(&BabyBear::from_uniform_bytes(bytes))
    }
}

impl SimdField for NeonBabyBear {
    type Scalar = BabyBear;

    const PACK_SIZE: usize = BABY_BEAR_PACK_SIZE;

    #[inline]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * *challenge
    }

    #[inline(always)]
    fn pack_full(x: &BabyBear) -> NeonBabyBear {
        NeonBabyBear {
            v: unsafe {
                // Safety: memory representation of [x; BABY_BEAR_PACK_SIZE]
                // is 16 u32s, which can be reinterpreted as 4 uint32x4_t.
                transmute::<[BabyBear; BABY_BEAR_PACK_SIZE], [uint32x4_t; 4]>(
                    [*x; BABY_BEAR_PACK_SIZE],
                )
            },
        }
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        debug_assert!(base_vec.len() == BABY_BEAR_PACK_SIZE);
        let ret: [Self::Scalar; BABY_BEAR_PACK_SIZE] = base_vec.try_into().unwrap();
        Self {
            // Transmute is reinterpreting an array of scalars in Montgomery form to an AVX register
            v: unsafe { transmute::<[Self::Scalar; BABY_BEAR_PACK_SIZE], [uint32x4_t; 4]>(ret) },
        }
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        // Transmute is reinterpreting an AVX register to an array of scalars in Montgomery form
        let ret =
            unsafe { transmute::<[uint32x4_t; 4], [Self::Scalar; BABY_BEAR_PACK_SIZE]>(self.v) };
        ret.to_vec()
    }
}

impl From<BabyBear> for NeonBabyBear {
    #[inline(always)]
    fn from(x: BabyBear) -> Self {
        NeonBabyBear::pack_full(&x)
    }
}

impl Debug for NeonBabyBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let unpacked = self.unpack();
        if unpacked.iter().all(|x| *x == unpacked[0]) {
            write!(f, "uint32x4_t<16 x {:?}>", unpacked[0])
        } else {
            write!(f, "uint32x4_t<{unpacked:?}>")
        }
    }
}

impl Default for NeonBabyBear {
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for NeonBabyBear {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            transmute::<[uint32x4_t; 4], [u32; 16]>(mod_reduce_epi32x4_twice(&self.v))
                == transmute::<[uint32x4_t; 4], [u32; 16]>(mod_reduce_epi32x4_twice(&other.v))
        }
    }
}

impl Eq for NeonBabyBear {}

impl Mul<&BabyBear> for NeonBabyBear {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: &BabyBear) -> Self::Output {
        self * NeonBabyBear::pack_full(rhs)
    }
}

impl Mul<BabyBear> for NeonBabyBear {
    type Output = NeonBabyBear;
    #[inline(always)]
    #[allow(clippy::op_ref)]
    fn mul(self, rhs: BabyBear) -> Self::Output {
        self * &rhs
    }
}

impl Add<BabyBear> for NeonBabyBear {
    type Output = NeonBabyBear;
    #[inline(always)]
    fn add(self, rhs: BabyBear) -> Self::Output {
        self + NeonBabyBear::pack_full(&rhs)
    }
}

impl From<u32> for NeonBabyBear {
    #[inline(always)]
    fn from(value: u32) -> Self {
        // BabyBear::new converts to Montgomery form
        NeonBabyBear::pack_full(&BabyBear::new(value))
    }
}

impl Neg for NeonBabyBear {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        NeonBabyBear {
            v: [
                p3_instructions::neg(self.v[0]),
                p3_instructions::neg(self.v[1]),
                p3_instructions::neg(self.v[2]),
                p3_instructions::neg(self.v[3]),
            ],
        }
    }
}

#[inline(always)]
fn add_internal(a: &NeonBabyBear, b: &NeonBabyBear) -> NeonBabyBear {
    NeonBabyBear {
        v: [
            p3_instructions::add(a.v[0], b.v[0]),
            p3_instructions::add(a.v[1], b.v[1]),
            p3_instructions::add(a.v[2], b.v[2]),
            p3_instructions::add(a.v[3], b.v[3]),
        ],
    }
}

#[inline(always)]
fn sub_internal(a: &NeonBabyBear, b: &NeonBabyBear) -> NeonBabyBear {
    NeonBabyBear {
        v: [
            p3_instructions::sub(a.v[0], b.v[0]),
            p3_instructions::sub(a.v[1], b.v[1]),
            p3_instructions::sub(a.v[2], b.v[2]),
            p3_instructions::sub(a.v[3], b.v[3]),
        ],
    }
}

#[inline]
fn mul_internal(a: &NeonBabyBear, b: &NeonBabyBear) -> NeonBabyBear {
    if !a.is_canonical() || !b.is_canonical() {
        panic!("mul_internal: input is not canonical\n{a:?}\n{b:?}");
    }

    NeonBabyBear {
        v: [
            p3_instructions::mul(a.v[0], b.v[0]),
            p3_instructions::mul(a.v[1], b.v[1]),
            p3_instructions::mul(a.v[2], b.v[2]),
            p3_instructions::mul(a.v[3], b.v[3]),
        ],
    }
}

impl Hash for NeonBabyBear {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            state.write(transmute::<[uint32x4_t; 4], [u8; 64]>(self.v).as_ref());
        }
    }
}

mod p3_instructions {
    use std::{
        arch::{aarch64::*, asm},
        hint::unreachable_unchecked,
        mem::transmute,
    };

    use super::PACKED_MOD;

    const PACKED_MU: int32x4_t = unsafe { transmute::<[i32; 4], int32x4_t>([-0x77ffffff; 4]) };

    /// No-op. Prevents the compiler from deducing the value of the vector.
    ///
    /// Similar to `std::hint::black_box`, it can be used to stop the compiler applying undesirable
    /// "optimizations". Unlike the built-in `black_box`, it does not force the value to be written
    /// to and then read from the stack.
    #[inline]
    #[must_use]
    fn confuse_compiler(x: uint32x4_t) -> uint32x4_t {
        let y;
        unsafe {
            asm!(
                "/*{0:v}*/",
                inlateout(vreg) x => y,
                options(nomem, nostack, preserves_flags, pure),
            );
            // Below tells the compiler the semantics of this so it can still do constant folding,
            // etc. You may ask, doesn't it defeat the point of the inline asm block to
            // tell the compiler what it does? The answer is that we still inhibit the
            // transform we want to avoid, so apparently not. Idk, LLVM works in
            // mysterious ways.
            if transmute::<uint32x4_t, [u32; 4]>(x) != transmute::<uint32x4_t, [u32; 4]>(y) {
                unreachable_unchecked();
            }
        }
        y
    }

    /// Add two vectors of Monty31 field elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn add(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
        // We want this to compile to:
        //      add   t.4s, lhs.4s, rhs.4s
        //      sub   u.4s, t.4s, P.4s
        //      umin  res.4s, t.4s, u.4s
        // throughput: .75 cyc/vec (5.33 els/cyc)
        // latency: 6 cyc

        //   Let `t := lhs + rhs`. We want to return `t mod P`. Recall that `lhs` and `rhs` are in
        // `0, ..., P - 1`, so `t` is in `0, ..., 2 P - 2 (< 2^32)`. It suffices to return `t` if
        // `t < P` and `t - P` otherwise.
        //   Let `u := (t - P) mod 2^32` and `r := unsigned_min(t, u)`.
        //   If `t` is in `0, ..., P - 1`, then `u` is in `(P - 1 <) 2^32 - P, ..., 2^32 - 1` and
        // `r = t`. Otherwise `t` is in `P, ..., 2 P - 2`, `u` is in `0, ..., P - 2 (< P)` and `r =
        // u`. Hence, `r` is `t` if `t < P` and `t - P` otherwise, as desired.

        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let t = vaddq_u32(lhs, rhs);
            let u = vsubq_u32(t, PACKED_MOD);
            vminq_u32(t, u)
        }
    }

    /// Subtract vectors of Monty31 field elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn sub(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
        // We want this to compile to:
        //      sub   res.4s, lhs.4s, rhs.4s
        //      cmhi  underflow.4s, rhs.4s, lhs.4s
        //      mls   res.4s, underflow.4s, P.4s
        // throughput: .75 cyc/vec (5.33 els/cyc)
        // latency: 5 cyc

        //   Let `d := lhs - rhs`. We want to return `d mod P`.
        //   Since `lhs` and `rhs` are both in `0, ..., P - 1`, `d` is in `-P + 1, ..., P - 1`. It
        // suffices to return `d + P` if `d < 0` and `d` otherwise.
        //   Equivalently, we return `d + P` if `rhs > lhs` and `d` otherwise.  Observe that this
        // permits us to perform all calculations `mod 2^32`, so define `diff := d mod 2^32`.
        //   Let `underflow` be `-1 mod 2^32` if `rhs > lhs` and `0` otherwise.
        //   Finally, let `r := (diff - underflow * P) mod 2^32` and observe that
        // `r = (diff + P) mod 2^32` if `rhs > lhs` and `diff` otherwise, as desired.
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let diff = vsubq_u32(lhs, rhs);
            let underflow = vcltq_u32(lhs, rhs);
            // We really want to emit a `mls` instruction here. The compiler knows that `underflow`
            // is either 0 or -1 and will try to do an `and` and `add` instead, which is
            // slower on the M1. The `confuse_compiler` prevents this "optimization".
            vmlsq_u32(diff, confuse_compiler(underflow), PACKED_MOD)
        }
    }

    /// Negate a vector of Monty31 field elements in canonical form.
    /// If the inputs are not in canonical form, the result is undefined.
    #[inline]
    #[must_use]
    pub(super) fn neg(val: uint32x4_t) -> uint32x4_t {
        // We want this to compile to:
        //      sub   t.4s, P.4s, val.4s
        //      cmeq  is_zero.4s, val.4s, #0
        //      bic   res.4s, t.4s, is_zero.4s
        // throughput: .75 cyc/vec (5.33 els/cyc)
        // latency: 4 cyc

        // This has the same throughput as `sub(0, val)` but slightly lower latency.

        //   We want to return (-val) mod P. This is equivalent to returning `0` if `val = 0` and
        // `P - val` otherwise, since `val` is in `0, ..., P - 1`.
        //   Let `t := P - val` and let `is_zero := (-1) mod 2^32` if `val = 0` and `0` otherwise.
        //   We return `r := t & ~is_zero`, which is `t` if `val > 0` and `0` otherwise, as desired.
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let t = vsubq_u32(PACKED_MOD, val);
            let is_zero = vceqzq_u32(val);
            vbicq_u32(t, is_zero)
        }
    }

    // MONTGOMERY MULTIPLICATION
    //   This implementation is based on [1] but with changes. The reduction is as follows:
    //
    // Constants: P < 2^31
    //            B = 2^32
    //            μ = P^-1 mod B
    // Input: -P^2 <= C <= P^2
    // Output: -P < D < P such that D = C B^-1 (mod P)
    // Define:
    //   smod_B(a) = r, where -B/2 <= r <= B/2 - 1 and r = a (mod B).
    // Algorithm:
    //   1. Q := smod_B(μ C)
    //   2. D := (C - Q P) / B
    //
    // We first show that the division in step 2. is exact. It suffices to show that C = Q P (mod
    // B). By definition of Q, smod_B, and μ, we have Q P = smod_B(μ C) P = μ C P = P^-1 C P = C
    // (mod B).
    //
    // We also have C - Q P = C (mod P), so thus D = C B^-1 (mod P).
    //
    // It remains to show that D is in the correct range. It suffices to show that -P B < C - Q P <
    // P B. We know that -P^2 <= C <= P^2 and (-B / 2) P <= Q P <= (B/2 - 1) P. Then
    // (1 - B/2) P - P^2 <= C - Q P <= (B/2) P + P^2. Now, P < B/2, so B/2 + P < B and
    // (B/2) P + P^2 < P B; also B/2 - 1 + P < B, so -P B < (1 - B/2) P - P^2.
    // Hence, -P B < C - Q P < P B as desired.
    //
    // [1] Modern Computer Arithmetic, Richard Brent and Paul Zimmermann, Cambridge University
    // Press,     2010, algorithm 2.7.

    #[inline]
    #[must_use]
    fn mulby_mu(val: int32x4_t) -> int32x4_t {
        // We want this to compile to:
        //      mul      res.4s, val.4s, MU.4s
        // throughput: .25 cyc/vec (16 els/cyc)
        // latency: 3 cyc

        unsafe { vmulq_s32(val, PACKED_MU) }
    }

    #[inline]
    #[must_use]
    fn get_c_hi(lhs: int32x4_t, rhs: int32x4_t) -> int32x4_t {
        // We want this to compile to:
        //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
        // throughput: .25 cyc/vec (16 els/cyc)
        // latency: 3 cyc

        unsafe {
            // Get bits 31, ..., 62 of C. Note that `sqdmulh` saturates when the product doesn't fit
            // in an `i63`, but this cannot happen here due to our bounds on `lhs` and
            // `rhs`.
            vqdmulhq_s32(lhs, rhs)
        }
    }

    #[inline]
    #[must_use]
    fn get_qp_hi(lhs: int32x4_t, mu_rhs: int32x4_t) -> int32x4_t {
        // We want this to compile to:
        //      mul      q.4s, lhs.4s, mu_rhs.4s
        //      sqdmulh  qp_hi.4s, q.4s, P.4s
        // throughput: .5 cyc/vec (8 els/cyc)
        // latency: 6 cyc

        unsafe {
            // Form `Q`.
            let q = vmulq_s32(lhs, mu_rhs);

            // Gets bits 31, ..., 62 of Q P. Again, saturation is not an issue because `P` is not
            // -2**31.
            vqdmulhq_s32(q, vreinterpretq_s32_u32(PACKED_MOD))
        }
    }

    #[inline]
    #[must_use]
    fn get_d(c_hi: int32x4_t, qp_hi: int32x4_t) -> int32x4_t {
        // We want this to compile to:
        //      shsub    res.4s, c_hi.4s, qp_hi.4s
        // throughput: .25 cyc/vec (16 els/cyc)
        // latency: 2 cyc

        unsafe {
            // Form D. Note that `c_hi` is C >> 31 and `qp_hi` is (Q P) >> 31, whereas we want
            // (C - Q P) >> 32, so we need to subtract and divide by 2. Luckily NEON has an
            // instruction for that! The lowest bit of `c_hi` and `qp_hi` is the same,
            // so the division is exact.
            vhsubq_s32(c_hi, qp_hi)
        }
    }

    #[inline]
    #[must_use]
    fn get_reduced_d(c_hi: int32x4_t, qp_hi: int32x4_t) -> uint32x4_t {
        // We want this to compile to:
        //      shsub    res.4s, c_hi.4s, qp_hi.4s
        //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
        //      mls      res.4s, underflow.4s, P.4s
        // throughput: .75 cyc/vec (5.33 els/cyc)
        // latency: 5 cyc

        unsafe {
            let d = vreinterpretq_u32_s32(get_d(c_hi, qp_hi));

            // Finally, we reduce D to canonical form. D is negative iff `c_hi > qp_hi`, so if
            // that's the case then we add P. Note that if `c_hi > qp_hi` then
            // `underflow` is -1, so we must _subtract_ `underflow` * P.
            let underflow = vcltq_s32(c_hi, qp_hi);
            vmlsq_u32(d, confuse_compiler(underflow), PACKED_MOD)
        }
    }

    #[inline]
    #[must_use]
    pub(super) fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
        // We want this to compile to:
        //      sqdmulh  c_hi.4s, lhs.4s, rhs.4s
        //      mul      mu_rhs.4s, rhs.4s, MU.4s
        //      mul      q.4s, lhs.4s, mu_rhs.4s
        //      sqdmulh  qp_hi.4s, q.4s, P.4s
        //      shsub    res.4s, c_hi.4s, qp_hi.4s
        //      cmgt     underflow.4s, qp_hi.4s, c_hi.4s
        //      mls      res.4s, underflow.4s, P.4s
        // throughput: 1.75 cyc/vec (2.29 els/cyc)
        // latency: (lhs->) 11 cyc, (rhs->) 14 cyc

        unsafe {
            // No-op. The inputs are non-negative so we're free to interpret them as signed numbers.
            let lhs = vreinterpretq_s32_u32(lhs);
            let rhs = vreinterpretq_s32_u32(rhs);

            let mu_rhs = mulby_mu(rhs);
            let c_hi = get_c_hi(lhs, rhs);
            let qp_hi = get_qp_hi(lhs, mu_rhs);
            get_reduced_d(c_hi, qp_hi)
        }
    }
}
