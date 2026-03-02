use std::{
    io::{Read, Write},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use arith::{field_common, FFTField, Field, SimdField};
use ethnum::U256;
use rand::RngCore;
use serdes::{ExpSerde, SerdeResult};

use crate::goldilocks::p2_instructions::{assume, branch_hint, reduce128, try_inverse_u64};

// Goldilocks field modulus: 2^64 - 2^32 + 1
pub const GOLDILOCKS_MOD: u64 = 0xFFFFFFFF00000001;
/// 2^32 - 1
pub const EPSILON: u64 = 0xffffffff;

#[inline(always)]
pub(crate) fn mod_reduce_u64(x: u64) -> u64 {
    if x >= GOLDILOCKS_MOD {
        x - GOLDILOCKS_MOD
    } else {
        x
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Goldilocks {
    pub v: u64,
}

impl PartialEq for Goldilocks {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        mod_reduce_u64(self.v) == mod_reduce_u64(other.v)
    }
}

impl Eq for Goldilocks {}

impl PartialOrd for Goldilocks {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Goldilocks {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        mod_reduce_u64(self.v).cmp(&mod_reduce_u64(other.v))
    }
}

field_common!(Goldilocks);

impl ExpSerde for Goldilocks {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        // normalize the element: both 0 and Modulus are valid internal representations
        let v = mod_reduce_u64(self.v);
        writer.write_all(&v.to_le_bytes())?;
        Ok(())
    }

    #[inline(always)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 8];
        reader.read_exact(&mut u)?;
        let mut v = u64::from_le_bytes(u);
        v = mod_reduce_u64(v);
        Ok(Goldilocks { v })
    }
}

impl Field for Goldilocks {
    const NAME: &'static str = "Goldilocks";

    const SIZE: usize = 64 / 8;

    const ZERO: Self = Goldilocks { v: 0 };

    const ONE: Self = Goldilocks { v: 1 };

    const INV_2: Self = Goldilocks {
        v: 0x7FFFFFFF80000001,
    }; // (2^63 - 2^31 + 1)

    const FIELD_SIZE: usize = 64;

    const MODULUS: U256 = U256([GOLDILOCKS_MOD as u128, 0]);

    #[inline(always)]
    fn zero() -> Self {
        Goldilocks { v: 0 }
    }

    #[inline(always)]
    fn one() -> Self {
        Goldilocks { v: 1 }
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.v == 0 || self.v == GOLDILOCKS_MOD
    }

    #[inline(always)]
    fn random_unsafe(mut rng: impl RngCore) -> Self {
        rng.next_u64().into()
    }

    #[inline(always)]
    fn random_bool(mut rng: impl RngCore) -> Self {
        (rng.next_u64() & 1).into()
    }

    #[inline(always)]
    fn to_u256(&self) -> U256 {
        U256([mod_reduce_u64(self.v) as u128, 0])
    }

    #[inline(always)]
    fn from_u256(value: U256) -> Self {
        let value = value % Self::MODULUS;
        // TODO: this is a hack to get the low 64 bits of the u256
        // TODO: we should remove the assumption that the top bits are 0s
        let (_high, low) = value.into_words();
        let mut v = low as u64;
        v = mod_reduce_u64(v);
        Goldilocks { v }
    }

    #[inline(always)]
    fn inv(&self) -> Option<Self> {
        self.try_inverse()
    }

    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        assert!(self.v <= u32::MAX as u64);
        self.v as u32
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        let mut v = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        v = mod_reduce_u64(v);
        Goldilocks { v }
    }
}

impl Neg for Goldilocks {
    type Output = Goldilocks;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        Goldilocks::ZERO - self
    }
}

impl From<u32> for Goldilocks {
    #[inline(always)]
    fn from(x: u32) -> Self {
        Goldilocks { v: x as u64 }
    }
}

impl From<u64> for Goldilocks {
    #[inline(always)]
    fn from(x: u64) -> Self {
        Goldilocks {
            v: mod_reduce_u64(x),
        }
    }
}

impl Goldilocks {
    #[inline(always)]
    fn try_inverse(&self) -> Option<Self> {
        try_inverse_u64(&self.v).map(|v| Goldilocks { v })
    }

    #[inline(always)]
    pub fn mul_by_7(&self) -> Self {
        *self * Self { v: 7 }
    }
}

#[inline(always)]
/// credit: plonky2
fn add_internal(a: &Goldilocks, b: &Goldilocks) -> Goldilocks {
    let (sum, over) = a.v.overflowing_add(b.v);
    let (mut sum, over) = sum.overflowing_add((over as u64) * EPSILON);
    if over {
        // NB: self.0 > Self::ORDER && rhs.0 > Self::ORDER is necessary but not sufficient for
        // double-overflow.
        // This assume does two things:
        //  1. If compiler knows that either self.0 or rhs.0 <= ORDER, then it can skip this check.
        //  2. Hints to the compiler how rare this double-overflow is (thus handled better with a
        //     branch).
        assume(a.v > GOLDILOCKS_MOD && b.v > GOLDILOCKS_MOD);
        branch_hint();
        sum += EPSILON; // Cannot overflow.
    }
    Goldilocks { v: sum }
}

#[inline(always)]
fn sub_internal(a: &Goldilocks, b: &Goldilocks) -> Goldilocks {
    let (diff, under) = a.v.overflowing_sub(b.v);
    let (mut diff, under) = diff.overflowing_sub((under as u64) * EPSILON);
    if under {
        // NB: self.0 < EPSILON - 1 && rhs.0 > Self::ORDER is necessary but not sufficient for
        // double-underflow.
        // This assume does two things:
        //  1. If compiler knows that either self.0 >= EPSILON - 1 or rhs.0 <= ORDER, then it can
        //     skip this check.
        //  2. Hints to the compiler how rare this double-underflow is (thus handled better with a
        //     branch).
        assume(a.v < EPSILON - 1 && b.v > GOLDILOCKS_MOD);
        branch_hint();
        diff -= EPSILON; // Cannot underflow.
    }
    Goldilocks { v: diff }
}

#[inline(always)]
fn mul_internal(a: &Goldilocks, b: &Goldilocks) -> Goldilocks {
    reduce128((a.v as u128) * (b.v as u128))
}

impl std::hash::Hash for Goldilocks {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(mod_reduce_u64(self.v));
    }
}

impl SimdField for Goldilocks {
    type Scalar = Self;

    const PACK_SIZE: usize = 1;

    #[inline(always)]
    fn scale(&self, challenge: &Self::Scalar) -> Self {
        *self * challenge
    }

    #[inline(always)]
    fn pack_full(base: &Self::Scalar) -> Self {
        *base
    }

    #[inline(always)]
    fn pack(base_vec: &[Self::Scalar]) -> Self {
        assert_eq!(base_vec.len(), 1);
        base_vec[0]
    }

    #[inline(always)]
    fn unpack(&self) -> Vec<Self::Scalar> {
        vec![*self]
    }
}

impl FFTField for Goldilocks {
    const TWO_ADICITY: usize = 32; // 2^32 divides p-1

    /// The `2^s` root of unity.
    ///
    /// It can be calculated by exponentiating `Self::MULTIPLICATIVE_GENERATOR` by `t`,
    /// where `t = (modulus - 1) >> Self::S`.
    #[inline(always)]
    fn root_of_unity() -> Self {
        Goldilocks {
            v: 0x185629dcda58878c,
        } // 5 is a primitive root of order 2^32
    }
}

pub(crate) mod p2_instructions {
    //! Credit: the majority of the code is borrowed or inspired from Plonky2 with modifications.

    use std::arch::asm;
    use std::hint::unreachable_unchecked;

    use arith::Field;

    use crate::{Goldilocks, EPSILON, GOLDILOCKS_MOD};

    /// Reduces to a 64-bit value. The result might not be in canonical form; it could be in between
    /// the field order and `2^64`.
    #[inline]
    pub(crate) fn reduce128(x: u128) -> Goldilocks {
        let (x_lo, x_hi) = split(x); // This is a no-op
        let x_hi_hi = x_hi >> 32;
        let x_hi_lo = x_hi & EPSILON;

        let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
        if borrow {
            branch_hint(); // A borrow is exceedingly rare. It is faster to branch.
            t0 -= EPSILON; // Cannot underflow.
        }
        let t1 = x_hi_lo * EPSILON;
        let t2 = unsafe { add_no_canonicalize_trashing_input(t0, t1) };
        Goldilocks { v: t2 }
    }

    #[inline(always)]
    pub(super) fn assume(p: bool) {
        debug_assert!(p);
        if !p {
            unsafe {
                unreachable_unchecked();
            }
        }
    }

    /// Try to force Rust to emit a branch. Example:
    ///     if x > 2 {
    ///         y = foo();
    ///         branch_hint();
    ///     } else {
    ///         y = bar();
    ///     }
    /// This function has no semantics. It is a hint only.
    #[inline(always)]
    pub(super) fn branch_hint() {
        unsafe {
            asm!("", options(nomem, nostack, preserves_flags));
        }
    }

    /// Fast addition modulo ORDER for x86-64.
    /// This function is marked unsafe for the following reasons:
    ///   - It is only correct if x + y < 2**64 + ORDER = 0x1ffffffff00000001.
    ///   - It is only faster in some circumstances. In particular, on x86 it overwrites both inputs
    ///     in the registers, so its use is not recommended when either input will be used again.
    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
        let res_wrapped: u64;
        let adjustment: u64;
        asm!(
            "add {0}, {1}",
            // Trick. The carry flag is set iff the addition overflowed.
            // sbb x, y does x := x - y - CF. In our case, x and y are both {1:e}, so it simply does
            // {1:e} := 0xffffffff on overflow and {1:e} := 0 otherwise. {1:e} is the low 32 bits of
            // {1}; the high 32-bits are zeroed on write. In the end, we end up with 0xffffffff in {1}
            // on overflow; this happens be EPSILON.
            // Note that the CPU does not realize that the result of sbb x, x does not actually depend
            // on x. We must write the result to a register that we know to be ready. We have a
            // dependency on {1} anyway, so let's use it.
            "sbb {1:e}, {1:e}",
            inlateout(reg) x => res_wrapped,
            inlateout(reg) y => adjustment,
            options(pure, nomem, nostack),
        );
        assume(x != 0 || (res_wrapped == y && adjustment == 0));
        assume(y != 0 || (res_wrapped == x && adjustment == 0));
        // Add EPSILON == subtract ORDER.
        // Cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
        res_wrapped + adjustment
    }

    #[inline(always)]
    #[cfg(not(target_arch = "x86_64"))]
    pub(crate) unsafe fn add_no_canonicalize_trashing_input(x: u64, y: u64) -> u64 {
        use crate::EPSILON;

        let (res_wrapped, carry) = x.overflowing_add(y);
        // Below cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
        res_wrapped + EPSILON * (carry as u64)
    }

    #[inline]
    pub(crate) fn split(x: u128) -> (u64, u64) {
        (x as u64, (x >> 64) as u64)
    }

    /// Try to invert an element in a prime field.
    ///
    /// The algorithm below is the "plus-minus-inversion" method
    /// with an "almost Montgomery inverse" flair. See Handbook of
    /// Elliptic and Hyperelliptic Cryptography, Algorithms 11.6
    /// and 11.12.
    #[allow(clippy::many_single_char_names)]
    pub(super) fn try_inverse_u64(x: &u64) -> Option<u64> {
        let mut f = *x;
        let mut g = GOLDILOCKS_MOD;
        // NB: These two are very rarely such that their absolute
        // value exceeds (p-1)/2; we are paying the price of i128 for
        // the whole calculation, just for the times they do
        // though. Measurements suggest a further 10% time saving if c
        // and d could be replaced with i64's.
        let mut c = 1i128;
        let mut d = 0i128;

        if f == 0 || f == GOLDILOCKS_MOD {
            return None;
        }

        // f and g must always be odd.
        let mut k = f.trailing_zeros();
        f >>= k;
        if f == 1 {
            return Some(inverse_2exp(k as usize));
        }

        // The first two iterations are unrolled. This is to handle
        // the case where f and g are both large and f+g can
        // overflow. log2(max{f,g}) goes down by at least one each
        // iteration though, so after two iterations we can be sure
        // that f+g won't overflow.

        // Iteration 1:
        safe_iteration(&mut f, &mut g, &mut c, &mut d, &mut k);

        if f == 1 {
            // c must be -1 or 1 here.
            if c == -1 {
                return Some(GOLDILOCKS_MOD - inverse_2exp(k as usize));
            }
            debug_assert!(c == 1, "bug in try_inverse_u64");
            return Some(inverse_2exp(k as usize));
        }

        // Iteration 2:
        safe_iteration(&mut f, &mut g, &mut c, &mut d, &mut k);

        // Remaining iterations:
        while f != 1 {
            unsafe {
                unsafe_iteration(&mut f, &mut g, &mut c, &mut d, &mut k);
            }
        }

        // The following two loops adjust c so it's in the canonical range
        // [0, F::ORDER).

        // The maximum number of iterations observed here is 2; should
        // prove this.
        while c < 0 {
            c += GOLDILOCKS_MOD as i128;
        }

        // The maximum number of iterations observed here is 1; should
        // prove this.
        while c >= GOLDILOCKS_MOD as i128 {
            c -= GOLDILOCKS_MOD as i128;
        }

        // Precomputing the binary inverses rather than using inverse_2exp
        // saves ~5ns on my machine.
        let res = Goldilocks { v: c as u64 }
            * Goldilocks {
                v: inverse_2exp(k as usize),
            };
        debug_assert!(
            Goldilocks { v: *x } * res == Goldilocks::ONE,
            "bug in try_inverse_u64"
        );
        Some(res.v)
    }

    /// This is a 'safe' iteration for the modular inversion algorithm. It
    /// is safe in the sense that it will produce the right answer even
    /// when f + g >= 2^64.
    #[inline(always)]
    fn safe_iteration(f: &mut u64, g: &mut u64, c: &mut i128, d: &mut i128, k: &mut u32) {
        if f < g {
            std::mem::swap(f, g);
            std::mem::swap(c, d);
        }
        if *f & 3 == *g & 3 {
            // f - g = 0 (mod 4)
            *f -= *g;
            *c -= *d;

            // kk >= 2 because f is now 0 (mod 4).
            let kk = f.trailing_zeros();
            *f >>= kk;
            *d <<= kk;
            *k += kk;
        } else {
            // f + g = 0 (mod 4)
            *f = (*f >> 2) + (*g >> 2) + 1u64;
            *c += *d;
            let kk = f.trailing_zeros();
            *f >>= kk;
            *d <<= kk + 2;
            *k += kk + 2;
        }
    }

    /// This is an 'unsafe' iteration for the modular inversion
    /// algorithm. It is unsafe in the sense that it might produce the
    /// wrong answer if f + g >= 2^64.
    #[inline(always)]
    unsafe fn unsafe_iteration(f: &mut u64, g: &mut u64, c: &mut i128, d: &mut i128, k: &mut u32) {
        if *f < *g {
            std::mem::swap(f, g);
            std::mem::swap(c, d);
        }
        if *f & 3 == *g & 3 {
            // f - g = 0 (mod 4)
            *f -= *g;
            *c -= *d;
        } else {
            // f + g = 0 (mod 4)
            *f += *g;
            *c += *d;
        }

        // kk >= 2 because f is now 0 (mod 4).
        let kk = f.trailing_zeros();
        *f >>= kk;
        *d <<= kk;
        *k += kk;
    }

    /// Compute the inverse of 2^exp in this field.
    #[inline]
    fn inverse_2exp(exp: usize) -> u64 {
        // Let p = char(F). Since 2^exp is in the prime subfield, i.e. an
        // element of GF_p, its inverse must be as well. Thus we may add
        // multiples of p without changing the result. In particular,
        // 2^-exp = 2^-exp - p 2^-exp
        //        = 2^-exp (1 - p)
        //        = p - (p - 1) / 2^exp

        // If this field's two adicity, t, is at least exp, then 2^exp divides
        // p - 1, so this division can be done with a simple bit shift. If
        // exp > t, we repeatedly multiply by 2^-t and reduce exp until it's in
        // the right range.

        // NB: The only reason this is split into two cases is to save
        // the multiplication (and possible calculation of
        // inverse_2_pow_adicity) in the usual case that exp <=
        // TWO_ADICITY. Can remove the branch and simplify if that
        // saving isn't worth it.
        let res = if exp > 32 {
            // NB: This should be a compile-time constant
            // MODULUS - ((MODULUS - 1) >> 32)
            let inverse_2_pow_adicity = Goldilocks {
                v: 0xfffffffe00000002,
            };

            let mut res = inverse_2_pow_adicity;
            let mut e = exp - 32;

            while e > 32 {
                res *= inverse_2_pow_adicity;
                e -= 32;
            }
            res * Goldilocks {
                v: GOLDILOCKS_MOD - ((GOLDILOCKS_MOD - 1) >> e),
            }
        } else {
            Goldilocks {
                v: GOLDILOCKS_MOD - ((GOLDILOCKS_MOD - 1) >> exp),
            }
        };
        res.v
    }

    // pub(crate) fn sqrt_tonelli_shanks(f: &Goldilocks, tm1d2: u64) -> Option<Goldilocks> {
    //     // w = self^((t - 1) // 2)
    //     let w = f.exp(tm1d2 as u128);

    //     let mut v = Goldilocks::S;
    //     let mut x = w * f;
    //     let mut b = x * w;

    //     // Initialize z as the 2^S root of unity.
    //     let mut z = Goldilocks::ROOT_OF_UNITY;

    //     for max_v in (1..=Goldilocks::S).rev() {
    //         let mut k = 1;
    //         let mut tmp = b.square();
    //         let mut j_less_than_v: Choice = 1.into();

    //         for j in 2..max_v {
    //             let tmp_is_one = tmp.ct_eq(&Goldilocks::ONE);
    //             let squared = Goldilocks::conditional_select(&tmp, &z, tmp_is_one).square();
    //             tmp = Goldilocks::conditional_select(&squared, &tmp, tmp_is_one);
    //             let new_z = Goldilocks::conditional_select(&z, &squared, tmp_is_one);
    //             j_less_than_v &= !j.ct_eq(&v);
    //             k = u32::conditional_select(&j, &k, tmp_is_one);
    //             z = Goldilocks::conditional_select(&z, &new_z, j_less_than_v);
    //         }

    //         let result = x * z;
    //         x = Goldilocks::conditional_select(&result, &x, b.ct_eq(&Goldilocks::ONE));
    //         z = z.square();
    //         b *= z;
    //         v = k;
    //     }
    //     CtOption::new(
    //         x,
    //         (x * x).ct_eq(f), // Only return Some if it's the square root.
    //     )
    // }
}
