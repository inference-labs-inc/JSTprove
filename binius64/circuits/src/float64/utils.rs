// Copyright 2025 Irreducible Inc.
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

/// Simple view of a decoded binary64 payload.
#[derive(Clone, Copy)]
pub struct Fp64Parts {
    pub sign: Wire, // Sign as MSB-bool (bit63 set for negative, 0 otherwise)
    pub exp: Wire,  // unbiased field bits (0..=0x7FF)
    pub frac: Wire, // 52-bit payload

    // Common classifiers as MSB-bools (true = MSB=1, false = 0):
    pub is_nan: Wire,  // exp==0x7FF && frac!=0
    pub is_inf: Wire,  // exp==0x7FF && frac==0
    pub is_zero: Wire, // exp==0 && frac==0
    pub is_sub: Wire,  // subnormal: exp==0 && frac!=0
    pub is_norm: Wire, // normal: exp!=0 && exp!=0x7FF
}

/// Creates a wire containing the constant value 0.
pub fn zero(b: &CircuitBuilder) -> Wire {
    b.add_constant_64(0)
}

/// Creates a wire containing the constant value 1.
pub fn one(b: &CircuitBuilder) -> Wire {
    b.add_constant_64(1)
}

/// Extracts bit `i` from `x` as an MSB-bool (true = bit63 set, false = 0).
///
/// Returns `((x >> i) & 1) << 63`.
#[inline]
pub fn bit_msb01(b: &CircuitBuilder, x: Wire, i: u32) -> Wire {
    b.shl(bit_lsb(b, x, i), 63)
}

/// Moves MSB to LSB position zeroing all other bits.
#[inline]
pub fn msb_to_lsb01(b: &CircuitBuilder, b_msb: Wire) -> Wire {
    b.shr(b_msb, 63)
}

/// Performs integer addition: `a + b`.
///
/// This is a wrapper around the circuit builder's integer addition that handles
/// carry-in/carry-out automatically with zero carry-in.
pub fn iadd(builder: &CircuitBuilder, a: Wire, b: Wire) -> Wire {
    let (s, _c) = builder.iadd_cin_cout(a, b, zero(builder));
    s
}

/// Performs integer subtraction: `a - b`.
///
/// This is a wrapper around the circuit builder's integer subtraction that handles
/// borrow-in/borrow-out automatically with zero borrow-in.
pub fn isub(builder: &CircuitBuilder, a: Wire, b: Wire) -> Wire {
    let (d, _b) = builder.isub_bin_bout(a, b, zero(builder));
    d
}

/// Extracts bit `i` from `x` as a 0/1 value.
///
/// Returns `(x >> i) & 1`.
pub fn bit_lsb(b: &CircuitBuilder, x: Wire, i: u32) -> Wire {
    b.band(b.shr(x, i), one(b))
}

/// Performs variable right shift with sticky bit tracking.
///
/// This implements a barrel shifter that can shift by any amount 0-63,
/// with optional saturation. All bits shifted out are OR'd together to
/// create a "sticky" bit that tracks whether any precision was lost.
///
/// # Parameters
/// - `x`: Value to shift
/// - `d`: Shift amount (0-63, or optionally saturated at 63)
/// - `saturate_at_63`: If true, shifts >= 64 are treated as 63; if false, they wrap
///
/// # Returns
/// - First wire: The shifted value
/// - Second wire: Sticky bit (all-1 mask if any bits were lost, all-0 otherwise)
pub fn var_shr_with_sticky(
    b: &CircuitBuilder,
    x: Wire,
    d: Wire,
    saturate_at_63: bool,
) -> (Wire, Wire) {
    let c63 = b.add_constant_64(63);
    let c64 = b.add_constant_64(64);
    let d_eff = if saturate_at_63 {
        let lt64 = b.icmp_ult(d, c64);
        b.select(lt64, d, c63)
    } else {
        b.band(d, c63)
    };

    let mut v = x;
    let mut sticky01 = zero(b); // 0/1 sticky

    // Stage 32 (bit 5)
    {
        let cond01 = bit_lsb(b, d_eff, 5);
        let cond = bit_msb01(b, d_eff, 5);
        let lost = b.band(v, b.add_constant_64((1u64 << 32) - 1));
        let lost_ne0 = b.icmp_ne(lost, zero(b));
        let lost_nz01 = msb_to_lsb01(b, lost_ne0);
        sticky01 = b.bor(sticky01, b.band(lost_nz01, cond01));
        let shifted = b.shr(v, 32);
        v = b.select(cond, shifted, v);
    }
    // Stage 16 (bit 4)
    {
        let cond01 = bit_lsb(b, d_eff, 4);
        let cond = bit_msb01(b, d_eff, 4);
        let lost = b.band(v, b.add_constant_64((1u64 << 16) - 1));
        let lost_ne0 = b.icmp_ne(lost, zero(b));
        let lost_nz01 = msb_to_lsb01(b, lost_ne0);
        sticky01 = b.bor(sticky01, b.band(lost_nz01, cond01));
        let shifted = b.shr(v, 16);
        v = b.select(cond, shifted, v);
    }
    // Stage 8 (bit 3)
    {
        let cond01 = bit_lsb(b, d_eff, 3);
        let cond = bit_msb01(b, d_eff, 3);
        let lost = b.band(v, b.add_constant_64((1u64 << 8) - 1));
        let lost_ne0 = b.icmp_ne(lost, zero(b));
        let lost_nz01 = msb_to_lsb01(b, lost_ne0);
        sticky01 = b.bor(sticky01, b.band(lost_nz01, cond01));
        let shifted = b.shr(v, 8);
        v = b.select(cond, shifted, v);
    }
    // Stage 4 (bit 2)
    {
        let cond01 = bit_lsb(b, d_eff, 2);
        let cond = bit_msb01(b, d_eff, 2);
        let lost = b.band(v, b.add_constant_64((1u64 << 4) - 1));
        let lost_ne0 = b.icmp_ne(lost, zero(b));
        let lost_nz01 = msb_to_lsb01(b, lost_ne0);
        sticky01 = b.bor(sticky01, b.band(lost_nz01, cond01));
        let shifted = b.shr(v, 4);
        v = b.select(cond, shifted, v);
    }
    // Stage 2 (bit 1)
    {
        let cond01 = bit_lsb(b, d_eff, 1);
        let cond = bit_msb01(b, d_eff, 1);
        let lost = b.band(v, b.add_constant_64((1u64 << 2) - 1));
        let lost_ne0 = b.icmp_ne(lost, zero(b));
        let lost_nz01 = msb_to_lsb01(b, lost_ne0);
        sticky01 = b.bor(sticky01, b.band(lost_nz01, cond01));
        let shifted = b.shr(v, 2);
        v = b.select(cond, shifted, v);
    }
    // Stage 1 (bit 0)
    {
        let cond01 = bit_lsb(b, d_eff, 0);
        let cond = bit_msb01(b, d_eff, 0);
        let lost = b.band(v, b.add_constant_64(1));
        let lost_ne0 = b.icmp_ne(lost, zero(b));
        let lost_nz01 = msb_to_lsb01(b, lost_ne0);
        sticky01 = b.bor(sticky01, b.band(lost_nz01, cond01));
        let shifted = b.shr(v, 1);
        v = b.select(cond, shifted, v);
    }

    (v, sticky01)
}

/// Performs variable left shift.
///
/// This implements a barrel shifter that can shift left by any amount 0-63.
/// Shift amounts >= 64 are masked to 0-63 range (i.e., `d & 63`).
///
/// # Parameters
/// - `x`: Value to shift
/// - `d`: Shift amount (effectively `d & 63`)
///
/// # Returns
/// The left-shifted value `x << (d & 63)`
pub fn var_shl(b: &CircuitBuilder, x: Wire, d: Wire) -> Wire {
    let d_eff = b.band(d, b.add_constant_64(63));
    let mut v = x;

    {
        let cond = bit_msb01(b, d_eff, 5); // 32
        let shifted = b.shl(v, 32);
        v = b.select(cond, shifted, v);
    }
    {
        let cond = bit_msb01(b, d_eff, 4); // 16
        let shifted = b.shl(v, 16);
        v = b.select(cond, shifted, v);
    }
    {
        let cond = bit_msb01(b, d_eff, 3); // 8
        let shifted = b.shl(v, 8);
        v = b.select(cond, shifted, v);
    }
    {
        let cond = bit_msb01(b, d_eff, 2); // 4
        let shifted = b.shl(v, 4);
        v = b.select(cond, shifted, v);
    }
    {
        let cond = bit_msb01(b, d_eff, 1); // 2
        let shifted = b.shl(v, 2);
        v = b.select(cond, shifted, v);
    }
    {
        let cond = bit_msb01(b, d_eff, 0); // 1
        let shifted = b.shl(v, 1);
        v = b.select(cond, shifted, v);
    }

    v
}

/// Count leading zeroes in `x`
pub fn clz64(b: &CircuitBuilder, x: Wire) -> Wire {
    let mut n = zero(b);
    let mut y = x;

    // step(32)
    {
        let t = b.shr(y, 32);
        let z = b.icmp_eq(t, zero(b));
        let add32 = b.add_constant_64(32);
        n = iadd(b, n, b.select(z, add32, zero(b)));
        y = b.select(z, b.shl(y, 32), y);
    }
    // step(16)
    {
        let t = b.shr(y, 48);
        let z = b.icmp_eq(t, zero(b));
        let add16 = b.add_constant_64(16);
        n = iadd(b, n, b.select(z, add16, zero(b)));
        y = b.select(z, b.shl(y, 16), y);
    }
    // step(8)
    {
        let t = b.shr(y, 56);
        let z = b.icmp_eq(t, zero(b));
        let add8 = b.add_constant_64(8);
        n = iadd(b, n, b.select(z, add8, zero(b)));
        y = b.select(z, b.shl(y, 8), y);
    }
    // step(4)
    {
        let t = b.shr(y, 60);
        let z = b.icmp_eq(t, zero(b));
        let add4 = b.add_constant_64(4);
        n = iadd(b, n, b.select(z, add4, zero(b)));
        y = b.select(z, b.shl(y, 4), y);
    }
    // step(2)
    {
        let t = b.shr(y, 62);
        let z = b.icmp_eq(t, zero(b));
        let add2 = b.add_constant_64(2);
        n = iadd(b, n, b.select(z, add2, zero(b)));
        y = b.select(z, b.shl(y, 2), y);
    }
    // step(1)
    {
        let t = b.shr(y, 63);
        let z = b.icmp_eq(t, zero(b));
        n = iadd(b, n, b.select(z, one(b), zero(b)));
    }
    n
}

/// Build the 53-bit integer significand and effective exponent.
///
/// For multiplication, we use 53-bit integers (including hidden bit for normals)
/// rather than the 64-bit extended format used for addition.
///
/// - Normals: `sig53 = (1<<52) | frac`, `exp_eff = exp`
/// - Subnormals: `sig53 = frac`, `exp_eff = 1`
///
/// # Parameters
/// - `p`: Parts from `fp64_unpack`
///
/// # Returns
/// - `(sig53, exp_eff)`: 53-bit significand and effective exponent
pub fn fp64_sig53_and_exp(b: &CircuitBuilder, p: &Fp64Parts) -> (Wire, Wire) {
    let one52 = b.add_constant_64(1u64 << 52);
    let sig_norm = b.bor(one52, p.frac);
    let sig = b.select(p.is_norm, sig_norm, p.frac);
    let exp_eff = b.select(p.is_norm, p.exp, one(b)); // subnormals use exp=1
    (sig, exp_eff)
}

/// Right-shift a 128-bit value by a small constant and return the low 64 bits.
///
/// Given `p = (hi << 64) | lo`, computes `(p >> s).lo64 = (lo >> s) | (hi << (64 - s))`
///
/// # Parameters
/// - `hi`: High 64 bits of 128-bit value
/// - `lo`: Low 64 bits of 128-bit value
/// - `s`: Shift amount (must be 0 < s < 64)
///
/// # Returns
/// Low 64 bits of the right-shifted result
#[inline]
pub fn shr128_to_u64_const(b: &CircuitBuilder, hi: Wire, lo: Wire, s: u32) -> Wire {
    debug_assert!(s > 0 && s < 64);
    let lo_part = b.shr(lo, s);
    let hi_part = b.shl(hi, 64 - s);
    b.bor(lo_part, hi_part)
}

/// Extract sticky bit from the k least-significant bits of a value.
///
/// Returns 0/1 (in LSB) indicating whether any of the k least-significant
/// bits of `lo` are set to 1.
///
/// # Parameters
/// - `lo`: Input value
/// - `k`: Number of LSBs to check (must be k < 64)
///
/// # Returns
/// 0/1 value: 1 if any of k LSBs are set, 0 otherwise
#[inline]
pub fn sticky_from_low_k(b: &CircuitBuilder, lo: Wire, k: u32) -> Wire {
    debug_assert!(k < 64);
    let mask = b.add_constant_64((1u64 << k) - 1);
    let ne0 = b.icmp_ne(b.band(lo, mask), zero(b));
    msb_to_lsb01(b, ne0)
}

/// Unpack and classify a binary64 word.
///
/// Input:
/// - `x`: 64-bit IEEE-754 encoding
///
/// Output:
/// - `Fp64Parts` with fields:
///   - `sign`: MSB-bool of sign bit (i.e., `x & (1<<63)`, either 0 or 0x8000..)
///   - `exp = (x >> 52) & 0x7FF`
///   - `frac = x & ((1<<52)-1)`
///   - `is_nan`: exp==0x7FF && frac!=0
///   - `is_inf`: exp==0x7FF && frac==0
///   - `is_zero`: exp==0 && frac==0
///   - `is_sub`: exp==0 && frac!=0
///   - `is_norm`: exp!=0 && exp!=0x7FF
///
/// All booleans are in MSB-bool format.
pub fn fp64_unpack(b: &CircuitBuilder, x: Wire) -> Fp64Parts {
    let exp_m = b.add_constant_64(0x7FF);
    let frac_m = b.add_constant_64((1u64 << 52) - 1);

    let sign = x;
    let exp = b.band(b.shr(x, 52), exp_m);
    let frac = b.band(x, frac_m);

    let exp_is_max = b.icmp_eq(exp, exp_m); // MSB-bool
    let exp_is_zero = b.icmp_eq(exp, zero(b)); // MSB-bool
    let frac_is_zero = b.icmp_eq(frac, zero(b)); // MSB-bool

    let is_nan = b.band(exp_is_max, b.bnot(frac_is_zero));
    let is_inf = b.band(exp_is_max, frac_is_zero);
    let is_zero = b.band(exp_is_zero, frac_is_zero);
    let is_sub = b.band(exp_is_zero, b.bnot(frac_is_zero));
    let is_norm = b.bnot(b.bor(exp_is_max, exp_is_zero));

    Fp64Parts {
        sign,
        exp,
        frac,
        is_nan,
        is_inf,
        is_zero,
        is_sub,
        is_norm,
    }
}

/// Pre-round **underflow** handling: if `exp<=0`, right shift by `k=1-exp`
/// and fold sticky into bit0. Also prepare the exponent for rounding geometry (Exp=1).
///
/// Input: `(res_sig, res_exp)`
///
/// Output:
/// - `(sig_round_base, exp_round_base, exp_lt_1_mask)` where `exp_round_base = (exp<=0 ? 1 : exp)`.
pub fn fp64_underflow_shift(
    b: &CircuitBuilder,
    res_sig: Wire,
    res_exp: Wire,
) -> (Wire, Wire, Wire) {
    let c1 = one(b);
    let keep = b.bnot(c1);

    let exp_lt_1 = b.icmp_ult(res_exp, c1);
    let k = isub(b, c1, res_exp); // 1 - exp
    let k_use = b.select(exp_lt_1, k, zero(b));

    let (mut sig_u, st) = var_shr_with_sticky(b, res_sig, k_use, true);
    let bit0 = b.bor(b.band(sig_u, c1), b.band(st, c1));
    sig_u = b.bor(b.band(sig_u, keep), bit0);

    let sig_round_base = b.select(exp_lt_1, sig_u, res_sig);
    let exp_round_base = b.select(exp_lt_1, c1, res_exp);
    (sig_round_base, exp_round_base, exp_lt_1)
}

/// Round-to-nearest, ties-to-even (RN-even).
///
/// Geometry:
/// - Integer bit at 63; we interpret bits:
///   - LSB of target mantissa at bit 11
///   - Guard=10, Round=9, Sticky=bit 0 (already folded)
///
/// Input: `(sig_base, exp_base)`
///
/// Output:
/// - `(mant_final_53, exp_after_round, mant_overflow_mask)`
///   - `mant_final_53` is a 53-bit value (includes hidden 1 for normals)
///   - If mant overflowed to 54 bits, we shift right 1 and increment exponent.
pub fn fp64_round_rne(b: &CircuitBuilder, sig_base: Wire, exp_base: Wire) -> (Wire, Wire, Wire) {
    let lsb = bit_lsb(b, sig_base, 11);
    let g: Wire = bit_lsb(b, sig_base, 10);
    let r = bit_lsb(b, sig_base, 9);
    let s = b.band(sig_base, one(b));
    let r_or_s = b.bor(r, s);
    let tie_or_gt = b.bor(r_or_s, lsb);
    let round_up01 = b.band(g, tie_or_gt); // 0/1

    let mant_trunc = b.shr(sig_base, 11);
    let mant_rounded = iadd(b, mant_trunc, round_up01);

    let overflow01 = bit_lsb(b, mant_rounded, 53); // 0/1
    let overflow_msb = bit_msb01(b, mant_rounded, 53);
    let mant_final_53 = b.select(overflow_msb, b.shr(mant_rounded, 1), mant_rounded);
    let exp_after = iadd(b, exp_base, overflow01);

    (mant_final_53, exp_after, overflow_msb)
}

/// Pack finite (normal / subnormal) and apply overflow-to-Inf if needed.
///
/// Input:
/// - `sign` (0/1), `mant_final_53`, `exp_after_round`
/// - `stayed_sub_mask`: all-1 iff we are in subnormal regime **and** there was no mantissa overflow
///
/// Output:
/// - `finite_or_inf`: packed 64-bit result (finite or +/−Inf on overflow)
pub fn fp64_pack_finite_or_inf(
    b: &CircuitBuilder,
    sign: Wire,
    mant_final_53: Wire,
    exp_after_round: Wire,
    stayed_sub_mask: Wire,
) -> Wire {
    let frac_m = b.add_constant_64((1u64 << 52) - 1);
    let exp_2047 = b.add_constant_64(0x7FF);

    let frac = b.band(mant_final_53, frac_m);
    let sign = b.band(sign, b.add_constant(Word::MSB_ONE));
    let packed_sub = b.bor(sign, frac);

    let exp_sh = b.shl(exp_after_round, 52);
    let packed_norm = b.bor(b.bor(sign, exp_sh), frac);

    let finite_packed = b.select(stayed_sub_mask, packed_sub, packed_norm);

    // If mantissa is zero, the result is a signed zero regardless of exp_after_round.
    let mant_is_zero = b.icmp_eq(mant_final_53, zero(b));
    let finite_or_zero = b.select(mant_is_zero, sign, finite_packed);

    // overflow_to_inf when exp_after_round >= 2047
    let overflow_to_inf = b.icmp_ule(exp_2047, exp_after_round);
    let inf_payload = b.shl(exp_2047, 52);
    let packed_inf = b.bor(sign, inf_payload);

    b.select(overflow_to_inf, packed_inf, finite_or_zero)
}

#[cfg(test)]
pub mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use proptest::bool;

    use super::*;

    // Reference implementations for testing (shared between add.rs and utils.rs tests)
    pub struct Fp64UnpackResult {
        pub sign: u64,
        pub exp: u64,
        pub frac: u64,
        pub is_nan: u64,  // MSB-bool
        pub is_inf: u64,  // MSB-bool
        pub is_norm: u64, // MSB-bool
    }

    pub fn ref_fp64_unpack(x: u64) -> Fp64UnpackResult {
        let sign = x & (1u64 << 63);
        let exp = (x >> 52) & 0x7FF;
        let frac = x & ((1u64 << 52) - 1);

        let exp_is_max = if exp == 0x7FF { 1u64 << 63 } else { 0 };
        let exp_is_zero = if exp == 0 { 1u64 << 63 } else { 0 };
        let frac_is_zero = if frac == 0 { 1u64 << 63 } else { 0 };

        // MSB-bool logic
        let is_nan = if exp_is_max != 0 && frac_is_zero == 0 {
            1u64 << 63
        } else {
            0
        };
        let is_inf = if exp_is_max != 0 && frac_is_zero != 0 {
            1u64 << 63
        } else {
            0
        };
        let is_norm = if exp_is_max == 0 && exp_is_zero == 0 {
            1u64 << 63
        } else {
            0
        };

        Fp64UnpackResult {
            sign,
            exp,
            frac,
            is_nan,
            is_inf,
            is_norm,
        }
    }

    pub fn ref_fp64_underflow_shift(res_sig: u64, res_exp: u64) -> (u64, u64, u64) {
        let exp_lt_1 = if res_exp < 1 { 1u64 << 63 } else { 0 };

        if exp_lt_1 != 0 {
            let k = 1u64.wrapping_sub(res_exp);
            let k_use = std::cmp::min(k, 63);

            let sig_shifted = res_sig >> k_use;
            let lost_bits = res_sig & ((1u64 << k_use) - 1);
            let sticky = if lost_bits != 0 { 1 } else { 0 };
            let bit0 = (sig_shifted & 1) | sticky;
            let sig_u = (sig_shifted & !1) | bit0;

            (sig_u, 1, exp_lt_1)
        } else {
            (res_sig, res_exp, exp_lt_1)
        }
    }

    pub fn ref_fp64_round_rne(sig_base: u64, exp_base: u64) -> (u64, u64, u64) {
        let lsb = (sig_base >> 11) & 1;
        let g = (sig_base >> 10) & 1;
        let r = (sig_base >> 9) & 1;
        let s = sig_base & 1;

        let r_or_s = r | s;
        let tie_or_gt = r_or_s | lsb;
        let round_up = g & tie_or_gt;

        let mant_trunc = sig_base >> 11;
        let mant_rounded = mant_trunc.wrapping_add(round_up);

        let overflow = (mant_rounded >> 53) & 1;
        let overflow_msb = if overflow != 0 { 1u64 << 63 } else { 0 };
        let mant_final_53 = if overflow_msb != 0 {
            mant_rounded >> 1
        } else {
            mant_rounded
        };
        let exp_after = exp_base.wrapping_add(overflow);

        (mant_final_53, exp_after, overflow_msb)
    }

    pub fn ref_fp64_pack_finite_or_inf(
        sign_msb: u64,
        mant_final_53: u64,
        exp_after_round: u64,
        stayed_sub_mask_msb: u64,
    ) -> u64 {
        let sign_hi = sign_msb & (1u64 << 63);
        let frac = mant_final_53 & ((1u64 << 52) - 1);

        let packed_sub = sign_hi | frac;
        let exp_sh = exp_after_round << 52;
        let packed_norm = sign_hi | exp_sh | frac;

        let finite_packed = if stayed_sub_mask_msb != 0 {
            packed_sub
        } else {
            packed_norm
        };

        // If mantissa is zero, this is a signed zero regardless of exponent
        let finite_or_zero = if mant_final_53 == 0 {
            sign_hi
        } else {
            finite_packed
        };

        let overflow_to_inf = exp_after_round >= 0x7FF;
        if overflow_to_inf {
            let inf_payload = 0x7FFu64 << 52;
            sign_hi | inf_payload
        } else {
            finite_or_zero
        }
    }

    #[test]
    fn test_bit_lsb() {
        let builder = CircuitBuilder::new();
        let input = builder.add_inout();
        let output0 = bit_lsb(&builder, input, 0);
        let output1 = bit_lsb(&builder, input, 1);
        let output63 = bit_lsb(&builder, input, 63);
        let expected0 = builder.add_inout();
        let expected1 = builder.add_inout();
        let expected63 = builder.add_inout();
        builder.assert_eq("test_output0", output0, expected0);
        builder.assert_eq("test_output1", output1, expected1);
        builder.assert_eq("test_output63", output63, expected63);

        let circuit = builder.build();

        let test_cases = [
            (0b101, 1, 0, 0),              // bit 0=1, bit 1=0, bit 63=0
            (0b110, 0, 1, 0),              // bit 0=0, bit 1=1, bit 63=0
            (0x8000000000000001, 1, 0, 1), // bit 0=1, bit 1=0, bit 63=1
        ];

        for (val, exp0, exp1, exp63) in test_cases {
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val);
            w[expected0] = Word(exp0);
            w[expected1] = Word(exp1);
            w[expected63] = Word(exp63);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_clz64() {
        let builder = CircuitBuilder::new();
        let input = builder.add_inout();
        let output = clz64(&builder, input);
        let expected = builder.add_inout();
        builder.assert_eq("test_output", output, expected);

        let circuit = builder.build();

        let test_cases = [
            (0x8000000000000000u64, 0),  // Top bit set
            (0x4000000000000000u64, 1),  // Second bit set
            (0x0000000000000001u64, 63), // Only bottom bit set
            (0xFFFFFFFFFFFFFFFFu64, 0),  // All bits set
            (0x0000000000008000u64, 48), // Bit 15 set
        ];

        for (val, expected_clz) in test_cases {
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val);
            w[expected] = Word(expected_clz);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_var_shl() {
        let builder = CircuitBuilder::new();
        let input = builder.add_inout();
        let shift = builder.add_inout();
        let output = var_shl(&builder, input, shift);
        let expected = builder.add_inout();
        builder.assert_eq("test_output", output, expected);

        let circuit = builder.build();

        let test_cases = [
            (1, 0, 1),        // No shift
            (1, 1, 2),        // Shift left by 1
            (1, 8, 256),      // Shift left by 8
            (0xFF, 4, 0xFF0), // Shift 0xFF left by 4
            (1, 64, 1),       // Shift amount wraps (64 & 63 = 0)
        ];

        for (val, shift_amt, expected_result) in test_cases {
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val);
            w[shift] = Word(shift_amt);
            w[expected] = Word(expected_result);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_var_shr_with_sticky() {
        let builder = CircuitBuilder::new();
        let input = builder.add_inout();
        let shift = builder.add_inout();
        let (output, sticky) = var_shr_with_sticky(&builder, input, shift, false);
        let expected_out = builder.add_inout();
        let expected_sticky = builder.add_inout();
        builder.assert_eq("test_output", output, expected_out);
        builder.assert_eq("test_sticky", sticky, expected_sticky);

        let circuit = builder.build();

        let test_cases = [
            (8, 1, 4, 0),      // 8 >> 1 = 4, no bits lost
            (7, 1, 3, 1),      // 7 >> 1 = 3, bit lost (sticky = 1)
            (0xFF, 4, 0xF, 1), // 0xFF >> 4 = 0xF, bits lost
            (0xF0, 4, 0xF, 0), // 0xF0 >> 4 = 0xF, no bits lost
        ];

        for (val, shift_amt, expected_result, expected_sticky_val) in test_cases {
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val);
            w[shift] = Word(shift_amt);
            w[expected_out] = Word(expected_result);
            w[expected_sticky] = Word(expected_sticky_val);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_sig53_and_exp() {
        // fp64_unpack is now available through the parent module import

        let test_values = vec![1.0f64, -1.0f64, 2.0f64, f64::MIN_POSITIVE, 0.5f64];

        for val in test_values {
            let builder = CircuitBuilder::new();
            let input = builder.add_inout();
            let parts = fp64_unpack(&builder, input);
            let (sig, exp_eff) = fp64_sig53_and_exp(&builder, &parts);
            let expected_sig = builder.add_inout();
            let expected_exp = builder.add_inout();

            builder.assert_eq("sig", sig, expected_sig);
            builder.assert_eq("exp_eff", exp_eff, expected_exp);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val.to_bits());

            // Reference calculation
            let bits = val.to_bits();
            let exp = (bits >> 52) & 0x7FF;
            let frac = bits & ((1u64 << 52) - 1);
            let is_norm = exp != 0 && exp != 0x7FF;

            let (exp_sig, exp_exp) = if is_norm {
                ((1u64 << 52) | frac, exp)
            } else {
                (frac, 1)
            };

            w[expected_sig] = Word(exp_sig);
            w[expected_exp] = Word(exp_exp);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_shr128_to_u64_const() {
        let test_cases = [
            // (hi, lo, shift, expected)
            (0x0123456789ABCDEFu64, 0xFEDCBA9876543210u64, 8),
            (0xFFFFFFFFFFFFFFFFu64, 0x0000000000000000u64, 32),
            (0x0000000000000000u64, 0xFFFFFFFFFFFFFFFFu64, 16),
            (0x8000000000000000u64, 0x0000000000000001u64, 1),
            (0x1234567890ABCDEFu64, 0x1111111111111111u64, 4),
        ];

        for (hi, lo, shift) in test_cases {
            let builder = CircuitBuilder::new();
            let hi_wire = builder.add_inout();
            let lo_wire = builder.add_inout();
            let result = shr128_to_u64_const(&builder, hi_wire, lo_wire, shift);
            let expected_wire = builder.add_inout();
            builder.assert_eq("shr128_result", result, expected_wire);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[hi_wire] = Word(hi);
            w[lo_wire] = Word(lo);

            // Reference calculation: ((hi as u128) << 64 | lo as u128) >> shift
            let p = ((hi as u128) << 64) | (lo as u128);
            let expected = (p >> shift) as u64;
            w[expected_wire] = Word(expected);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_sticky_from_low_k() {
        let test_cases = [
            // (value, k, expected)
            (0x0000000000000000u64, 4, 0), // No bits set
            (0x0000000000000001u64, 1, 1), // Bit 0 set
            (0x0000000000000002u64, 1, 0), // Bit 0 not set, but bit 1 is
            (0x0000000000000002u64, 2, 1), // Bit 1 set in 2 LSBs
            (0x000000000000000Fu64, 4, 1), // All 4 LSBs set
            (0x0000000000000010u64, 4, 0), // Bit 4 set, but not in 4 LSBs
            (0x00000000000000FFu64, 8, 1), // All 8 LSBs set
            (0x0000000000000100u64, 8, 0), // Bit 8 set, but not in 8 LSBs
        ];

        for (value, k, expected_val) in test_cases {
            let builder = CircuitBuilder::new();
            let input = builder.add_inout();
            let result = sticky_from_low_k(&builder, input, k);
            let expected = builder.add_inout();
            builder.assert_eq("sticky_result", result, expected);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[input] = Word(value);
            w[expected] = Word(expected_val);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_unpack() {
        let test_values = vec![
            0.0f64,
            -0.0f64,
            1.0f64,
            -1.0f64,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE,
        ];

        for val in test_values {
            let builder = CircuitBuilder::new();
            let input = builder.add_inout();
            let result = fp64_unpack(&builder, input);

            // Create expected outputs
            let expected_sign = builder.add_inout();
            let expected_exp = builder.add_inout();
            let expected_frac = builder.add_inout();
            let expected_is_nan = builder.add_inout();
            let expected_is_inf = builder.add_inout();
            let expected_is_norm = builder.add_inout();

            // Compare only the MSB of sign; sign is represented as MSB-bool
            let bool_mask = builder.add_constant(Word::MSB_ONE);
            builder.assert_eq("sign", builder.band(result.sign, bool_mask), expected_sign);
            builder.assert_eq("exp", result.exp, expected_exp);
            builder.assert_eq("frac", result.frac, expected_frac);
            builder.assert_eq(
                "is_nan",
                builder.band(result.is_nan, bool_mask),
                expected_is_nan,
            );
            builder.assert_eq(
                "is_inf",
                builder.band(result.is_inf, bool_mask),
                expected_is_inf,
            );
            builder.assert_eq(
                "is_norm",
                builder.band(result.is_norm, bool_mask),
                expected_is_norm,
            );

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val.to_bits());

            // Get expected values from reference implementation
            let result = ref_fp64_unpack(val.to_bits());

            w[expected_sign] = Word(result.sign);
            w[expected_exp] = Word(result.exp);
            w[expected_frac] = Word(result.frac);
            w[expected_is_nan] = Word(result.is_nan);
            w[expected_is_inf] = Word(result.is_inf);
            w[expected_is_norm] = Word(result.is_norm);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_underflow_shift() {
        let test_cases = [
            // (res_sig, res_exp)
            (0x8000000000000000u64, 5), // Normal case, no underflow
            (0x8000000000000000u64, 0), // Underflow case, exp = 0
            (0x4000000000000000u64, -5i64 as u64), // Significant underflow
        ];

        for (res_sig, res_exp) in test_cases {
            let builder = CircuitBuilder::new();
            let res_sig_wire = builder.add_inout();
            let res_exp_wire = builder.add_inout();

            let (sig_round_base, exp_round_base, exp_lt_1) =
                fp64_underflow_shift(&builder, res_sig_wire, res_exp_wire);

            let expected_sig = builder.add_inout();
            let expected_exp = builder.add_inout();
            let expected_lt1 = builder.add_inout();
            let bool_mask = builder.add_constant(Word::MSB_ONE);

            builder.assert_eq("sig_round_base", sig_round_base, expected_sig);
            builder.assert_eq("exp_round_base", exp_round_base, expected_exp);
            builder.assert_eq("exp_lt_1", builder.band(exp_lt_1, bool_mask), expected_lt1);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[res_sig_wire] = Word(res_sig);
            w[res_exp_wire] = Word(res_exp);

            let (ref_sig, ref_exp, ref_lt1) = ref_fp64_underflow_shift(res_sig, res_exp);
            w[expected_sig] = Word(ref_sig);
            w[expected_exp] = Word(ref_exp);
            w[expected_lt1] = Word(ref_lt1);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    /// Helper: semantic equality for f64 bit patterns, treating any-NaN as equal
    pub fn f64_bits_semantic_eq(a_bits: u64, b_bits: u64) -> bool {
        let a = f64::from_bits(a_bits);
        let b = f64::from_bits(b_bits);
        if a.is_nan() && b.is_nan() {
            true
        } else {
            a_bits == b_bits
        }
    }

    #[test]
    fn test_fp64_round_rne() {
        let test_cases = [
            // (sig_base, exp_base) - test round-to-nearest-even
            (0x8000000000000000u64, 1023), // No rounding needed
            (0x8000000000000800u64, 1023), /* Round up (G=1, R=0, S=0, LSB=0 -> ties to even =
                                            * no round) */
            (0x8000000000001800u64, 1023), // Round up (G=1, R=1, S=0 -> round up)
            (0x8000000000000C00u64, 1023), // Round up (G=1, R=0, S=1 -> round up)
        ];

        for (sig_base, exp_base) in test_cases {
            let builder = CircuitBuilder::new();
            let sig_base_wire = builder.add_inout();
            let exp_base_wire = builder.add_inout();

            let (mant_final_53, exp_after_round, mant_overflow_mask) =
                fp64_round_rne(&builder, sig_base_wire, exp_base_wire);

            let expected_mant = builder.add_inout();
            let expected_exp = builder.add_inout();
            let expected_overflow = builder.add_inout();

            builder.assert_eq("mant_final_53", mant_final_53, expected_mant);
            builder.assert_eq("exp_after_round", exp_after_round, expected_exp);
            builder.assert_eq("mant_overflow_mask", mant_overflow_mask, expected_overflow);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[sig_base_wire] = Word(sig_base);
            w[exp_base_wire] = Word(exp_base);

            let (ref_mant, ref_exp, ref_overflow) = ref_fp64_round_rne(sig_base, exp_base);
            w[expected_mant] = Word(ref_mant);
            w[expected_exp] = Word(ref_exp);
            w[expected_overflow] = Word(ref_overflow);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }
}
