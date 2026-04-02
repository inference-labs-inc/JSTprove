// Copyright 2025 Irreducible Inc.
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::utils::*;

/// IEEE-754 double (binary64) addition built from small testable blocks.
/// Returns a 64-bit IEEE-754 encoding.
///
/// Behavior summary:
/// - Fully handles normals, subnormals, zeros (signed), infinities, NaNs
/// - Rounding mode: round-to-nearest, ties-to-even
/// - Sticky tracking across alignment and underflow right-shifts
/// - Exact cancellation (+x) + (-x) → +0
/// - Overflow → ±Inf with computed sign
pub fn float64_add(builder: &CircuitBuilder, a: Wire, b: Wire) -> Wire {
    // unpack & classify
    let pa = fp64_unpack(builder, a);
    let pb = fp64_unpack(builder, b);

    // extended sig & effective exp
    let (sig_a, exp_a) = fp64_ext_sig_and_exp(builder, &pa);
    let (sig_b, exp_b) = fp64_ext_sig_and_exp(builder, &pb);

    // order by exponent
    let (
        sig_a_ordered,
        exp_a_ordered,
        sign_a_ordered,
        sig_b_ordered,
        exp_b_ordered,
        sign_b_ordered,
        _swapped,
    ) = fp64_order_by_exp(builder, sig_a, exp_a, pa.sign, sig_b, exp_b, pb.sign);

    // alignment with sticky→bit0
    let d = isub(builder, exp_a_ordered, exp_b_ordered);
    let s_b = fp64_align_with_sticky(builder, sig_b_ordered, d);

    // same/different sign split
    let diff_sign = builder.bxor(sign_a_ordered, sign_b_ordered);

    // add path
    let (sum_norm, exp_add) = fp64_add_path(builder, sig_a_ordered, s_b, exp_a_ordered);

    // sub path
    let (diff_norm, exp_sub, sign_sub, mags_equal) = fp64_sub_path(
        builder,
        sig_a_ordered,
        s_b,
        exp_a_ordered,
        sign_a_ordered,
        sign_b_ordered,
    );

    // merge + cancellation -> +0
    let (res_sig, res_exp, res_sign) = fp64_merge_and_cancel(
        builder,
        diff_sign,
        sum_norm,
        exp_add,
        sign_a_ordered,
        diff_norm,
        exp_sub,
        sign_sub,
        mags_equal,
    );

    // underflow pre-round right-shift if exp<=0
    let (sig_round_base, exp_round_base, exp_lt_1) =
        fp64_underflow_shift(builder, res_sig, res_exp);

    // round to nearest even
    let (mant_final_53, exp_after_round, mant_overflow_mask) =
        fp64_round_rne(builder, sig_round_base, exp_round_base);

    // Pack finite or overflow to inf.
    // Subnormal regime if:
    //   - we were below 1 (exp_lt_1), OR
    //   - base exponent == 1 and the integer bit (bit 63) is 0 (no hidden 1)
    let msb01 = bit_lsb(builder, sig_round_base, 63); // 0/1
    let msb_is_zero = builder.icmp_eq(msb01, zero(builder)); // MSB-bool
    let base_is_one = builder.icmp_eq(exp_round_base, one(builder));
    let in_sub_regime = builder.bor(exp_lt_1, builder.band(base_is_one, msb_is_zero));
    let stayed_sub_mask = builder.band(in_sub_regime, builder.bnot(mant_overflow_mask));
    let finite_or_inf = fp64_pack_finite_or_inf(
        builder,
        res_sign,
        mant_final_53,
        exp_after_round,
        stayed_sub_mask,
    );

    // operand-driven specials (NaN, operand infinities)
    fp64_finish_specials(builder, &pa, &pb, finite_or_inf)
}

/// Build an **extended significand** and **effective exponent**.
///
/// Convention:
/// - We keep the integer bit at position 63 and leave **11 low bits** for G/R/S and headroom. i.e.,
///   `sig_ext = significand << 11`.
///
/// For normals:  sig = ((1<<52) | frac) << 11,  exp_eff = exp
/// For subnorms:  sig = (frac           ) << 11, exp_eff = 1
///
/// Input:
/// - `p`: parts from `fp64_unpack`
///
/// Output:
/// - `(sig_ext, exp_eff)` using the above convention.
fn fp64_ext_sig_and_exp(b: &CircuitBuilder, p: &Fp64Parts) -> (Wire, Wire) {
    let one52 = b.add_constant_64(1u64 << 52);

    let sig_norm = b.shl(b.bor(one52, p.frac), 11);
    let sig_sub = b.shl(p.frac, 11);
    let sig = b.select(p.is_norm, sig_norm, sig_sub);

    let exp_eff = b.select(p.is_norm, p.exp, one(b)); // subnormals use exp=1
    (sig, exp_eff)
}

/// Order two operands by **effective exponent** (A has >= exponent).
///
/// Input:
/// - `(sig_a, exp_a, sign_a)`, `(sig_b, exp_b, sign_b)`
///
/// Output:
/// - `(sig_a, exp_a, sign_a, sig_b, exp_b, sign_b, swapped_mask)` where `swapped_mask` is all-1 if
///   we swapped A/B (i.e., A.exp < B.exp).
fn fp64_order_by_exp(
    b: &CircuitBuilder,
    sig_a: Wire,
    exp_a: Wire,
    sign_a: Wire,
    sig_b: Wire,
    exp_b: Wire,
    sign_b: Wire,
) -> (Wire, Wire, Wire, Wire, Wire, Wire, Wire) {
    let a_lt_b = b.icmp_ult(exp_a, exp_b);

    let exp_a_ordered = b.select(a_lt_b, exp_b, exp_a);
    let exp_b_ordered = b.select(a_lt_b, exp_a, exp_b);
    let sig_a_ordered = b.select(a_lt_b, sig_b, sig_a);
    let sig_b_ordered = b.select(a_lt_b, sig_a, sig_b);
    let sign_a_ordered = b.select(a_lt_b, sign_b, sign_a);
    let sign_b_ordered = b.select(a_lt_b, sign_a, sign_b);

    (
        sig_a_ordered,
        exp_a_ordered,
        sign_a_ordered,
        sig_b_ordered,
        exp_b_ordered,
        sign_b_ordered,
        a_lt_b,
    )
}

/// Align `sig_b` to `exp_a` by right shifting with **sticky** folded to bit 0.
///
/// Input:
/// - `sig_b`: extended significand per our convention
/// - `d   = exp_a - exp_b`
/// - Saturates large shifts to 63; accumulates sticky over all discarded bits.
///
/// Output:
/// - `aligned` where bit0 = old_bit0 | sticky (i.e., precise sticky folding).
fn fp64_align_with_sticky(b: &CircuitBuilder, sig_b: Wire, d: Wire) -> Wire {
    let (mut v, sticky) = var_shr_with_sticky(b, sig_b, d, true);

    let one_const = one(b);
    let keep = b.bnot(one_const); // ~1 = clear bit0
    let bit0 = b.bor(b.band(v, one_const), b.band(sticky, one_const));
    v = b.bor(b.band(v, keep), bit0);
    v
}

/// Magnitude add path (same signs), with one-bit renormalize if carry=1.
///
/// Input:
/// - `sig_a`, `s_b`, `exp_a`
///
/// Output:
/// - `(sum_norm, exp_add)` If carry occurs, result is shifted right by 1; new sticky = (old R) |
///   (old S).
fn fp64_add_path(b: &CircuitBuilder, sig_a: Wire, s_b: Wire, exp_a: Wire) -> (Wire, Wire) {
    let (sum_raw, carry_mask) = b.iadd_cin_cout(sig_a, s_b, zero(b));

    // Detect final carry-out (bit63 of carry mask) as 0/1 and as a select mask
    let carry_bit01 = bit_lsb(b, carry_mask, 63); // 0/1
    let carry_sel_mask = bit_msb01(b, carry_mask, 63); // use carry's MSB directly as MSB-bool

    // if carry: shift 1 and update sticky := old R | old S
    let sum_shift1 = b.shr(sum_raw, 1);
    let old_r = bit_lsb(b, sum_raw, 9);
    let old_s = b.band(sum_raw, one(b));
    let new_s = b.bor(old_r, old_s);

    // Inject carry into bit63 when renormalizing: (carry_bit01 << 63)
    let carry_hi = b.shl(carry_bit01, 63);
    let sum_shift1_with_carry = b.bor(sum_shift1, carry_hi);

    let keep = b.bnot(one(b));
    let sum_shift1clr = b.band(sum_shift1_with_carry, keep);
    let sum_norm = b.select(carry_sel_mask, b.bor(sum_shift1clr, new_s), sum_raw);

    // Increment exponent by carry (0/1)
    let exp_add = iadd(b, exp_a, carry_bit01);
    (sum_norm, exp_add)
}

/// Magnitude subtraction path (different signs), normalized by CLZ.
///
/// Chooses `big-small` by comparing magnitudes; sign takes the sign of the larger magnitude.
/// Normalizes by shifting left up to `min(clz(diff), exp_a-1)`.
///
/// Input:
/// - `sig_a`, `s_b`, `exp_a`, `sign_a`, `sign_b`
///
/// Output:
/// - `(diff_norm, exp_sub, sign_sub, mags_equal_mask)`
///   - If `sig_a == s_b`, `diff_norm` may be 0; caller can force +0 behavior using
///     `mags_equal_mask`.
fn fp64_sub_path(
    b: &CircuitBuilder,
    sig_a: Wire,
    s_b: Wire,
    exp_a: Wire,
    sign_a: Wire,
    sign_b: Wire,
) -> (Wire, Wire, Wire, Wire) {
    let a_lt_b = b.icmp_ult(sig_a, s_b);
    let big = b.select(a_lt_b, s_b, sig_a);
    let small = b.select(a_lt_b, sig_a, s_b);
    let diff_raw = isub(b, big, small);
    let sign_sub = b.select(a_lt_b, sign_b, sign_a);
    let mags_eq = b.icmp_eq(sig_a, s_b);

    let lz = clz64(b, diff_raw);
    let c64 = b.add_constant_64(64);
    let c63 = b.add_constant_64(63);
    let lt64 = b.icmp_ult(lz, c64);
    let lz_clamped = b.select(lt64, lz, c63);

    let exp_a_m1 = isub(b, exp_a, one(b));
    let lz_lt_expm1 = b.icmp_ult(lz_clamped, exp_a_m1);
    let sh = b.select(lz_lt_expm1, lz_clamped, exp_a_m1);

    let diff_norm = var_shl(b, diff_raw, sh);
    let exp_sub = isub(b, exp_a, sh);

    (diff_norm, exp_sub, sign_sub, mags_eq)
}

/// Merge add/sub paths and handle **exact cancellation → +0**.
///
/// Input:
/// - `same_sign_mask`
/// - add: `(sum_norm, exp_add, sign_a)`
/// - sub: `(diff_norm, exp_sub, sign_sub, mags_equal_mask)`
///
/// Output:
/// - `(res_sig, res_exp, res_sign)` with cancellation mapped to +0.
#[allow(clippy::too_many_arguments)]
fn fp64_merge_and_cancel(
    b: &CircuitBuilder,
    diff_sign: Wire,
    sum_norm: Wire,
    exp_add: Wire,
    sign_a: Wire,
    diff_norm: Wire,
    exp_sub: Wire,
    sign_sub: Wire,
    mags_equal: Wire,
) -> (Wire, Wire, Wire) {
    let res_sig_pre = b.select(diff_sign, diff_norm, sum_norm);
    let res_exp_pre = b.select(diff_sign, exp_sub, exp_add);
    let res_sign_pre = b.select(diff_sign, sign_sub, sign_a);

    // different signs and equal magnitudes => +0
    let cancel = b.band(diff_sign, mags_equal);

    let res_sig = b.select(cancel, zero(b), res_sig_pre);
    let res_exp = b.select(cancel, zero(b), res_exp_pre); // exp ignored for zero
    let res_sign = b.select(cancel, zero(b), res_sign_pre); // +0
    (res_sig, res_exp, res_sign)
}

/// Final special-case selection: NaN and operand Infinities.
///
/// Precedence: **NaN > operand-Inf > (finite/overflow result)**.
///
/// - If any NaN, or (+Inf)+(-Inf), return canonical quiet NaN: 0x7FF8_0000_0000_0000
/// - Else if any operand is Inf, return that infinity with its operand sign
/// - Else return `finite_or_inf`
///
/// Inputs:
/// - `pa`, `pb`: parts from `fp64_unpack` for a and b
/// - `finite_or_inf`: result from `fp64_pack_finite_or_inf` (includes overflow to Inf)
///
/// Output:
/// - final packed double
fn fp64_finish_specials(
    b: &CircuitBuilder,
    pa: &Fp64Parts,
    pb: &Fp64Parts,
    finite_or_inf: Wire,
) -> Wire {
    let qnan = b.add_constant_64(0x7FF8_0000_0000_0000);
    let exp_2047 = b.add_constant_64(0x7FF);
    let inf_payload = b.shl(exp_2047, 52);

    let diff_sign = b.bxor(pa.sign, pb.sign);
    let opp_inf_nan = b.band(b.band(pa.is_inf, pb.is_inf), diff_sign);
    let any_nan = b.bor(pa.is_nan, pb.is_nan);
    let nan_mask = b.bor(any_nan, opp_inf_nan);

    // If any operand is infinity (with either same signs or only one inf), pick its sign.
    let any_inf = b.bor(pa.is_inf, pb.is_inf);
    let inf_sign = b.select(pb.is_inf, pb.sign, pa.sign);
    let packed_inf = b.bor(b.band(inf_sign, b.add_constant(Word::MSB_ONE)), inf_payload);

    let with_inf = b.select(any_inf, packed_inf, finite_or_inf);
    b.select(nan_mask, qnan, with_inf)
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};

    use super::*;
    use crate::float64::utils::tests::{
        f64_bits_semantic_eq, ref_fp64_pack_finite_or_inf, ref_fp64_round_rne,
        ref_fp64_underflow_shift, ref_fp64_unpack,
    };

    fn ref_fp64_ext_sig_and_exp(_sign: u64, exp: u64, frac: u64, is_norm: u64) -> (u64, u64) {
        let sig = if is_norm != 0 {
            ((1u64 << 52) | frac) << 11
        } else {
            frac << 11
        };

        let exp_eff = if is_norm != 0 { exp } else { 1 };

        (sig, exp_eff)
    }

    fn ref_fp64_order_by_exp(
        sig_a: u64,
        exp_a: u64,
        sign_a: u64,
        sig_b: u64,
        exp_b: u64,
        sign_b: u64,
    ) -> (u64, u64, u64, u64, u64, u64, u64) {
        if exp_a < exp_b {
            (sig_b, exp_b, sign_b, sig_a, exp_a, sign_a, 1u64 << 63)
        } else {
            (sig_a, exp_a, sign_a, sig_b, exp_b, sign_b, 0)
        }
    }

    fn ref_fp64_align_with_sticky(sig_b: u64, d: u64) -> u64 {
        let d_eff = std::cmp::min(d, 63);
        if d_eff == 0 {
            return sig_b;
        }

        let shifted = sig_b >> d_eff;
        let lost_bits = sig_b & ((1u64 << d_eff) - 1);
        let sticky = if lost_bits != 0 { 1 } else { 0 };

        // Set bit 0 to be old bit 0 OR sticky
        let old_bit0 = shifted & 1;
        let new_bit0 = old_bit0 | sticky;
        (shifted & !1) | new_bit0
    }

    fn ref_fp64_add_path(sig_a: u64, s_b: u64, exp_a: u64) -> (u64, u64) {
        // Reference implementation following the provided pseudocode exactly
        // function fp64_add_path(sig_a, s_b, exp_a) -> (sum_norm, exp_add):

        // 1) Add magnitudes
        let sum_raw = sig_a.wrapping_add(s_b);

        // 2) Detect 1-bit carry out of the 64-bit lane
        // (classic unsigned carry detection)
        let carry01 = if sum_raw < sig_a { 1 } else { 0 };

        // 3) If carry: renormalize by shifting right one, and update sticky := old_R | old_S
        if carry01 == 1 {
            // Old rounding geometry before the shift
            let old_r = (sum_raw >> 9) & 1; // bit 9
            let old_s = sum_raw & 1; // bit 0 (already a sticky-OR)

            // Shift right and preserve the carry into bit 63
            let shifted = (sum_raw >> 1) | (1u64 << 63);
            let new_s = old_r | old_s; // new sticky for bit 0 after shift
            let sum_norm = (shifted & !1) | new_s; // clear bit0, then OR in newS

            let exp_add = exp_a + 1;

            (sum_norm, exp_add)
        } else {
            // No carry: already normalized
            let sum_norm = sum_raw;
            let exp_add = exp_a;

            (sum_norm, exp_add)
        }
    }

    fn ref_fp64_sub_path(
        sig_a: u64,
        s_b: u64,
        exp_a: u64,
        sign_a: u64,
        sign_b: u64,
    ) -> (u64, u64, u64, u64) {
        // Match the circuit logic exactly
        let a_lt_b = if sig_a < s_b { 1u64 << 63 } else { 0 };
        let big = if a_lt_b != 0 { s_b } else { sig_a };
        let small = if a_lt_b != 0 { sig_a } else { s_b };
        let diff_raw = big.wrapping_sub(small);
        let sign_sub = if a_lt_b != 0 { sign_b } else { sign_a };
        let mags_eq = if sig_a == s_b { 1u64 << 63 } else { 0 };

        // Use clz64 reference implementation
        let lz = if diff_raw == 0 {
            64
        } else {
            diff_raw.leading_zeros() as u64
        };
        let c64 = 64;
        let c63 = 63;
        let lt64 = if lz < c64 { 1u64 << 63 } else { 0 };
        let lz_clamped = if lt64 != 0 { lz } else { c63 };

        let exp_a_m1 = exp_a.wrapping_sub(1);
        let lz_lt_expm1 = if lz_clamped < exp_a_m1 { 1u64 << 63 } else { 0 };
        let sh = if lz_lt_expm1 != 0 {
            lz_clamped
        } else {
            exp_a_m1
        };

        // Use variable left shift
        let shift_amt = std::cmp::min(sh, 63); // Simulate var_shl behavior
        let diff_norm = diff_raw << shift_amt;
        let exp_sub = exp_a.wrapping_sub(sh);

        (diff_norm, exp_sub, sign_sub, mags_eq)
    }

    #[allow(clippy::too_many_arguments)]
    fn ref_fp64_merge_and_cancel(
        diff_sign: u64,
        sum_norm: u64,
        exp_add: u64,
        sign_a: u64,
        diff_norm: u64,
        exp_sub: u64,
        sign_sub: u64,
        mags_equal: u64,
    ) -> (u64, u64, u64) {
        let (res_sig_pre, res_exp_pre, res_sign_pre) = if diff_sign & (1u64 << 63) == 0 {
            (sum_norm, exp_add, sign_a)
        } else {
            (diff_norm, exp_sub, sign_sub)
        };

        // Handle exact cancellation -> +0
        let cancel = if diff_sign & (1u64 << 63) != 0 && mags_equal != 0 {
            1u64 << 63
        } else {
            0
        };

        let res_sig = if cancel != 0 { 0 } else { res_sig_pre };
        let res_exp = if cancel != 0 { 0 } else { res_exp_pre };
        let res_sign = if cancel != 0 { 0 } else { res_sign_pre };

        (res_sig, res_exp, res_sign)
    }

    fn ref_fp64_finish_specials(
        pa_is_nan: u64,
        pa_is_inf: u64,
        pa_sign_msb: u64,
        pb_is_nan: u64,
        pb_is_inf: u64,
        pb_sign_msb: u64,
        finite_or_inf: u64,
    ) -> u64 {
        let qnan = 0x7FF8_0000_0000_0000u64;

        let any_nan = (pa_is_nan | pb_is_nan) != 0;
        let same_sign = pa_sign_msb == pb_sign_msb;
        let opp_inf_nan = (pa_is_inf != 0) && (pb_is_inf != 0) && !same_sign;
        let nan_case = any_nan || opp_inf_nan;

        if nan_case {
            return qnan;
        }

        let any_inf = (pa_is_inf | pb_is_inf) != 0;
        if any_inf {
            let inf_msb = if pb_is_inf != 0 {
                pb_sign_msb
            } else {
                pa_sign_msb
            } & (1u64 << 63);
            let inf_payload = 0x7FFu64 << 52;
            return inf_msb | inf_payload;
        }

        finite_or_inf
    }

    fn ref_float64_add(a_bits: u64, b_bits: u64) -> u64 {
        // Unpack & classify operands
        let pa = ref_fp64_unpack(a_bits);
        let pb = ref_fp64_unpack(b_bits);

        // Extended significands and effective exponents
        let (sig_a0, exp_a0) = ref_fp64_ext_sig_and_exp(pa.sign, pa.exp, pa.frac, pa.is_norm);
        let (sig_b0, exp_b0) = ref_fp64_ext_sig_and_exp(pb.sign, pb.exp, pb.frac, pb.is_norm);

        // Order by exponent
        let (sig_a, exp_a, sign_a, sig_b, exp_b, sign_b, _swapped) =
            ref_fp64_order_by_exp(sig_a0, exp_a0, pa.sign, sig_b0, exp_b0, pb.sign);

        // Align B to A with sticky folded into bit0
        let d = exp_a.wrapping_sub(exp_b);
        let s_b_align = ref_fp64_align_with_sticky(sig_b, d);

        // Choose path by sign (both MSB-bools already)
        let diff_sign = sign_a ^ sign_b;
        let (sum_norm, exp_add) = ref_fp64_add_path(sig_a, s_b_align, exp_a);
        let (diff_norm, exp_sub, sign_sub, mags_equal) =
            ref_fp64_sub_path(sig_a, s_b_align, exp_a, sign_a, sign_b);

        // Merge + exact cancellation to +0
        let (res_sig, res_exp, res_sign) = ref_fp64_merge_and_cancel(
            diff_sign, sum_norm, exp_add, sign_a, diff_norm, exp_sub, sign_sub, mags_equal,
        );

        // Pre-round underflow handling
        let (sig_round_base, exp_round_base, exp_lt_1) = ref_fp64_underflow_shift(res_sig, res_exp);

        // Round to nearest ties-to-even
        let (mant_final_53, exp_after_round, mant_overflow_mask) =
            ref_fp64_round_rne(sig_round_base, exp_round_base);

        // Pack finite or overflow to Inf
        // Subnormal regime if:
        //   - we were below 1 (exp_lt_1), OR
        //   - base exponent == 1 and the integer bit (bit 63) is 0 (no hidden 1)
        let msb = (sig_round_base >> 63) & 1;
        let base_is_one = exp_round_base == 1;
        let in_sub_regime = (exp_lt_1 != 0) || (base_is_one && msb == 0);
        let stayed_sub_mask = if in_sub_regime && mant_overflow_mask == 0 {
            1u64 << 63
        } else {
            0
        };
        let finite_or_inf =
            ref_fp64_pack_finite_or_inf(res_sign, mant_final_53, exp_after_round, stayed_sub_mask);

        // Final specials overlay
        ref_fp64_finish_specials(
            pa.is_nan,
            pa.is_inf,
            pa.sign,
            pb.is_nan,
            pb.is_inf,
            pb.sign,
            finite_or_inf,
        )
    }

    #[test]
    fn test_fp64_align_with_sticky() {
        let test_cases = [
            // (sig_b, d, expected_result)
            (0x8000000000000000u64, 1), // Simple shift right by 1
            (0x8000000000000001u64, 1), // Shift right by 1 with sticky
            (0xFFFFFFFFFFFFFFFFu64, 4), // Shift right by 4 with sticky
            (0x1000000000000000u64, 0), // No shift
            (0x123456789ABCDEFFu64, 8), // Shift by 8, lost bits set sticky
        ];

        for (sig_b, d) in test_cases {
            let builder = CircuitBuilder::new();
            let sig_b_wire = builder.add_inout();
            let d_wire = builder.add_inout();
            let result = fp64_align_with_sticky(&builder, sig_b_wire, d_wire);
            let expected_wire = builder.add_inout();
            builder.assert_eq("align_result", result, expected_wire);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[sig_b_wire] = Word(sig_b);
            w[d_wire] = Word(d);
            w[expected_wire] = Word(ref_fp64_align_with_sticky(sig_b, d));

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_iadd_simple() {
        let builder = CircuitBuilder::new();
        let a = builder.add_inout();
        let b = builder.add_inout();
        let result = iadd(&builder, a, b);
        let expected = builder.add_inout();
        builder.assert_eq("iadd_result", result, expected);

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();
        w[a] = Word(1000);
        w[b] = Word(1);
        w[expected] = Word(1001);

        circuit.populate_wire_witness(&mut w).unwrap();
        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_fp64_add_path() {
        let test_cases = [
            // (sig_a, s_b, exp_a)
            (0x8000000000000000u64, 0x4000000000000000u64, 1023), // Normal addition
            (0xFFFFFFFFFFFFFFFFu64, 0x0000000000000001u64, 1000), // Addition with carry
            (0x1000000000000000u64, 0x1000000000000000u64, 500),  // Equal values
            // 1.5 + 1.5 geometry in extended sig space: each is 0xC000.., expect renorm carry
            (0xC000000000000000u64, 0xC000000000000000u64, 1023),
        ];

        for (sig_a, s_b, exp_a) in test_cases.iter() {
            let builder = CircuitBuilder::new();
            let sig_a_wire = builder.add_inout();
            let s_b_wire = builder.add_inout();
            let exp_a_wire = builder.add_inout();
            let (sum_norm, exp_add) = fp64_add_path(&builder, sig_a_wire, s_b_wire, exp_a_wire);
            let expected_sum = builder.add_inout();
            let expected_exp = builder.add_inout();
            builder.assert_eq("sum_norm", sum_norm, expected_sum);
            builder.assert_eq("exp_add", exp_add, expected_exp);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[sig_a_wire] = Word(*sig_a);
            w[s_b_wire] = Word(*s_b);
            w[exp_a_wire] = Word(*exp_a);

            let (ref_sum, ref_exp) = ref_fp64_add_path(*sig_a, *s_b, *exp_a);
            w[expected_sum] = Word(ref_sum);
            w[expected_exp] = Word(ref_exp);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_sub_path() {
        let test_cases = [
            // (sig_a, s_b, exp_a, sign_a, sign_b)
            (0x8000000000000000u64, 0x4000000000000000u64, 1023, 0, 1), // Normal subtraction
            (0x4000000000000000u64, 0x8000000000000000u64, 1020, 0, 1), // B > A
            (0x8000000000000000u64, 0x8000000000000000u64, 1023, 0, 1), // Equal magnitudes
            (0x1000000000000000u64, 0x0800000000000000u64, 1000, 1, 0), // Small difference
        ];

        for (sig_a, s_b, exp_a, sign_a, sign_b) in test_cases {
            let builder = CircuitBuilder::new();
            let sig_a_wire = builder.add_inout();
            let s_b_wire = builder.add_inout();
            let exp_a_wire = builder.add_inout();
            let sign_a_wire = builder.add_inout();
            let sign_b_wire = builder.add_inout();

            let (diff_norm, exp_sub, sign_sub, mags_equal) = fp64_sub_path(
                &builder,
                sig_a_wire,
                s_b_wire,
                exp_a_wire,
                sign_a_wire,
                sign_b_wire,
            );

            let expected_diff = builder.add_inout();
            let expected_exp = builder.add_inout();
            let expected_sign = builder.add_inout();
            let expected_mags = builder.add_inout();
            let bool_mask = builder.add_constant(Word::MSB_ONE);

            builder.assert_eq("diff_norm", diff_norm, expected_diff);
            builder.assert_eq("exp_sub", exp_sub, expected_exp);
            builder.assert_eq("sign_sub", sign_sub, expected_sign);
            builder.assert_eq(
                "mags_equal",
                builder.band(mags_equal, bool_mask),
                expected_mags,
            );

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[sig_a_wire] = Word(sig_a);
            w[s_b_wire] = Word(s_b);
            w[exp_a_wire] = Word(exp_a);
            w[sign_a_wire] = Word(sign_a);
            w[sign_b_wire] = Word(sign_b);

            let (ref_diff, ref_exp, ref_sign, ref_mags) =
                ref_fp64_sub_path(sig_a, s_b, exp_a, sign_a, sign_b);
            w[expected_diff] = Word(ref_diff);
            w[expected_exp] = Word(ref_exp);
            w[expected_sign] = Word(ref_sign);
            w[expected_mags] = Word(ref_mags);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_merge_and_cancel() {
        let test_cases = [
            // (same_sign, sum_norm, exp_add, sign_a, diff_norm, exp_sub, sign_sub, mags_equal)
            (
                1u64 << 63,
                0x8000000000000000u64,
                1024,
                0,
                0x4000000000000000u64,
                1020,
                1,
                0,
            ), /* Same sign - use add path */
            (
                0,
                0x8000000000000000u64,
                1024,
                0,
                0x4000000000000000u64,
                1020,
                1,
                0,
            ), /* Different
                * sign - use
                * sub path */
            (
                0,
                0x8000000000000000u64,
                1024,
                0,
                0x4000000000000000u64,
                1020,
                1,
                1u64 << 63,
            ), /* Exact cancellation */
        ];

        for (same_sign, sum_norm, exp_add, sign_a, diff_norm, exp_sub, sign_sub, mags_equal) in
            test_cases
        {
            let builder = CircuitBuilder::new();
            let same_sign_wire = builder.add_inout();
            let sum_norm_wire = builder.add_inout();
            let exp_add_wire = builder.add_inout();
            let sign_a_wire = builder.add_inout();
            let diff_norm_wire = builder.add_inout();
            let exp_sub_wire = builder.add_inout();
            let sign_sub_wire = builder.add_inout();
            let mags_equal_wire = builder.add_inout();

            let (res_sig, res_exp, res_sign) = fp64_merge_and_cancel(
                &builder,
                same_sign_wire,
                sum_norm_wire,
                exp_add_wire,
                sign_a_wire,
                diff_norm_wire,
                exp_sub_wire,
                sign_sub_wire,
                mags_equal_wire,
            );

            let expected_sig = builder.add_inout();
            let expected_exp = builder.add_inout();
            let expected_sign = builder.add_inout();

            builder.assert_eq("res_sig", res_sig, expected_sig);
            builder.assert_eq("res_exp", res_exp, expected_exp);
            builder.assert_eq("res_sign", res_sign, expected_sign);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[same_sign_wire] = Word(same_sign);
            w[sum_norm_wire] = Word(sum_norm);
            w[exp_add_wire] = Word(exp_add);
            w[sign_a_wire] = Word(sign_a);
            w[diff_norm_wire] = Word(diff_norm);
            w[exp_sub_wire] = Word(exp_sub);
            w[sign_sub_wire] = Word(sign_sub);
            w[mags_equal_wire] = Word(mags_equal);

            let (ref_sig, ref_exp, ref_sign) = ref_fp64_merge_and_cancel(
                same_sign, sum_norm, exp_add, sign_a, diff_norm, exp_sub, sign_sub, mags_equal,
            );
            w[expected_sig] = Word(ref_sig);
            w[expected_exp] = Word(ref_exp);
            w[expected_sign] = Word(ref_sign);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_pack_finite_or_inf() {
        let test_cases = [
            // (sign, mant_final_53, exp_after_round, stayed_sub_mask)
            (0, 0x1000000000000000u64, 1023, 0), // Normal positive number
            (1, 0x1800000000000000u64, 1000, 0), // Normal negative number
            (0, 0x0000000000001000u64, 1, 1u64 << 63), // Subnormal positive
            (0, 0x1000000000000000u64, 0x7FF, 0), // Overflow to infinity
        ];

        for (sign, mant_final_53, exp_after_round, stayed_sub_mask) in test_cases {
            let builder = CircuitBuilder::new();
            let sign_wire = builder.add_inout();
            let mant_wire = builder.add_inout();
            let exp_wire = builder.add_inout();
            let stayed_sub_wire = builder.add_inout();

            // Convert 0/1 sign input into MSB-bool for the packer
            let sign_msb = builder.shl(sign_wire, 63);
            let result =
                fp64_pack_finite_or_inf(&builder, sign_msb, mant_wire, exp_wire, stayed_sub_wire);

            let expected_result = builder.add_inout();
            builder.assert_eq("pack_result", result, expected_result);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[sign_wire] = Word(sign);
            w[mant_wire] = Word(mant_final_53);
            w[exp_wire] = Word(exp_after_round);
            w[stayed_sub_wire] = Word(stayed_sub_mask);

            let ref_result = ref_fp64_pack_finite_or_inf(
                sign << 63,
                mant_final_53,
                exp_after_round,
                stayed_sub_mask,
            );
            w[expected_result] = Word(ref_result);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_finish_specials() {
        let test_cases = [
            // (pa_is_nan, pa_is_inf, pa_sign, pb_is_nan, pb_is_inf, pb_sign, finite_or_inf)
            (0, 0, 0, 0, 0, 0, 0x3FF0000000000000u64), // Normal case
            (1u64 << 63, 0, 0, 0, 0, 0, 0x4000000000000000u64), // A is NaN
            (0, 0, 0, 1u64 << 63, 0, 1, 0x4000000000000000u64), // B is NaN
            (0, 1u64 << 63, 0, 0, 0, 1, 0x4000000000000000u64), // A is +inf
            (0, 0, 1, 0, 1u64 << 63, 1, 0x4000000000000000u64), // B is -inf
            (0, 1u64 << 63, 0, 0, 1u64 << 63, 1, 0x4000000000000000u64), // +inf + (-inf) = NaN
        ];

        for (pa_is_nan, pa_is_inf, pa_sign, pb_is_nan, pb_is_inf, pb_sign, finite_or_inf) in
            test_cases
        {
            let builder = CircuitBuilder::new();
            let pa = Fp64Parts {
                sign: builder.add_inout(),
                exp: builder.add_inout(),
                frac: builder.add_inout(),
                is_nan: builder.add_inout(),
                is_inf: builder.add_inout(),
                is_zero: builder.add_inout(),
                is_sub: builder.add_inout(),
                is_norm: builder.add_inout(),
            };
            let pb = Fp64Parts {
                sign: builder.add_inout(),
                exp: builder.add_inout(),
                frac: builder.add_inout(),
                is_nan: builder.add_inout(),
                is_inf: builder.add_inout(),
                is_zero: builder.add_inout(),
                is_sub: builder.add_inout(),
                is_norm: builder.add_inout(),
            };
            let finite_or_inf_wire = builder.add_inout();

            let result = fp64_finish_specials(&builder, &pa, &pb, finite_or_inf_wire);
            let expected_result = builder.add_inout();
            builder.assert_eq("finish_result", result, expected_result);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();

            // Fill in all the pa/pb fields (most are unused for this test)
            // Provide MSB-bool sign values
            w[pa.sign] = Word(pa_sign << 63);
            w[pa.exp] = Word(0);
            w[pa.frac] = Word(0);
            w[pa.is_nan] = Word(pa_is_nan);
            w[pa.is_inf] = Word(pa_is_inf);
            w[pa.is_zero] = Word(0);
            w[pa.is_sub] = Word(0);
            w[pa.is_norm] = Word(0);

            w[pb.sign] = Word(pb_sign << 63);
            w[pb.exp] = Word(0);
            w[pb.frac] = Word(0);
            w[pb.is_nan] = Word(pb_is_nan);
            w[pb.is_inf] = Word(pb_is_inf);
            w[pb.is_zero] = Word(0);
            w[pb.is_sub] = Word(0);
            w[pb.is_norm] = Word(0);

            w[finite_or_inf_wire] = Word(finite_or_inf);

            let ref_result = ref_fp64_finish_specials(
                pa_is_nan,
                pa_is_inf,
                pa_sign << 63,
                pb_is_nan,
                pb_is_inf,
                pb_sign << 63,
                finite_or_inf,
            );
            w[expected_result] = Word(ref_result);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_ext_sig_and_exp() {
        let test_values = vec![1.0f64, -1.0f64, 2.0f64, f64::MIN_POSITIVE, 0.0f64];

        for val in test_values {
            let builder = CircuitBuilder::new();
            let input = builder.add_inout();
            let parts = fp64_unpack(&builder, input);
            let (sig, exp_eff) = fp64_ext_sig_and_exp(&builder, &parts);
            let expected_sig = builder.add_inout();
            let expected_exp = builder.add_inout();

            builder.assert_eq("sig", sig, expected_sig);
            builder.assert_eq("exp_eff", exp_eff, expected_exp);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[input] = Word(val.to_bits());

            // Get reference values
            let unpack_result = ref_fp64_unpack(val.to_bits());
            let (exp_sig, exp_exp) = ref_fp64_ext_sig_and_exp(
                unpack_result.sign,
                unpack_result.exp,
                unpack_result.frac,
                unpack_result.is_norm,
            );

            w[expected_sig] = Word(exp_sig);
            w[expected_exp] = Word(exp_exp);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_fp64_order_by_exp() {
        let test_cases = [
            // (sig_a, exp_a, sign_a, sig_b, exp_b, sign_b)
            (100, 5, 1, 200, 10, 0),  // a < b, should swap
            (100, 10, 1, 200, 5, 0),  // a >= b, no swap
            (300, 15, 0, 400, 15, 1), // equal exp, no swap
        ];

        for (sig_a_val, exp_a_val, sign_a_val, sig_b_val, exp_b_val, sign_b_val) in test_cases {
            let builder = CircuitBuilder::new();
            let sig_a = builder.add_inout();
            let exp_a = builder.add_inout();
            let sign_a = builder.add_inout();
            let sig_b = builder.add_inout();
            let exp_b = builder.add_inout();
            let sign_b = builder.add_inout();

            let (sig_a_out, exp_a_out, sign_a_out, sig_b_out, exp_b_out, sign_b_out, swapped) =
                fp64_order_by_exp(&builder, sig_a, exp_a, sign_a, sig_b, exp_b, sign_b);

            let expected_sig_a = builder.add_inout();
            let expected_exp_a = builder.add_inout();
            let expected_sign_a = builder.add_inout();
            let expected_sig_b = builder.add_inout();
            let expected_exp_b = builder.add_inout();
            let expected_sign_b = builder.add_inout();
            let expected_swapped = builder.add_inout();
            let bool_mask = builder.add_constant(Word::MSB_ONE);

            builder.assert_eq("sig_a", sig_a_out, expected_sig_a);
            builder.assert_eq("exp_a", exp_a_out, expected_exp_a);
            builder.assert_eq("sign_a", sign_a_out, expected_sign_a);
            builder.assert_eq("sig_b", sig_b_out, expected_sig_b);
            builder.assert_eq("exp_b", exp_b_out, expected_exp_b);
            builder.assert_eq("sign_b", sign_b_out, expected_sign_b);
            builder.assert_eq(
                "swapped",
                builder.band(swapped, bool_mask),
                expected_swapped,
            );

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();

            w[sig_a] = Word(sig_a_val);
            w[exp_a] = Word(exp_a_val);
            w[sign_a] = Word(sign_a_val);
            w[sig_b] = Word(sig_b_val);
            w[exp_b] = Word(exp_b_val);
            w[sign_b] = Word(sign_b_val);

            // Get expected values from reference implementation
            let (exp_sig_a, exp_exp_a, exp_sign_a, exp_sig_b, exp_exp_b, exp_sign_b, exp_swapped) =
                ref_fp64_order_by_exp(
                    sig_a_val, exp_a_val, sign_a_val, sig_b_val, exp_b_val, sign_b_val,
                );

            w[expected_sig_a] = Word(exp_sig_a);
            w[expected_exp_a] = Word(exp_exp_a);
            w[expected_sign_a] = Word(exp_sign_a);
            w[expected_sig_b] = Word(exp_sig_b);
            w[expected_exp_b] = Word(exp_exp_b);
            w[expected_sign_b] = Word(exp_sign_b);
            w[expected_swapped] = Word(exp_swapped);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }

    #[test]
    fn test_float64_add_basic() {
        let tiny_sub = f64::from_bits(1); // smallest subnormal 2^-1074
        let half_ulp_at_1 = 2f64.powi(-53);
        let one_ulp_at_1 = 2f64.powi(-52);
        let big = 1e300f64;
        let small = 1e-300f64;
        let min_norm = f64::MIN_POSITIVE; // 2^-1022
        let half_min_norm = min_norm / 2.0; // subnormal

        let test_cases: &[(f64, f64)] = &[
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, -0.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 2.0),
            (1.5, 1.5),
            (2.0, -0.5),
            // Rounding behavior near 1.0
            (1.0, half_ulp_at_1), // tie, LSB even -> stays 1.0
            (1.0, one_ulp_at_1),  // round up -> nextafter(1, +inf)
            // Subnormals and underflow path
            (tiny_sub, tiny_sub),           // sticky folding
            (half_min_norm, half_min_norm), // rises to min normal
            (min_norm, tiny_sub),           // tiny add that should not lose sticky
            // Large exponent differences (alignment saturates + sticky accumulates)
            (1.0, 2f64.powi(-1000)),
            (2f64.powi(100), 2f64.powi(-1000)),
            // Big/small normals
            (big, big),   // overflow to +inf
            (-big, -big), // overflow to -inf
            (small, small),
            // Infinities
            (f64::INFINITY, 1.0),
            (-f64::INFINITY, -1.0),
            (f64::INFINITY, -f64::INFINITY), // -> qNaN
            // NaNs (canonical qNaN regardless of payload)
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ];

        for (i, (a_val, b_val)) in test_cases.iter().copied().enumerate() {
            let ref_result = ref_float64_add(a_val.to_bits(), b_val.to_bits());
            let native = (a_val + b_val).to_bits();
            assert!(
                f64_bits_semantic_eq(ref_result, native),
                "Test case {}: pseudo {} != native {}",
                i,
                ref_result,
                native
            );

            let builder = CircuitBuilder::new();
            let a = builder.add_inout();
            let b = builder.add_inout();
            let result = float64_add(&builder, a, b);
            let expected = builder.add_inout();
            builder.assert_eq(format!("float64_add_case_{}", i), result, expected);

            let circuit = builder.build();
            let mut w = circuit.new_witness_filler();
            w[a] = Word(a_val.to_bits());
            w[b] = Word(b_val.to_bits());

            let expected_val = ref_result;
            w[expected] = Word(expected_val);

            circuit.populate_wire_witness(&mut w).unwrap();
            let cs = circuit.constraint_system();
            verify_constraints(cs, &w.into_value_vec()).unwrap();
        }
    }
}
