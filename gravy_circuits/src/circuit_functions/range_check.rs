//! range_check.rs — Radix‑decomposition helpers and signed‑range checks
//! for Expander Compiler Collection (ECC) circuits.
//!
//! --------------------------------------------------------------------
//! Quick glossary & assumptions
//! --------------------------------------------------------------------
//! * We work over a prime field 𝔽ₚ chosen by the backend; every `Variable`
//!   is the least–non‑negative residue of an integer in [0, p‑1].
//! * Outside the circuit we fix an integer h with 0 ≤ h < p and assume
//!   the _true_ integer value `a` satisfies  _h − p ≤ a ≤ h − 1_.  The
//!   default choice is `h = (p + 1)/2` (balanced interval).
//! * When we speak of an **upper‑bound triple** (r₁, κ₁, B₁) we require
//!      r₁^κ₁ ≤ r₁^κ₁ − 1 − B₁ + h ≤ p   and   B₁ ≤ (r₁−1)·r₁^{κ₁−1}.
//!   For a **lower‑bound triple** (r₂, κ₂, B₂) we require
//!      r₂^κ₂ ≤ B₂ + h ≤ p   and   B₂ ≤ (r₂−1)·r₂^{κ₂−1}.
//! * The caller pre‑computes the least residues  `B1_bar ≡ B₁ (mod p)`
//!   and  `B2_bar ≡ B₂ (mod p)`  and passes them as circuit constants
//!   (`Variable`s constructed with `api.constant(..)`).
//! * All helper names use **"radix"** instead of "base" to avoid the
//!   ambiguous word "base field".
//!
//! --------------------------------------------------------------------
//! Public helpers
//! --------------------------------------------------------------------
//! * `to_binary`                – witness κ least‑significant bits.
//! * `assert_is_binary_digit`   – degree‑2 constraint b(b−1)=0.
//! * `from_binary`              – constrain binary witness + reconstruct.
//! * `to_radix`                 – witness κ digits in radix r ≥ 2.
//! * `assert_is_radix_digit`    – digit validity via LUT or vanishing poly.
//! * `from_radix`               – constrain witness + reconstruct.
//! * `range_check`              – optional upper + lower signed range check.
//!
//! Every function returns the reconstructed value so that the caller can
//! chain further computations (e.g. ReLU).

use expander_compiler::frontend::*;

// ---------------------------------------------------------------------
// ░░  Binary helpers  ░░
// ---------------------------------------------------------------------

/// Return `kappa` least‑significant bits of `x` **without constraints**.
/// Bits are little‑endian: `d[0]` is the 2⁰‑bit.
pub fn to_binary<C: Config, B: RootAPI<C>>(
    api: &mut B,
    x: Variable,
    kappa: usize,
) -> Vec<Variable> {
    let mut q = x;
    let mut digits = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        // ECC offers bit‑wise helpers.
        let bit = api.unconstrained_bit_and(q, 1);
        digits.push(bit);
        q = api.unconstrained_shift_r(q, 1);
    }
    digits
}

/// Enforce that a `Variable` is a single bit via the degree‑2 vanishing
/// polynomial  b(b−1) = 0.
pub fn assert_is_binary_digit<C: Config, B: RootAPI<C>>(
    api: &mut B,
    bit: Variable,
) {
    let one = api.constant(1);
    let bit_minus_one = api.sub(bit, one);
    let product = api.mul(bit, bit_minus_one);
    api.assert_is_zero(product);
}

/// Constrain a binary witness `bits` (little‑endian) **and** assert that
/// they reconstruct `x`.  Returns the reconstructed value so callers can
/// re‑use it.
pub fn from_binary<C: Config, B: RootAPI<C>>(
    api: &mut B,
    bits: &[Variable],
    x: Variable,
) -> Variable {
    for &b in bits {
        assert_is_binary_digit(api, b);
    }

    let mut acc = api.constant(0);
    for (i, &b) in bits.iter().enumerate() {
        let coeff = api.constant(1 << i);
        acc = api.add(acc, api.mul(coeff, b));
    }

    api.assert_is_equal(acc, x);
    acc
}

// ---------------------------------------------------------------------
// ░░  Generic‑radix helpers  ░░
// ---------------------------------------------------------------------

/// Witness the `kappa` least‑significant digits of `x` in radix `radix`.
/// Digits are little‑endian: `d[0]` is the radix⁰ digit.
pub fn to_radix<C: Config, B: RootAPI<C>>(
    api: &mut B,
    x: Variable,
    radix: u32,
    kappa: usize,
) -> Vec<Variable> {
    assert!(radix >= 2, "radix must be ≥ 2");
    let mut q = x;
    let mut digits = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let d = api.unconstrained_mod(q, radix);
        digits.push(d);
        q = api.unconstrained_int_div(q, radix);
    }
    digits
}

/// Assert that `digit ∈ {0,…,radix−1}` using either
///   * a Lookup table (if `use_lookup == true`), or
///   * the degree‑`radix` vanishing polynomial otherwise.
pub fn assert_is_radix_digit<C: Config, B: RootAPI<C>>(
    api: &mut B,
    digit: Variable,
    radix: u32,
    use_lookup: bool,
    table: &mut Option<LogUpRangeProofTable>,
) {
    if use_lookup {
        // Either use caller‑supplied table or create a throw‑away one.
        let t = if let Some(t_ref) = table.as_mut() {
            t_ref
        } else {
            let nb_bits = (32 - radix.leading_zeros()) as usize;
            table.insert(LogUpRangeProofTable::new(nb_bits));
            table.as_mut().unwrap()
        };
        t.rangeproof(api, digit, radix as usize);
    } else {
        // Vanishing polynomial Π_{k=0}^{radix-1} (digit‑k).
        let mut poly = api.constant(1);
        for k in 0..radix {
            let term = if k == 0 {
                digit
            } else {
                api.sub(digit, api.constant(k))
            };
            poly = api.mul(poly, term);
        }
        api.assert_is_zero(poly);
    }
}

/// Constrain a radix‑witness `digits` (little‑endian) **and** assert it
/// reconstructs `x`.  Returns the reconstructed value.
pub fn from_radix<C: Config, B: RootAPI<C>>(
    api: &mut B,
    digits: &[Variable],
    radix: u32,
    x: Variable,
    use_lookup: bool,
    table: &mut Option<LogUpRangeProofTable>,
) -> Variable {
    for &d in digits {
        assert_is_radix_digit(api, d, radix, use_lookup, table);
    }

    let mut acc = api.constant(0);
    for (i, &d) in digits.iter().enumerate() {
        let coeff = api.constant(radix.pow(i as u32));
        acc = api.add(acc, api.mul(coeff, d));
    }

    api.assert_is_equal(acc, x);
    acc
}

// ---------------------------------------------------------------------
// ░░  Range check  ░░
// ---------------------------------------------------------------------

/// Range check.
///
/// * `a`            – secret/witness value (least residue).
/// * `upper`        – `Some((radix, kappa, B1_bar))` to enforce`a ≤ B₁`,
///                    or `None` to skip upper bound.
/// * `lower`        – `Some((radix, kappa, B2_bar))` to enforce`a ≥ −B₂`,
///                    or `None` to skip lower bound.
/// * `use_lookup`   – select lookup‑table vs. polynomial digit validity.
/// * `table`        – optional lookup table to reuse; if absent and
///                    `use_lookup == true` a temporary table is built.
///
/// Returns `(opt_upper_shift, opt_lower_shift)` where each component is
/// the reconstructed non‑negative value checked by the corresponding
/// bound, or `None` if that bound was not requested.
pub fn range_check<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: Variable,
    upper: Option<(u32, usize, Variable)>,
    lower: Option<(u32, usize, Variable)>,
    use_lookup: bool,
    table: &mut Option<LogUpRangeProofTable>,
) -> (Option<Variable>, Option<Variable>) {
    // ------------ upper bound -------------
    let upper_val = upper.map(|(radix, kappa, b1_bar)| {
        // compute x = B1_bar − a  (mod p)
        let x = api.sub(b1_bar, a);
        let digs = to_radix(api, x, radix, kappa);
        from_radix(api, &digs, radix, x, use_lookup, table)
    });

    // ------------ lower bound -------------
    let lower_val = lower.map(|(radix, kappa, b2_bar)| {
        // compute x = B2_bar + a  (mod p)
        let x = api.add(b2_bar, a);
        let digs = to_radix(api, x, radix, kappa);
        from_radix(api, &digs, radix, x, use_lookup, table)
    });

    (upper_val, lower_val)
}