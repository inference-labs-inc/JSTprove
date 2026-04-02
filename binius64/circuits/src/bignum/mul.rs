// Copyright 2025 Irreducible Inc.
use std::iter;

use binius_core::consts::WORD_SIZE_BITS;
use binius_frontend::{CircuitBuilder, Wire};

use super::{
    addsub::{add_with_carry_out, compute_stack_adds, compute_stack_adds_with_carry_outs, sub},
    biguint::BigUint,
};

/// Multiply two arbitrary-sized `BigUint`s using textbook algorithm.
///
/// Produces `O(a.len() * b.len())` constraints.
///
/// Computes `a * b` where both inputs are `BigUint`s. The result will have
/// `a.limbs.len() + b.limbs.len()` limbs to accommodate the full product
/// without overflow.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand `BigUint`
/// * `b` - Second operand `BigUint`
///
/// # Returns
/// Product `BigUint` with `a.limbs.len() + b.limbs.len()` limbs
pub fn textbook_mul(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> BigUint {
    // Multiply argument's limbs pairwise.
    //
    // The accumulator has exactly a.limbs.len() + b.limbs.len() slots to hold
    // all partial products
    let mut accumulator = vec![vec![]; a.limbs.len() + b.limbs.len()];
    for (i, &ai) in a.limbs.iter().enumerate() {
        for (j, &bj) in b.limbs.iter().enumerate() {
            let (hi, lo) = builder.imul(ai, bj);
            let k = i + j;
            accumulator[k].push(lo);
            accumulator[k + 1].push(hi);
        }
    }
    compute_stack_adds(builder, &accumulator)
}

/// Square an arbitrary-sized `BigUint` using textbook algorithm.
///
/// Computes `a * a` using an optimized algorithm that takes advantage of the symmetry
/// in squaring (each cross-product appears twice). This is roughly twice more efficient
/// than using `textbook_mul`, though still quadratic in the number of constraints.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - The `BigUint` to be squared
///
/// # Returns
/// The square of `a` as a `BigUint` with `2 * a.limbs.len()` limbs
pub fn textbook_square(builder: &CircuitBuilder, a: &BigUint) -> BigUint {
    let mut accumulator = vec![vec![]; a.limbs.len() + a.limbs.len()];
    for (i, &ai) in a.limbs.iter().enumerate() {
        for (j, &aj) in a.limbs.iter().enumerate().skip(i) {
            let (hi, lo) = builder.imul(ai, aj);
            accumulator[i + j].push(lo);
            accumulator[i + j + 1].push(hi);
            if i != j {
                // Off-diagonal elements appear twice
                accumulator[i + j].push(lo);
                accumulator[i + j + 1].push(hi);
            }
        }
    }
    compute_stack_adds(builder, &accumulator)
}

/// BigUint size at which Karatsuba becomes better than textbook multiplication.
const KARATSUBA_LIMBS_THRESHOLD: usize = 8;

/// Multiply two arbitrary-sized `BigUint`s using textbook algorithm.
///
/// This method attempts to pick the most efficient multiplication algorithm.
///
/// Computes `a * b` where both inputs are `BigUint`s. The result will have
/// `a.limbs.len() + b.limbs.len()` limbs to accommodate the full product
/// without overflow.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand `BigUint`
/// * `b` - Second operand `BigUint`
///
/// # Returns
/// Product `BigUint` with `a.limbs.len() + b.limbs.len()` limbs
pub fn optimal_mul(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> BigUint {
    let n = a.limbs.len();
    if n == b.limbs.len() && n.is_power_of_two() && n >= KARATSUBA_LIMBS_THRESHOLD {
        karatsuba_mul(builder, a, b)
    } else {
        textbook_mul(builder, a, b)
    }
}

/// Square an arbitrary-sized `BigUint`.
///
/// This method attempts to pick the most efficient multiplication algorithm.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - The `BigUint` to be squared
///
/// # Returns
/// The square of `a` as a `BigUint` with `2 * a.limbs.len()` limbs
pub fn optimal_sqr(builder: &CircuitBuilder, a: &BigUint) -> BigUint {
    let n = a.limbs.len();
    if n.is_power_of_two() && n >= KARATSUBA_LIMBS_THRESHOLD {
        karatsuba_mul(builder, a, a)
    } else {
        textbook_square(builder, a)
    }
}

/// Multiply two `BigUint`s with po2 number of limbs using Karatsuba (aka Toom-22).
///
/// Whereas `textbook_mul` and `textbook_square` require $O(n^2)$ constraints for $n$ limbs,
/// this method is asymptotically more efficient with $O(n^{log_2 3}) = O(n^{1.58})$,
/// however due to larger constant factor it's beneficial for longer `BigUint`s only.
///
/// Computes `a * b` where both inputs are `BigUint`s. The result will have
/// `a.limbs.len() + b.limbs.len()` limbs to accommodate the full product
/// without overflow.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand `BigUint`
/// * `b` - Second operand `BigUint`
///
/// # Returns
/// Product `BigUint` with `a.limbs.len() + b.limbs.len()` limbs
pub fn karatsuba_mul(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> BigUint {
    let n = a.limbs.len();

    // Preconditions
    assert!(n.is_power_of_two());
    assert_eq!(b.limbs.len(), n);

    if n == 1 {
        // Base case
        let (hi, lo) = builder.imul(a.limbs[0], b.limbs[0]);
        return BigUint {
            limbs: vec![lo, hi],
        };
    }

    // a(t) = a_0 + t * a_∞
    // b(t) = b_0 + t * b_∞
    // for t = 2^(WORD_SIZE_BITS*n/2), a = a(t), b = b(t)
    let n_half = n >> 1;
    let (a_0, a_inf) = a.clone().split_at_limbs(n_half);
    let (b_0, b_inf) = b.clone().split_at_limbs(n_half);

    // Recursively multiply (a_0, b_0) and (a_∞, b_∞)
    let ab_0 = karatsuba_mul(builder, &a_0, &b_0);
    let ab_inf = karatsuba_mul(builder, &a_inf, &b_inf);

    // Compute a(1) = a_0 + a_∞, b(1) = b_0 + b_∞, split into 2*n limbs and carry out bit
    let (a_1, a_1_carry_out) = add_with_carry_out(builder, &a_0, &a_inf);
    let (b_1, b_1_carry_out) = add_with_carry_out(builder, &b_0, &b_inf);

    // Recursively multiply a(1) and b(1), _without_ carry out bits
    let ab_1 = karatsuba_mul(builder, &a_1, &b_1);

    // Multiply a(1) and b(1) carry out bits only
    let ab_1_carry_out = builder.shr(
        builder.band(a_1_carry_out, b_1_carry_out),
        (WORD_SIZE_BITS - 1) as u32,
    );

    // Addition chain for overflow_sum = ab_0 + ab_inf * 2^n + ab_1 * 2^n/2
    let mut product_stacks = vec![Vec::new(); 2 * n];
    // 0 ... n/2 ... n ... 3n/2 ... 2n
    // [    ab_0     )
    //               [    ab_inf     )
    //       [    ab_1      )
    //               [ a_1  )     -- if b_1 carry out not zero
    //               [ b_1  )     -- if a_1 carry out not zero
    //                      ab_1_carry_out
    add_to_stacks(&mut product_stacks[0..], &ab_0);
    add_to_stacks(&mut product_stacks[n..], &ab_inf);
    add_to_stacks(&mut product_stacks[n_half..], &ab_1);
    add_to_stacks(
        &mut product_stacks[n..],
        &a_1.zero_unless(builder, b_1_carry_out),
    );
    add_to_stacks(
        &mut product_stacks[n..],
        &b_1.zero_unless(builder, a_1_carry_out),
    );
    add_to_stacks(
        &mut product_stacks[n + n_half..],
        &BigUint {
            limbs: [ab_1_carry_out].to_vec(),
        },
    );
    let (mut overflow_sum, carry_outs) =
        compute_stack_adds_with_carry_outs(builder, &product_stacks);

    // overflow_sum needs 2n+1 limbs, highest limb is 0 or 1, so we can bit-or them together
    let carry_out = carry_outs
        .into_iter()
        .reduce(|lhs, rhs| builder.bor(lhs, rhs))
        .expect("at least one carry out");
    overflow_sum
        .limbs
        .push(builder.shr(carry_out, (WORD_SIZE_BITS - 1) as u32));

    // now we need to subtract (ab_0 + ab_inf) * 2^n/2, separate the lower limbs that won't get
    // affected
    let (ab_lo_invariant, ab_hi_overflow) = overflow_sum.split_at_limbs(n_half);
    let remainder_len = ab_hi_overflow.limbs.len();
    let ab_0_padded = ab_0.zero_extend(builder, remainder_len);
    let ab_inf_padded = ab_inf.zero_extend(builder, remainder_len);

    let ab_hi_overflow = sub(
        builder,
        &sub(builder, &ab_hi_overflow, &ab_0_padded),
        &ab_inf_padded,
    );
    let (ab_hi, overflow_limb) = ab_hi_overflow.split_at_limbs(2 * n - n_half);

    // The extra limb of overflow_limb should be zero, because product fits into 2n limbs
    assert_eq!(overflow_limb.limbs.len(), 1);
    builder.assert_zero("karatsuba_mul_overflow_limb", overflow_limb.limbs[0]);

    ab_lo_invariant.concat_limbs(&ab_hi)
}

fn add_to_stacks(limb_stacks: &mut [Vec<Wire>], a: &BigUint) {
    assert!(limb_stacks.len() >= a.limbs.len());
    for (limb_stack, &limb) in iter::zip(limb_stacks, &a.limbs) {
        limb_stack.push(limb);
    }
}
