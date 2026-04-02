// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

use super::biguint::BigUint;

/// Add two equally-sized `BigUints`s with carry propagation.
///
/// See `add_with_carry_out`; this routine additionally asserts that no overflow happens.
pub fn add(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> BigUint {
    let (sum, carry_out) = add_with_carry_out(builder, a, b);
    // Assert that no overflow happened.
    builder.assert_false("add_carry_out_zero", carry_out);
    sum
}

/// Add two equally-sized `BigUints`s with carry propagation.
///
/// Computes `a + b` with proper carry handling between limbs. The result
/// has the same number of limbs as the inputs. Overflow beyond the most
/// significant limb is checked and must be zero.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - First operand
/// * `b` - Second operand (must have same number of limbs as `a`)
///
/// # Returns
/// A tuple of `BigUint` sum of size equal to the inputs and a carry out boolean wire.
///
/// # Panics
/// - Panics if `a` and `b` have different number of limbs
pub fn add_with_carry_out(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> (BigUint, Wire) {
    assert_eq!(
        a.limbs.len(),
        b.limbs.len(),
        "add_with_carry_out: inputs must have the same number of limbs"
    );

    let mut accumulator = vec![vec![]; a.limbs.len()];
    for i in 0..a.limbs.len() {
        accumulator[i].push(a.limbs[i]);
        accumulator[i].push(b.limbs[i]);
    }

    let (sum, carry_out) = compute_stack_adds_with_carry_outs(builder, &accumulator);
    assert_eq!(carry_out.len(), if a.limbs.is_empty() { 0 } else { 1 });
    (
        sum,
        carry_out
            .first()
            .copied()
            .unwrap_or(builder.add_constant(Word::ZERO)),
    )
}

/// Subtracts two equally-sized `BigUints`s with carry propagation.
///
/// Computes `a - b` with proper borrow handling between limbs. The result
/// has the same number of limbs as the inputs. Underflow beyond the most
/// significant limb is checked and must be zero. This implies that `a >= b`
/// and the difference remains unsigned.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `a` - minuend
/// * `b` - subtrahend (must have same number of limbs as `a`)
///
/// # Returns
/// Difference as a `BigUint` with the same number of limbs as the inputs
///
/// # Panics
/// - Panics if `a` and `b` have different number of limbs
pub fn sub(builder: &CircuitBuilder, a: &BigUint, b: &BigUint) -> BigUint {
    assert_eq!(
        a.limbs.len(),
        b.limbs.len(),
        "sub: inputs must have the same number of limbs"
    );

    let zero = builder.add_constant(Word::ZERO);

    let mut diff_limbs = Vec::with_capacity(a.limbs.len());

    let mut borrow_in = zero;
    for (&a_limb, &b_limb) in a.limbs.iter().zip(&b.limbs) {
        let (diff_limb, borrow_out) = builder.isub_bin_bout(a_limb, b_limb, borrow_in);
        diff_limbs.push(diff_limb);

        borrow_in = borrow_out;
    }

    // Assert the final borrow is zero (i.e no underflow).
    //
    // It requires checking the MSB of the `borrow_in`
    let borrow_out_msb = builder.shr(borrow_in, 63);
    builder.assert_eq("sub_borrow_out", borrow_out_msb, zero);

    BigUint { limbs: diff_limbs }
}

/// Computes multi-operand addition with carry propagation across limb positions.
///
/// See `compute_stack_adds_with_carry_outs`; this routine additionally asserts that no
/// carry out happens.
pub(super) fn compute_stack_adds(builder: &CircuitBuilder, limb_stacks: &[Vec<Wire>]) -> BigUint {
    let (sum, carry_outs) = compute_stack_adds_with_carry_outs(builder, limb_stacks);

    // Assert all final carries are zero (i.e no overflow).
    //
    // It is sufficient to check the MSB of each wire in `carries` because:
    //
    // - The `carries` vector stores carry_out from each iadd_cin_cout gate.
    // - The carry bit for each addition is stored in the MSB of the carry_out wire.
    for (i, carry_out) in carry_outs.into_iter().enumerate() {
        builder.assert_false(format!("compute_stack_adds_carry_zero_{i}"), carry_out);
    }

    sum
}

/// Computes multi-operand addition with carry propagation across limb positions.
///
/// This function is the core of bignum arithmetic, handling the addition of multiple
/// values at each limb position with proper carry propagation to higher limbs.
/// It's used by other bignum operations to resolve partial products and sums.
///
/// # Arguments
/// * `builder` - Circuit builder for constraint generation
/// * `limb_stacks` - Array where `limb_stacks[i]` contains all values to be added at limb position
///   `i`.
///
/// # Returns
/// The sum produced by the addition chain, as well as a number of boolean wires representing carry
/// outs in the most significant limb; converting these booleans to 0/1 integers and summing
/// produces the value of overflow limb.
pub(super) fn compute_stack_adds_with_carry_outs(
    builder: &CircuitBuilder,
    limb_stacks: &[Vec<Wire>],
) -> (BigUint, Vec<Wire>) {
    let mut sums = Vec::new();
    let mut carries = Vec::new();
    let zero = builder.add_constant(Word::ZERO);

    for limb_stack in limb_stacks {
        let mut limb_stack = limb_stack.clone();
        let mut new_carries = Vec::new();

        // Pad stack to handle incoming carries
        if limb_stack.len() < carries.len() + 1 {
            limb_stack.resize(carries.len() + 1, zero);
        }

        while limb_stack.len() >= 2 {
            let carry_in = carries.pop().unwrap_or(zero);
            let x = limb_stack.pop().expect("limb_stack.len() >= 2");
            let y = limb_stack.pop().expect("limb_stack.len() >= 2");

            let (sum, cout) = builder.iadd_cin_cout(x, y, carry_in);
            limb_stack.push(sum);
            new_carries.push(cout);
        }

        sums.push(limb_stack[0]);
        assert!(limb_stack.len() == 1);
        assert!(carries.is_empty());
        carries = new_carries;
    }

    (BigUint { limbs: sums }, carries)
}
