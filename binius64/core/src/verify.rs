// Copyright 2025 Irreducible Inc.
//! Routines for checking whether the
//! [constraint system][`crate::constraint_system::ConstraintSystem`] is satisfied with the given
//! [value vector][`ValueVec`].

use crate::{
    constraint_system::{
        AndConstraint, ConstraintSystem, MulConstraint, ShiftVariant, ShiftedValueIndex, ValueVec,
    },
    word::Word,
};

/// Evaluates a shifted value given a word
#[inline]
pub fn eval_shifted_word(word: Word, shift_variant: ShiftVariant, amount: usize) -> Word {
    match shift_variant {
        ShiftVariant::Sll => word << (amount as u32),
        ShiftVariant::Slr => word >> (amount as u32),
        ShiftVariant::Sar => word.sar(amount as u32),
        ShiftVariant::Rotr => word.rotr(amount as u32),
        ShiftVariant::Sll32 => word.sll32(amount as u32),
        ShiftVariant::Srl32 => word.srl32(amount as u32),
        ShiftVariant::Sra32 => word.sra32(amount as u32),
        ShiftVariant::Rotr32 => word.rotr32(amount as u32),
    }
}

/// Evaluates an operand (XOR of shifted values) using a ValueVec
pub fn eval_operand(witness: &ValueVec, operand: &[ShiftedValueIndex]) -> Word {
    operand.iter().fold(Word::ZERO, |acc, sv| {
        let word = witness[sv.value_index];
        let shifted_word = eval_shifted_word(word, sv.shift_variant, sv.amount);
        acc ^ shifted_word
    })
}

/// Verifies that an AND constraint is satisfied: (A & B) ^ C = 0
pub fn verify_and_constraint(witness: &ValueVec, constraint: &AndConstraint) -> Result<(), String> {
    let Word(a) = eval_operand(witness, &constraint.a);
    let Word(b) = eval_operand(witness, &constraint.b);
    let Word(c) = eval_operand(witness, &constraint.c);

    let result = (a & b) ^ c;
    if result != 0 {
        Err(format!(
            "AND constraint failed: ({a:016x} & {b:016x}) ^ {c:016x} = {result:016x} (expected 0)",
        ))
    } else {
        Ok(())
    }
}

/// Verifies that a MUL constraint is satisfied: A * B = (HI << 64) | LO
pub fn verify_mul_constraint(witness: &ValueVec, constraint: &MulConstraint) -> Result<(), String> {
    let Word(a) = eval_operand(witness, &constraint.a);
    let Word(b) = eval_operand(witness, &constraint.b);
    let Word(lo) = eval_operand(witness, &constraint.lo);
    let Word(hi) = eval_operand(witness, &constraint.hi);

    let a_val = a as u128;
    let b_val = b as u128;
    let product = a_val * b_val;

    let expected_lo = (product & 0xFFFFFFFFFFFFFFFF) as u64;
    let expected_hi = (product >> 64) as u64;

    if lo != expected_lo || hi != expected_hi {
        Err(format!(
            "MUL constraint failed: {a:016x} * {b:016x} = {hi:016x}{lo:016x} (expected {expected_hi:016x}{expected_lo:016x})",
        ))
    } else {
        Ok(())
    }
}

/// Verifies all constraints in a constraint system are satisfied by the witness
pub fn verify_constraints(cs: &ConstraintSystem, witness: &ValueVec) -> Result<(), String> {
    cs.value_vec_layout
        .validate()
        .map_err(|e| format!("ValueVec layout validation failed: {e}"))?;

    // First check that the witness correctly populated the constants section.
    for (index, constant) in cs.constants.iter().enumerate() {
        if witness.get(index) != *constant {
            return Err(format!(
                "Constant at index {index} does not match expected value {:016x} in value vec",
                constant.as_u64()
            ));
        }
    }
    for (i, constraint) in cs.and_constraints.iter().enumerate() {
        verify_and_constraint(witness, constraint)
            .map_err(|e| format!("AND constraint {i} failed: {e}"))?;
    }
    for (i, constraint) in cs.mul_constraints.iter().enumerate() {
        verify_mul_constraint(witness, constraint)
            .map_err(|e| format!("MUL constraint {i} failed: {e}"))?;
    }
    Ok(())
}
