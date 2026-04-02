// Copyright 2025 Irreducible Inc.
//! Hosts error definitions for the core crate.

use crate::consts::MIN_WORDS_PER_SEGMENT;

/// Constraint system related error.
#[allow(missing_docs)] // errors are self-documenting
#[derive(Debug, thiserror::Error)]
pub enum ConstraintSystemError {
    #[error("the total length of the value vector must be a power of two")]
    ValueVecLenNotPowerOfTwo,
    #[error("the public input segment must have power of two length")]
    PublicInputPowerOfTwo,
    #[error(
        "the public input segment must be at least {MIN_WORDS_PER_SEGMENT} words, got: {pub_input_size}"
    )]
    PublicInputTooShort { pub_input_size: usize },
    #[error("the data length doesn't match layout. Expected: {expected}, Actual: {actual}")]
    ValueVecLenMismatch { expected: usize, actual: usize },
    #[error(
        "{constraint_type} #{constraint_index} uses non canonical shift in its {operand_name} operand"
    )]
    NonCanonicalShift {
        constraint_type: &'static str,
        constraint_index: usize,
        operand_name: &'static str,
    },
    #[error(
        "{constraint_type} #{constraint_index} refers to padding in its {operand_name} operand"
    )]
    PaddingValueIndex {
        constraint_type: &'static str,
        operand_name: &'static str,
        constraint_index: usize,
    },
    #[error(
        "{constraint_type} #{constraint_index} uses shift amount n={shift_amount}>=64 {operand_name} operand"
    )]
    ShiftAmountTooLarge {
        constraint_type: &'static str,
        constraint_index: usize,
        operand_name: &'static str,
        shift_amount: usize,
    },
    #[error(
        "{constraint_type} #{constraint_index} refers to out-of-range value index in {operand_name} operand (index {value_index} >= total length {total_len})"
    )]
    OutOfRangeValueIndex {
        constraint_type: &'static str,
        constraint_index: usize,
        operand_name: &'static str,
        value_index: u32,
        total_len: usize,
    },
}
