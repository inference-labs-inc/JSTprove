// Copyright 2025 Irreducible Inc.
//! Protocol-level constants for the Binius64 constraint system.

/// Computes the base-2 logarithm of a value, panicking if it's not a power of 2.
///
/// We redefine this function here instead of using the one from binius-utils
/// to avoid bringing in unnecessary dependencies and cruft.
const fn checked_log_2(val: usize) -> usize {
    assert!(val.is_power_of_two(), "Value is not a power of 2");
    val.ilog2() as usize
}

/// The protocol proves constraint systems over 64-bit words.
pub const WORD_SIZE_BYTES: usize = 8;

/// The protocol proves constraint systems over 64-bit words.
pub const WORD_SIZE_BITS: usize = WORD_SIZE_BYTES * 8;

/// log2 of [`WORD_SIZE_BITS`].
pub const LOG_WORD_SIZE_BITS: usize = checked_log_2(WORD_SIZE_BITS);

/// The minimum number of words per segment.
///
/// This is the minimum size requirement for public input segments in the constraint system.
pub const MIN_WORDS_PER_SEGMENT: usize = 2;

/// Minimum number of AND constraints (must be a power of 2).
/// This is a protocol requirement for the constraint system.
pub const MIN_AND_CONSTRAINTS: usize = 8;

/// Minimum number of MUL constraints (must be a power of 2).
/// This is a protocol requirement for the constraint system.
pub const MIN_MUL_CONSTRAINTS: usize = 1;

/// The number of bits in a byte.
pub const BYTE_BITS: usize = 8;

/// log2 of [`BYTE_BITS`].
pub const LOG_BYTE_BITS: usize = checked_log_2(BYTE_BITS);
