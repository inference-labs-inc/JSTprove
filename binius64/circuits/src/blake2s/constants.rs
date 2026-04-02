// Copyright 2025 Irreducible Inc.
//! Blake2s constants
//!
//! This module contains the constants used in the Blake2s hash function,
//! including initialization vectors and the message permutation schedule.

/// Blake2s initialization vectors derived from the fractional parts of
/// square roots of the first 8 prime numbers (2, 3, 5, 7, 11, 13, 17, 19).
///
/// These values are specified in RFC 7693 Section 2.6 and provide the
/// initial state for the hash computation. They ensure that even empty
/// messages produce non-zero hashes.
pub const IV: [u32; 8] = [
    0x6A09E667, // Frac(sqrt(2)) - first 32 bits
    0xBB67AE85, // Frac(sqrt(3))
    0x3C6EF372, // Frac(sqrt(5))
    0xA54FF53A, // Frac(sqrt(7))
    0x510E527F, // Frac(sqrt(11))
    0x9B05688C, // Frac(sqrt(13))
    0x1F83D9AB, // Frac(sqrt(17))
    0x5BE0CD19, // Frac(sqrt(19))
];

/// SIGMA permutation schedule for message word selection.
///
/// Defines which message words are used in each G-function call during
/// the 10 rounds of compression. This permutation ensures that all message
/// words influence the output multiple times in different combinations,
/// providing cryptographic diffusion as specified in RFC 7693 Section 3.1.
///
/// Each row represents one round, with 16 indices selecting message words
/// for the 8 G-function calls (2 words per G-function).
pub const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], // Round 0
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3], // Round 1
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4], // Round 2
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8], // Round 3
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13], // Round 4
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9], // Round 5
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11], // Round 6
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10], // Round 7
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5], // Round 8
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0], // Round 9
];
