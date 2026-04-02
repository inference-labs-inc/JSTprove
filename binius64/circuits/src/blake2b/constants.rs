// Copyright 2025 Irreducible Inc.
/// BLAKE2b constants
/// Initialization vectors from SHA-512 (fractional parts of square roots of primes 2..19)
pub const IV: [u64; 8] = [
    0x6A09E667F3BCC908, // sqrt(2)
    0xBB67AE8584CAA73B, // sqrt(3)
    0x3C6EF372FE94F82B, // sqrt(5)
    0xA54FF53A5F1D36F1, // sqrt(7)
    0x510E527FADE682D1, // sqrt(11)
    0x9B05688C2B3E6C1F, // sqrt(13)
    0x1F83D9ABFB41BD6B, // sqrt(17)
    0x5BE0CD19137E2179, // sqrt(19)
];

/// SIGMA permutation schedule for message words
/// 12 rounds total: rounds 0-9 have unique permutations, rounds 10-11 reuse 0-1
pub const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    // Rounds 10-11 reuse SIGMA[0] and SIGMA[1]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// Rotation constants for the G mixing function
pub const R1: u32 = 32;
pub const R2: u32 = 24;
pub const R3: u32 = 16;
pub const R4: u32 = 63;

/// Algorithm parameters
pub const ROUNDS: usize = 12;
pub const BLOCK_BYTES: usize = 128;
pub const STATE_WORDS: usize = 8;
pub const MAX_OUTPUT_BYTES: usize = 64;
