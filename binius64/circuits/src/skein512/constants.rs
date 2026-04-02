// Copyright 2025 Irreducible Inc.
// Type codes from Skein specification Table 6:
pub const TWEAK_TYPE_CFG: u64 = 4;
pub const TWEAK_TYPE_MSG: u64 = 48;
pub const TWEAK_TYPE_OUT: u64 = 63;

// Skein/Threefish-512 uses 72 rounds, with rotation schedule repeating every 8 rounds.
pub const R512: [[u32; 4]; 8] = [
    [46, 36, 19, 37],
    [33, 27, 14, 42],
    [17, 49, 36, 39],
    [44, 9, 54, 56],
    [39, 30, 34, 24],
    [13, 50, 10, 17],
    [25, 29, 39, 43],
    [8, 35, 56, 22],
];

// Parity constant for the extended key u64 k[8] = C240 ^ sum(k0..k7)
pub const C240: u64 = 0x1BD11BDAA9FC1A22;
