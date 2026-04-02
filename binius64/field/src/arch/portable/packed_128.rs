// Copyright 2024-2025 Irreducible Inc.

use crate::arch::{
    BitwiseAndStrategy,
    portable::packed_macros::{portable_macros::*, *},
};

pub type M128 = u128;

define_packed_binary_fields!(
    underlier: M128,
    packed_fields: [
        packed_field {
            name: PackedBinaryField128x1b,
            scalar: BinaryField1b,
            mul: (BitwiseAndStrategy),
            square: (BitwiseAndStrategy),
            invert: (BitwiseAndStrategy),
            mul_alpha: (BitwiseAndStrategy),
            transform: (PackedStrategy),
        },
    ]
);
