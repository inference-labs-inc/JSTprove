// Copyright 2024-2025 Irreducible Inc.

use super::m128::M128;
use crate::arch::{
    BitwiseAndStrategy,
    portable::packed_macros::{portable_macros::*, *},
};

define_packed_binary_fields!(
    underlier: M128,
    packed_fields: [
        packed_field {
            name: PackedBinaryField128x1b,
            scalar: BinaryField1b,
            mul:       (BitwiseAndStrategy),
            square:    (BitwiseAndStrategy),
            invert:    (BitwiseAndStrategy),
            mul_alpha: (BitwiseAndStrategy),
            transform: (None),
        },
    ]
);
