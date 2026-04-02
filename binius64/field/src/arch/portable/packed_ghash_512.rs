// Copyright 2024-2025 Irreducible Inc.

use super::{
    packed_512::M512,
    packed_macros::{portable_macros::*, *},
};
use crate::arch::strategies::ScaledStrategy;

define_packed_binary_fields!(
    underlier: M512,
    packed_fields: [
        packed_field {
            name: PackedBinaryGhash4x128b,
            scalar: BinaryField128bGhash,
            mul:       (ScaledStrategy),
            square:    (ScaledStrategy),
            invert:    (ScaledStrategy),
            mul_alpha: (None),
            transform: (None),
        },
    ]
);
