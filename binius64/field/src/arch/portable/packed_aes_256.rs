// Copyright 2024-2025 Irreducible Inc.

use super::{
    packed_256::M256,
    packed_macros::{portable_macros::*, *},
};
use crate::arch::strategies::ScaledStrategy;

define_packed_binary_fields!(
    underlier: M256,
    packed_fields: [
        packed_field {
            name: PackedAESBinaryField32x8b,
            scalar: AESTowerField8b,
            mul:       (ScaledStrategy),
            square:    (ScaledStrategy),
            invert:    (ScaledStrategy),
            mul_alpha: (ScaledStrategy),
            transform: (ScaledStrategy),
        },
    ]
);
