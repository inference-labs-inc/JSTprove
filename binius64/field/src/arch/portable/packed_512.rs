// Copyright 2024-2025 Irreducible Inc.

use crate::{
    arch::{
        M256,
        portable::packed_macros::{portable_macros::*, *},
        strategies::ScaledStrategy,
    },
    underlier::ScaledUnderlier,
};

pub type M512 = ScaledUnderlier<M256, 2>;

define_packed_binary_fields!(
    underlier: M512,
    packed_fields: [
        packed_field {
            name: PackedBinaryField512x1b,
            scalar: BinaryField1b,
            mul:       (ScaledStrategy),
            square:    (ScaledStrategy),
            invert:    (ScaledStrategy),
            mul_alpha: (ScaledStrategy),
            transform: (ScaledStrategy),
        },
    ]
);
