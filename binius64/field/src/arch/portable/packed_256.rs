// Copyright 2023-2025 Irreducible Inc.

use crate::{
    arch::{
        M128,
        portable::packed_macros::{portable_macros::*, *},
        strategies::ScaledStrategy,
    },
    underlier::ScaledUnderlier,
};

pub type M256 = ScaledUnderlier<M128, 2>;

define_packed_binary_fields!(
    underlier: M256,
    packed_fields: [
        packed_field {
            name: PackedBinaryField256x1b,
            scalar: BinaryField1b,
            mul:       (ScaledStrategy),
            square:    (ScaledStrategy),
            invert:    (ScaledStrategy),
            mul_alpha: (ScaledStrategy),
            transform: (ScaledStrategy),
        },
    ]
);
