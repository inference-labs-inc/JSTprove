// Copyright 2024-2025 Irreducible Inc.

use crate::{
    arch::{
        PairwiseTableStrategy,
        portable::packed_macros::{portable_macros::*, *},
    },
    arithmetic_traits::{impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with},
};

define_packed_binary_fields!(
    underlier: u16,
    packed_fields: [
        packed_field {
            name: PackedAESBinaryField2x8b,
            scalar: AESTowerField8b,
            mul: (PairwiseTableStrategy),
            square: (PairwiseTableStrategy),
            invert: (PairwiseTableStrategy),
            mul_alpha: (PairwiseTableStrategy),
            transform: (PackedStrategy),
        },
    ]
);
