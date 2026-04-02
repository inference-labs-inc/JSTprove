// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use crate::{
    arch::{
        PairwiseTableStrategy,
        portable::packed_macros::{portable_macros::*, *},
    },
    arithmetic_traits::{impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with},
};

define_packed_binary_fields!(
    underlier: u64,
    packed_fields: [
        packed_field {
            name: PackedAESBinaryField8x8b,
            scalar: AESTowerField8b,
            mul:       (if gfni_x86 PackedAESBinaryField16x8b else PairwiseTableStrategy),
            square:    (if gfni_x86 PackedAESBinaryField16x8b else PairwiseTableStrategy),
            invert:    (if gfni_x86 PackedAESBinaryField16x8b else PairwiseTableStrategy),
            mul_alpha: (PairwiseTableStrategy),
            transform: (PackedStrategy),
        },
    ]
);
