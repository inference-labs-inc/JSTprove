// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use super::m256::M256;
use crate::{
    arch::portable::packed_macros::{portable_macros::*, *},
    arithmetic_traits::{impl_invert_with, impl_mul_alpha_with, impl_mul_with, impl_square_with},
};

define_packed_binary_fields!(
    underlier: M256,
    packed_fields: [
        packed_field {
            name: PackedAESBinaryField32x8b,
            scalar: AESTowerField8b,
            mul:       (if gfni GfniStrategy else PairwiseTableStrategy),
            square:    (if gfni ReuseMultiplyStrategy else PairwiseTableStrategy),
            invert:    (if gfni GfniStrategy else PairwiseTableStrategy),
            mul_alpha: (if gfni ReuseMultiplyStrategy else PairwiseTableStrategy),
            transform: (if gfni GfniStrategy else SimdStrategy),
        },
    ]
);
