// Copyright 2024-2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::ops::Mul;

use super::{
    m128::M128,
    simd_arithmetic::{
        packed_aes_16x8b_invert_or_zero, packed_aes_16x8b_mul_alpha, packed_aes_16x8b_multiply,
        packed_aes_16x8b_square,
    },
};
use crate::{
    aes_field::AESTowerField8b,
    arch::portable::packed::PackedPrimitiveType,
    arithmetic_traits::{InvertOrZero, MulAlpha, Square},
    underlier::WithUnderlier,
};

pub type PackedAESBinaryField16x8b = PackedPrimitiveType<M128, AESTowerField8b>;

impl Mul for PackedAESBinaryField16x8b {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        crate::tracing::trace_multiplication!(PackedAESBinaryField16x8b);

        self.mutate_underlier(|underlier| packed_aes_16x8b_multiply(underlier, rhs.to_underlier()))
    }
}

impl Square for PackedAESBinaryField16x8b {
    fn square(self) -> Self {
        self.mutate_underlier(packed_aes_16x8b_square)
    }
}

impl InvertOrZero for PackedAESBinaryField16x8b {
    fn invert_or_zero(self) -> Self {
        self.mutate_underlier(packed_aes_16x8b_invert_or_zero)
    }
}

impl MulAlpha for PackedAESBinaryField16x8b {
    fn mul_alpha(self) -> Self {
        self.mutate_underlier(packed_aes_16x8b_mul_alpha)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::packed::PackedField;

    proptest! {
        #[test]
        fn test_square_equals_self_mul_self(a_val in any::<u128>()) {
            let a = PackedAESBinaryField16x8b::from_underlier(a_val.into());

            let squared = Square::square(a);

            for i in 0..PackedAESBinaryField16x8b::WIDTH {
                assert_eq!(squared.get(i), a.get(i) * a.get(i));
            }
        }
    }
}
