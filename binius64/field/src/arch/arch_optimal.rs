// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

use crate::{BinaryField1b, PackedSubfield};

cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        cfg_if! {
            if #[cfg(target_feature = "avx512f")] {
                pub type OptimalPackedB128 = crate::PackedBinaryGhash4x128b;

            } else if #[cfg(target_feature = "avx2")] {
                pub type OptimalPackedB128 = crate::PackedBinaryGhash2x128b;

            } else if #[cfg(target_feature = "sse2")] {
                pub type OptimalPackedB128 = crate::PackedBinaryGhash1x128b;
            }
        }
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon", target_feature = "aes"))] {
        pub type OptimalPackedB128 = crate::PackedBinaryGhash1x128b;

    } else {
        pub type OptimalPackedB128 = crate::PackedBinaryGhash1x128b;
    }
}

pub type OptimalB128 = crate::BinaryField128bGhash;
pub type OptimalPackedB1 = PackedSubfield<OptimalPackedB128, BinaryField1b>;
