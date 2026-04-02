// Copyright 2025 Irreducible Inc.

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(target_feature = "simd128")] {
        mod m128;
        pub mod packed_ghash_128;

        pub use super::portable::{packed_ghash_256};
    } else {
        pub use super::portable::{packed_ghash_128, packed_ghash_256};
    }
}
