// Copyright 2023-2025 Irreducible Inc.

use cfg_if::cfg_if;

#[cfg(target_feature = "gfni")]
mod gfni;

mod simd;

cfg_if! {
    if #[cfg(target_feature = "sse2")] {
        pub(super) mod m128;
        pub mod packed_128;
        pub mod packed_aes_128;
        pub mod packed_ghash_128;
    } else {
        pub use super::portable::packed_128;
        pub use super::portable::packed_aes_128;
        pub use super::portable::packed_ghash_128;
    }
}

cfg_if! {
    if #[cfg(target_feature = "avx2")] {
        pub use m256::M256;
        pub(super) mod m256;
        pub mod packed_256;
        pub mod packed_aes_256;
        pub mod packed_ghash_256;
    } else {
        pub use super::portable::packed_256::{self, M256};
        pub use super::portable::packed_aes_256;
        pub use super::portable::packed_ghash_256;
    }
}

cfg_if! {
    if #[cfg(target_feature = "avx512f")] {
        pub(super) mod m512;
        pub mod packed_512;
        pub mod packed_aes_512;
        pub mod packed_ghash_512;
    } else {
        pub use super::portable::packed_512;
        pub use super::portable::packed_aes_512;
        pub use super::portable::packed_ghash_512;
    }
}
