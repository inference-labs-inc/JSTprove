// Copyright 2025 Irreducible Inc.
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
pub mod aarch64;
pub mod portable64;
