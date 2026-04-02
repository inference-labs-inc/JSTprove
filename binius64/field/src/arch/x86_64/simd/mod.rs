// Copyright 2024-2025 Irreducible Inc.

pub mod simd_arithmetic;

#[cfg(target_feature = "sse2")]
mod m128;
#[cfg(target_feature = "avx2")]
mod m256;
#[cfg(target_feature = "avx512f")]
mod m512;
