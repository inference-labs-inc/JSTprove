// Copyright 2024-2025 Irreducible Inc.

use core::arch::x86_64::*;

use gfni_arithmetics::GfniType;

use super::*;
use crate::arch::x86_64::m256::M256;

impl GfniType for M256 {
    #[inline(always)]
    fn gf2p8affine_epi64_epi8(x: Self, a: Self) -> Self {
        unsafe { _mm256_gf2p8affine_epi64_epi8::<0>(x.0, a.0) }.into()
    }

    #[inline(always)]
    fn gf2p8mul_epi8(a: Self, b: Self) -> Self {
        unsafe { _mm256_gf2p8mul_epi8(a.0, b.0) }.into()
    }

    #[inline(always)]
    fn gf2p8affineinv_epi64_epi8(x: Self, a: Self) -> Self {
        unsafe { _mm256_gf2p8affineinv_epi64_epi8::<0>(x.0, a.0) }.into()
    }
}
