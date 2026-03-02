use std::ops::{Add, Mul};

use arith::ExtensionField;
use gf2::GF2x8;

use crate::GF2_128;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
pub type GF2_128x8 = neon::NeonGF2_128x8;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type GF2_128x8 = avx512::AVX512GF2_128x8;

// Fallback, use avx2
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod avx256;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type GF2_128x8 = avx256::AVX256GF2_128x8;

impl Ord for GF2_128x8 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for GF2_128x8 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for GF2_128x8 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for GF2_128x8 is not supported")
    }
}

impl Mul<GF2_128> for GF2_128x8 {
    type Output = GF2_128x8;

    #[inline(always)]
    fn mul(self, rhs: GF2_128) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl Add<GF2x8> for GF2_128x8 {
    type Output = GF2_128x8;

    #[inline(always)]
    fn add(self, rhs: GF2x8) -> Self::Output {
        self.add_by_base_field(&rhs)
    }
}
