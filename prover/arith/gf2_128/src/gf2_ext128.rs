use std::ops::Mul;

use arith::ExtensionField;
use gf2::GF2x8;

use crate::GF2_128x8;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
pub type GF2_128 = neon::NeonGF2_128;

#[cfg(target_arch = "x86_64")]
mod avx;
#[cfg(target_arch = "x86_64")]
pub type GF2_128 = avx::AVXGF2_128;

impl Ord for GF2_128 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for GF2_128 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for GF2_128 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for GF2_128 is not supported")
    }
}

impl Mul<GF2x8> for GF2_128 {
    type Output = GF2_128x8;

    #[inline(always)]
    fn mul(self, rhs: GF2x8) -> Self::Output {
        GF2_128x8::from(self).mul_by_base_field(&rhs)
    }
}
