// A BabyBearx16 stores 512 bits of data.
// With AVX it stores a single __m512i element.
// With NEON it stores four uint32x4_t elements.

use arith::FFTField;

#[cfg(target_arch = "aarch64")]
mod babybear_neon;
#[cfg(target_arch = "aarch64")]
pub type BabyBearx16 = babybear_neon::NeonBabyBear;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod babybear_avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type BabyBearx16 = babybear_avx512::AVXBabyBear;

// Fallback, use avx2
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod babybear_avx256;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type BabyBearx16 = babybear_avx256::AVXBabyBear;

impl Ord for BabyBearx16 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for BabyBearx16 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for BabyBearx16 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for BabyBearx16 is not supported")
    }
}

impl FFTField for BabyBearx16 {
    const TWO_ADICITY: usize = 27;

    fn root_of_unity() -> Self {
        Self::from(0x1a427a41)
    }
}
