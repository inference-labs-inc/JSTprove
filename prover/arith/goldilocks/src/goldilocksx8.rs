// A Goldilocksx8 stores 512 bits of data.
// With AVX it stores a single __m512i element.
// With NEON it stores 8 u64 elements.
// NEON doesn't use and arm instructions and can be used as a fallback for other architectures.

#[cfg(target_arch = "aarch64")]
mod goldilocks_neon;
#[cfg(target_arch = "aarch64")]
pub type Goldilocksx8 = goldilocks_neon::NeonGoldilocks;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod goldilocks_avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type Goldilocksx8 = goldilocks_avx512::AVXGoldilocks;

// Fallback, use avx2
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod goldilocks_avx256;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type Goldilocksx8 = goldilocks_avx256::AVXGoldilocks;

impl Ord for Goldilocksx8 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for Goldilocksx8 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Goldilocksx8 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for Goldilocksx8 is not supported")
    }
}
