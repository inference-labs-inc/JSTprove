// A M31x16 stores 512 bits of data.
// With AVX it stores a single __m512i element.
// With NEON it stores four uint32x4_t elements.

#[cfg(target_arch = "aarch64")]
mod m31_neon;
#[cfg(target_arch = "aarch64")]
pub type M31x16 = m31_neon::NeonM31;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod m31_avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type M31x16 = m31_avx512::AVXM31;

// Fallback, use avx2
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod m31_avx256;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub type M31x16 = m31_avx256::AVXM31;

impl Ord for M31x16 {
    #[inline(always)]
    fn cmp(&self, _: &Self) -> std::cmp::Ordering {
        unimplemented!("Ord for M31x16 is not supported")
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for M31x16 {
    #[inline(always)]
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        unimplemented!("PartialOrd for M31x16 is not supported")
    }
}
