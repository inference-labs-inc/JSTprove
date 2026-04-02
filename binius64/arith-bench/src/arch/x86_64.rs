// Copyright 2025 Irreducible Inc.
use std::arch::x86_64::*;

use seq_macro::seq;

use crate::underlier::{PackedUnderlier, Underlier};

#[cfg(target_feature = "sse2")]
impl Underlier for __m128i {
    const BITS: usize = 128;

    #[inline]
    fn and(a: Self, b: Self) -> Self {
        unsafe { _mm_and_si128(a, b) }
    }

    #[inline]
    fn xor(a: Self, b: Self) -> Self {
        unsafe { _mm_xor_si128(a, b) }
    }

    #[inline]
    fn zero() -> Self {
        unsafe { _mm_setzero_si128() }
    }

    #[inline]
    fn is_equal(a: Self, b: Self) -> bool {
        unsafe {
            let cmp = _mm_cmpeq_epi8(a, b);
            _mm_movemask_epi8(cmp) == 0xFFFF
        }
    }

    fn random(mut rng: impl rand::RngCore) -> Self {
        let mut bytes = [0u8; 16];
        rng.fill_bytes(&mut bytes);
        unsafe { _mm_loadu_si128(bytes.as_ptr() as *const __m128i) }
    }
}

#[cfg(target_feature = "sse2")]
impl PackedUnderlier<u64> for __m128i {
    const LOG_WIDTH: usize = 1; // 2^1 = 2 elements

    #[inline]
    fn get(self, index: usize) -> u64 {
        assert!(index < 2, "index out of bounds");
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => _mm_extract_epi64(self, N) as u64,)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: u64) -> Self {
        assert!(index < 2, "index out of bounds");
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => _mm_insert_epi64(self, val as i64, N),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: u64) -> Self {
        unsafe { _mm_set1_epi64x(val as i64) }
    }
}

#[cfg(target_feature = "sse4.1")]
impl PackedUnderlier<u8> for __m128i {
    const LOG_WIDTH: usize = 4; // 2^4 = 16 elements

    #[inline]
    fn get(self, index: usize) -> u8 {
        assert!(index < 16, "index out of bounds");
        unsafe {
            seq!(N in 0..16 {
                match index {
                    #(N => _mm_extract_epi8(self, N) as u8,)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: u8) -> Self {
        assert!(index < 16, "index out of bounds");
        unsafe {
            seq!(N in 0..16 {
                match index {
                    #(N => _mm_insert_epi8(self, val as i32, N),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: u8) -> Self {
        unsafe { _mm_set1_epi8(val as i8) }
    }
}

#[cfg(target_feature = "sse2")]
impl PackedUnderlier<u128> for __m128i {
    const LOG_WIDTH: usize = 0; // 2^0 = 1 element

    #[inline]
    fn get(self, index: usize) -> u128 {
        assert!(index == 0, "index out of bounds");
        unsafe { std::mem::transmute(self) }
    }

    #[inline]
    fn set(self, index: usize, val: u128) -> Self {
        assert!(index == 0, "index out of bounds");
        unsafe { std::mem::transmute(val) }
    }

    #[inline]
    fn broadcast(val: u128) -> Self {
        unsafe { std::mem::transmute(val) }
    }
}

#[cfg(all(target_feature = "gfni", target_feature = "sse2"))]
impl crate::underlier::OpsGfni for __m128i {
    #[inline]
    fn gf2p8mul(a: Self, b: Self) -> Self {
        unsafe { _mm_gf2p8mul_epi8(a, b) }
    }

    #[inline]
    fn gf2p8affine<const B: i32>(a: Self, b: Self) -> Self {
        unsafe { _mm_gf2p8affine_epi64_epi8::<B>(a, b) }
    }

    #[inline]
    fn gf2p8affineinv<const B: i32>(a: Self, b: Self) -> Self {
        unsafe { _mm_gf2p8affineinv_epi64_epi8::<B>(a, b) }
    }
}

#[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
impl crate::underlier::OpsClmul for __m128i {
    #[inline]
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
        unsafe { _mm_clmulepi64_si128::<IMM8>(a, b) }
    }

    #[inline]
    fn duplicate_hi_64(a: Self) -> Self {
        unsafe { _mm_shuffle_epi32::<0xEE>(a) }
    }

    #[inline]
    fn swap_hi_lo_64(a: Self) -> Self {
        unsafe { _mm_shuffle_epi32::<0x4E>(a) }
    }

    #[inline]
    fn extract_hi_lo_64(a: Self, b: Self) -> Self {
        unsafe {
            _mm_castps_si128(_mm_shuffle_ps::<0x4E>(
                _mm_castsi128_ps(a),
                _mm_castsi128_ps(b),
            ))
        }
    }

    #[inline]
    fn unpacklo_epi64(a: Self, b: Self) -> Self {
        unsafe { _mm_unpacklo_epi64(a, b) }
    }

    #[inline]
    fn unpackhi_epi64(a: Self, b: Self) -> Self {
        unsafe { _mm_unpackhi_epi64(a, b) }
    }

    #[inline]
    fn slli_si128<const IMM8: i32>(a: Self) -> Self {
        unsafe { _mm_slli_si128::<IMM8>(a) }
    }

    #[inline]
    fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
        unsafe { _mm_slli_epi64::<IMM8>(a) }
    }

    #[inline]
    fn movepi64_mask(a: Self) -> Self {
        unsafe {
            // Shuffle to move the high dword of each qword to positions 0 and 1
            // Pattern 0xF5 = 11110101 = [3, 3, 1, 1] - takes elements 1 and 3
            let shuffled = _mm_shuffle_epi32::<0xF5>(a);
            // Arithmetic shift right by 31 to create masks from sign bits
            _mm_srai_epi32(shuffled, 31)
        }
    }
}

#[cfg(target_feature = "avx2")]
impl Underlier for __m256i {
    const BITS: usize = 256;

    #[inline]
    fn and(a: Self, b: Self) -> Self {
        unsafe { _mm256_and_si256(a, b) }
    }

    #[inline]
    fn xor(a: Self, b: Self) -> Self {
        unsafe { _mm256_xor_si256(a, b) }
    }

    #[inline]
    fn zero() -> Self {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline]
    fn is_equal(a: Self, b: Self) -> bool {
        unsafe {
            let cmp = _mm256_cmpeq_epi8(a, b);
            _mm256_movemask_epi8(cmp) as u32 == 0xFFFFFFFF
        }
    }

    fn random(mut rng: impl rand::RngCore) -> Self {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        unsafe { _mm256_loadu_si256(bytes.as_ptr() as *const __m256i) }
    }
}

#[cfg(target_feature = "avx2")]
impl PackedUnderlier<__m128i> for __m256i {
    const LOG_WIDTH: usize = 1; // 2^1 = 2 elements

    #[inline]
    fn get(self, index: usize) -> __m128i {
        assert!(index < 2, "index out of bounds");
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => _mm256_extracti128_si256::<N>(self),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: __m128i) -> Self {
        assert!(index < 2, "index out of bounds");
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => _mm256_inserti128_si256::<N>(self, val),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: __m128i) -> Self {
        unsafe { _mm256_broadcastsi128_si256(val) }
    }
}

#[cfg(target_feature = "avx")]
impl PackedUnderlier<u64> for __m256i {
    const LOG_WIDTH: usize = 2; // 2^2 = 4 elements

    #[inline]
    fn get(self, index: usize) -> u64 {
        assert!(index < 4, "index out of bounds");
        unsafe {
            seq!(N in 0..4 {
                match index {
                    #(N => _mm256_extract_epi64(self, N) as u64,)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: u64) -> Self {
        assert!(index < 4, "index out of bounds");
        unsafe {
            seq!(N in 0..4 {
                match index {
                    #(N => _mm256_insert_epi64(self, val as i64, N),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: u64) -> Self {
        unsafe { _mm256_set1_epi64x(val as i64) }
    }
}

#[cfg(target_feature = "avx2")]
impl PackedUnderlier<u8> for __m256i {
    const LOG_WIDTH: usize = 5; // 2^5 = 32 elements

    #[inline]
    fn get(self, index: usize) -> u8 {
        assert!(index < 32, "index out of bounds");
        unsafe {
            seq!(N in 0..32 {
                match index {
                    #(N => _mm256_extract_epi8(self, N) as u8,)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: u8) -> Self {
        assert!(index < 32, "index out of bounds");
        unsafe {
            seq!(N in 0..32 {
                match index {
                    #(N => _mm256_insert_epi8(self, val as i8, N),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: u8) -> Self {
        unsafe { _mm256_set1_epi8(val as i8) }
    }
}

#[cfg(target_feature = "avx")]
impl PackedUnderlier<u128> for __m256i {
    const LOG_WIDTH: usize = 1; // 2^1 = 2 elements

    #[inline]
    fn get(self, index: usize) -> u128 {
        assert!(index < 2, "index out of bounds");
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => std::mem::transmute::<__m128i, u128>(_mm256_extracti128_si256::<N>(self)),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: u128) -> Self {
        assert!(index < 2, "index out of bounds");
        let val_m128 = unsafe { std::mem::transmute::<u128, __m128i>(val) };
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => _mm256_inserti128_si256::<N>(self, val_m128),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: u128) -> Self {
        let val_m128 = unsafe { std::mem::transmute::<u128, __m128i>(val) };
        unsafe { _mm256_broadcastsi128_si256(val_m128) }
    }
}

#[cfg(all(target_feature = "gfni", target_feature = "avx"))]
impl crate::underlier::OpsGfni for __m256i {
    #[inline]
    fn gf2p8mul(a: Self, b: Self) -> Self {
        unsafe { _mm256_gf2p8mul_epi8(a, b) }
    }

    #[inline]
    fn gf2p8affine<const B: i32>(x: Self, a: Self) -> Self {
        unsafe { _mm256_gf2p8affine_epi64_epi8::<B>(x, a) }
    }

    #[inline]
    fn gf2p8affineinv<const B: i32>(x: Self, a: Self) -> Self {
        unsafe { _mm256_gf2p8affineinv_epi64_epi8::<B>(x, a) }
    }
}

#[cfg(all(
    target_feature = "vpclmulqdq",
    target_feature = "avx2",
    target_feature = "sse2"
))]
impl crate::underlier::OpsClmul for __m256i {
    #[inline]
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
        unsafe { _mm256_clmulepi64_epi128::<IMM8>(a, b) }
    }

    #[inline]
    fn duplicate_hi_64(a: Self) -> Self {
        unsafe { _mm256_shuffle_epi32::<0xEE>(a) }
    }

    #[inline]
    fn swap_hi_lo_64(a: Self) -> Self {
        unsafe { _mm256_shuffle_epi32::<0x4E>(a) }
    }

    #[inline]
    fn extract_hi_lo_64(a: Self, b: Self) -> Self {
        unsafe {
            _mm256_castps_si256(_mm256_shuffle_ps::<0x4E>(
                _mm256_castsi256_ps(a),
                _mm256_castsi256_ps(b),
            ))
        }
    }

    #[inline]
    fn unpacklo_epi64(a: Self, b: Self) -> Self {
        unsafe { _mm256_unpacklo_epi64(a, b) }
    }

    #[inline]
    fn unpackhi_epi64(a: Self, b: Self) -> Self {
        unsafe { _mm256_unpackhi_epi64(a, b) }
    }

    #[inline]
    fn slli_si128<const IMM8: i32>(a: Self) -> Self {
        unsafe { _mm256_slli_si256::<IMM8>(a) }
    }

    #[inline]
    fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
        unsafe { _mm256_slli_epi64::<IMM8>(a) }
    }

    #[inline]
    fn movepi64_mask(a: Self) -> Self {
        unsafe {
            // Shuffle to move the high dword of each qword to positions 0, 1, 2, 3
            // Pattern 0xF5 = 11110101 = [3, 3, 1, 1] - takes elements 1 and 3 from each 128-bit
            // lane
            let shuffled = _mm256_shuffle_epi32::<0xF5>(a);
            // Arithmetic shift right by 31 to create masks from sign bits
            _mm256_srai_epi32(shuffled, 31)
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    #[cfg(any(
        all(target_feature = "pclmulqdq", target_feature = "sse2"),
        all(
            target_feature = "vpclmulqdq",
            target_feature = "avx2",
            target_feature = "sse2"
        )
    ))]
    use crate::ghash::{ONE, mul_clmul as ghash_mul};
    #[cfg(any(
        all(target_feature = "pclmulqdq", target_feature = "sse2"),
        all(
            target_feature = "vpclmulqdq",
            target_feature = "avx2",
            target_feature = "sse2"
        )
    ))]
    use crate::monbijou::{
        MONBIJOU_128B_ONE, MONBIJOU_ONE, mul_128b_clmul as monbijou_128b_mul,
        mul_clmul as monbijou_mul,
    };
    #[cfg(any(
        all(target_feature = "pclmulqdq", target_feature = "sse2"),
        all(
            target_feature = "vpclmulqdq",
            target_feature = "avx2",
            target_feature = "sse2"
        )
    ))]
    use crate::polyval::{MONTGOMERY_ONE, mul_clmul as polyval_mul};
    #[cfg(any(
        all(target_feature = "gfni", target_feature = "sse2"),
        all(target_feature = "gfni", target_feature = "avx")
    ))]
    use crate::rijndael::gfni::mul;
    #[cfg(target_feature = "avx2")]
    use crate::test_utils::GetSetOp;
    #[cfg(all(
        test,
        any(
            all(target_feature = "gfni", target_feature = "sse2"),
            all(target_feature = "gfni", target_feature = "avx"),
            all(target_feature = "pclmulqdq", target_feature = "sse2"),
            all(
                target_feature = "vpclmulqdq",
                target_feature = "avx2",
                target_feature = "sse2"
            )
        )
    ))]
    use crate::test_utils::multiplication_tests::{
        test_mul_associative, test_mul_commutative, test_mul_distributive, test_mul_identity,
    };
    use crate::test_utils::{arb_get_set_op, test_packed_underlier_get_set_behaves_like_vec};

    // Strategy for generating __m128i values
    #[allow(dead_code)]
    fn arb_m128i() -> impl Strategy<Value = __m128i> {
        any::<u128>().prop_map(|val| unsafe { std::mem::transmute::<u128, __m128i>(val) })
    }

    // Strategy for generating __m256i values
    #[allow(dead_code)]
    fn arb_m256i() -> impl Strategy<Value = __m256i> {
        (any::<u128>(), any::<u128>()).prop_map(|(low, high)| unsafe {
            let low_m128 = std::mem::transmute::<u128, __m128i>(low);
            let high_m128 = std::mem::transmute::<u128, __m128i>(high);
            _mm256_set_m128i(high_m128, low_m128)
        })
    }

    // Strategy for generating GetSetOp<__m128i> with valid indices
    #[cfg(target_feature = "avx2")]
    fn arb_get_set_op_m128i(width: usize) -> impl Strategy<Value = GetSetOp<__m128i>> {
        prop_oneof![
            // 50% Get operations
            (0..width).prop_map(|index| GetSetOp::Get { index }),
            // 50% Set operations
            (0..width, arb_m128i()).prop_map(|(index, val)| GetSetOp::Set { index, val })
        ]
    }

    proptest! {
        #[test]
        #[cfg(target_feature = "sse4.1")]
        fn test_m128i_as_packed_u8_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u8>(16), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m128i, u8>(ops);
        }

        #[test]
        #[cfg(target_feature = "sse2")]
        fn test_m128i_as_packed_u64_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u64>(2), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m128i, u64>(ops);
        }

        #[test]
        #[cfg(target_feature = "sse2")]
        fn test_m128i_as_packed_u128_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u128>(1), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m128i, u128>(ops);
        }

        #[test]
        #[cfg(target_feature = "avx2")]
        fn test_m256i_as_packed_u8_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u8>(32), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m256i, u8>(ops);
        }

        #[test]
        #[cfg(target_feature = "avx")]
        fn test_m256i_as_packed_u64_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u64>(4), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m256i, u64>(ops);
        }

        #[test]
        #[cfg(target_feature = "avx")]
        fn test_m256i_as_packed_u128_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u128>(2), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m256i, u128>(ops);
        }

        #[test]
        #[cfg(target_feature = "avx2")]
        fn test_m256i_as_packed_m128i_proptest(
            ops in prop::collection::vec(arb_get_set_op_m128i(2), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<__m256i, __m128i>(ops);
        }

        // GF(2^8) multiplication property tests for __m128i
        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "sse2"))]
        fn test_m128i_gf2p8mul_commutative_proptest(
            a in arb_m128i(),
            b in arb_m128i()
        ) {
            test_mul_commutative(a, b, crate::underlier::OpsGfni::gf2p8mul, "GF(2^8)");
        }

        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "sse2"))]
        fn test_m128i_gf2p8mul_associative_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_associative(a, b, c, crate::underlier::OpsGfni::gf2p8mul, "GF(2^8)");
        }

        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "sse2"))]
        fn test_m128i_gf2p8mul_distributive_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_distributive(a, b, c, crate::underlier::OpsGfni::gf2p8mul, "GF(2^8)");
        }

        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "sse2", target_feature = "sse4.1"))]
        fn test_m128i_gf2p8mul_identity_proptest(
            a in arb_m128i()
        ) {
            test_mul_identity(a, 0x01u8, mul, "GF(2^8)");
        }

        // GF(2^8) multiplication property tests for __m256i
        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "avx"))]
        fn test_m256i_gf2p8mul_commutative_proptest(
            a in arb_m256i(),
            b in arb_m256i()
        ) {
            test_mul_commutative(a, b, crate::underlier::OpsGfni::gf2p8mul, "GF(2^8)");
        }

        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "avx"))]
        fn test_m256i_gf2p8mul_associative_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_associative(a, b, c, crate::underlier::OpsGfni::gf2p8mul, "GF(2^8)");
        }

        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "avx"))]
        fn test_m256i_gf2p8mul_distributive_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_distributive(a, b, c, crate::underlier::OpsGfni::gf2p8mul, "GF(2^8)");
        }

        #[test]
        #[cfg(all(target_feature = "gfni", target_feature = "avx", target_feature = "avx2"))]
        fn test_m256i_gf2p8mul_identity_proptest(
            a in arb_m256i()
        ) {
            test_mul_identity(a, 0x01u8, mul, "GF(2^8)");
        }

        // Polynomial Montgomery multiplication property tests for __m128i
        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_polyval_mul_commutative_proptest(
            a in arb_m128i(),
            b in arb_m128i()
        ) {
            test_mul_commutative(a, b, polyval_mul, "POLYVAL");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_polyval_mul_associative_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_associative(a, b, c, polyval_mul, "POLYVAL");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_polyval_mul_identity_proptest(
            a in arb_m128i()
        ) {
            test_mul_identity(a, MONTGOMERY_ONE, polyval_mul, "POLYVAL");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_polyval_mul_distributive_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_distributive(a, b, c, polyval_mul, "POLYVAL");
        }

        // Polynomial Montgomery multiplication property tests for __m256i
        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_polyval_mul_commutative_proptest(
            a in arb_m256i(),
            b in arb_m256i()
        ) {
            test_mul_commutative(a, b, polyval_mul, "POLYVAL");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_polyval_mul_associative_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_associative(a, b, c, polyval_mul, "POLYVAL");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_polyval_mul_identity_proptest(
            a in arb_m256i()
        ) {
            test_mul_identity(a, MONTGOMERY_ONE, polyval_mul, "POLYVAL");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_polyval_mul_distributive_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_distributive(a, b, c, polyval_mul, "POLYVAL");
        }

        // GHASH multiplication property tests for __m128i
        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_ghash_mul_commutative_proptest(
            a in arb_m128i(),
            b in arb_m128i()
        ) {
            test_mul_commutative(a, b, ghash_mul, "GHASH");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_ghash_mul_associative_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_associative(a, b, c, ghash_mul, "GHASH");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_ghash_mul_identity_proptest(
            a in arb_m128i()
        ) {

            test_mul_identity(a, ONE, ghash_mul, "GHASH");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_ghash_mul_distributive_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_distributive(a, b, c, ghash_mul, "GHASH");
        }

        // GHASH multiplication property tests for __m256i
        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_ghash_mul_commutative_proptest(
            a in arb_m256i(),
            b in arb_m256i()
        ) {
            test_mul_commutative(a, b, ghash_mul, "GHASH");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_ghash_mul_associative_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_associative(a, b, c, ghash_mul, "GHASH");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_ghash_mul_identity_proptest(
            a in arb_m256i()
        ) {
            test_mul_identity(a, ONE, ghash_mul, "GHASH");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_ghash_mul_distributive_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_distributive(a, b, c, ghash_mul, "GHASH");
        }

        // Monbijou multiplication property tests for __m128i
        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_mul_commutative_proptest(
            a in arb_m128i(),
            b in arb_m128i()
        ) {
            test_mul_commutative(a, b, monbijou_mul, "Monbijou");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_mul_associative_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_associative(a, b, c, monbijou_mul, "Monbijou");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_mul_identity_proptest(
            a in arb_m128i()
        ) {
            test_mul_identity(a, MONBIJOU_ONE, monbijou_mul, "Monbijou");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_mul_distributive_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_distributive(a, b, c, monbijou_mul, "Monbijou");
        }

        // Monbijou multiplication property tests for __m256i
        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_mul_commutative_proptest(
            a in arb_m256i(),
            b in arb_m256i()
        ) {
            test_mul_commutative(a, b, monbijou_mul, "Monbijou");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_mul_associative_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_associative(a, b, c, monbijou_mul, "Monbijou");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_mul_identity_proptest(
            a in arb_m256i()
        ) {
            test_mul_identity(a, MONBIJOU_ONE, monbijou_mul, "Monbijou");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_mul_distributive_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_distributive(a, b, c, monbijou_mul, "Monbijou");
        }

        // Monbijou 128-bit multiplication property tests for __m128i
        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_128b_mul_commutative_proptest(
            a in arb_m128i(),
            b in arb_m128i()
        ) {
            test_mul_commutative(a, b, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_128b_mul_associative_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_associative(a, b, c, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_128b_mul_identity_proptest(
            a in arb_m128i()
        ) {
            test_mul_identity(a, MONBIJOU_128B_ONE, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
        fn test_m128i_monbijou_128b_mul_distributive_proptest(
            a in arb_m128i(),
            b in arb_m128i(),
            c in arb_m128i()
        ) {
            test_mul_distributive(a, b, c, monbijou_128b_mul, "Monbijou 128b");
        }

        // Monbijou 128-bit multiplication property tests for __m256i
        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_128b_mul_commutative_proptest(
            a in arb_m256i(),
            b in arb_m256i()
        ) {
            test_mul_commutative(a, b, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_128b_mul_associative_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_associative(a, b, c, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_128b_mul_identity_proptest(
            a in arb_m256i()
        ) {
            test_mul_identity(a, MONBIJOU_128B_ONE, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        #[cfg(all(target_feature = "vpclmulqdq", target_feature = "avx2", target_feature = "sse2"))]
        fn test_m256i_monbijou_128b_mul_distributive_proptest(
            a in arb_m256i(),
            b in arb_m256i(),
            c in arb_m256i()
        ) {
            test_mul_distributive(a, b, c, monbijou_128b_mul, "Monbijou 128b");
        }
    }

    #[test]
    #[cfg(all(target_feature = "pclmulqdq", target_feature = "sse2"))]
    fn test_m128i_movepi64_mask() {
        use crate::underlier::{OpsClmul, Underlier};

        unsafe {
            // Test all zeros - expect all zeros in the mask
            let zeros = _mm_setzero_si128();
            let mask_zeros = <__m128i as OpsClmul>::movepi64_mask(zeros);
            assert!(<__m128i as Underlier>::is_equal(
                mask_zeros,
                _mm_setzero_si128()
            ));

            // Test with negative values (sign bit set) - expect all 1s in all positions
            let neg_ones = _mm_set1_epi64x(-1);
            let mask_neg = <__m128i as OpsClmul>::movepi64_mask(neg_ones);
            // With shuffle pattern 0xF5 = [3, 3, 1, 1], both qwords have negative high dwords
            // So we expect 0xFFFFFFFF in all four positions
            let expected_neg = _mm_set_epi32(-1, -1, -1, -1);
            assert!(<__m128i as Underlier>::is_equal(mask_neg, expected_neg));

            // Test mixed values - lane 0 positive, lane 1 negative
            let mixed = _mm_set_epi64x(-1, 1);
            let mask_mixed = <__m128i as OpsClmul>::movepi64_mask(mixed);
            // Lane 0 (low qword) has positive high dword (element 1)
            // Lane 1 (high qword) has negative high dword (element 3)
            // Shuffle 0xF5 duplicates: positions 0,1 get element 1 (0), positions 2,3 get element 3
            // (-1)
            let expected_mixed = _mm_set_epi32(-1, -1, 0, 0);
            assert!(<__m128i as Underlier>::is_equal(mask_mixed, expected_mixed));

            // Test another mixed pattern - lane 0 negative, lane 1 positive
            let mixed2 = _mm_set_epi64x(1, -1);
            let mask_mixed2 = <__m128i as OpsClmul>::movepi64_mask(mixed2);
            // Lane 0 (low qword) has negative high dword (element 1)
            // Lane 1 (high qword) has positive high dword (element 3)
            // Shuffle 0xF5 duplicates: positions 0,1 get element 1 (-1), positions 2,3 get element
            // 3 (0)
            let expected_mixed2 = _mm_set_epi32(0, 0, -1, -1);
            assert!(<__m128i as Underlier>::is_equal(
                mask_mixed2,
                expected_mixed2
            ));
        }
    }

    #[test]
    #[cfg(all(
        target_feature = "vpclmulqdq",
        target_feature = "avx2",
        target_feature = "sse2"
    ))]
    fn test_m256i_movepi64_mask() {
        use crate::underlier::{OpsClmul, Underlier};

        unsafe {
            // Test all zeros - expect all zeros in the mask
            let zeros = _mm256_setzero_si256();
            let mask_zeros = <__m256i as OpsClmul>::movepi64_mask(zeros);
            assert!(<__m256i as Underlier>::is_equal(
                mask_zeros,
                _mm256_setzero_si256()
            ));

            // Test with negative values (sign bit set) - expect all 1s in all positions
            let neg_ones = _mm256_set1_epi64x(-1);
            let mask_neg = <__m256i as OpsClmul>::movepi64_mask(neg_ones);
            // With shuffle pattern 0xF5 = [3, 3, 1, 1] applied to each 128-bit lane
            // All qwords have negative high dwords, so we expect 0xFFFFFFFF in all eight positions
            let expected_neg = _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
            assert!(<__m256i as Underlier>::is_equal(mask_neg, expected_neg));

            // Test mixed values - lanes 0,2 positive, lanes 1,3 negative
            let mixed = _mm256_set_epi64x(-1, 1, -1, 1);
            let mask_mixed = <__m256i as OpsClmul>::movepi64_mask(mixed);
            // Lower 128-bit lane: qword0=1 (positive high), qword1=-1 (negative high)
            // Upper 128-bit lane: qword2=1 (positive high), qword3=-1 (negative high)
            // Shuffle 0xF5 duplicates in each lane:
            // Lower lane: positions 0,1 get element 1 (0), positions 2,3 get element 3 (-1)
            // Upper lane: positions 4,5 get element 1 (0), positions 6,7 get element 3 (-1)
            let expected_mixed = _mm256_set_epi32(-1, -1, 0, 0, -1, -1, 0, 0);
            assert!(<__m256i as Underlier>::is_equal(mask_mixed, expected_mixed));

            // Test another mixed pattern - lanes 0,2 negative, lanes 1,3 positive
            let mixed2 = _mm256_set_epi64x(1, -1, 1, -1);
            let mask_mixed2 = <__m256i as OpsClmul>::movepi64_mask(mixed2);
            // Lower 128-bit lane: qword0=-1 (negative high), qword1=1 (positive high)
            // Upper 128-bit lane: qword2=-1 (negative high), qword3=1 (positive high)
            // Shuffle 0xF5 duplicates in each lane:
            // Lower lane: positions 0,1 get element 1 (-1), positions 2,3 get element 3 (0)
            // Upper lane: positions 4,5 get element 1 (-1), positions 6,7 get element 3 (0)
            let expected_mixed2 = _mm256_set_epi32(0, 0, -1, -1, 0, 0, -1, -1);
            assert!(<__m256i as Underlier>::is_equal(
                mask_mixed2,
                expected_mixed2
            ));
        }
    }
}
