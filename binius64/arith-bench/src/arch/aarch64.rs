// Copyright 2025 Irreducible Inc.
use std::arch::aarch64::*;

use seq_macro::seq;

use crate::underlier::{PackedUnderlier, Underlier};

impl Underlier for uint64x2_t {
    const BITS: usize = 128;

    #[inline]
    fn and(a: Self, b: Self) -> Self {
        unsafe { vandq_u64(a, b) }
    }

    #[inline]
    fn xor(a: Self, b: Self) -> Self {
        unsafe { veorq_u64(a, b) }
    }

    #[inline]
    fn zero() -> Self {
        unsafe { vdupq_n_u64(0) }
    }

    #[inline]
    fn is_equal(a: Self, b: Self) -> bool {
        unsafe {
            let cmp = vceqq_u64(a, b);
            vgetq_lane_u64(cmp, 0) == u64::MAX && vgetq_lane_u64(cmp, 1) == u64::MAX
        }
    }

    fn random(mut rng: impl rand::RngCore) -> Self {
        use rand::Rng;

        let value: u128 = rng.random();
        unsafe { std::mem::transmute::<_, uint64x2_t>(value) }
    }
}

impl PackedUnderlier<u64> for uint64x2_t {
    const LOG_WIDTH: usize = 1; // 2^1 = 2 elements

    #[inline]
    fn get(self, index: usize) -> u64 {
        assert!(index < 2, "index out of bounds");
        unsafe {
            seq!(N in 0..2 {
                match index {
                    #(N => vgetq_lane_u64(self, N),)*
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
                    #(N => vsetq_lane_u64(val, self, N),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn broadcast(val: u64) -> Self {
        unsafe { vdupq_n_u64(val) }
    }
}

impl PackedUnderlier<u8> for uint64x2_t {
    const LOG_WIDTH: usize = 4; // 2^4 = 16 elements

    #[inline]
    fn get(self, index: usize) -> u8 {
        assert!(index < 16, "index out of bounds");
        unsafe {
            let bytes = vreinterpretq_u8_u64(self);
            seq!(N in 0..16 {
                match index {
                    #(N => vgetq_lane_u8(bytes, N),)*
                    _ => unreachable!(),
                }
            })
        }
    }

    #[inline]
    fn set(self, index: usize, val: u8) -> Self {
        assert!(index < 16, "index out of bounds");
        unsafe {
            let mut bytes = vreinterpretq_u8_u64(self);
            seq!(N in 0..16 {
                match index {
                    #(N => { bytes = vsetq_lane_u8(val, bytes, N); },)*
                    _ => unreachable!(),
                }
            });
            vreinterpretq_u64_u8(bytes)
        }
    }

    #[inline]
    fn broadcast(val: u8) -> Self {
        unsafe {
            let bytes = vdupq_n_u8(val);
            vreinterpretq_u64_u8(bytes)
        }
    }
}

impl PackedUnderlier<u128> for uint64x2_t {
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

impl crate::underlier::OpsClmul for uint64x2_t {
    #[inline]
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self {
        let result = match IMM8 {
            0x00 => unsafe { vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0)) },
            0x11 => unsafe { vmull_p64(vgetq_lane_u64(a, 1), vgetq_lane_u64(b, 1)) },
            0x10 => unsafe { vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 1)) },
            0x01 => unsafe { vmull_p64(vgetq_lane_u64(a, 1), vgetq_lane_u64(b, 0)) },
            _ => panic!("Unsupported IMM8 value for clmulepi64"),
        };

        unsafe { std::mem::transmute(result) }
    }

    #[inline]
    fn duplicate_hi_64(a: Self) -> Self {
        unsafe { vdupq_n_u64(vgetq_lane_u64(a, 1)) }
    }

    #[inline]
    fn swap_hi_lo_64(a: Self) -> Self {
        unsafe { vextq_u64(a, a, 1) }
    }

    #[inline]
    fn extract_hi_lo_64(a: Self, b: Self) -> Self {
        unsafe {
            vcombine_u64(
                vcreate_u64(vgetq_lane_u64(a, 1)),
                vcreate_u64(vgetq_lane_u64(b, 0)),
            )
        }
    }

    #[inline]
    fn unpacklo_epi64(a: Self, b: Self) -> Self {
        unsafe { vzip1q_u64(a, b) }
    }

    #[inline]
    fn unpackhi_epi64(a: Self, b: Self) -> Self {
        unsafe { vzip2q_u64(a, b) }
    }

    #[inline]
    fn slli_si128<const IMM8: i32>(a: Self) -> Self {
        // Shift left by IMM8 bytes
        unsafe {
            match IMM8 {
                0 => a,
                1..16 => {
                    let a_bytes: uint8x16_t = std::mem::transmute(a);
                    let zero: uint8x16_t = vdupq_n_u8(0);
                    let shifted: uint8x16_t = vextq_u8::<IMM8>(zero, a_bytes);
                    std::mem::transmute::<uint8x16_t, uint64x2_t>(shifted)
                }
                16.. => vdupq_n_u64(0),
                _ => {
                    // For other byte shifts, use a more complex approach
                    // This is a simplified implementation
                    a
                }
            }
        }
    }

    #[inline]
    fn slli_epi64<const IMM8: i32>(a: Self) -> Self {
        unsafe { vshlq_n_u64::<IMM8>(a) }
    }

    #[inline]
    fn movepi64_mask(a: Self) -> Self {
        unsafe {
            let a = std::mem::transmute::<uint64x2_t, uint32x4_t>(a);
            // Get the odd lanes (upper 32 bits of each 64-bit element)
            let odd_lanes = vtrn2q_u32(a, a);
            // Arithmetic shift right to broadcast the sign bit
            std::mem::transmute(vshrq_n_s32(
                std::mem::transmute::<uint32x4_t, int32x4_t>(odd_lanes),
                31,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        ghash::{ONE, mul_clmul as ghash_mul},
        monbijou::{
            MONBIJOU_128B_ONE, MONBIJOU_ONE, mul_128b_clmul as monbijou_128b_mul,
            mul_clmul as monbijou_mul,
        },
        polyval::{MONTGOMERY_ONE, mul_clmul as polyval_mul},
        rijndael::vmull::mul as rijndael_mul,
        test_utils::{
            arb_get_set_op,
            multiplication_tests::{
                test_mul_associative, test_mul_commutative, test_mul_distributive,
                test_mul_identity,
            },
            test_packed_underlier_get_set_behaves_like_vec,
        },
    };

    // Strategy for generating uint64x2_t values
    fn arb_uint64x2_t() -> impl Strategy<Value = uint64x2_t> {
        any::<u128>().prop_map(|val| unsafe { std::mem::transmute::<u128, uint64x2_t>(val) })
    }

    proptest! {
        #[test]
        fn test_uint64x2_t_as_packed_u8_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u8>(16), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<uint64x2_t, u8>(ops);
        }

        #[test]
        fn test_uint64x2_t_as_packed_u64_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u64>(2), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<uint64x2_t, u64>(ops);
        }

        #[test]
        fn test_uint64x2_t_as_packed_u128_proptest(
            ops in prop::collection::vec(arb_get_set_op::<u128>(1), 0..100)
        ) {
            test_packed_underlier_get_set_behaves_like_vec::<uint64x2_t, u128>(ops);
        }

        // Polynomial Montgomery multiplication property tests for uint64x2_t
        #[test]
        fn test_uint64x2_t_polyval_mul_commutative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t()
        ) {
            test_mul_commutative(a, b, polyval_mul, "POLYVAL");
        }

        #[test]
        fn test_uint64x2_t_polyval_mul_associative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_associative(a, b, c, polyval_mul, "POLYVAL");
        }

        #[test]
        fn test_uint64x2_t_polyval_mul_identity_proptest(
            a in arb_uint64x2_t()
        ) {
            test_mul_identity(a, MONTGOMERY_ONE, polyval_mul, "POLYVAL");
        }

        #[test]
        fn test_uint64x2_t_polyval_mul_distributive_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_distributive(a, b, c, polyval_mul, "POLYVAL");
        }

        // GHASH multiplication property tests for uint64x2_t
        #[test]
        fn test_uint64x2_t_ghash_mul_commutative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t()
        ) {
            test_mul_commutative(a, b, ghash_mul, "GHASH");
        }

        #[test]
        fn test_uint64x2_t_ghash_mul_associative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_associative(a, b, c, ghash_mul, "GHASH");
        }

        #[test]
        fn test_uint64x2_t_ghash_mul_identity_proptest(
            a in arb_uint64x2_t()
        ) {
            test_mul_identity(a, ONE, ghash_mul, "GHASH");
        }

        #[test]
        fn test_uint64x2_t_ghash_mul_distributive_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_distributive(a, b, c, ghash_mul, "GHASH");
        }

        // Monbijou multiplication property tests for uint64x2_t
        #[test]
        fn test_uint64x2_t_monbijou_mul_commutative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t()
        ) {
            test_mul_commutative(a, b, monbijou_mul, "Monbijou");
        }

        #[test]
        fn test_uint64x2_t_monbijou_mul_associative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_associative(a, b, c, monbijou_mul, "Monbijou");
        }

        #[test]
        fn test_uint64x2_t_monbijou_mul_identity_proptest(
            a in arb_uint64x2_t()
        ) {
            test_mul_identity(a, MONBIJOU_ONE, monbijou_mul, "Monbijou");
        }

        #[test]
        fn test_uint64x2_t_monbijou_mul_distributive_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_distributive(a, b, c, monbijou_mul, "Monbijou");
        }

        // Monbijou 128-bit multiplication property tests for uint64x2_t
        #[test]
        fn test_uint64x2_t_monbijou_128b_mul_commutative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t()
        ) {
            test_mul_commutative(a, b, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        fn test_uint64x2_t_monbijou_128b_mul_associative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_associative(a, b, c, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        fn test_uint64x2_t_monbijou_128b_mul_identity_proptest(
            a in arb_uint64x2_t()
        ) {
            test_mul_identity(a, MONBIJOU_128B_ONE, monbijou_128b_mul, "Monbijou 128b");
        }

        #[test]
        fn test_uint64x2_t_monbijou_128b_mul_distributive_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_distributive(a, b, c, monbijou_128b_mul, "Monbijou 128b");
        }

        // GF(2^8) Rijndael multiplication property tests for uint64x2_t
        #[test]
        fn test_uint64x2_t_rijndael_mul_commutative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t()
        ) {
            test_mul_commutative(a, b, rijndael_mul, "Rijndael");
        }

        #[test]
        fn test_uint64x2_t_rijndael_mul_associative_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_associative(a, b, c, rijndael_mul, "Rijndael");
        }

        #[test]
        fn test_uint64x2_t_rijndael_mul_identity_proptest(
            a in arb_uint64x2_t()
        ) {
            test_mul_identity(a, 0x01u8, rijndael_mul, "Rijndael");
        }

        #[test]
        fn test_uint64x2_t_rijndael_mul_distributive_proptest(
            a in arb_uint64x2_t(),
            b in arb_uint64x2_t(),
            c in arb_uint64x2_t()
        ) {
            test_mul_distributive(a, b, c, rijndael_mul, "Rijndael");
        }
    }

    #[test]
    fn test_uint64x2_t_movepi64_mask() {
        use crate::underlier::{OpsClmul, Underlier};

        unsafe {
            // Test all zeros - expect all zeros in the mask
            let zeros = vdupq_n_u64(0);
            let mask_zeros = <uint64x2_t as OpsClmul>::movepi64_mask(zeros);
            assert!(<uint64x2_t as Underlier>::is_equal(
                mask_zeros,
                vdupq_n_u64(0)
            ));

            // Test with negative values (sign bit set) - expect all 1s in all positions
            let neg_ones = vdupq_n_u64(u64::MAX);
            let mask_neg = <uint64x2_t as OpsClmul>::movepi64_mask(neg_ones);
            let expected_neg = vdupq_n_u64(u64::MAX);
            assert!(<uint64x2_t as Underlier>::is_equal(mask_neg, expected_neg));

            // Test mixed values - lane 0 positive, lane 1 negative
            let mixed = vsetq_lane_u64(u64::MAX, vsetq_lane_u64(1, vdupq_n_u64(0), 0), 1);
            let mask_mixed = <uint64x2_t as OpsClmul>::movepi64_mask(mixed);
            let expected_mixed = vsetq_lane_u64(u64::MAX, vsetq_lane_u64(0, vdupq_n_u64(0), 0), 1);
            assert!(<uint64x2_t as Underlier>::is_equal(
                mask_mixed,
                expected_mixed
            ));

            // Test another mixed pattern - lane 0 negative, lane 1 positive
            let mixed2 = vsetq_lane_u64(1, vsetq_lane_u64(u64::MAX, vdupq_n_u64(0), 0), 1);
            let mask_mixed2 = <uint64x2_t as OpsClmul>::movepi64_mask(mixed2);
            let expected_mixed2 = vsetq_lane_u64(0, vsetq_lane_u64(u64::MAX, vdupq_n_u64(0), 0), 1);
            assert!(<uint64x2_t as Underlier>::is_equal(
                mask_mixed2,
                expected_mixed2
            ));
        }
    }

    #[test]
    fn test_uint64x2_t_basic_ops() {
        use crate::underlier::Underlier;

        unsafe {
            // Test basic operations
            let a = vsetq_lane_u64(0xDEADBEEF, vsetq_lane_u64(0x12345678, vdupq_n_u64(0), 0), 1);
            let b = vsetq_lane_u64(0xCAFEBABE, vsetq_lane_u64(0x87654321, vdupq_n_u64(0), 0), 1);

            // Test AND
            let and_result = <uint64x2_t as Underlier>::and(a, b);
            let expected_and_low = 0x12345678 & 0x87654321;
            let expected_and_high = 0xDEADBEEF & 0xCAFEBABE;
            assert_eq!(vgetq_lane_u64(and_result, 0), expected_and_low);
            assert_eq!(vgetq_lane_u64(and_result, 1), expected_and_high);

            // Test XOR
            let xor_result = <uint64x2_t as Underlier>::xor(a, b);
            let expected_xor_low = 0x12345678 ^ 0x87654321;
            let expected_xor_high = 0xDEADBEEF ^ 0xCAFEBABE;
            assert_eq!(vgetq_lane_u64(xor_result, 0), expected_xor_low);
            assert_eq!(vgetq_lane_u64(xor_result, 1), expected_xor_high);

            // Test zero
            let zero_result = <uint64x2_t as Underlier>::zero();
            assert_eq!(vgetq_lane_u64(zero_result, 0), 0);
            assert_eq!(vgetq_lane_u64(zero_result, 1), 0);

            // Test equality
            let a_copy =
                vsetq_lane_u64(0xDEADBEEF, vsetq_lane_u64(0x12345678, vdupq_n_u64(0), 0), 1);
            assert!(<uint64x2_t as Underlier>::is_equal(a, a_copy));
            assert!(!<uint64x2_t as Underlier>::is_equal(a, b));
        }
    }

    #[test]
    fn test_uint64x2_t_ops_clmul() {
        use crate::underlier::OpsClmul;

        unsafe {
            let a = vsetq_lane_u64(0xDEADBEEF, vsetq_lane_u64(0x12345678, vdupq_n_u64(0), 0), 1);
            let b = vsetq_lane_u64(0xCAFEBABE, vsetq_lane_u64(0x87654321, vdupq_n_u64(0), 0), 1);

            // Test unpack operations
            let unpack_lo = <uint64x2_t as OpsClmul>::unpacklo_epi64(a, b);
            assert_eq!(vgetq_lane_u64(unpack_lo, 0), 0x12345678);
            assert_eq!(vgetq_lane_u64(unpack_lo, 1), 0x87654321);

            let unpack_hi = <uint64x2_t as OpsClmul>::unpackhi_epi64(a, b);
            assert_eq!(vgetq_lane_u64(unpack_hi, 0), 0xDEADBEEF);
            assert_eq!(vgetq_lane_u64(unpack_hi, 1), 0xCAFEBABE);

            // Test shift operations
            let shift_1 = <uint64x2_t as OpsClmul>::slli_epi64::<1>(a);
            assert_eq!(vgetq_lane_u64(shift_1, 0), 0x12345678 << 1);
            assert_eq!(vgetq_lane_u64(shift_1, 1), 0xDEADBEEF << 1);

            let shift_8 = <uint64x2_t as OpsClmul>::slli_epi64::<8>(a);
            assert_eq!(vgetq_lane_u64(shift_8, 0), 0x12345678 << 8);
            assert_eq!(vgetq_lane_u64(shift_8, 1), 0xDEADBEEF << 8);

            // Test byte shift (slli_si128)
            let byte_shift_8 = <uint64x2_t as OpsClmul>::slli_si128::<8>(a);
            assert_eq!(vgetq_lane_u64(byte_shift_8, 0), 0);
            assert_eq!(vgetq_lane_u64(byte_shift_8, 1), 0x12345678);
        }
    }
}
