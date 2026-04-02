// Copyright 2025 Irreducible Inc.
use proptest::prelude::*;

use crate::test_utils::{arb_get_set_op, test_packed_underlier_get_set_behaves_like_vec};

proptest! {
    #[test]
    fn test_u64_as_packed_u8_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u8>(8), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u64, u8>(ops);
    }

    #[test]
    fn test_u32_as_packed_u8_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u8>(4), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u32, u8>(ops);
    }

    #[test]
    fn test_u16_as_packed_u8_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u8>(2), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u16, u8>(ops);
    }

    #[test]
    fn test_u128_as_packed_u8_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u8>(16), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u128, u8>(ops);
    }

    #[test]
    fn test_u64_as_packed_u16_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u16>(4), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u64, u16>(ops);
    }

    #[test]
    fn test_u32_as_packed_u16_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u16>(2), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u32, u16>(ops);
    }

    #[test]
    fn test_u128_as_packed_u16_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u16>(8), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u128, u16>(ops);
    }

    #[test]
    fn test_u64_as_packed_u32_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u32>(2), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u64, u32>(ops);
    }

    #[test]
    fn test_u128_as_packed_u32_proptest(
        ops in prop::collection::vec(arb_get_set_op::<u32>(4), 0..100)
    ) {
        test_packed_underlier_get_set_behaves_like_vec::<u128, u32>(ops);
    }
}
