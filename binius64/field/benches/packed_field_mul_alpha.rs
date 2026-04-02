// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
    arch::{
        packed_128::*, packed_256::*, packed_512::*, packed_aes_128::*, packed_aes_256::*,
        packed_aes_512::*, packed_ghash_128::*, packed_ghash_256::*, packed_ghash_512::*,
    },
    arithmetic_traits::MulAlpha,
};
use cfg_if::cfg_if;
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn mul_alpha_main<T: MulAlpha>(val: T) -> T {
    val.mul_alpha()
}

cfg_if! {
    if #[cfg(feature = "benchmark_alternative_strategies")] {
        use binius_field::{
            arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy,
                PairwiseTableStrategy, },
            arithmetic_traits::TaggedMulAlpha
        };

        fn mul_alpha_pairwise<T: TaggedMulAlpha<PairwiseStrategy>>(val: T) -> T {
            val.mul_alpha()
        }

        fn mul_alpha_pairwise_recursive<T: TaggedMulAlpha<PairwiseRecursiveStrategy>>(val: T) -> T {
            val.mul_alpha()
        }

        fn mul_alpha_pairwise_table<T: TaggedMulAlpha<PairwiseTableStrategy>>(val: T) -> T {
            val.mul_alpha()
        }

        fn mul_alpha_packed<T: TaggedMulAlpha<PackedStrategy>>(val: T) -> T {
            val.mul_alpha()
        }

        benchmark_packed_operation!(
            op_name @ mul_alpha,
            bench_type @ unary_op,
            strategies @ (
                (main, MulAlpha, mul_alpha_main),
                (pairwise, TaggedMulAlpha::<PairwiseStrategy>, mul_alpha_pairwise),
                (pairwise_recursive, TaggedMulAlpha::<PairwiseRecursiveStrategy>, mul_alpha_pairwise_recursive),
                (pairwise_table, TaggedMulAlpha::<PairwiseTableStrategy>, mul_alpha_pairwise_table),
                (packed, TaggedMulAlpha::<PackedStrategy>, mul_alpha_packed),
            )
        );
    } else {
        benchmark_packed_operation!(
            op_name @ mul_alpha,
            bench_type @ unary_op,
            strategies @ (
                (main, MulAlpha, mul_alpha_main),
            )
        );
    }
}

criterion_main!(mul_alpha);
