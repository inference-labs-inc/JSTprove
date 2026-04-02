// Copyright 2024-2025 Irreducible Inc.

mod packed_field_utils;

use std::ops::Mul;

use binius_field::arch::{
    packed_128::*, packed_256::*, packed_512::*, packed_aes_128::*, packed_aes_256::*,
    packed_aes_512::*, packed_ghash_128::*, packed_ghash_256::*, packed_ghash_512::*,
};
use cfg_if::cfg_if;
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

/// This trait is needed to specify `Mul` constraint only
trait SelfMul: Mul<Self, Output = Self> + Sized {}

impl<T: Mul<Self, Output = Self> + Sized> SelfMul for T {}

fn mul_main<T: SelfMul>(lhs: T, rhs: T) -> T {
    lhs * rhs
}

cfg_if! {
    if #[cfg(feature = "benchmark_alternative_strategies")] {
        use binius_field::{
            arch::{PackedStrategy, PairwiseRecursiveStrategy, PairwiseStrategy,
                PairwiseTableStrategy, },
            arithmetic_traits::TaggedMul
        };

        fn mul_pairwise<T: TaggedMul<PairwiseStrategy>>(lhs: T, rhs: T) -> T {
            TaggedMul::<PairwiseStrategy>::mul(lhs, rhs)
        }

        fn mul_pairwise_table<T: TaggedMul<PairwiseTableStrategy>>(lhs: T, rhs: T) -> T {
            TaggedMul::<PairwiseTableStrategy>::mul(lhs, rhs)
        }

        fn mul_pairwise_recursive<T: TaggedMul<PairwiseRecursiveStrategy>>(lhs: T, rhs: T) -> T {
            TaggedMul::<PairwiseRecursiveStrategy>::mul(lhs, rhs)
        }

        fn mul_packed<T: TaggedMul<PackedStrategy>>(lhs: T, rhs: T) -> T {
            TaggedMul::<PackedStrategy>::mul(lhs, rhs)
        }

        benchmark_packed_operation!(
            op_name @ multiply,
            bench_type @ binary_op,
            strategies @ (
                (main, SelfMul, mul_main),
                (pairwise, TaggedMul::<PairwiseStrategy>, mul_pairwise),
                (pairwise_recursive, TaggedMul::<PairwiseRecursiveStrategy>, mul_pairwise_recursive),
                (pairwise_table, TaggedMul::<PairwiseTableStrategy>, mul_pairwise_table),
                (packed, TaggedMul::<PackedStrategy>, mul_packed),
            )
        );
    } else {
        benchmark_packed_operation!(
            op_name @ multiply,
            bench_type @ binary_op,
            strategies @ (
                (main, SelfMul, mul_main),
            )
        );
    }
}

criterion_main!(multiply);
