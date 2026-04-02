// Copyright 2025 Irreducible Inc.

mod packed_field_utils;

use std::ops::Mul;

use binius_field::arch::{OptimalPackedB1, OptimalPackedB128};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

trait SelfMul: Mul<Self, Output = Self> + Sized {}

impl<T: Mul<Self, Output = Self> + Sized> SelfMul for T {}

fn mul_main<T: SelfMul>(lhs: T, rhs: T) -> T {
    lhs * rhs
}

benchmark_packed_operation!(
    op_name @ main_binary_ops,
    bench_type @ binary_op,
    strategies @ (
        (mul_main, SelfMul, mul_main),
    ),
    packed_fields @ [
        OptimalPackedB1
        OptimalPackedB128
    ]
);

criterion_main!(main_binary_ops);
