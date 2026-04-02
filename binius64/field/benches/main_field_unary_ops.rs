// Copyright 2025 Irreducible Inc.

mod packed_field_utils;

use binius_field::{
    PackedField,
    arch::{OptimalPackedB1, OptimalPackedB128},
};
use criterion::criterion_main;
use packed_field_utils::benchmark_packed_operation;

fn invert<T: PackedField>(val: T) -> T {
    val.invert_or_zero()
}

fn square<T: PackedField>(val: T) -> T {
    val.square()
}

benchmark_packed_operation!(
    op_name @ main_unary_ops,
    bench_type @ unary_op,
    strategies @ (
        (main_invert, PackedField, invert),
        (main_square, PackedField, square),
    ),
    packed_fields @ [
        OptimalPackedB1
        OptimalPackedB128
    ]
);

criterion_main!(main_unary_ops);
