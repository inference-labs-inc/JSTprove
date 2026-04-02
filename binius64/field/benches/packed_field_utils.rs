// Copyright 2024-2025 Irreducible Inc.

use criterion::{BenchmarkGroup, measurement::WallTime};

pub fn run_benchmark<R>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    func: impl Fn() -> Batch<R>,
) {
    group.bench_function(name, |bench| bench.iter(&func));
}

/// This number is chosen for values to fit into L1 cache
pub const BATCH_SIZE: usize = 32;
pub type Batch<T> = [T; BATCH_SIZE];

/// Helper macro to benchmark several operation strategies on many packed types
#[allow(unused_macros)]
macro_rules! benchmark_packed_operation {
	// run benchmark for a single type for single strategy
	(@declare_func op_name @ $op_name:ident, bench_type @ binary_op, $packed_field:ty, $strategy_name:ident, $constraint:path, $func:ident) => {
		paste::paste! {
            #[allow(non_snake_case)]
			#[inline(never)]
			fn [<$op_name $packed_field $strategy_name>](group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                a: &$crate::packed_field_utils::Batch<$packed_field>,
                b: &$crate::packed_field_utils::Batch<$packed_field>) {
				#[allow(unused)]
				trait BenchmarkFallback {
					const ENABLED: bool = false;

					fn bench<T>(_: T, _: T) -> T { unreachable!() }
				}

				impl<T> BenchmarkFallback for T {}

				struct BenchmarkImpl<T>(T);

				#[allow(unused)]
				impl<T: $constraint + Copy> BenchmarkImpl<T>{
					const ENABLED: bool = true;

					fn bench(a: T, b: T) -> T {
						$func(a, b)
					}
				}

				// use trick similar to the `impls` crate to run benchmark only if constraint
				// is satisfied.
				if BenchmarkImpl::<$packed_field>::ENABLED {
					$crate::packed_field_utils::run_benchmark(group, &stringify!($strategy_name),
						|| -> $crate::packed_field_utils::Batch<$packed_field> {
							std::array::from_fn(|i| BenchmarkImpl::<$packed_field>::bench(a[i], b[i]))
						});
				}
			}
		}
	};
	(@declare_func op_name @ $op_name:ident, bench_type @ unary_op, $packed_field:ty, $strategy_name:ident, $constraint:path, $func:ident) => {
		paste::paste! {
            #[allow(non_snake_case)]
			#[inline(never)]
			fn [<$op_name $packed_field $strategy_name>](group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                a: &$crate::packed_field_utils::Batch<$packed_field>,
                _b: &$crate::packed_field_utils::Batch<$packed_field>) {
				#[allow(unused)]
				trait BenchmarkFallback {
					const ENABLED: bool = false;

					fn bench<T>(_: T) -> T { unreachable!() }
				}

				impl<T> BenchmarkFallback for T {}

				struct BenchmarkImpl<T>(T);

				#[allow(unused)]
				impl<T: $constraint + Copy> BenchmarkImpl<T>{
					const ENABLED: bool = true;

					#[inline(always)]
					fn bench(a: T) -> T {
						$func(a)
					}
				}

				// use trick similar to the `impls` crate to run benchmark only if constraint
				// is satisfied.
				if BenchmarkImpl::<$packed_field>::ENABLED {
					$crate::packed_field_utils::run_benchmark(group, &stringify!($strategy_name),
						|| -> $crate::packed_field_utils::Batch<$packed_field> {
							std::array::from_fn(|i| BenchmarkImpl::<$packed_field>::bench(a[i]))
						});
				}
			}
		}
	};
	(@declare_func op_name @ $op_name:ident, bench_type @ transformation, $packed_field:ty, $strategy_name:ident, $constraint:path, $func:ident) => {
		paste::paste! {
            #[allow(non_snake_case)]
			#[inline(never)]
			fn [<$op_name $packed_field $strategy_name>](group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
                a: &$crate::packed_field_utils::Batch<$packed_field>,
                _b: &$crate::packed_field_utils::Batch<$packed_field>) {
				struct EmptyTransformation{}

				impl<I, O> Transformation<I, O> for EmptyTransformation {
					fn transform(&self, _: &I) -> O {
						unreachable!();
					}
				}

				#[allow(unused)]
				trait BenchmarkFallback {
					const ENABLED: bool = false;

					fn make_packed_transformation<T>(_: T) -> EmptyTransformation {
						EmptyTransformation {}
					}
				}

				impl<T> BenchmarkFallback for T {}

				struct BenchmarkImpl<T>(T);

				#[allow(unused)]
				impl<T: $constraint + Copy> BenchmarkImpl<T>{
					const ENABLED: bool = true;

					#[inline(always)]
					fn make_packed_transformation(_: T) -> impl Transformation<T, T> {
						$func::<T>()
					}
				}

				// use trick similar to the `impls` crate to run benchmark only if constraint
				// is satisfied.
				if BenchmarkImpl::<$packed_field>::ENABLED {
					let transformation = BenchmarkImpl::<$packed_field>::make_packed_transformation(<$packed_field>::default());
					$crate::packed_field_utils::run_benchmark(group, &stringify!($strategy_name),
						|| -> $crate::packed_field_utils::Batch<$packed_field> {
							std::array::from_fn(|i| transformation.transform(&a[i]))
						});
				}
			}
		}
	};
	// run benchmark on a single type for all strategies
	(@declare_func $packed_field:ty, op_name @ $op_name:ident, bench_type @ $benchmark_type:ident, strategies @ ($(($strategy_name:ident, $constraint:path, $func:ident),)*)) => {
		$(
			benchmark_packed_operation!(@declare_func op_name @ $op_name, bench_type @ $benchmark_type, $packed_field, $strategy_name, $constraint, $func);
		)*
	};
	(@run_func $group:ident, $packed_field:ty, $a:ident, $b:ident, op_name @ $op_name:ident, strategies @ ($(($strategy_name:ident, $constraint:path, $func:ident),)*)) => {
		$(
			paste::paste! {
				[<$op_name $packed_field $strategy_name>](&mut $group, &$a, &$b);
			}
		)*
	};
	// Run list of strategies for the list of fields
	(op_name @ $op_name:ident, bench_type @ $benchmark_type:ident, strategies @ $strategies:tt, packed_fields @ [$($packed_field:ident)*]) => {
		$(
			benchmark_packed_operation!(@declare_func $packed_field, op_name @ $op_name, bench_type @ $benchmark_type, strategies @ $strategies);
		)*

        $(
            #[allow(non_snake_case)]
			#[inline(never)]
            fn $packed_field(c: &mut criterion::Criterion) {
				use binius_field::Random;

                let mut group = c.benchmark_group(format!("{}/{}", stringify!($op_name), stringify!($packed_field)));
                group.warm_up_time(core::time::Duration::from_secs(1));
                group.measurement_time(core::time::Duration::from_secs(3));
                group.throughput(criterion::Throughput::Elements((<$packed_field as binius_field::PackedField>::WIDTH *  $crate::packed_field_utils::BATCH_SIZE) as _));

                let mut rng = rand::rng();
                let a: $crate::packed_field_utils::Batch<$packed_field> = std::array::from_fn(|_| <$packed_field>::random(&mut rng));
                let b: $crate::packed_field_utils::Batch<$packed_field> = std::array::from_fn(|_| <$packed_field>::random(&mut rng));

                benchmark_packed_operation!(@run_func group, $packed_field, a, b, op_name @ $op_name, strategies @ $strategies);

                group.finish();
            }
		)*

        criterion::criterion_group!($op_name, $($packed_field),*);
	};
	// Run given strategies on the full list of types
	(op_name @ $op_name:ident, bench_type @ $benchmark_type:ident, strategies @ $strategies:tt) => {
		benchmark_packed_operation!(
			op_name @ $op_name,
			bench_type @ $benchmark_type,
			strategies @ $strategies,
			packed_fields @ [
				// B1
				PackedBinaryField128x1b
				PackedBinaryField256x1b
				PackedBinaryField512x1b

				// AES
				PackedAESBinaryField16x8b
				PackedAESBinaryField32x8b
				PackedAESBinaryField64x8b

				// GHASH
				PackedBinaryGhash1x128b
				PackedBinaryGhash2x128b
				PackedBinaryGhash4x128b
			]);
	};
}

#[allow(unused_imports)]
pub(crate) use benchmark_packed_operation;
