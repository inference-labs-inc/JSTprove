// Copyright 2025 Irreducible Inc.

use binius_field::{
    BinaryField, FieldOps, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b,
    PackedBinaryGhash4x128b, PackedField,
};
use binius_math::{
    ntt::{
        AdditiveNTT, DomainContext, NeighborsLastBreadthFirst, NeighborsLastMultiThread,
        NeighborsLastSingleThread,
        domain_context::{GaoMateerOnTheFly, GaoMateerPreExpanded},
    },
    test_utils::random_field_buffer,
};
use binius_utils::{env::boolean_env_flag_set, rayon::ThreadPoolBuilder};
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};

/// `Standard` means it reports the standard input_size / time throughput.
/// `Multiplication` means it reports num_multiplications / time instead as throughput.
///
/// The `Multiplication` variant is useful to compare against the raw multiplication throughput
/// (from the field benchmarks), so one can see the overhead of the NTT.
#[derive(Copy, Clone)]
enum ThroughputVariant {
    Standard,
    Multiplication,
}

/// Benches different NTT implementations with a specific `PackedField` and specific parameter
/// choice.
///
/// `log_y` is computed automatically from `log_x`, `log_z`, and the size of `data`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::single_element_loop)]
fn bench_ntts<F: BinaryField, P: PackedField<Scalar = F>>(
    group: &mut BenchmarkGroup<WallTime>,
    throughput_var: ThroughputVariant,
    log_d: usize,
    domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
    domain_context_name: &str,
    skip_early: usize,
    skip_late: usize,
) {
    let mut rng = rand::rng();

    let parameter = format!("log_d={log_d}/skip_early={skip_early}/skip_late={skip_late}");

    let throughput = match throughput_var {
        ThroughputVariant::Standard => Throughput::Bytes(((F::N_BITS / 8) << log_d) as u64),
        ThroughputVariant::Multiplication => {
            Throughput::Elements(num_muls(log_d, skip_early, skip_late))
        }
    };
    group.throughput(throughput);

    for log_base_len in [10] {
        let ntt_name = format!("singlethread/log_base_len={log_base_len}/{domain_context_name}");
        group.bench_function(BenchmarkId::new(&ntt_name, &parameter), |b| {
            let ntt = NeighborsLastSingleThread {
                domain_context,
                log_base_len,
            };

            let mut data = random_field_buffer::<P>(&mut rng, log_d);
            b.iter(|| ntt.forward_transform(data.to_mut(), skip_early, skip_late))
        });
    }

    let ntt_name = format!("neighbors_last_breadth_first/{domain_context_name}");
    group.bench_function(BenchmarkId::new(&ntt_name, &parameter), |b| {
        let ntt = NeighborsLastBreadthFirst { domain_context };

        let mut data = random_field_buffer::<P>(&mut rng, log_d);
        b.iter(|| ntt.forward_transform(data.to_mut(), skip_early, skip_late))
    });

    for log_num_shares in [0, 3] {
        for log_base_len in [4, 8, 12, 16] {
            let num_threads = 1 << log_num_shares;
            let ntt_name = format!(
                "multithread/threads={num_threads}/log_base_len={log_base_len}/{domain_context_name}"
            );
            group.bench_function(BenchmarkId::new(&ntt_name, &parameter), |b| {
                let ntt = NeighborsLastMultiThread {
                    domain_context,
                    log_base_len,
                    log_num_shares,
                };

                let mut data = random_field_buffer::<P>(&mut rng, log_d);

                let thread_pool = ThreadPoolBuilder::new()
                    .num_threads(num_threads)
                    .build()
                    .unwrap();
                thread_pool.install(|| {
                    b.iter(|| ntt.forward_transform(data.to_mut(), skip_early, skip_late))
                })
            });
        }
    }
}

/// Macro that generates benchmarks for a matrix of parameters.
///
/// This macro generates benchmark code for all combinations of:
/// - DomainContext implementations
/// - PackedField types
/// - log_d values
/// - skip_early/skip_late pairs
///
/// Implementation uses macro recursion to process domain contexts sequentially,
/// avoiding Rust's nested repetition depth limitations.
macro_rules! bench_ntt_matrix {
	// Internal helper: process a single domain context with all fields
	(
		@single_dc
		criterion = $c:expr,
		throughput = $throughput_var:expr,
		domain_context = ($dc_name:expr, $dc_type:ty, $dc_constructor:expr),
		fields = [ $( ($field_type:ty, $field_name:expr) ),* $(,)? ],
		log_d = $log_d_array:expr,
		skip_params = $skip_params_array:expr
	) => {
		// Single level of repetition over fields
		$(
			{
				type F = <$field_type as FieldOps>::Scalar;
				let mut group = $c.benchmark_group($field_name);

				for log_d in $log_d_array {
					let domain_context: $dc_type = $dc_constructor(log_d);
					let domain_context_name = $dc_name;

					if log_d >= 24 {
						group.sample_size(10);
					} else if log_d >= 20 {
						group.sample_size(40);
					}

					for (skip_early, skip_late) in $skip_params_array {
						bench_ntts::<F, $field_type>(
							&mut group,
							$throughput_var,
							log_d,
							&domain_context,
							domain_context_name,
							skip_early,
							skip_late,
						);
					}
				}
			}
		)*
	};

	// Main entry point: process domain contexts recursively
	(
		criterion = $c:expr,
		throughput = $throughput_var:expr,
		domain_contexts = [
			($dc_name:expr, $dc_type:ty, $dc_constructor:expr)
			$(, ($rest_dc_name:expr, $rest_dc_type:ty, $rest_dc_constructor:expr) )*
			$(,)?
		],
		fields = $fields:tt,
		log_d = $log_d_array:expr,
		skip_params = $skip_params_array:expr
		$(,)?
	) => {
		// Process first domain context
		bench_ntt_matrix! {
			@single_dc
			criterion = $c,
			throughput = $throughput_var,
			domain_context = ($dc_name, $dc_type, $dc_constructor),
			fields = $fields,
			log_d = $log_d_array,
			skip_params = $skip_params_array
		}

		// Recurse on remaining domain contexts
		$(
			bench_ntt_matrix! {
				@single_dc
				criterion = $c,
				throughput = $throughput_var,
				domain_context = ($rest_dc_name, $rest_dc_type, $rest_dc_constructor),
				fields = $fields,
				log_d = $log_d_array,
				skip_params = $skip_params_array
			}
		)*
	};
}

/// Calls benchmarks with all parameter combinations.
fn bench_fields(c: &mut Criterion) {
    let throughput_var = determine_throughput_variant();

    bench_ntt_matrix! {
        criterion = c,
        throughput = throughput_var,
        domain_contexts = [
            ("on-the-fly", GaoMateerOnTheFly<_>, GaoMateerOnTheFly::generate),
            ("pre-expanded", GaoMateerPreExpanded<_>, GaoMateerPreExpanded::generate),
        ],
        fields = [
            (PackedBinaryGhash1x128b, "1xGhash"),
            (PackedBinaryGhash2x128b, "2xGhash"),
            (PackedBinaryGhash4x128b, "4xGhash"),
        ],
        log_d = [16, 20, 24],
        skip_params = [(0, 0), (4, 0), (0, 4)],
    }
}

/// Gives the number of raw field multiplications that are done for an NTT with specific parameters.
fn num_muls(log_d: usize, skip_early: usize, skip_late: usize) -> u64 {
    let num_rounds = log_d - skip_late - skip_early;
    let muls_per_round = 1u64 << (log_d - 1);

    num_rounds as u64 * muls_per_round
}

/// Determine the throughput variant based on an environment variable.
fn determine_throughput_variant() -> ThroughputVariant {
    const VAR_NAME: &str = "NTT_MUL_THROUGHPUT";

    if boolean_env_flag_set(VAR_NAME) {
        println!("{VAR_NAME} is activated - using *multiplication* throughput");
        ThroughputVariant::Multiplication
    } else {
        println!("{VAR_NAME} is NOT activated - using *standard* throughput");
        println!(
            "NOTE: Use {VAR_NAME}=1 to see multiplication throughput instead of normal throughput"
        );
        ThroughputVariant::Standard
    }
}

criterion_group!(default, bench_fields);
criterion_main!(default);
