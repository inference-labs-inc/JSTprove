use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::op_specs::{split_initializers, OpInputSpec};
use crate::onnx_builder::build_single_op_model;
use crate::runner::{ConformanceRunner, TestCase, TestResult};

/// Default seed suite: a fixed set of seeds committed to the codebase.
/// Covers baseline, adversarial shapes, boundary values, and extremes.
/// Adding new seeds at the end is non-breaking; do NOT reorder or remove existing entries.
pub const DEFAULT_SEEDS: &[u64] = &[
    0,           // baseline
    1,           // second baseline
    42,          // historical "magic" seed
    0xDEAD,      // large values
    0xBEEF,      // edge shapes
    0x1234_5678, // mixed
    u64::MAX,    // extreme seed
    0xCAFE_BABE,
    0x0102_0304,
    0xFFFF_0000,
];

/// Assembles `TestCase`s from an `OpInputSpec` and a list of seeds.
pub struct TestCaseBuilder {
    pub spec: OpInputSpec,
}

impl TestCaseBuilder {
    pub fn new(spec: OpInputSpec) -> Self {
        Self { spec }
    }

    /// Generate `n` test cases, each seeded with `base_seed + i`.
    ///
    /// Determinism contract: identical results across runs, platforms, and Rust versions
    /// because we use `ChaCha8Rng::seed_from_u64`.
    pub fn build_cases(&self, base_seed: u64, n: usize) -> Vec<TestCase> {
        (0..n)
            .map(|i| self.build_one(base_seed.wrapping_add(i as u64)))
            .collect()
    }

    /// Build a single `TestCase` from a seed.
    pub fn build_one(&self, seed: u64) -> TestCase {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let spec = &self.spec;

        // Sample shapes and values for every input slot
        let sampled_shapes: Vec<Vec<usize>> = spec
            .inputs
            .iter()
            .scan((), |_, t| {
                // For Add: make both inputs have the same shape (use first shape for all)
                Some(t.shape.sample(&mut rng))
            })
            .collect();

        let sampled_data: Vec<Vec<i64>> = spec
            .inputs
            .iter()
            .zip(sampled_shapes.iter())
            .map(|(t, shape)| {
                let n: usize = shape.iter().product::<usize>().max(1);
                t.values.sample_tensor(n, &mut rng)
            })
            .collect();

        // Split into dynamic inputs and ONNX initializers
        let (initializers, dynamic_indices) =
            split_initializers(spec, &sampled_data, &sampled_shapes);

        // Build input_shapes tuples: (name, dims_as_i64, elem_type)
        // elem_type 7 = INT64 (bypasses alpha-quantization in JSTProve)
        let input_shapes: Vec<(&str, Vec<i64>, i32)> = dynamic_indices
            .iter()
            .map(|&i| {
                let dims: Vec<i64> = sampled_shapes[i].iter().map(|&d| d as i64).collect();
                (spec.inputs[i].name, dims, 7i32)
            })
            .collect();

        // For output shapes we need to compute the output shape from the op.
        // For simple element-wise ops, output shape == first input shape.
        // For Gemm [M, K] × [K, N] → [M, N].
        let output_shape = compute_output_shape(spec, &sampled_shapes, &dynamic_indices);

        // Flatten dynamic input data for TestCase (initializer data is baked into ONNX)
        let flat_inputs: Vec<Vec<i64>> = dynamic_indices
            .iter()
            .map(|&i| sampled_data[i].clone())
            .collect();

        // Build input_shapes slices for the builder function
        let input_shapes_ref: Vec<(&str, &[i64], i32)> = input_shapes
            .iter()
            .map(|(name, dims, et)| (*name, dims.as_slice(), *et))
            .collect();

        let output_shapes_ref: &[(&str, &[i64], i32)] = &[("output", &output_shape, 7)];

        let onnx_bytes = build_single_op_model(
            spec.op_name,
            &input_shapes_ref,
            output_shapes_ref,
            &spec.attrs,
            &initializers,
        )
        .expect("build_single_op_model failed in TestCaseBuilder");

        TestCase {
            op_name: spec.op_name,
            seed,
            onnx_bytes,
            inputs: flat_inputs,
            tolerance: spec.tolerance.clone(),
        }
    }
}

/// Compute the output shape for a generated test case.
/// For element-wise ops (Relu, Add), output shape = first dynamic input shape.
/// For Gemm, output shape = [M, N] where A=[M,K], B=[K,N].
fn compute_output_shape(
    spec: &OpInputSpec,
    sampled_shapes: &[Vec<usize>],
    dynamic_indices: &[usize],
) -> Vec<i64> {
    match spec.op_name {
        "Gemm" => {
            // A is dynamic (index 0), B is initializer (index 1)
            // A: [M, K], B: [K, N] → output [M, N]
            let a_shape = &sampled_shapes[0];
            let b_shape = &sampled_shapes[1];
            if a_shape.len() == 2 && b_shape.len() == 2 {
                vec![a_shape[0] as i64, b_shape[1] as i64]
            } else {
                // Fallback: use first dynamic input shape
                first_dynamic_shape(sampled_shapes, dynamic_indices)
            }
        }
        _ => {
            // Element-wise: output shape = first dynamic input shape
            first_dynamic_shape(sampled_shapes, dynamic_indices)
        }
    }
}

fn first_dynamic_shape(sampled_shapes: &[Vec<usize>], dynamic_indices: &[usize]) -> Vec<i64> {
    if let Some(&i) = dynamic_indices.first() {
        sampled_shapes[i].iter().map(|&d| d as i64).collect()
    } else if !sampled_shapes.is_empty() {
        sampled_shapes[0].iter().map(|&d| d as i64).collect()
    } else {
        vec![1]
    }
}

/// Attempt to produce a simpler `TestCase` that also triggers the same failure.
///
/// Strategy:
/// 1. Try halving each input tensor's values element-by-element.
/// 2. Try replacing input tensors with shorter versions.
/// Stop after 50 shrink attempts. Return the smallest failing case found.
pub fn shrink(original: &TestCase, runner: &ConformanceRunner) -> TestCase {
    const MAX_ATTEMPTS: usize = 50;

    let mut best = original.clone();
    let mut attempts = 0;

    // Phase 1: try shrinking values (halve each value)
    let shrunken_inputs: Vec<Vec<i64>> = best
        .inputs
        .iter()
        .map(|v| v.iter().map(|&x| x / 2).collect())
        .collect();

    let candidate = TestCase {
        inputs: shrunken_inputs.clone(),
        onnx_bytes: best.onnx_bytes.clone(),
        op_name: best.op_name,
        seed: best.seed,
        tolerance: best.tolerance.clone(),
    };
    attempts += 1;
    if is_still_failing(&candidate, runner) {
        best = candidate;
    }

    // Phase 2: try zeroing out individual elements
    'outer: for input_idx in 0..best.inputs.len() {
        for elem_idx in 0..best.inputs[input_idx].len() {
            if attempts >= MAX_ATTEMPTS {
                break 'outer;
            }
            let orig_val = best.inputs[input_idx][elem_idx];
            if orig_val == 0 {
                continue;
            }
            let mut new_inputs = best.inputs.clone();
            new_inputs[input_idx][elem_idx] = 0;
            let candidate = TestCase {
                inputs: new_inputs,
                onnx_bytes: best.onnx_bytes.clone(),
                op_name: best.op_name,
                seed: best.seed,
                tolerance: best.tolerance.clone(),
            };
            attempts += 1;
            if is_still_failing(&candidate, runner) {
                best = candidate;
            }
        }
    }

    // Phase 3: try replacing large values with smaller magnitudes
    'outer2: for input_idx in 0..best.inputs.len() {
        for elem_idx in 0..best.inputs[input_idx].len() {
            if attempts >= MAX_ATTEMPTS {
                break 'outer2;
            }
            let orig_val = best.inputs[input_idx][elem_idx];
            let smaller = orig_val / 2;
            if smaller == orig_val {
                continue;
            }
            let mut new_inputs = best.inputs.clone();
            new_inputs[input_idx][elem_idx] = smaller;
            let candidate = TestCase {
                inputs: new_inputs,
                onnx_bytes: best.onnx_bytes.clone(),
                op_name: best.op_name,
                seed: best.seed,
                tolerance: best.tolerance.clone(),
            };
            attempts += 1;
            if is_still_failing(&candidate, runner) {
                best = candidate;
            }
        }
    }

    best
}

fn is_still_failing(case: &TestCase, runner: &ConformanceRunner) -> bool {
    matches!(runner.run(case), TestResult::Fail(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::op_specs::relu_spec;

    #[test]
    fn build_cases_is_deterministic() {
        let spec = relu_spec();
        let builder = TestCaseBuilder::new(spec);

        let run_a = builder.build_cases(0, 5);
        let run_b = builder.build_cases(0, 5);

        for (a, b) in run_a.iter().zip(run_b.iter()) {
            assert_eq!(a.inputs, b.inputs, "inputs must be identical for same seed");
            assert_eq!(a.seed, b.seed);
        }
    }

    #[test]
    fn build_cases_different_seeds_differ() {
        let spec = relu_spec();
        let builder = TestCaseBuilder::new(spec);

        let case0 = builder.build_one(0);
        let case1 = builder.build_one(1);
        // With overwhelming probability, two different seeds produce different inputs
        assert_ne!(
            case0.inputs, case1.inputs,
            "different seeds should produce different inputs"
        );
    }

    #[test]
    fn build_one_produces_valid_onnx() {
        let spec = relu_spec();
        let builder = TestCaseBuilder::new(spec);
        let case = builder.build_one(42);
        assert!(
            !case.onnx_bytes.is_empty(),
            "onnx_bytes should not be empty"
        );
        assert_eq!(case.op_name, "Relu");
    }

    #[test]
    fn default_seeds_count() {
        assert!(DEFAULT_SEEDS.len() >= 7, "should have at least 7 seeds");
    }

    #[test]
    fn shrink_returns_original_when_no_failure() {
        // A passing case: shrink should return the original (unchanged)
        let spec = relu_spec();
        let builder = TestCaseBuilder::new(spec);
        let case = builder.build_one(0);

        // Use reference_only runner — always passes, so shrink can't find a "smaller failure"
        let runner = ConformanceRunner {
            reference_only: true,
        };
        let shrunken = shrink(&case, &runner);

        // shrink returns original when no failure is found
        assert_eq!(shrunken.seed, case.seed);
    }
}
