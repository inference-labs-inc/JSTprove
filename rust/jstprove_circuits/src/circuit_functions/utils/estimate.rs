use std::collections::HashMap;
use std::hash::BuildHasher;

use super::onnx_model::{Architecture, CircuitParams};
use crate::circuit_functions::gadgets::DEFAULT_LOGUP_CHUNK_BITS;

#[derive(Debug, Clone)]
pub struct EstimationConfig {
    pub n_bits: usize,
    pub chunk_bits: usize,
    pub scale_exponent: usize,
    pub freivalds_reps: usize,
    pub rescale_config: HashMap<String, bool>,
}

impl EstimationConfig {
    #[must_use]
    pub fn from_circuit_params(params: &CircuitParams, default_n_bits: usize) -> Self {
        Self {
            n_bits: default_n_bits,
            chunk_bits: params.logup_chunk_bits.unwrap_or(DEFAULT_LOGUP_CHUNK_BITS),
            scale_exponent: params.scale_exponent as usize,
            freivalds_reps: params.freivalds_reps,
            rescale_config: params.rescale_config.clone(),
        }
    }

    #[must_use]
    pub fn bn254_defaults() -> Self {
        Self {
            n_bits: 64,
            chunk_bits: DEFAULT_LOGUP_CHUNK_BITS,
            scale_exponent: 18,
            freivalds_reps: 1,
            rescale_config: HashMap::new(),
        }
    }

    #[must_use]
    pub fn goldilocks_defaults() -> Self {
        Self {
            n_bits: 31,
            chunk_bits: DEFAULT_LOGUP_CHUNK_BITS,
            scale_exponent: 18,
            freivalds_reps: 1,
            rescale_config: HashMap::new(),
        }
    }

    fn rescale_queries_per_element(&self) -> u64 {
        let remainder = self.scale_exponent.div_ceil(self.chunk_bits);
        let quotient = self.n_bits.div_ceil(self.chunk_bits);
        (remainder + quotient) as u64
    }

    fn relu_queries_per_element(&self) -> u64 {
        self.n_bits.div_ceil(self.chunk_bits) as u64
    }
}

#[derive(Debug, Clone)]
pub struct LayerEstimate {
    pub name: String,
    pub op_type: String,
    pub output_elements: u64,
    pub arithmetic: u64,
    pub range_check_queries: u64,
}

impl LayerEstimate {
    #[must_use]
    pub fn total(&self) -> u64 {
        self.arithmetic.saturating_add(self.range_check_queries)
    }
}

#[derive(Debug, Clone)]
pub struct CircuitEstimate {
    pub layers: Vec<LayerEstimate>,
    pub total_arithmetic: u64,
    pub total_range_queries: u64,
}

impl CircuitEstimate {
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total_arithmetic
            .saturating_add(self.total_range_queries)
    }
}

fn product(shape: &[usize]) -> u64 {
    shape.iter().map(|&d| d.max(1) as u64).product()
}

#[allow(clippy::cast_possible_truncation)]
fn matmul_cost(rows: u64, inner: u64, cols: u64, freivalds_reps: usize) -> u64 {
    let rows_w = u128::from(rows);
    let inner_w = u128::from(inner);
    let cols_w = u128::from(cols);
    let cost_full = 2 * rows_w * inner_w * cols_w - rows_w * cols_w;

    if freivalds_reps > 0 {
        let pair_sum = rows_w * inner_w + rows_w * cols_w + inner_w * cols_w;
        let overhead = inner_w + 2 * rows_w;
        if 2 * pair_sum > overhead {
            let delta = 2 * pair_sum - overhead;
            let cost_freivalds = u128::from(freivalds_reps as u64) * delta;
            if cost_freivalds < cost_full {
                return cost_freivalds.min(u128::from(u64::MAX)) as u64;
            }
        }
    }
    cost_full.min(u128::from(u64::MAX)) as u64
}

fn resolve_shape<'a, S: BuildHasher>(
    name: &str,
    shapes: &'a HashMap<String, Vec<usize>, S>,
) -> Option<&'a Vec<usize>> {
    shapes.get(name)
}

fn estimate_linear<S: BuildHasher>(
    input_names: &[String],
    output_names: &[String],
    shapes: &HashMap<String, Vec<usize>, S>,
    out_elems: u64,
    rescale_queries: u64,
    config: &EstimationConfig,
) -> (u64, u64, u64) {
    let in0_shape = input_names.first().and_then(|n| resolve_shape(n, shapes));
    let out_shape = output_names.first().and_then(|n| resolve_shape(n, shapes));
    let (ell, m, n) = matmul_dims(in0_shape, out_shape);
    let batch = matmul_batch_from_output(out_shape, ell, n);
    let arith = batch.saturating_mul(matmul_cost(ell, m, n, config.freivalds_reps));
    (out_elems, arith, rescale_queries)
}

fn estimate_conv<S: BuildHasher>(
    input_names: &[String],
    shapes: &HashMap<String, Vec<usize>, S>,
    out_elems: u64,
    rescale_queries: u64,
) -> (u64, u64, u64) {
    let in1_shape = input_names.get(1).and_then(|n| resolve_shape(n, shapes));
    let kernel_volume = in1_shape.map_or(9, |w| {
        if w.len() >= 4 {
            w[2..].iter().map(|&d| d as u64).product()
        } else {
            1
        }
    });
    let channels_per_group = in1_shape.map_or(1, |w| if w.len() >= 2 { w[1] as u64 } else { 1 });
    let inner_dim = kernel_volume.saturating_mul(channels_per_group);
    let arith = out_elems.saturating_mul(inner_dim.saturating_mul(2).saturating_sub(1));
    (out_elems, arith, rescale_queries)
}

fn estimate_elementwise_activation(
    out_elems: u64,
    queries_mult: u64,
    arith_mult: u64,
    config: &EstimationConfig,
) -> (u64, u64, u64) {
    let queries = out_elems.saturating_mul(
        config
            .relu_queries_per_element()
            .saturating_mul(queries_mult),
    );
    let arith = out_elems.saturating_mul(arith_mult);
    (out_elems, arith, queries)
}

fn estimate_pool<S: BuildHasher>(
    op_type: &str,
    input_names: &[String],
    shapes: &HashMap<String, Vec<usize>, S>,
    out_shape: Option<&Vec<usize>>,
    out_elems: u64,
    config: &EstimationConfig,
) -> (u64, u64, u64) {
    let in0_shape = input_names.first().and_then(|n| resolve_shape(n, shapes));
    if op_type == "MaxPool" {
        let kernel_elems = in0_shape.map_or(4, |s| {
            if s.len() >= 4 {
                s[2..].iter().map(|&d| d as u64).product::<u64>().max(1)
            } else {
                4
            }
        });
        let queries = out_elems
            .saturating_mul(kernel_elems.saturating_mul(config.relu_queries_per_element()));
        let arith = out_elems.saturating_mul(kernel_elems);
        (out_elems, arith, queries)
    } else {
        let pool_size = if op_type == "GlobalAveragePool" {
            in0_shape.map_or(1, |s| {
                if s.len() >= 4 {
                    s[2..].iter().map(|&d| d as u64).product()
                } else {
                    1
                }
            })
        } else {
            in0_shape.map_or(4, |s| {
                if s.len() >= 4 {
                    let in_spatial: u64 = s[2..].iter().map(|&d| d as u64).product();
                    let out_spatial = out_shape.map_or(1, |o| {
                        if o.len() >= 4 {
                            o[2..].iter().map(|&d| d as u64).product()
                        } else {
                            1
                        }
                    });
                    if out_spatial > 0 {
                        in_spatial / out_spatial
                    } else {
                        in_spatial
                    }
                } else {
                    4
                }
            })
        };
        let arith = out_elems.saturating_mul(pool_size);
        let queries = out_elems.saturating_mul(config.rescale_queries_per_element());
        (out_elems, arith, queries)
    }
}

fn estimate_reduce<S: BuildHasher>(
    op_type: &str,
    input_names: &[String],
    shapes: &HashMap<String, Vec<usize>, S>,
    out_elems: u64,
    config: &EstimationConfig,
) -> (u64, u64, u64) {
    let in0_shape = input_names.first().and_then(|n| resolve_shape(n, shapes));
    let in_elems = in0_shape.map_or(out_elems, |s| product(s));

    match op_type {
        "ReduceMax" => {
            let reduction_factor = if out_elems > 0 {
                in_elems / out_elems
            } else {
                in_elems
            };
            let queries = out_elems
                .saturating_mul(reduction_factor.saturating_mul(config.relu_queries_per_element()));
            (out_elems, in_elems, queries)
        }
        "ReduceMean" => {
            let queries = out_elems.saturating_mul(config.rescale_queries_per_element());
            (out_elems, in_elems, queries)
        }
        _ => (out_elems, in_elems, 0),
    }
}

#[allow(clippy::too_many_lines)]
fn estimate_single_op<S: BuildHasher>(
    op_type: &str,
    input_names: &[String],
    output_names: &[String],
    shapes: &HashMap<String, Vec<usize>, S>,
    config: &EstimationConfig,
    is_rescale: bool,
) -> (u64, u64, u64) {
    let out_shape = output_names.first().and_then(|n| resolve_shape(n, shapes));
    let out_elems = out_shape.map_or(0, |s| product(s));

    let rescale_queries = if is_rescale {
        out_elems.saturating_mul(config.rescale_queries_per_element())
    } else {
        0
    };

    match op_type {
        "MatMul" | "Gemm" => estimate_linear(
            input_names,
            output_names,
            shapes,
            out_elems,
            rescale_queries,
            config,
        ),

        "Conv" | "ConvTranspose" => estimate_conv(input_names, shapes, out_elems, rescale_queries),

        "ReLU" | "Relu" | "LeakyRelu" | "LeakyReLU" => {
            estimate_elementwise_activation(out_elems, 1, 1, config)
        }

        "Sigmoid" | "Tanh" | "Gelu" | "Erf" | "HardSwish" | "Hardswish" => {
            estimate_elementwise_activation(out_elems, 3, 3, config)
        }

        "Exp" | "Log" | "Sqrt" | "Pow" | "Sin" | "Cos" => {
            estimate_elementwise_activation(out_elems, 2, 2, config)
        }

        "Softmax" => {
            let lane_size = out_shape.map_or(1, |s| *s.last().unwrap_or(&1) as u64);
            let n_lanes = if lane_size > 0 {
                out_elems / lane_size
            } else {
                out_elems
            };
            let per_lane_arith = lane_size.saturating_mul(4);
            let per_lane_queries =
                lane_size.saturating_mul(config.relu_queries_per_element().saturating_mul(3));
            let arith = n_lanes.saturating_mul(per_lane_arith);
            let queries = n_lanes.saturating_mul(per_lane_queries);
            (out_elems, arith, queries)
        }

        "LayerNormalization"
        | "InstanceNormalization"
        | "GroupNormalization"
        | "BatchNormalization" => {
            let arith = out_elems.saturating_mul(3);
            let queries = out_elems.saturating_mul(config.rescale_queries_per_element());
            (out_elems, arith, queries)
        }

        "Mul" => (out_elems, out_elems, rescale_queries),

        "Div" => {
            let queries = out_elems.saturating_mul(config.rescale_queries_per_element());
            (out_elems, out_elems, queries)
        }

        "Add" | "Sub" => (out_elems, out_elems, 0),

        "Max" | "Min" | "Clip" => {
            let queries = out_elems.saturating_mul(config.relu_queries_per_element());
            (out_elems, out_elems.saturating_mul(2), queries)
        }

        "Where" => (out_elems, out_elems.saturating_mul(2), 0),

        "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
            estimate_pool(op_type, input_names, shapes, out_shape, out_elems, config)
        }

        "ReduceMean" | "ReduceSum" | "ReduceMax" => {
            estimate_reduce(op_type, input_names, shapes, out_elems, config)
        }

        "TopK" => {
            let in0_shape = input_names.first().and_then(|n| resolve_shape(n, shapes));
            let in_elems = in0_shape.map_or(out_elems, |s| product(s));
            let queries = in_elems.saturating_mul(config.relu_queries_per_element());
            (out_elems, in_elems.saturating_mul(2), queries)
        }

        "Resize" | "GridSample" => {
            let arith = out_elems.saturating_mul(4);
            let queries = out_elems.saturating_mul(config.rescale_queries_per_element());
            (out_elems, arith, queries)
        }

        _ => (out_elems, 0, 0),
    }
}

fn matmul_dims(a_shape: Option<&Vec<usize>>, out_shape: Option<&Vec<usize>>) -> (u64, u64, u64) {
    let (ell, inner) = a_shape.map_or((1, 1), |s| {
        let rank = s.len();
        if rank >= 2 {
            (s[rank - 2] as u64, s[rank - 1] as u64)
        } else if rank == 1 {
            (1, s[0] as u64)
        } else {
            (1, 1)
        }
    });
    let cols = out_shape.map_or(inner, |s| {
        let rank = s.len();
        if rank >= 1 { s[rank - 1] as u64 } else { 1 }
    });
    (ell, inner, cols)
}

fn matmul_batch_from_output(out_shape: Option<&Vec<usize>>, ell: u64, cols: u64) -> u64 {
    out_shape.map_or(1, |s| {
        let total = product(s);
        let per_batch = ell.saturating_mul(cols).max(1);
        total / per_batch
    })
}

#[must_use]
pub fn estimate_circuit(
    params: &CircuitParams,
    architecture: &Architecture,
    default_n_bits: usize,
) -> CircuitEstimate {
    let config = EstimationConfig::from_circuit_params(params, default_n_bits);
    let shapes = super::onnx_model::collect_all_shapes(&architecture.architecture, &params.inputs);
    estimate_from_layers(&architecture.architecture, &shapes, &config)
}

#[must_use]
pub fn estimate_from_layers<S: BuildHasher>(
    layers: &[super::onnx_types::ONNXLayer],
    shapes: &HashMap<String, Vec<usize>, S>,
    config: &EstimationConfig,
) -> CircuitEstimate {
    let mut result = CircuitEstimate {
        layers: Vec::with_capacity(layers.len()),
        total_arithmetic: 0,
        total_range_queries: 0,
    };

    for layer in layers {
        let is_rescale = config
            .rescale_config
            .get(&layer.name)
            .copied()
            .unwrap_or(true);

        let (out_elems, arith, queries) = estimate_single_op(
            &layer.op_type,
            &layer.inputs,
            &layer.outputs,
            shapes,
            config,
            is_rescale,
        );

        result.total_arithmetic = result.total_arithmetic.saturating_add(arith);
        result.total_range_queries = result.total_range_queries.saturating_add(queries);

        result.layers.push(LayerEstimate {
            name: layer.name.clone(),
            op_type: layer.op_type.clone(),
            output_elements: out_elems,
            arithmetic: arith,
            range_check_queries: queries,
        });
    }

    result
}

#[must_use]
pub fn estimate_op_constraints(
    op_type: &str,
    input_shapes: &[Vec<usize>],
    output_shapes: &[Vec<usize>],
    config: &EstimationConfig,
) -> u64 {
    let mut shapes = HashMap::new();
    let input_names: Vec<String> = input_shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let name = format!("__input_{i}");
            shapes.insert(name.clone(), s.clone());
            name
        })
        .collect();
    let output_names: Vec<String> = output_shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let name = format!("__output_{i}");
            shapes.insert(name.clone(), s.clone());
            name
        })
        .collect();

    let (_, arith, queries) =
        estimate_single_op(op_type, &input_names, &output_names, &shapes, config, true);
    arith.saturating_add(queries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_direct_cheaper_than_freivalds_for_small_matrices() {
        let cost_direct = matmul_cost(2, 3, 2, 0);
        assert_eq!(cost_direct, 2 * 2 * 3 * 2 - 2 * 2);

        let cost_with_f = matmul_cost(2, 3, 2, 1);
        assert_eq!(cost_with_f, cost_direct);
    }

    #[test]
    fn matmul_freivalds_cheaper_for_large_matrices() {
        let cost_no_f = matmul_cost(128, 256, 128, 0);
        let cost_f = matmul_cost(128, 256, 128, 1);
        assert!(cost_f < cost_no_f);
    }

    #[test]
    fn zero_cost_ops_return_zero() {
        let config = EstimationConfig::bn254_defaults();
        for op in &[
            "Reshape",
            "Transpose",
            "Flatten",
            "Identity",
            "Squeeze",
            "Unsqueeze",
            "Constant",
            "Shape",
            "Cast",
        ] {
            let cost = estimate_op_constraints(op, &[vec![4, 8]], &[vec![4, 8]], &config);
            assert_eq!(cost, 0, "{op} should be zero-cost");
        }
    }

    #[test]
    fn relu_cost_scales_with_elements() {
        let config = EstimationConfig::bn254_defaults();
        let cost_small = estimate_op_constraints("ReLU", &[vec![1, 10]], &[vec![1, 10]], &config);
        let cost_large = estimate_op_constraints("ReLU", &[vec![1, 100]], &[vec![1, 100]], &config);
        assert_eq!(cost_large, cost_small * 10);
    }

    #[test]
    fn matmul_estimate_matches_formula() {
        let config = EstimationConfig {
            freivalds_reps: 0,
            ..EstimationConfig::bn254_defaults()
        };
        let cost = estimate_op_constraints(
            "MatMul",
            &[vec![4, 8], vec![8, 16]],
            &[vec![4, 16]],
            &config,
        );
        let expected_arith = 2 * 4 * 8 * 16 - 4 * 16;
        let expected_rescale = 4 * 16 * config.rescale_queries_per_element();
        assert_eq!(cost, expected_arith + expected_rescale);
    }

    #[test]
    fn conv_estimate_uses_kernel_volume() {
        let config = EstimationConfig::bn254_defaults();
        let cost = estimate_op_constraints(
            "Conv",
            &[vec![1, 3, 32, 32], vec![16, 3, 3, 3]],
            &[vec![1, 16, 30, 30]],
            &config,
        );
        let out_elems: u64 = 16 * 30 * 30;
        let kernel_vol: u64 = 3 * 3;
        let c_per_group: u64 = 3;
        let inner = kernel_vol * c_per_group;
        let expected_arith = out_elems * (inner * 2 - 1);
        let expected_rescale = out_elems * config.rescale_queries_per_element();
        assert_eq!(cost, expected_arith + expected_rescale);
    }

    #[test]
    fn rescale_queries_bn254_default() {
        let config = EstimationConfig::bn254_defaults();
        assert_eq!(config.rescale_queries_per_element(), 8);
    }

    #[test]
    fn rescale_queries_goldilocks_default() {
        let config = EstimationConfig::goldilocks_defaults();
        let expected = 18usize.div_ceil(12) + 31usize.div_ceil(12);
        assert_eq!(config.rescale_queries_per_element(), expected as u64);
    }

    #[test]
    fn estimate_op_constraints_standalone() {
        let config = EstimationConfig::bn254_defaults();
        let cost =
            estimate_op_constraints("Add", &[vec![1, 64], vec![1, 64]], &[vec![1, 64]], &config);
        assert_eq!(cost, 64);
    }

    #[test]
    fn batched_matmul_scales_with_batch() {
        let config = EstimationConfig {
            freivalds_reps: 0,
            ..EstimationConfig::bn254_defaults()
        };
        let cost_single = estimate_op_constraints(
            "MatMul",
            &[vec![4, 8], vec![8, 16]],
            &[vec![4, 16]],
            &config,
        );
        let cost_batched = estimate_op_constraints(
            "MatMul",
            &[vec![2, 4, 8], vec![2, 8, 16]],
            &[vec![2, 4, 16]],
            &config,
        );
        let single_arith = 2 * 4 * 8 * 16 - 4 * 16;
        let single_rescale = 4u64 * 16 * config.rescale_queries_per_element();
        assert_eq!(cost_single, single_arith + single_rescale);
        let batched_rescale = 2u64 * 4 * 16 * config.rescale_queries_per_element();
        assert_eq!(cost_batched, 2 * single_arith + batched_rescale);
    }
}
