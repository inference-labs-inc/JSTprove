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
    pub fn for_field(n_bits: usize) -> Self {
        Self {
            n_bits,
            chunk_bits: DEFAULT_LOGUP_CHUNK_BITS,
            scale_exponent: 18,
            freivalds_reps: 1,
            rescale_config: HashMap::new(),
        }
    }

    #[must_use]
    pub fn bn254_defaults() -> Self {
        Self::for_field(64)
    }

    #[must_use]
    pub fn goldilocks_defaults() -> Self {
        Self::for_field(31)
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

#[derive(Debug, Clone, Copy)]
struct OpEstimate {
    out_elems: u64,
    arithmetic: u64,
    range_check_queries: u64,
}

impl OpEstimate {
    const fn zero(out_elems: u64) -> Self {
        Self {
            out_elems,
            arithmetic: 0,
            range_check_queries: 0,
        }
    }

    fn elementwise(
        out_elems: u64,
        arith_mult: u64,
        query_kind: QueryKind,
        config: &EstimationConfig,
    ) -> Self {
        Self {
            out_elems,
            arithmetic: out_elems.saturating_mul(arith_mult),
            range_check_queries: query_kind.queries_for(out_elems, config),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum QueryKind {
    None,
    Rescale,
    Relu(u64),
    RescaleIfEnabled(u64),
}

impl QueryKind {
    fn queries_for(self, out_elems: u64, config: &EstimationConfig) -> u64 {
        match self {
            Self::None => 0,
            Self::Rescale => out_elems.saturating_mul(config.rescale_queries_per_element()),
            Self::Relu(mult) => {
                out_elems.saturating_mul(config.relu_queries_per_element().saturating_mul(mult))
            }
            Self::RescaleIfEnabled(rescale_queries) => rescale_queries,
        }
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

fn spatial_product(shape: &[usize]) -> u64 {
    if shape.len() >= 4 {
        shape[2..].iter().map(|&d| d as u64).product()
    } else {
        1
    }
}

fn last_dim(shape: &[usize]) -> u64 {
    shape.last().map_or(1, |&d| d as u64)
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

struct ShapeCtx<'a> {
    input_shapes: &'a [&'a [usize]],
    output_shapes: &'a [&'a [usize]],
}

impl<'a> ShapeCtx<'a> {
    fn in0(&self) -> Option<&'a [usize]> {
        self.input_shapes.first().copied()
    }

    fn in1(&self) -> Option<&'a [usize]> {
        self.input_shapes.get(1).copied()
    }

    fn out0(&self) -> Option<&'a [usize]> {
        self.output_shapes.first().copied()
    }

    fn out_elems(&self) -> u64 {
        self.out0().map_or(0, product)
    }
}

fn estimate_linear(
    ctx: &ShapeCtx<'_>,
    out_elems: u64,
    rescale_queries: u64,
    config: &EstimationConfig,
) -> OpEstimate {
    let (ell, inner) = ctx.in0().map_or((1, 1), |s| {
        let rank = s.len();
        if rank >= 2 {
            (s[rank - 2] as u64, s[rank - 1] as u64)
        } else if rank == 1 {
            (1, s[0] as u64)
        } else {
            (1, 1)
        }
    });
    let cols = ctx.out0().map_or(inner, last_dim);
    let batch = ctx.out0().map_or(1, |s| {
        let total = product(s);
        let per_batch = ell.saturating_mul(cols).max(1);
        total / per_batch
    });
    OpEstimate {
        out_elems,
        arithmetic: batch.saturating_mul(matmul_cost(ell, inner, cols, config.freivalds_reps)),
        range_check_queries: rescale_queries,
    }
}

fn estimate_conv(ctx: &ShapeCtx<'_>, out_elems: u64, rescale_queries: u64) -> OpEstimate {
    let kernel_volume = ctx
        .in1()
        .map_or(9, |w| if w.len() >= 4 { spatial_product(w) } else { 1 });
    let channels_per_group = ctx
        .in1()
        .map_or(1, |w| if w.len() >= 2 { w[1] as u64 } else { 1 });
    let inner_dim = kernel_volume.saturating_mul(channels_per_group);
    OpEstimate {
        out_elems,
        arithmetic: out_elems.saturating_mul(inner_dim.saturating_mul(2).saturating_sub(1)),
        range_check_queries: rescale_queries,
    }
}

fn estimate_pool(
    op_type: &str,
    ctx: &ShapeCtx<'_>,
    out_elems: u64,
    config: &EstimationConfig,
) -> OpEstimate {
    if op_type == "MaxPool" {
        let kernel_elems = ctx.in0().map_or(4, |s| spatial_product(s).max(1));
        return OpEstimate::elementwise(
            out_elems,
            kernel_elems,
            QueryKind::Relu(kernel_elems),
            config,
        );
    }
    let pool_size = if op_type == "GlobalAveragePool" {
        ctx.in0().map_or(1, spatial_product)
    } else {
        let in_spatial = ctx.in0().map_or(4, spatial_product);
        let out_spatial = ctx.out0().map_or(1, spatial_product).max(1);
        in_spatial / out_spatial
    };
    OpEstimate::elementwise(out_elems, pool_size, QueryKind::Rescale, config)
}

fn estimate_reduce(
    op_type: &str,
    ctx: &ShapeCtx<'_>,
    out_elems: u64,
    config: &EstimationConfig,
) -> OpEstimate {
    let in_elems = ctx.in0().map_or(out_elems, product);
    let query_kind = match op_type {
        "ReduceMax" => {
            let reduction_factor = if out_elems > 0 {
                in_elems / out_elems
            } else {
                in_elems
            };
            QueryKind::Relu(reduction_factor)
        }
        "ReduceMean" => QueryKind::Rescale,
        _ => QueryKind::None,
    };
    OpEstimate {
        out_elems,
        arithmetic: in_elems,
        range_check_queries: query_kind.queries_for(out_elems, config),
    }
}

fn estimate_softmax(
    out_elems: u64,
    out_shape: Option<&[usize]>,
    config: &EstimationConfig,
) -> OpEstimate {
    let lane_size = out_shape.map_or(1, last_dim);
    let n_lanes = if lane_size > 0 {
        out_elems / lane_size
    } else {
        out_elems
    };
    OpEstimate {
        out_elems,
        arithmetic: n_lanes.saturating_mul(lane_size.saturating_mul(4)),
        range_check_queries: n_lanes.saturating_mul(
            lane_size.saturating_mul(config.relu_queries_per_element().saturating_mul(3)),
        ),
    }
}

fn estimate_single_op(
    op_type: &str,
    ctx: &ShapeCtx<'_>,
    config: &EstimationConfig,
    is_rescale: bool,
) -> OpEstimate {
    let out_elems = ctx.out_elems();
    let rescale_queries = if is_rescale {
        out_elems.saturating_mul(config.rescale_queries_per_element())
    } else {
        0
    };

    match op_type {
        "MatMul" | "Gemm" => estimate_linear(ctx, out_elems, rescale_queries, config),

        "Conv" | "ConvTranspose" => estimate_conv(ctx, out_elems, rescale_queries),

        "Softmax" => estimate_softmax(out_elems, ctx.out0(), config),

        "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
            estimate_pool(op_type, ctx, out_elems, config)
        }

        "ReduceMean" | "ReduceSum" | "ReduceMax" => {
            estimate_reduce(op_type, ctx, out_elems, config)
        }

        "TopK" => {
            let in_elems = ctx.in0().map_or(out_elems, product);
            OpEstimate::elementwise(out_elems, 2, QueryKind::Relu(1), config)
                .with_arithmetic(in_elems.saturating_mul(2))
                .with_queries(in_elems.saturating_mul(config.relu_queries_per_element()))
        }

        "Resize" | "GridSample" => {
            OpEstimate::elementwise(out_elems, 4, QueryKind::Rescale, config)
        }

        "ReLU" | "Relu" | "LeakyRelu" | "LeakyReLU" => {
            OpEstimate::elementwise(out_elems, 1, QueryKind::Relu(1), config)
        }

        "Sigmoid" | "Tanh" | "Gelu" | "Erf" | "HardSwish" | "Hardswish" => {
            OpEstimate::elementwise(out_elems, 3, QueryKind::Relu(3), config)
        }

        "Exp" | "Log" | "Sqrt" | "Pow" | "Sin" | "Cos" => {
            OpEstimate::elementwise(out_elems, 2, QueryKind::Relu(2), config)
        }

        "LayerNormalization"
        | "InstanceNormalization"
        | "GroupNormalization"
        | "BatchNormalization" => {
            OpEstimate::elementwise(out_elems, 15, QueryKind::Relu(6), config)
        }

        "Mul" => OpEstimate::elementwise(
            out_elems,
            1,
            QueryKind::RescaleIfEnabled(rescale_queries),
            config,
        ),
        "Div" => OpEstimate::elementwise(out_elems, 1, QueryKind::Rescale, config),
        "Add" | "Sub" => OpEstimate::elementwise(out_elems, 1, QueryKind::None, config),
        "Max" | "Min" | "Clip" => OpEstimate::elementwise(out_elems, 2, QueryKind::Relu(1), config),
        "Where" => OpEstimate::elementwise(out_elems, 2, QueryKind::None, config),

        _ => OpEstimate::zero(out_elems),
    }
}

impl OpEstimate {
    fn with_arithmetic(mut self, arith: u64) -> Self {
        self.arithmetic = arith;
        self
    }

    fn with_queries(mut self, queries: u64) -> Self {
        self.range_check_queries = queries;
        self
    }
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

        let in_shapes: Vec<&[usize]> = layer
            .inputs
            .iter()
            .filter_map(|n| shapes.get(n).map(Vec::as_slice))
            .collect();
        let out_shapes: Vec<&[usize]> = layer
            .outputs
            .iter()
            .filter_map(|n| shapes.get(n).map(Vec::as_slice))
            .collect();

        let ctx = ShapeCtx {
            input_shapes: &in_shapes,
            output_shapes: &out_shapes,
        };
        let est = estimate_single_op(&layer.op_type, &ctx, config, is_rescale);

        result.total_arithmetic = result.total_arithmetic.saturating_add(est.arithmetic);
        result.total_range_queries = result
            .total_range_queries
            .saturating_add(est.range_check_queries);

        result.layers.push(LayerEstimate {
            name: layer.name.clone(),
            op_type: layer.op_type.clone(),
            output_elements: est.out_elems,
            arithmetic: est.arithmetic,
            range_check_queries: est.range_check_queries,
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
    let in_refs: Vec<&[usize]> = input_shapes.iter().map(Vec::as_slice).collect();
    let out_refs: Vec<&[usize]> = output_shapes.iter().map(Vec::as_slice).collect();
    let ctx = ShapeCtx {
        input_shapes: &in_refs,
        output_shapes: &out_refs,
    };
    let est = estimate_single_op(op_type, &ctx, config, true);
    est.arithmetic.saturating_add(est.range_check_queries)
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

    #[test]
    fn gemm_transpose_uses_output_cols() {
        let config = EstimationConfig {
            freivalds_reps: 0,
            ..EstimationConfig::bn254_defaults()
        };
        let cost = estimate_op_constraints(
            "Gemm",
            &[vec![1, 4096], vec![60, 4096]],
            &[vec![1, 60]],
            &config,
        );
        let expected_arith = 2 * 4096 * 60 - 60;
        let expected_rescale = 60 * config.rescale_queries_per_element();
        assert_eq!(cost, expected_arith + expected_rescale);
    }
}
