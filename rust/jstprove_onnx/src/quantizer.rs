use std::collections::HashMap;

use anyhow::Result;

use super::graph::{LayerGraph, LayerNode, OpType};
use super::ops;
use super::parser::{AttrValue, TensorData};

pub const DEFAULT_SCALE_BASE: u64 = 2;
pub const DEFAULT_SCALE_EXPONENT: u32 = 18;
pub const N_BITS_BN254: u32 = 64;

// Goldilocks: 64-bit prime field. Intermediate products (alpha^2 * bound) must
// not overflow, so usable bits = field_bits / 2 - 1.
const GOLDILOCKS_FIELD_BITS: u32 = 64;
pub const N_BITS_GOLDILOCKS: u32 = GOLDILOCKS_FIELD_BITS / 2 - 1;

// GoldilocksExt2: degree-2 extension → 128-bit arithmetic. Same overflow rule.
const GOLDILOCKS_EXT2_FIELD_BITS: u32 = GOLDILOCKS_FIELD_BITS * 2;
pub const N_BITS_GOLDILOCKS_EXT2: u32 = GOLDILOCKS_EXT2_FIELD_BITS / 2 - 1;

pub const MIN_USEFUL_EXPONENT: u32 = 8;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScaleConfig {
    pub base: u64,
    pub exponent: u32,
    pub alpha: i64,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        let alpha = (DEFAULT_SCALE_BASE as i64).pow(DEFAULT_SCALE_EXPONENT);
        Self {
            base: DEFAULT_SCALE_BASE,
            exponent: DEFAULT_SCALE_EXPONENT,
            alpha,
        }
    }
}

impl ScaleConfig {
    pub fn new(base: u64, exponent: u32) -> Self {
        let alpha = (base as i64).pow(exponent);
        Self {
            base,
            exponent,
            alpha,
        }
    }

    #[must_use]
    pub fn exponent_for_digits(target_digits: u32) -> u32 {
        (f64::from(target_digits) * std::f64::consts::LOG2_10).ceil() as u32
    }

    #[must_use]
    pub fn max_safe_exponent(n_bits: u32, max_bound: f64) -> u32 {
        let log2_accum = if max_bound > 1.0 {
            max_bound.log2().ceil() as u32
        } else {
            0
        };
        n_bits.saturating_sub(log2_accum.saturating_add(1)) / 2
    }

    #[must_use]
    pub fn max_safe_digits(n_bits: u32, max_bound: f64) -> u32 {
        let max_exp = Self::max_safe_exponent(n_bits, max_bound);
        (f64::from(max_exp) / std::f64::consts::LOG2_10).floor() as u32
    }

    #[must_use]
    pub fn adaptive(n_bits: u32, max_bound: f64) -> Self {
        let exponent = Self::max_safe_exponent(n_bits, max_bound);
        Self::new(DEFAULT_SCALE_BASE, exponent)
    }

    pub fn for_precision(target_digits: u32, n_bits: u32, max_bound: f64) -> Result<Self> {
        let exponent = Self::exponent_for_digits(target_digits);
        let max_exp = Self::max_safe_exponent(n_bits, max_bound);
        let max_digits = Self::max_safe_digits(n_bits, max_bound);
        anyhow::ensure!(
            exponent <= max_exp,
            "requested {} decimal digits requires exponent={}, but field n_bits={} \
             with max accumulation {:.2} only supports exponent up to {} ({} digits)",
            target_digits,
            exponent,
            n_bits,
            max_bound,
            max_exp,
            max_digits,
        );
        Ok(Self::new(DEFAULT_SCALE_BASE, exponent))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizedModel {
    pub graph: LayerGraph,
    pub scale_config: ScaleConfig,
    #[serde(default)]
    pub n_bits_config: HashMap<String, usize>,
}

impl QuantizedModel {
    pub fn apply_observed_n_bits(&mut self, overrides: &HashMap<String, usize>) {
        for layer in &mut self.graph.layers {
            if let Some(&obs) = overrides.get(&layer.name) {
                layer.n_bits = Some(obs);
                self.n_bits_config.insert(layer.name.clone(), obs);
            }
        }
    }
}

/// # Errors
/// Returns an error if the requested precision exceeds the field capacity,
/// or if ONNX graph analysis or quantization fails.
pub fn quantize_model_for_precision(
    mut graph: LayerGraph,
    target_digits: u32,
    n_bits: u32,
) -> Result<QuantizedModel> {
    let max_bound = compute_max_bound(&mut graph)?;
    let config = ScaleConfig::for_precision(target_digits, n_bits, max_bound)?;
    quantize_model(graph, &config)
}

pub fn quantize_model(mut graph: LayerGraph, config: &ScaleConfig) -> Result<QuantizedModel> {
    let alpha = config.alpha;

    fold_all_batchnorms(&mut graph)?;
    rewrite_pow_sqrt(&mut graph);

    let n_bits_config = compute_bounds(&graph, config)?;

    for layer in &mut graph.layers {
        quantize_layer_weights(layer, alpha)?;
    }

    for layer in &mut graph.layers {
        layer.n_bits = n_bits_config.get(&layer.name).copied();
    }

    Ok(QuantizedModel {
        graph,
        scale_config: config.clone(),
        n_bits_config,
    })
}

fn quantize_layer_weights(layer: &mut LayerNode, alpha: i64) -> Result<()> {
    let scale_plan = ops::get_scale_plan(layer.op_type);

    for (&input_pos, &scale_factor) in &scale_plan {
        if let Some(input_name) = layer.inputs.get(input_pos) {
            if let Some(weight) = layer.weights.get_mut(input_name) {
                if !weight.float_data.is_empty() {
                    let factor = alpha.pow(scale_factor as u32);
                    let scaled: Vec<i64> = weight
                        .float_data
                        .iter()
                        .map(|v| (*v * factor as f64).round() as i64)
                        .collect();
                    weight.int_data = scaled;
                    weight.float_data.clear();
                }
            }
        }
    }

    Ok(())
}

fn fold_all_batchnorms(graph: &mut LayerGraph) -> Result<()> {
    for layer in &mut graph.layers {
        if layer.op_type == OpType::BatchNormalization && layer.inputs.len() >= 5 {
            fold_batchnorm_params(layer)?;
        }
    }
    Ok(())
}

fn fold_batchnorm_params(layer: &mut LayerNode) -> Result<()> {
    let epsilon = layer.get_float_attr("epsilon").unwrap_or(1e-5) as f64;

    let scale_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing scale input", layer.name))?
        .clone();
    let bias_name = layer
        .inputs
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing bias input", layer.name))?
        .clone();
    let mean_name = layer
        .inputs
        .get(3)
        .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing mean input", layer.name))?
        .clone();
    let var_name = layer
        .inputs
        .get(4)
        .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing var input", layer.name))?
        .clone();

    let get_floats = |name: &str| -> Result<Vec<f64>> {
        let td = layer
            .weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing weight '{}'", layer.name, name))?;
        if !td.float_data.is_empty() {
            Ok(td.float_data.clone())
        } else {
            Ok(td.int_data.iter().map(|&v| v as f64).collect())
        }
    };

    let scale = get_floats(&scale_name)?;
    let bias = get_floats(&bias_name)?;
    let mean = get_floats(&mean_name)?;
    let var = get_floats(&var_name)?;

    let c = scale.len();
    anyhow::ensure!(
        bias.len() == c && mean.len() == c && var.len() == c,
        "BatchNorm {} parameter length mismatch: scale={}, bias={}, mean={}, var={}",
        layer.name,
        c,
        bias.len(),
        mean.len(),
        var.len()
    );

    let mut mul = Vec::with_capacity(c);
    let mut add = Vec::with_capacity(c);
    for i in 0..c {
        let v = var[i] + epsilon;
        anyhow::ensure!(
            v > 0.0,
            "BatchNorm {} channel {} has non-positive variance+epsilon: var={}, eps={}",
            layer.name,
            i,
            var[i],
            epsilon
        );
        let m = scale[i] / v.sqrt();
        mul.push(m);
        add.push(bias[i] - mean[i] * m);
    }

    let mul_tensor_name = format!("{}_folded_mul", layer.name);
    let add_tensor_name = format!("{}_folded_add", layer.name);

    layer.weights.insert(
        mul_tensor_name.clone(),
        TensorData {
            name: mul_tensor_name.clone(),
            dims: vec![c as i64],
            data_type: 1,
            float_data: mul,
            int_data: vec![],
        },
    );
    layer.weights.insert(
        add_tensor_name.clone(),
        TensorData {
            name: add_tensor_name.clone(),
            dims: vec![c as i64],
            data_type: 1,
            float_data: add,
            int_data: vec![],
        },
    );

    let stale = [scale_name, bias_name, mean_name, var_name];

    let primary_input = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("BatchNorm {} has no inputs", layer.name))?
        .clone();
    layer.inputs = vec![primary_input, mul_tensor_name, add_tensor_name];

    for name in &stale {
        layer.weights.remove(name);
    }

    Ok(())
}

/// # Errors
/// Returns an error if layer bound propagation encounters invalid weight data.
pub fn compute_max_bound(graph: &mut LayerGraph) -> Result<f64> {
    fold_all_batchnorms(graph)?;
    rewrite_pow_sqrt(graph);
    compute_max_bound_inner(graph)
}

fn rewrite_pow_sqrt(graph: &mut LayerGraph) {
    for layer in &mut graph.layers {
        if layer.op_type != OpType::Pow {
            continue;
        }
        let is_sqrt = layer
            .inputs
            .get(1)
            .and_then(|name| layer.weights.get(name))
            .is_some_and(|td| {
                let vals = td.as_f64_vec();
                vals.len() == 1 && (vals[0] - 0.5).abs() < 1e-9
            });
        if is_sqrt {
            let exp_name = layer.inputs[1].clone();
            layer.weights.remove(&exp_name);
            layer.inputs.truncate(1);
            layer.op_type = OpType::Sqrt;
        }
    }
}

fn compute_max_bound_inner(graph: &LayerGraph) -> Result<f64> {
    let mut bounds: HashMap<String, f64> = HashMap::new();
    let mut max_bound: f64 = 1.0;

    // compute_max_bound runs before the final ScaleConfig is known, so use the
    // default alpha for Log-layer bound propagation (ln(2^18) ≈ 12.47).
    let default_alpha_f64 = (DEFAULT_SCALE_BASE as f64).powi(DEFAULT_SCALE_EXPONENT as i32);

    let shapes = propagate_shapes(graph)?;

    for name in &graph.input_names {
        bounds.insert(name.clone(), 1.0);
    }

    for layer in graph.iter_topo() {
        let mut bound = compute_layer_bound(layer, &bounds, default_alpha_f64)?;

        if layer.op_type == OpType::ReduceSum {
            if let Some(in_name) = layer.inputs.first() {
                let m_in = bounds.get(in_name.as_str()).copied().unwrap_or(1.0);
                if let (Some(in_shape), Some(out_name)) =
                    (shapes.get(in_name.as_str()), layer.outputs.first())
                {
                    let in_total: usize = in_shape.iter().product();
                    let out_total: usize = shapes
                        .get(out_name.as_str())
                        .map(|s| s.iter().product())
                        .unwrap_or(1);
                    let factor = (in_total / out_total.max(1)).max(1) as f64;
                    bound = m_in * factor;
                }
            }
        }

        if layer.op_type == OpType::MatMul {
            if let (Some(in0_name), Some(in1_name)) = (layer.inputs.first(), layer.inputs.get(1)) {
                let no_weight = !layer.weights.contains_key(in0_name.as_str())
                    && !layer.weights.contains_key(in1_name.as_str());
                if no_weight {
                    let m_in = bounds.get(in0_name.as_str()).copied().unwrap_or(1.0);
                    let m_b = bounds.get(in1_name.as_str()).copied().unwrap_or(1.0);
                    let k = shapes
                        .get(in0_name.as_str())
                        .and_then(|s| s.last().copied())
                        .unwrap_or(1)
                        .max(1) as f64;
                    bound = m_in * m_b * k;
                }
            }
        }

        if layer.needs_rescale {
            max_bound = max_bound.max(bound);
        }

        let post_rescale_bound = if layer.needs_rescale { 1.0 } else { bound };

        for out_name in &layer.outputs {
            bounds.insert(out_name.clone(), post_rescale_bound);
        }
    }

    Ok(max_bound)
}

/// Elementwise numpy-style broadcast of two shapes (no error on incompatible dims — takes max).
fn broadcast_two(a: &[usize], b: &[usize]) -> Vec<usize> {
    let len = a.len().max(b.len());
    (0..len)
        .map(|i| {
            let ai = if i + a.len() >= len {
                a[i + a.len() - len]
            } else {
                1
            };
            let bi = if i + b.len() >= len {
                b[i + b.len() - len]
            } else {
                1
            };
            ai.max(bi)
        })
        .collect()
}

/// Build a map of tensor name → shape from the graph, using:
/// - Input shapes (from `graph.input_shapes`)
/// - Weight tensor shapes (from `layer.weights`)
/// - Simple topological shape propagation for intermediate tensors
///
/// This is used by `compute_bounds` so that reduction ops (e.g. ReduceSum)
/// can account for the actual number of elements being accumulated.
fn propagate_shapes(graph: &LayerGraph) -> Result<HashMap<String, Vec<usize>>> {
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // Seed with model-level input shapes.
    for (name, shape) in &graph.input_shapes {
        shapes.insert(name.clone(), shape.clone());
    }

    // Seed with all weight tensor shapes.
    for layer in &graph.layers {
        for (name, td) in &layer.weights {
            shapes.insert(name.clone(), td.shape());
        }
    }

    for layer in &graph.layers {
        if layer.op_type == OpType::Constant {
            if let Some(AttrValue::Tensor(td)) = layer.attributes.get("value") {
                for out_name in &layer.outputs {
                    shapes.insert(out_name.clone(), td.shape());
                }
            }
        }
    }

    // Propagate shapes topologically.
    for layer in graph.iter_topo() {
        let input_shape: Option<Vec<usize>> = layer
            .inputs
            .first()
            .and_then(|n| shapes.get(n.as_str()))
            .cloned();

        let out_shape: Vec<usize> = match layer.op_type {
            OpType::ReduceSum => {
                // Compute the output shape from axes + keepdims.
                if let Some(ref in_shape) = input_shape {
                    let rank = in_shape.len();

                    // Collect axes (from input[1] weight or from attribute).
                    let normalize_axis = |a: i64| -> usize {
                        let a = if a < 0 { a + rank as i64 } else { a };
                        (a as usize).min(rank.saturating_sub(1))
                    };
                    let axes: Vec<usize> = {
                        let from_weight = layer.inputs.get(1).and_then(|axes_name| {
                            layer.weights.get(axes_name.as_str()).map(|td| {
                                td.as_i64_vec()
                                    .iter()
                                    .map(|&a| normalize_axis(a))
                                    .collect::<Vec<usize>>()
                            })
                        });
                        match from_weight {
                            Some(v) if !v.is_empty() => v,
                            Some(_) => {
                                // Empty axes tensor: per ONNX spec, means reduce all
                                // unless noop_with_empty_axes == 1.
                                let noop = matches!(
                                    layer.attributes.get("noop_with_empty_axes"),
                                    Some(AttrValue::Int(1))
                                );
                                if noop {
                                    vec![]
                                } else {
                                    (0..rank).collect()
                                }
                            }
                            None => {
                                // Fallback: try attributes.
                                match layer.attributes.get("axes") {
                                    Some(AttrValue::Ints(v)) => {
                                        v.iter().map(|&a| normalize_axis(a)).collect()
                                    }
                                    // No axes = reduce all.
                                    _ => (0..rank).collect(),
                                }
                            }
                        }
                    };

                    let keepdims = match layer.attributes.get("keepdims") {
                        Some(AttrValue::Int(v)) => *v != 0,
                        _ => true,
                    };

                    if keepdims {
                        in_shape
                            .iter()
                            .enumerate()
                            .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
                            .collect()
                    } else {
                        in_shape
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
                            .collect()
                    }
                } else {
                    vec![]
                }
            }
            // MatMul: [.., M, K] × [.., K, N] → [broadcast(..), M, N].
            OpType::MatMul => {
                let s0 = layer.inputs.first().and_then(|n| shapes.get(n.as_str()));
                let s1 = layer.inputs.get(1).and_then(|n| shapes.get(n.as_str()));
                match (s0, s1) {
                    (Some(a), Some(b)) if a.len() >= 2 && b.len() >= 2 => {
                        let a_batch = &a[..a.len() - 2];
                        let b_batch = &b[..b.len() - 2];
                        let mut out = broadcast_two(a_batch, b_batch);
                        out.push(a[a.len() - 2]); // M
                        out.push(*b.last().unwrap()); // N
                        out
                    }
                    _ => input_shape.unwrap_or_default(),
                }
            }
            // Reshape: output shape is the target shape tensor (input[1]).
            // Resolve ONNX semantics: 0 → copy from input, -1 → infer from total.
            OpType::Reshape => {
                if let Some(shape_name) = layer.inputs.get(1) {
                    if let Some(td) = layer.weights.get(shape_name.as_str()) {
                        let raw = td.as_i64_vec();
                        let allowzero = matches!(
                            layer.attributes.get("allowzero"),
                            Some(AttrValue::Int(v)) if *v != 0
                        );
                        let in_shape: &[usize] = input_shape.as_deref().unwrap_or(&[]);
                        let input_total: usize = in_shape.iter().product();

                        // First pass: resolve 0 → copy and positives; -1 → infer; other negatives are invalid.
                        let mut dims: Vec<Option<usize>> = Vec::with_capacity(raw.len());
                        for (i, &d) in raw.iter().enumerate() {
                            if d == 0 {
                                dims.push(if allowzero {
                                    Some(0)
                                } else {
                                    Some(in_shape.get(i).copied().unwrap_or(0))
                                });
                            } else if d > 0 {
                                dims.push(Some(d as usize));
                            } else if d == -1 {
                                dims.push(None); // infer sentinel
                            } else {
                                anyhow::bail!(
                                    "Reshape layer '{}': invalid dimension {} at index {} (only -1 is a valid infer sentinel)",
                                    layer.name, d, i
                                );
                            }
                        }

                        // Infer the single -1 dimension when input total is known.
                        let n_unknown = dims.iter().filter(|d| d.is_none()).count();
                        if n_unknown == 1 && input_total > 0 {
                            let known: usize = dims.iter().filter_map(|&d| d).product();
                            if known == 0 {
                                anyhow::bail!(
                                    "Reshape layer '{}': cannot infer -1 dimension when known product is 0",
                                    layer.name
                                );
                            }
                            if input_total % known != 0 {
                                anyhow::bail!(
                                    "Reshape layer '{}': input total {} is not divisible by known dims product {}",
                                    layer.name, input_total, known
                                );
                            }
                            let inferred = input_total / known;
                            for d in &mut dims {
                                if d.is_none() {
                                    *d = Some(inferred);
                                    break;
                                }
                            }
                        } else if n_unknown > 1 {
                            anyhow::bail!(
                                "Reshape layer '{}': more than one -1 dimension is not allowed",
                                layer.name
                            );
                        }

                        // All None should be resolved by now; any remaining None is an error.
                        dims.into_iter()
                            .map(|d| {
                                d.ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "Reshape layer '{}': could not infer -1 dimension (input shape unknown)",
                                        layer.name
                                    )
                                })
                            })
                            .collect::<Result<Vec<usize>>>()?
                    } else {
                        input_shape.unwrap_or_default()
                    }
                } else {
                    input_shape.unwrap_or_default()
                }
            }
            // Concat: merge input shapes along the concat axis.
            OpType::Concat => {
                let axis = match layer.attributes.get("axis") {
                    Some(AttrValue::Int(v)) => *v,
                    _ => 0,
                };
                let in_shapes: Vec<&Vec<usize>> = layer
                    .inputs
                    .iter()
                    .filter_map(|n| shapes.get(n.as_str()))
                    .collect();
                if let Some(first) = in_shapes.first() {
                    let rank = first.len();
                    let ax = if axis < 0 {
                        (axis + rank as i64).max(0) as usize
                    } else {
                        (axis as usize).min(rank.saturating_sub(1))
                    };
                    let mut out = (*first).clone();
                    out[ax] = in_shapes
                        .iter()
                        .map(|s| s.get(ax).copied().unwrap_or(0))
                        .sum();
                    out
                } else {
                    input_shape.unwrap_or_default()
                }
            }
            // MaxPool/AveragePool: spatial dims shrink per kernel/stride/pad.
            OpType::MaxPool | OpType::AveragePool => {
                if let Some(ref in_shape) = input_shape {
                    if in_shape.len() >= 2 {
                        let kernel: Vec<usize> = layer
                            .get_ints_attr("kernel_shape")
                            .map(|v| v.iter().map(|&k| k as usize).collect())
                            .unwrap_or_default();
                        let strides: Vec<usize> = layer
                            .get_ints_attr("strides")
                            .map(|v| v.iter().map(|&s| s.max(1) as usize).collect())
                            .unwrap_or_else(|| vec![1; kernel.len()]);
                        let pads: Vec<usize> = layer
                            .get_ints_attr("pads")
                            .map(|v| v.iter().map(|&p| p.max(0) as usize).collect())
                            .unwrap_or_else(|| vec![0; kernel.len() * 2]);
                        let spatial_dims = kernel.len();
                        let mut out = in_shape[..2].to_vec();
                        for i in 0..spatial_dims {
                            let in_d = in_shape.get(2 + i).copied().unwrap_or(1);
                            let k = kernel.get(i).copied().unwrap_or(1).max(1);
                            let s = strides.get(i).copied().unwrap_or(1).max(1);
                            let ph = pads.get(i).copied().unwrap_or(0);
                            let pe = pads.get(spatial_dims + i).copied().unwrap_or(0);
                            let padded = in_d + ph + pe;
                            let out_d = padded.saturating_sub(k) / s + 1;
                            out.push(out_d);
                        }
                        out
                    } else {
                        in_shape.clone()
                    }
                } else {
                    vec![]
                }
            }
            // Conv: output = [N, C_out, H_out, W_out, ...].
            // Spatial dims computed via floor((in + 2*pad - dilation*(k-1) - 1)/stride) + 1.
            OpType::Conv => {
                if let Some(ref in_shape) = input_shape {
                    if let Some(w_name) = layer.inputs.get(1) {
                        if let Some(w) = layer.weights.get(w_name.as_str()) {
                            let w_shape = w.shape();
                            let c_out = w_shape.first().copied().unwrap_or(1);
                            let spatial_dims = in_shape.len().saturating_sub(2);
                            // Kernel spatial sizes from weight shape [C_out, C_in, k...].
                            let kernel: Vec<usize> = if w_shape.len() > 2 {
                                w_shape[2..].to_vec()
                            } else {
                                layer
                                    .get_ints_attr("kernel_shape")
                                    .map(|v| v.iter().map(|&k| k as usize).collect())
                                    .unwrap_or_else(|| vec![1; spatial_dims])
                            };
                            let strides: Vec<usize> = layer
                                .get_ints_attr("strides")
                                .map(|v| v.iter().map(|&s| s.max(1) as usize).collect())
                                .unwrap_or_else(|| vec![1; spatial_dims]);
                            let dilations: Vec<usize> = layer
                                .get_ints_attr("dilations")
                                .map(|v| v.iter().map(|&d| d.max(1) as usize).collect())
                                .unwrap_or_else(|| vec![1; spatial_dims]);
                            let pads: Vec<usize> = layer
                                .get_ints_attr("pads")
                                .map(|v| v.iter().map(|&p| p.max(0) as usize).collect())
                                .unwrap_or_else(|| vec![0; spatial_dims * 2]);
                            let mut out = vec![in_shape.first().copied().unwrap_or(1), c_out];
                            for i in 0..spatial_dims {
                                let in_d = in_shape.get(2 + i).copied().unwrap_or(1);
                                let k = kernel.get(i).copied().unwrap_or(1).max(1);
                                let s = strides.get(i).copied().unwrap_or(1).max(1);
                                let d = dilations.get(i).copied().unwrap_or(1).max(1);
                                let ph = pads.get(i).copied().unwrap_or(0);
                                let pe = pads.get(spatial_dims + i).copied().unwrap_or(0);
                                let effective_k = (k - 1) * d + 1;
                                let out_d = (in_d + ph + pe).saturating_sub(effective_k) / s + 1;
                                out.push(out_d);
                            }
                            out
                        } else {
                            in_shape.clone()
                        }
                    } else {
                        in_shape.clone()
                    }
                } else {
                    vec![]
                }
            }
            // ConvTranspose: output = [N, C_out/group * group, out_H, out_W, ...].
            OpType::ConvTranspose => {
                if let Some(ref in_shape) = input_shape {
                    if let Some(w_name) = layer.inputs.get(1) {
                        if let Some(w) = layer.weights.get(w_name.as_str()) {
                            let w_shape = w.shape();
                            // Weight: [C_in, C_out/group, *kernel]
                            let groups = layer.get_int_attr("group").unwrap_or(1).max(1) as usize;
                            let c_out = w_shape.get(1).copied().unwrap_or(1) * groups;
                            let spatial_dims = in_shape.len().saturating_sub(2);
                            let kernel: Vec<usize> = layer
                                .get_ints_attr("kernel_shape")
                                .map(|v| v.iter().map(|&k| k as usize).collect())
                                .unwrap_or_else(|| {
                                    if w_shape.len() >= 3 {
                                        w_shape[2..].to_vec()
                                    } else {
                                        vec![]
                                    }
                                });
                            let strides: Vec<usize> = layer
                                .get_ints_attr("strides")
                                .map(|v| v.iter().map(|&s| s.max(1) as usize).collect())
                                .unwrap_or_else(|| vec![1; spatial_dims]);
                            let pads: Vec<usize> = layer
                                .get_ints_attr("pads")
                                .map(|v| v.iter().map(|&p| p.max(0) as usize).collect())
                                .unwrap_or_else(|| vec![0; spatial_dims * 2]);
                            let dilations: Vec<usize> = layer
                                .get_ints_attr("dilations")
                                .map(|v| v.iter().map(|&d| d.max(1) as usize).collect())
                                .unwrap_or_else(|| vec![1; spatial_dims]);
                            let out_padding: Vec<usize> = layer
                                .get_ints_attr("output_padding")
                                .map(|v| v.iter().map(|&p| p.max(0) as usize).collect())
                                .unwrap_or_else(|| vec![0; spatial_dims]);
                            let mut out = vec![in_shape.first().copied().unwrap_or(1), c_out];
                            for i in 0..spatial_dims {
                                let in_d = in_shape.get(2 + i).copied().unwrap_or(1);
                                let k = kernel.get(i).copied().unwrap_or(1).max(1);
                                let s = strides.get(i).copied().unwrap_or(1).max(1);
                                let d = dilations.get(i).copied().unwrap_or(1).max(1);
                                let pad = pads.get(i).copied().unwrap_or(0)
                                    + pads.get(spatial_dims + i).copied().unwrap_or(0);
                                let op = out_padding.get(i).copied().unwrap_or(0);
                                let out_d = s * in_d.saturating_sub(1) + (k - 1) * d + 1 + op;
                                out.push(out_d.saturating_sub(pad));
                            }
                            out
                        } else {
                            in_shape.clone()
                        }
                    } else {
                        in_shape.clone()
                    }
                } else {
                    vec![]
                }
            }
            // Where: output shape is the broadcast of condition, X, and Y shapes.
            OpType::Where => {
                let s0 = layer.inputs.first().and_then(|n| shapes.get(n.as_str()));
                let s1 = layer.inputs.get(1).and_then(|n| shapes.get(n.as_str()));
                let s2 = layer.inputs.get(2).and_then(|n| shapes.get(n.as_str()));
                match (s0, s1, s2) {
                    (Some(c), Some(x), Some(y)) => broadcast_two(&broadcast_two(c, x), y),
                    (Some(a), Some(b), None)
                    | (Some(a), None, Some(b))
                    | (None, Some(a), Some(b)) => broadcast_two(a, b),
                    (Some(s), None, None) | (None, Some(s), None) | (None, None, Some(s)) => {
                        s.clone()
                    }
                    _ => input_shape.unwrap_or_default(),
                }
            }
            // Split: each output gets a slice along the split axis.
            OpType::Split => {
                if let Some(ref in_shape) = input_shape {
                    let rank = in_shape.len();
                    let axis_raw = match layer.attributes.get("axis") {
                        Some(AttrValue::Int(v)) => *v,
                        _ => 0,
                    };
                    let axis = if axis_raw < 0 {
                        (axis_raw + rank as i64).max(0) as usize
                    } else {
                        (axis_raw as usize).min(rank.saturating_sub(1))
                    };
                    let axis_dim = in_shape.get(axis).copied().unwrap_or(0);
                    let num_outputs = layer.outputs.len();
                    let split_sizes: Vec<usize> = if let Some(split_name) = layer.inputs.get(1) {
                        layer
                            .weights
                            .get(split_name.as_str())
                            .map(|td| td.as_i64_vec().iter().map(|&v| v.max(0) as usize).collect())
                    } else {
                        None
                    }
                    .or_else(|| match layer.attributes.get("split") {
                        Some(AttrValue::Ints(v)) => {
                            Some(v.iter().map(|&v| v.max(0) as usize).collect())
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        let each = if num_outputs > 0 {
                            axis_dim / num_outputs
                        } else {
                            0
                        };
                        vec![each; num_outputs]
                    });
                    for (out_name, &sz) in layer.outputs.iter().zip(split_sizes.iter()) {
                        let mut out = in_shape.clone();
                        if axis < out.len() {
                            out[axis] = sz;
                        }
                        shapes.insert(out_name.clone(), out);
                    }
                    continue;
                } else {
                    vec![]
                }
            }
            // Gemm: [M, K] x [K, N] → [M, N] (with optional transA/transB).
            OpType::Gemm => {
                let trans_a = layer.get_int_attr("transA").unwrap_or(0) != 0;
                let trans_b = layer.get_int_attr("transB").unwrap_or(0) != 0;
                let s0 = layer.inputs.first().and_then(|n| shapes.get(n.as_str()));
                let s1 = layer.inputs.get(1).and_then(|n| shapes.get(n.as_str()));
                match (s0, s1) {
                    (Some(a), Some(b)) if a.len() == 2 && b.len() == 2 => {
                        let m = if trans_a { a[1] } else { a[0] };
                        let n = if trans_b { b[0] } else { b[1] };
                        vec![m, n]
                    }
                    _ => input_shape.unwrap_or_default(),
                }
            }
            // Flatten: reshape to [d0*...*d_{axis-1}, d_{axis}*...*d_{n-1}].
            OpType::Flatten => {
                if let Some(ref in_shape) = input_shape {
                    let axis_raw = layer.get_int_attr("axis").unwrap_or(1);
                    let rank = in_shape.len();
                    let axis = if axis_raw < 0 {
                        (axis_raw + rank as i64).max(0) as usize
                    } else {
                        (axis_raw as usize).min(rank)
                    };
                    let d0: usize = in_shape[..axis].iter().product::<usize>().max(1);
                    let d1: usize = in_shape[axis..].iter().product::<usize>().max(1);
                    vec![d0, d1]
                } else {
                    vec![]
                }
            }
            // Transpose: permute dimensions.
            OpType::Transpose => {
                if let Some(ref in_shape) = input_shape {
                    if let Some(perm) = layer.get_ints_attr("perm") {
                        perm.iter()
                            .map(|&p| in_shape.get(p as usize).copied().unwrap_or(1))
                            .collect()
                    } else {
                        in_shape.iter().rev().copied().collect()
                    }
                } else {
                    vec![]
                }
            }
            // Squeeze: remove axes of size 1.
            OpType::Squeeze => {
                if let Some(ref in_shape) = input_shape {
                    let axes: Vec<usize> = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| {
                            td.as_i64_vec()
                                .iter()
                                .map(|&a| {
                                    if a < 0 {
                                        (a + in_shape.len() as i64) as usize
                                    } else {
                                        a as usize
                                    }
                                })
                                .collect()
                        })
                        .or_else(|| {
                            layer.get_ints_attr("axes").map(|v| {
                                v.iter()
                                    .map(|&a| {
                                        if a < 0 {
                                            (a + in_shape.len() as i64) as usize
                                        } else {
                                            a as usize
                                        }
                                    })
                                    .collect()
                            })
                        })
                        .unwrap_or_default();
                    if axes.is_empty() {
                        in_shape.iter().copied().filter(|&d| d != 1).collect()
                    } else {
                        in_shape
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
                            .collect()
                    }
                } else {
                    vec![]
                }
            }
            // Unsqueeze: insert axes of size 1.
            OpType::Unsqueeze => {
                if let Some(ref in_shape) = input_shape {
                    let axes: Vec<i64> = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec())
                        .or_else(|| layer.get_ints_attr("axes").map(|v| v.to_vec()))
                        .unwrap_or_default();
                    let out_rank = in_shape.len() + axes.len();
                    let mut normalized: Vec<usize> = axes
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (a + out_rank as i64) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect();
                    normalized.sort_unstable();
                    let mut out = in_shape.clone();
                    for &ax in &normalized {
                        let pos = ax.min(out.len());
                        out.insert(pos, 1);
                    }
                    out
                } else {
                    vec![]
                }
            }
            // Pad: adds padding to spatial dims.
            OpType::Pad => {
                if let Some(ref in_shape) = input_shape {
                    let pads_raw: Vec<i64> = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec())
                        .unwrap_or_default();
                    if pads_raw.is_empty() {
                        in_shape.clone()
                    } else {
                        let rank = in_shape.len();
                        in_shape
                            .iter()
                            .enumerate()
                            .map(|(i, &d)| {
                                let pad_begin =
                                    pads_raw.get(i).copied().unwrap_or(0).max(0) as usize;
                                let pad_end =
                                    pads_raw.get(rank + i).copied().unwrap_or(0).max(0) as usize;
                                d + pad_begin + pad_end
                            })
                            .collect()
                    }
                } else {
                    vec![]
                }
            }
            // Gather: index along axis — output has shape with axis dim replaced.
            OpType::Gather => {
                if let Some(ref in_shape) = input_shape {
                    let axis_raw = layer.get_int_attr("axis").unwrap_or(0);
                    let rank = in_shape.len();
                    let axis = if axis_raw < 0 {
                        (axis_raw + rank as i64).max(0) as usize
                    } else {
                        (axis_raw as usize).min(rank.saturating_sub(1))
                    };
                    let indices_shape = layer
                        .inputs
                        .get(1)
                        .and_then(|n| shapes.get(n.as_str()))
                        .cloned()
                        .unwrap_or_else(|| vec![1]);
                    let mut out: Vec<usize> = in_shape[..axis].to_vec();
                    out.extend_from_slice(&indices_shape);
                    if axis + 1 < in_shape.len() {
                        out.extend_from_slice(&in_shape[axis + 1..]);
                    }
                    out
                } else {
                    vec![]
                }
            }
            // Slice: reduces dims along sliced axes.
            OpType::Slice => {
                if let Some(ref in_shape) = input_shape {
                    let starts: Vec<i64> = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec())
                        .unwrap_or_default();
                    let ends: Vec<i64> = layer
                        .inputs
                        .get(2)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec())
                        .unwrap_or_default();
                    let axes: Vec<usize> = layer
                        .inputs
                        .get(3)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| {
                            td.as_i64_vec()
                                .iter()
                                .map(|&a| {
                                    if a < 0 {
                                        (a + in_shape.len() as i64) as usize
                                    } else {
                                        a as usize
                                    }
                                })
                                .collect()
                        })
                        .unwrap_or_else(|| (0..starts.len()).collect());
                    let steps: Vec<i64> = layer
                        .inputs
                        .get(4)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec())
                        .unwrap_or_else(|| vec![1; axes.len()]);
                    let mut out = in_shape.clone();
                    for (idx, &ax) in axes.iter().enumerate() {
                        if ax >= out.len() {
                            continue;
                        }
                        let dim = out[ax] as i64;
                        let s = starts.get(idx).copied().unwrap_or(0).clamp(-dim, dim);
                        let e = ends.get(idx).copied().unwrap_or(dim).clamp(-dim, dim);
                        let step = steps.get(idx).copied().unwrap_or(1).max(1);
                        let s_norm = if s < 0 { s + dim } else { s };
                        let e_norm = if e < 0 { e + dim } else { e };
                        let len = (e_norm - s_norm).max(0);
                        out[ax] = ((len + step - 1) / step).max(0) as usize;
                    }
                    out
                } else {
                    vec![]
                }
            }
            // Expand: broadcast to target shape.
            OpType::Expand => {
                if let Some(ref in_shape) = input_shape {
                    let target = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec().iter().map(|&v| v as usize).collect())
                        .unwrap_or_else(|| in_shape.clone());
                    broadcast_two(in_shape, &target)
                } else {
                    vec![]
                }
            }
            // Tile: repeat tensor along each axis.
            OpType::Tile => {
                if let Some(ref in_shape) = input_shape {
                    let repeats: Vec<usize> = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .map(|td| td.as_i64_vec().iter().map(|&v| v.max(1) as usize).collect())
                        .unwrap_or_default();
                    in_shape
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| d * repeats.get(i).copied().unwrap_or(1))
                        .collect()
                } else {
                    vec![]
                }
            }
            // TopK: output shape = input shape with last dim replaced by k.
            OpType::TopK => {
                if let Some(ref in_shape) = input_shape {
                    let axis_raw = layer.get_int_attr("axis").unwrap_or(-1);
                    let rank = in_shape.len();
                    let axis = if axis_raw < 0 {
                        (axis_raw + rank as i64).max(0) as usize
                    } else {
                        (axis_raw as usize).min(rank.saturating_sub(1))
                    };
                    let k = layer
                        .inputs
                        .get(1)
                        .and_then(|n| layer.weights.get(n.as_str()))
                        .and_then(|td| td.as_i64_vec().first().copied())
                        .unwrap_or(1)
                        .max(1) as usize;
                    let mut out = in_shape.clone();
                    if axis < out.len() {
                        out[axis] = k;
                    }
                    out
                } else {
                    vec![]
                }
            }
            OpType::GlobalAveragePool => {
                if let Some(ref in_shape) = input_shape {
                    if in_shape.len() >= 2 {
                        let mut out = in_shape[..2].to_vec();
                        out.extend(std::iter::repeat_n(1, in_shape.len() - 2));
                        out
                    } else {
                        in_shape.clone()
                    }
                } else {
                    vec![]
                }
            }
            _ => input_shape.unwrap_or_default(),
        };

        for out_name in &layer.outputs {
            shapes.insert(out_name.clone(), out_shape.clone());
        }
    }

    Ok(shapes)
}

fn compute_bounds(graph: &LayerGraph, config: &ScaleConfig) -> Result<HashMap<String, usize>> {
    let alpha = config.alpha;
    let mut bounds: HashMap<String, f64> = HashMap::new();
    let mut n_bits_config = HashMap::new();

    // Pre-compute tensor shapes for reduction-size estimation.
    let shapes = propagate_shapes(graph)?;

    for name in &graph.input_names {
        bounds.insert(name.clone(), 1.0);
    }

    let alpha_f64 = alpha as f64;

    for layer in graph.iter_topo() {
        let mut bound = compute_layer_bound(layer, &bounds, alpha_f64)?;

        // ReduceSum accumulates values: its bound is N * m_in where N is the
        // reduction factor.  The base compute_layer_bound conservatively
        // returns just m_in; override that here now that we have shapes.
        if layer.op_type == OpType::ReduceSum {
            if let Some(in_name) = layer.inputs.first() {
                let m_in = bounds.get(in_name.as_str()).copied().unwrap_or(1.0);
                if let (Some(in_shape), Some(out_name)) =
                    (shapes.get(in_name.as_str()), layer.outputs.first())
                {
                    let in_total: usize = in_shape.iter().product();
                    let out_total: usize = shapes
                        .get(out_name.as_str())
                        .map(|s| s.iter().product())
                        .unwrap_or(1);
                    let factor = (in_total / out_total.max(1)).max(1) as f64;
                    bound = m_in * factor;
                }
            }
        }

        // MatMul dynamic/dynamic: the base bound is m_in * m_b, but the dot product
        // accumulates K terms, so multiply by K (the contracting dimension).
        if layer.op_type == OpType::MatMul {
            if let (Some(in0_name), Some(in1_name)) = (layer.inputs.first(), layer.inputs.get(1)) {
                let no_weight = !layer.weights.contains_key(in0_name.as_str())
                    && !layer.weights.contains_key(in1_name.as_str());
                if no_weight {
                    let m_in = bounds.get(in0_name.as_str()).copied().unwrap_or(1.0);
                    let m_b = bounds.get(in1_name.as_str()).copied().unwrap_or(1.0);
                    let k = shapes
                        .get(in0_name.as_str())
                        .and_then(|s| s.last().copied())
                        .unwrap_or(1)
                        .max(1) as f64;
                    bound = m_in * m_b * k;
                }
            }
        }

        // Similarly, Add/Sub can accumulate two different branches, but
        // compute_layer_bound already returns m_a + m_b for those, which is correct.

        if layer.needs_rescale || is_range_check_op(layer.op_type) {
            // Guard: compute_n_bits computes (alpha as f64 * bound).log2().
            // If alpha * bound overflows f64 to +Inf, log2 returns +Inf and
            // the subsequent `as usize` saturates, producing an incorrect n_bits.
            // Catch this before it propagates.
            let safe_max = f64::MAX / (alpha as f64);
            if bound > safe_max {
                anyhow::bail!(
                    "layer {}: activation bound {bound:.3e} exceeds safe maximum {safe_max:.3e} \
                    (alpha={alpha} * bound overflows f64 in n_bits sizing); \
                    the model's activation range is too large to quantise safely",
                    layer.name
                );
            }
            let n_bits = compute_n_bits(alpha, bound);
            n_bits_config.insert(layer.name.clone(), n_bits);
        }

        // TopK output[0] = values (bound = m_in), output[1] = indices
        // (range [0, axis_len − 1], not quantised floats). Use a conservative
        // index bound of 1.0 so downstream ops do not inflate their n_bits
        // based on the values bound.
        if layer.op_type == OpType::TopK {
            if let Some(values_name) = layer.outputs.first() {
                bounds.insert(values_name.clone(), bound);
            }
            if let Some(indices_name) = layer.outputs.get(1) {
                bounds.insert(indices_name.clone(), 1.0_f64);
            }
        } else {
            for out_name in &layer.outputs {
                bounds.insert(out_name.clone(), bound);
            }
        }
    }

    Ok(n_bits_config)
}

fn compute_layer_bound(
    layer: &LayerNode,
    prev_bounds: &HashMap<String, f64>,
    alpha_f64: f64,
) -> Result<f64> {
    let get_input_bound = |idx: usize| -> f64 {
        let name = match layer.inputs.get(idx) {
            Some(n) => n,
            None => return 1.0,
        };
        if let Some(&b) = prev_bounds.get(name) {
            return b;
        }
        // Constant initializer: return absolute-max value as the bound.
        if let Some(w) = layer.weights.get(name.as_str()) {
            let vals = w.as_f64_vec();
            if !vals.is_empty() {
                return vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            }
        }
        1.0
    };

    let get_bias_bound = |input_idx: usize| -> f64 {
        layer
            .inputs
            .get(input_idx)
            .and_then(|name| layer.weights.get(name))
            .map(|b| {
                let vals = b.as_f64_vec();
                vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
            })
            .unwrap_or(0.0)
    };

    let max_output_channel_l1 = |input_idx: usize, out_channels: usize| -> Result<f64> {
        let name = layer.inputs.get(input_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: expected weight input at index {input_idx} but only {} inputs present",
                layer.name,
                layer.inputs.len(),
            )
        })?;
        let w = layer.weights.get(name).ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: weight tensor '{}' not found in initializers",
                layer.name,
                name,
            )
        })?;
        let vals = w.as_f64_vec();
        anyhow::ensure!(
            out_channels > 0 && !vals.is_empty(),
            "layer {}: weight tensor '{}' has out_channels={} vals.len()={}",
            layer.name,
            name,
            out_channels,
            vals.len(),
        );
        anyhow::ensure!(
            vals.len() % out_channels == 0,
            "layer {}: weight tensor length {} not divisible by out_channels {}",
            layer.name,
            vals.len(),
            out_channels,
        );
        let per_channel = vals.len() / out_channels;
        Ok((0..out_channels)
            .map(|oc| {
                vals[oc * per_channel..(oc + 1) * per_channel]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max))
    };

    match layer.op_type {
        OpType::Conv => {
            let m_in = get_input_bound(0);
            let weight_name = layer.inputs.get(1).ok_or_else(|| {
                anyhow::anyhow!("layer {}: Conv missing weight input at index 1", layer.name)
            })?;
            let w = layer.weights.get(weight_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {}: Conv weight tensor '{}' not found",
                    layer.name,
                    weight_name,
                )
            })?;
            let d = w.shape();
            let c_out = if d.len() >= 4 { d[0] } else { 1 };
            let weight = max_output_channel_l1(1, c_out)?;
            let bias_bound = get_bias_bound(2);
            Ok(weight * m_in + bias_bound)
        }
        OpType::Gemm => {
            let m_in = get_input_bound(0);
            let weight_name = layer.inputs.get(1).ok_or_else(|| {
                anyhow::anyhow!("layer {}: Gemm missing weight input at index 1", layer.name)
            })?;
            let w = layer.weights.get(weight_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {}: Gemm weight tensor '{}' not found",
                    layer.name,
                    weight_name,
                )
            })?;
            let trans_b = layer
                .get_int_attr("transB")
                .map(|v| v != 0)
                .unwrap_or(false);
            let d = w.shape();
            let n_out = if d.len() >= 2 {
                if trans_b {
                    d[0]
                } else {
                    d[1]
                }
            } else {
                1
            };
            let weight_l1 = if trans_b {
                max_output_channel_l1(1, n_out)?
            } else {
                let vals = w.as_f64_vec();
                if d.len() >= 2 {
                    let (rows, cols) = (d[0], d[1]);
                    anyhow::ensure!(
                        cols > 0 && !vals.is_empty() && vals.len() == rows * cols,
                        "layer {}: Gemm weight tensor size mismatch: vals.len()={} expected {}x{}={}",
                        layer.name, vals.len(), rows, cols, rows * cols,
                    );
                    (0..cols)
                        .map(|c| (0..rows).map(|r| vals[r * cols + c].abs()).sum::<f64>())
                        .fold(0.0_f64, f64::max)
                } else {
                    vals.iter().map(|v| v.abs()).sum::<f64>()
                }
            };
            let bias_bound = get_bias_bound(2);
            Ok(weight_l1 * m_in + bias_bound)
        }
        OpType::BatchNormalization => {
            let m_in = get_input_bound(0);
            let max_mul = layer
                .inputs
                .get(1)
                .and_then(|name| layer.weights.get(name))
                .map(|w| {
                    let vals = w.as_f64_vec();
                    vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                })
                .unwrap_or(1.0);
            let bias_bound = get_bias_bound(2);
            Ok(max_mul * m_in + bias_bound)
        }
        OpType::Mul => {
            let m_a = get_input_bound(0);
            let m_b = get_input_bound(1);
            Ok(m_a * m_b)
        }
        OpType::Add | OpType::Sub => {
            let m_a = get_input_bound(0);
            let m_b = get_input_bound(1);
            Ok(m_a + m_b)
        }
        OpType::Div => {
            let m_a = get_input_bound(0);
            let denom_weight = layer.inputs.get(1).and_then(|name| layer.weights.get(name));
            let min_abs_b = denom_weight
                .map(|w| {
                    w.as_f64_vec()
                        .iter()
                        .map(|v| v.abs())
                        .filter(|&v| v > 0.0)
                        .fold(f64::INFINITY, f64::min)
                })
                .filter(|&v| v.is_finite() && v > 0.0);
            let b = min_abs_b.unwrap_or(1.0);
            Ok(m_a / b)
        }
        OpType::Relu => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::MaxPool | OpType::Max | OpType::Min => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Clip => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Cast
        | OpType::Reshape
        | OpType::Flatten
        | OpType::Squeeze
        | OpType::Unsqueeze
        | OpType::Tile
        | OpType::Gather
        | OpType::Resize
        | OpType::GridSample
        | OpType::Transpose
        | OpType::Slice => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Concat => {
            let mut max_bound = 0.0_f64;
            for i in 0..layer.inputs.len() {
                max_bound = max_bound.max(get_input_bound(i));
            }
            Ok(max_bound)
        }
        // TopK selects K values from the input along an axis — output bound ≤ input bound.
        OpType::TopK => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        // Shape outputs raw dimension integers (not quantized floats); bound is irrelevant.
        OpType::Shape => Ok(1.0),
        // Log output can be negative (for real values < 1). The output of
        // log(x_q / α) * α for small positive inputs can be as large as
        // |ln(α)| * α in the negative direction. We conservatively bound the
        // magnitude as max(m_in, ln(α)) so downstream n_bits calculations
        // are large enough to represent the actual output range.
        OpType::Log => {
            let m_in = get_input_bound(0);
            Ok(m_in.max(alpha_f64.ln()))
        }
        // Expand is a broadcast passthrough — output bound equals input bound.
        OpType::Expand => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        // ReduceMean: mean ≤ max input.
        OpType::ReduceMean => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        // MatMul: like Gemm — if a constant weight is present, use its L1 norm.
        // If both inputs are runtime, use a conservative product bound.
        OpType::MatMul => {
            // Right factor constant [K, N]: max column-wise L1 * input[0] bound.
            if let Some(weight_name) = layer.inputs.get(1) {
                if let Some(w) = layer.weights.get(weight_name) {
                    let vals = w.as_f64_vec();
                    if !vals.is_empty() {
                        let m_in = get_input_bound(0);
                        let d = w.shape();
                        let n_out = if d.len() >= 2 { d[d.len() - 1] } else { 1 };
                        let per_out = if n_out > 0 { vals.len() / n_out } else { 1 };
                        let max_l1 = if n_out > 0 && per_out > 0 {
                            (0..n_out)
                                .map(|c| {
                                    vals.iter()
                                        .skip(c)
                                        .step_by(n_out)
                                        .map(|v| v.abs())
                                        .sum::<f64>()
                                })
                                .fold(0.0_f64, f64::max)
                        } else {
                            vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                        };
                        return Ok(max_l1 * m_in);
                    }
                }
            }
            // Left factor constant [M, K]: max row-wise L1 * input[1] bound.
            if let Some(weight_name) = layer.inputs.first() {
                if let Some(w) = layer.weights.get(weight_name) {
                    let vals = w.as_f64_vec();
                    if !vals.is_empty() {
                        let m_b = get_input_bound(1);
                        let d = w.shape();
                        let n_cols = if d.len() >= 2 { *d.last().unwrap() } else { 1 };
                        let n_rows = if n_cols > 0 { vals.len() / n_cols } else { 1 };
                        let max_row_l1 = if n_rows > 0 && n_cols > 0 {
                            (0..n_rows)
                                .map(|r| {
                                    vals[r * n_cols..(r + 1) * n_cols]
                                        .iter()
                                        .map(|v| v.abs())
                                        .sum::<f64>()
                                })
                                .fold(0.0_f64, f64::max)
                        } else {
                            vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                        };
                        return Ok(max_row_l1 * m_b);
                    }
                }
            }
            let m_in = get_input_bound(0);
            let m_b = get_input_bound(1);
            Ok(m_in * m_b)
        }
        // AveragePool: averaging reduces magnitude — bound ≤ input bound.
        OpType::AveragePool => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        // Pad: output is either from the input or from the constant fill value (input[2]).
        OpType::Pad => {
            let m_in = get_input_bound(0);
            let m_pad = get_input_bound(2).abs();
            Ok(m_in.max(m_pad))
        }
        // Split: splitting doesn't change value range.
        OpType::Split => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        // Where: output is one of X or Y.
        OpType::Where => {
            let m_x = get_input_bound(1);
            let m_y = get_input_bound(2);
            Ok(m_x.max(m_y))
        }
        // Pow: bound depends on the exponent.
        // If the exponent is a constant initializer scalar, use m_in^exponent.
        // Otherwise fall back to m_in^2 (conservative: exponent=2 is the common case).
        OpType::Pow => {
            let m_in = get_input_bound(0);
            let exp = layer
                .inputs
                .get(1)
                .and_then(|name| layer.weights.get(name))
                .and_then(|td| {
                    let vals = td.as_f64_vec();
                    if vals.len() == 1 {
                        Some(vals[0])
                    } else {
                        None
                    }
                });
            match exp {
                Some(e) if e >= 1.0 => Ok(m_in.powf(e)),
                Some(e) if e >= 0.0 => Ok(m_in.max(m_in.powf(e))),
                Some(e) => anyhow::bail!(
                    "layer {}: Pow exponent {e} is negative; \
                     cannot compute a safe quantization bound",
                    layer.name
                ),
                None => anyhow::bail!(
                    "layer {}: Pow exponent is not a compile-time constant initializer; \
                     dynamic exponents are unsupported for safe quantization — \
                     provide a constant initializer for the exponent input",
                    layer.name
                ),
            }
        }
        // Sqrt: sqrt(x) > x when 0 < x < 1, so the conservative bound is max(m_in, sqrt(m_in)).
        OpType::Sqrt => {
            let m_in = get_input_bound(0);
            Ok(m_in.max(m_in.sqrt()))
        }
        // Tanh: output is in (-1, 1).
        OpType::Tanh => Ok(1.0),
        // ReduceSum: sum of values — bound ≤ input bound (conservative).
        // The actual maximum is N * m_in where N is the reduction size,
        // but we don't know N here. Conservatively use m_in.
        OpType::ReduceSum => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        // Erf: output is in (-1, 1).
        OpType::Erf => Ok(1.0),
        // ConvTranspose: similar to Conv output bound.
        OpType::ConvTranspose => {
            let m_in = get_input_bound(0);
            let weight_name = layer.inputs.get(1).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {}: ConvTranspose missing weight input at index 1",
                    layer.name
                )
            })?;
            let w = layer.weights.get(weight_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {}: ConvTranspose weight tensor '{}' not found",
                    layer.name,
                    weight_name,
                )
            })?;
            let d = w.shape();
            // ConvTranspose weights have layout [C_in, C_out/group, *kernel].
            let c_in = if !d.is_empty() { d[0] } else { 1 };
            let c_out = if d.len() >= 2 { d[1] } else { 1 };
            let vals = w.as_f64_vec();
            if vals.is_empty() || c_out == 0 {
                return Ok(m_in);
            }
            let kernel_elems: usize = if d.len() > 2 {
                d[2..].iter().product()
            } else {
                1
            };
            if kernel_elems == 0 {
                return Ok(m_in);
            }
            // For each output channel oc, sum abs over all input channels ci and kernel
            // positions k using the flat index ((ci * c_out + oc) * kernel_elems + k).
            let max_l1 = (0..c_out)
                .map(|oc| {
                    let mut l1 = 0.0_f64;
                    for ci in 0..c_in {
                        for k in 0..kernel_elems {
                            l1 += vals[(ci * c_out + oc) * kernel_elems + k].abs();
                        }
                    }
                    l1
                })
                .fold(0.0_f64, f64::max);
            let bias_bound = get_bias_bound(2);
            Ok(max_l1 * m_in + bias_bound)
        }
        // exp(x) is monotonically increasing; max output is exp(max_input).
        OpType::Exp => {
            let m_in = get_input_bound(0);
            let bound = m_in.exp();
            if !bound.is_finite() {
                anyhow::bail!(
                    "layer {}: Exp input bound {m_in} produces a non-finite exp result; \
                    the model's activation range is too large to quantise safely",
                    layer.name
                );
            }
            Ok(bound)
        }
        // softmax / sigmoid outputs are in [0, 1].
        OpType::LeakyRelu => {
            let m_in = get_input_bound(0);
            let alpha = layer
                .attributes
                .get("alpha")
                .and_then(|v| match v {
                    AttrValue::Float(f) => Some(*f as f64),
                    _ => None,
                })
                .unwrap_or(0.01);
            Ok(m_in.max(alpha.abs() * m_in))
        }
        OpType::Identity => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Neg => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Softmax | OpType::Sigmoid => Ok(1.0),
        // GELU is not capped at 1.0; conservatively propagate input bound.
        OpType::Gelu => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Constant => Ok(1.0),
        OpType::HardSwish => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::GlobalAveragePool => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::InstanceNormalization => {
            let m_in = get_input_bound(0);
            let epsilon = layer.get_float_attr("epsilon").unwrap_or(1e-5_f32) as f64;
            let epsilon = epsilon.max(1e-12);
            let normalization_bound = 2.0 * m_in / epsilon.sqrt();
            let max_gamma = layer
                .inputs
                .get(1)
                .and_then(|name| layer.weights.get(name))
                .map(|w| {
                    let vals = w.as_f64_vec();
                    vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                })
                .unwrap_or(1.0);
            let max_beta = get_bias_bound(2);
            Ok(max_gamma * normalization_bound + max_beta)
        }
        OpType::GroupNormalization => {
            let m_in = get_input_bound(0);
            let epsilon = layer.get_float_attr("epsilon").unwrap_or(1e-5_f32) as f64;
            let epsilon = epsilon.max(1e-12);
            let normalization_bound = 2.0 * m_in / epsilon.sqrt();
            let max_gamma = layer
                .inputs
                .get(1)
                .and_then(|name| layer.weights.get(name))
                .map(|w| {
                    let vals = w.as_f64_vec();
                    vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                })
                .unwrap_or(1.0);
            let max_beta = get_bias_bound(2);
            Ok(max_gamma * normalization_bound + max_beta)
        }
        OpType::Not
        | OpType::And
        | OpType::Equal
        | OpType::Greater
        | OpType::Less
        | OpType::ConstantOfShape => Ok(1.0),
        OpType::Sin | OpType::Cos => Ok(1.0),
        OpType::Range => Ok(1.0),
        OpType::ReduceMax => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::ScatterND => {
            let m_in = get_input_bound(0);
            let m_updates = if layer.inputs.len() > 2 {
                get_input_bound(2)
            } else {
                m_in
            };
            Ok(m_in.max(m_updates))
        }
        OpType::GatherElements => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::LayerNormalization => {
            // After normalization the per-element output is bounded by roughly
            // max|γ| * (x − μ) / σ + max|β|. In the worst case (two values
            // symmetrically around the mean) the normalised value is bounded
            // by 2 * m_in / sqrt(ε), where ε is the LayerNorm epsilon.
            // We use that as a conservative normalization bound so that the
            // final n_bits calculation covers the actual output range.
            let m_in = get_input_bound(0);
            let epsilon = layer.get_float_attr("epsilon").unwrap_or(1e-5_f32) as f64;
            let epsilon = epsilon.max(1e-12); // guard against 0
            let normalization_bound = 2.0 * m_in / epsilon.sqrt();
            let max_gamma = layer
                .inputs
                .get(1)
                .and_then(|name| layer.weights.get(name))
                .map(|w| {
                    let vals = w.as_f64_vec();
                    vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
                })
                .unwrap_or(1.0);
            let max_beta = get_bias_bound(2);
            Ok(max_gamma * normalization_bound + max_beta)
        }
    }
}

fn is_range_check_op(op: OpType) -> bool {
    matches!(
        op,
        OpType::Relu
            | OpType::LeakyRelu
            | OpType::MaxPool
            | OpType::Max
            | OpType::Min
            | OpType::Clip
            | OpType::Exp
            | OpType::Softmax
            | OpType::Sigmoid
            | OpType::Gelu
            | OpType::Resize
            | OpType::GridSample
            | OpType::TopK
            | OpType::AveragePool
            | OpType::Sqrt
            | OpType::Tanh
            | OpType::Erf
            | OpType::Pow
            | OpType::HardSwish
            | OpType::GlobalAveragePool
            | OpType::Sin
            | OpType::Cos
    )
}

fn compute_n_bits(alpha: i64, bound: f64) -> usize {
    let val = (alpha as f64 * bound).abs();
    if val <= 1.0 {
        return 2;
    }
    (val.log2().ceil() as usize) + 1
}
