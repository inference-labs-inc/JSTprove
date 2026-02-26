use std::collections::HashMap;

use anyhow::Result;

use super::graph::{LayerGraph, LayerNode, OpType};
use super::ops;
use super::parser::TensorData;

pub const DEFAULT_SCALE_BASE: u64 = 2;
pub const DEFAULT_SCALE_EXPONENT: u32 = 18;

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

pub fn quantize_model(mut graph: LayerGraph, config: &ScaleConfig) -> Result<QuantizedModel> {
    let alpha = config.alpha;

    for layer in &mut graph.layers {
        if layer.op_type == OpType::BatchNormalization {
            fold_batchnorm_params(layer)?;
        }
    }

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

fn compute_bounds(graph: &LayerGraph, config: &ScaleConfig) -> Result<HashMap<String, usize>> {
    let alpha = config.alpha;
    let mut bounds: HashMap<String, f64> = HashMap::new();
    let mut n_bits_config = HashMap::new();

    for name in &graph.input_names {
        bounds.insert(name.clone(), 1.0);
    }

    for layer in graph.iter_topo() {
        let bound = compute_layer_bound(layer, &bounds)?;

        if layer.needs_rescale || is_range_check_op(layer.op_type) {
            let n_bits = compute_n_bits(alpha, bound);
            n_bits_config.insert(layer.name.clone(), n_bits);
        }

        for out_name in &layer.outputs {
            bounds.insert(out_name.clone(), bound);
        }
    }

    Ok(n_bits_config)
}

fn compute_layer_bound(layer: &LayerNode, prev_bounds: &HashMap<String, f64>) -> Result<f64> {
    let get_input_bound = |idx: usize| -> f64 {
        layer
            .inputs
            .get(idx)
            .and_then(|name| prev_bounds.get(name))
            .copied()
            .unwrap_or(1.0)
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
            match min_abs_b {
                Some(b) => Ok(m_a / b),
                None => anyhow::bail!(
                    "layer {}: Div denominator is not a constant initializer or contains zero; \
                     cannot compute a safe bound",
                    layer.name,
                ),
            }
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
        OpType::Cast | OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Constant => Ok(1.0),
    }
}

fn is_range_check_op(op: OpType) -> bool {
    matches!(
        op,
        OpType::Relu | OpType::MaxPool | OpType::Max | OpType::Min | OpType::Clip
    )
}

fn compute_n_bits(alpha: i64, bound: f64) -> usize {
    let val = (alpha as f64 * bound).abs();
    if val <= 1.0 {
        return 2;
    }
    (val.log2().ceil() as usize) + 1
}
