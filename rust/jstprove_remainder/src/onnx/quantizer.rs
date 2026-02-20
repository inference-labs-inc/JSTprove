use std::collections::HashMap;

use anyhow::Result;

use super::graph::{LayerGraph, LayerNode, OpType};
use super::ops;

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

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantizedModel {
    pub graph: LayerGraph,
    pub scale_config: ScaleConfig,
    pub n_bits_config: HashMap<String, usize>,
}

pub fn quantize_model(mut graph: LayerGraph, config: &ScaleConfig) -> Result<QuantizedModel> {
    let alpha = config.alpha;

    for layer in &mut graph.layers {
        quantize_layer_weights(layer, alpha)?;
    }

    let n_bits_config = compute_bounds(&graph, config)?;

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

fn compute_bounds(graph: &LayerGraph, config: &ScaleConfig) -> Result<HashMap<String, usize>> {
    let alpha = config.alpha;
    let mut bounds: HashMap<String, f64> = HashMap::new();
    let mut n_bits_config = HashMap::new();

    for layer in graph.iter_topo() {
        let bound = compute_layer_bound(layer, &bounds, alpha)?;

        if layer.needs_rescale || is_range_check_op(layer.op_type) {
            let rangecheck_bound = if layer.needs_rescale {
                bound
            } else {
                bound
            };
            let n_bits = compute_n_bits(alpha, rangecheck_bound);
            n_bits_config.insert(layer.name.clone(), n_bits);
            if layer.needs_rescale {
                bounds.insert(layer.name.clone(), rangecheck_bound / alpha as f64);
            } else {
                bounds.insert(layer.name.clone(), bound);
            }
        } else {
            bounds.insert(layer.name.clone(), bound);
        }
    }

    Ok(n_bits_config)
}

fn compute_layer_bound(
    layer: &LayerNode,
    prev_bounds: &HashMap<String, f64>,
    alpha: i64,
) -> Result<f64> {
    let get_input_bound = |idx: usize| -> f64 {
        layer
            .inputs
            .get(idx)
            .and_then(|name| prev_bounds.get(name))
            .copied()
            .unwrap_or(alpha as f64)
    };

    let get_weight_bound = |input_idx: usize| -> f64 {
        layer
            .inputs
            .get(input_idx)
            .and_then(|name| layer.weights.get(name))
            .map(|w| {
                let vals = w.as_i64_vec();
                vals.iter().map(|v| v.unsigned_abs()).sum::<u64>() as f64
            })
            .unwrap_or(1.0)
    };

    let get_bias_bound = |input_idx: usize| -> f64 {
        layer
            .inputs
            .get(input_idx)
            .and_then(|name| layer.weights.get(name))
            .map(|b| {
                let vals = b.as_i64_vec();
                vals.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0) as f64
            })
            .unwrap_or(0.0)
    };

    match layer.op_type {
        OpType::Conv => {
            let m_in = get_input_bound(0);
            let weight = get_weight_bound(1);
            let bias_bound = get_bias_bound(2);
            Ok(weight * m_in + bias_bound)
        }
        OpType::Gemm => {
            let m_in = get_input_bound(0);
            let weight_sum = get_weight_bound(1);
            let bias_bound = get_bias_bound(2);
            Ok(weight_sum * m_in + bias_bound)
        }
        OpType::BatchNormalization => {
            let m_in = get_input_bound(0);
            Ok(m_in * alpha as f64)
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
        OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
            let m_in = get_input_bound(0);
            Ok(m_in)
        }
        OpType::Constant => Ok(alpha as f64),
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
