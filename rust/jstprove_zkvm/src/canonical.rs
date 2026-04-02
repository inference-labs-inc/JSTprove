use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalModel {
    pub base: u64,
    pub exponent: u32,
    pub alpha: u64,
    pub input_dims: Vec<u32>,
    pub layers: Vec<CanonicalLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalLayer {
    pub op_type: u8,
    pub needs_rescale: bool,
    pub n_inputs: u32,
    pub weight_dims: Vec<u32>,
    pub weight_data: Vec<i64>,
    pub bias_dims: Vec<u32>,
    pub bias_data: Vec<i64>,
    pub int_attrs: Vec<i64>,
}

impl CanonicalModel {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.base.to_le_bytes());
        buf.extend_from_slice(&self.exponent.to_le_bytes());
        buf.extend_from_slice(&self.alpha.to_le_bytes());

        buf.extend_from_slice(&(self.input_dims.len() as u32).to_le_bytes());
        for &d in &self.input_dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }

        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            buf.push(layer.op_type);
            buf.push(u8::from(layer.needs_rescale));
            buf.extend_from_slice(&layer.n_inputs.to_le_bytes());

            buf.extend_from_slice(&(layer.weight_dims.len() as u32).to_le_bytes());
            for &d in &layer.weight_dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&(layer.weight_data.len() as u32).to_le_bytes());
            for &v in &layer.weight_data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf.extend_from_slice(&(layer.bias_dims.len() as u32).to_le_bytes());
            for &d in &layer.bias_dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&(layer.bias_data.len() as u32).to_le_bytes());
            for &v in &layer.bias_data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf.extend_from_slice(&(layer.int_attrs.len() as u32).to_le_bytes());
            for &v in &layer.int_attrs {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        buf
    }

    pub fn from_quantized_model_bytes(data: &[u8]) -> Result<Self> {
        let model: QuantizedModelCompat =
            rmp_serde::from_slice(data).context("deserializing QuantizedModel")?;

        let alpha = (model.scale_config.base as u64)
            .checked_pow(model.scale_config.exponent)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "alpha overflow: base={} exponent={}",
                    model.scale_config.base,
                    model.scale_config.exponent,
                )
            })?;

        let input_dims: Vec<u32> = model
            .graph
            .input_names
            .first()
            .and_then(|name| model.graph.input_shapes.get(name))
            .map(|s| s.iter().map(|&d| d as u32).collect())
            .unwrap_or_default();

        let mut layers = Vec::with_capacity(model.graph.layers.len());
        for layer in model.graph.layers.iter() {
            let (weight_dims, weight_data, bias_dims, bias_data) =
                extract_weights(&layer.weights, &layer.inputs);

            layers.push(CanonicalLayer {
                op_type: op_type_to_u8_from_enum(&layer.op_type),
                needs_rescale: layer.needs_rescale,
                n_inputs: layer.inputs.len() as u32,
                weight_dims,
                weight_data,
                bias_dims,
                bias_data,
                int_attrs: extract_int_attrs(&layer.attributes),
            });
        }

        let topo = &model.graph.topo_order;
        anyhow::ensure!(
            topo.len() == layers.len(),
            "topo_order length {} != layers length {}",
            topo.len(),
            layers.len(),
        );
        let mut seen = vec![false; layers.len()];
        for &idx in topo {
            anyhow::ensure!(
                idx < layers.len(),
                "topo_order index {} out of bounds ({})",
                idx,
                layers.len(),
            );
            anyhow::ensure!(!seen[idx], "topo_order contains duplicate index {}", idx,);
            seen[idx] = true;
        }
        let mut ordered_layers = Vec::with_capacity(layers.len());
        for &idx in topo {
            ordered_layers.push(layers[idx].clone());
        }

        Ok(CanonicalModel {
            base: model.scale_config.base,
            exponent: model.scale_config.exponent,
            alpha,
            input_dims,
            layers: ordered_layers,
        })
    }
}

fn floats_to_i64(float_data: &[f64]) -> Vec<i64> {
    float_data
        .iter()
        .map(|&f| {
            debug_assert!(
                f == f.floor(),
                "non-integer float {f} in weight data; quantization may be incomplete",
            );
            f as i64
        })
        .collect()
}

fn extract_weights(
    weights: &HashMap<String, TensorDataCompat>,
    inputs: &[String],
) -> (Vec<u32>, Vec<i64>, Vec<u32>, Vec<i64>) {
    let weight_td = inputs.get(1).and_then(|n| weights.get(n));
    let bias_td = inputs.get(2).and_then(|n| weights.get(n));

    let (wd, wv) = match weight_td {
        Some(td) => (
            td.dims.iter().map(|&d| d as u32).collect(),
            if !td.int_data.is_empty() {
                td.int_data.clone()
            } else {
                floats_to_i64(&td.float_data)
            },
        ),
        None => (vec![], vec![]),
    };
    let (bd, bv) = match bias_td {
        Some(td) => (
            td.dims.iter().map(|&d| d as u32).collect(),
            if !td.int_data.is_empty() {
                td.int_data.clone()
            } else {
                floats_to_i64(&td.float_data)
            },
        ),
        None => (vec![], vec![]),
    };
    (wd, wv, bd, bv)
}

fn extract_int_attrs(attrs: &HashMap<String, AttrValueCompat>) -> Vec<i64> {
    let mut keys: Vec<&String> = attrs.keys().collect();
    keys.sort();
    let mut out = Vec::new();
    for key in keys {
        match &attrs[key] {
            AttrValueCompat::Int(v) => out.push(*v),
            AttrValueCompat::Ints(v) => out.extend(v.iter().copied()),
            _ => {}
        }
    }
    out
}

fn op_type_to_u8_from_enum(op: &OpTypeCompat) -> u8 {
    match op {
        OpTypeCompat::Add => guest::OP_ADD,
        OpTypeCompat::Div => guest::OP_DIV,
        OpTypeCompat::Sub => guest::OP_SUB,
        OpTypeCompat::Mul => guest::OP_MUL,
        OpTypeCompat::Gemm => guest::OP_GEMM,
        OpTypeCompat::Conv => guest::OP_CONV,
        OpTypeCompat::Relu => guest::OP_RELU,
        OpTypeCompat::MaxPool => guest::OP_MAXPOOL,
        OpTypeCompat::BatchNormalization => guest::OP_BATCHNORM,
        OpTypeCompat::Max => guest::OP_MAX,
        OpTypeCompat::Min => guest::OP_MIN,
        OpTypeCompat::Cast => guest::OP_CAST,
        OpTypeCompat::Clip => guest::OP_CLIP,
        OpTypeCompat::Exp => guest::OP_EXP,
        OpTypeCompat::Reshape => guest::OP_RESHAPE,
        OpTypeCompat::Flatten => guest::OP_FLATTEN,
        OpTypeCompat::Squeeze => guest::OP_SQUEEZE,
        OpTypeCompat::Unsqueeze => guest::OP_UNSQUEEZE,
        OpTypeCompat::Constant => guest::OP_CONSTANT,
        OpTypeCompat::Softmax => guest::OP_SOFTMAX,
        OpTypeCompat::Sigmoid => guest::OP_SIGMOID,
        OpTypeCompat::Gelu => guest::OP_GELU,
        OpTypeCompat::Tile => guest::OP_TILE,
        OpTypeCompat::Gather => guest::OP_GATHER,
        OpTypeCompat::LayerNormalization => guest::OP_LAYERNORM,
        OpTypeCompat::Resize => guest::OP_RESIZE,
        OpTypeCompat::GridSample => guest::OP_GRIDSAMPLE,
        OpTypeCompat::Transpose => guest::OP_TRANSPOSE,
        OpTypeCompat::Concat => guest::OP_CONCAT,
        OpTypeCompat::Slice => guest::OP_SLICE,
        OpTypeCompat::TopK => guest::OP_TOPK,
        OpTypeCompat::Shape => guest::OP_SHAPE,
        OpTypeCompat::Log => guest::OP_LOG,
        OpTypeCompat::Expand => guest::OP_EXPAND,
        OpTypeCompat::ReduceMean => guest::OP_REDUCEMEAN,
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizedModelCompat {
    pub graph: LayerGraphCompat,
    pub scale_config: ScaleConfigCompat,
    #[serde(default)]
    pub n_bits_config: HashMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScaleConfigCompat {
    pub base: u64,
    pub exponent: u32,
    pub alpha: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerGraphCompat {
    pub layers: Vec<LayerNodeCompat>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub topo_order: Vec<usize>,
    pub input_shapes: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerNodeCompat {
    pub id: usize,
    pub name: String,
    pub op_type: OpTypeCompat,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weights: HashMap<String, TensorDataCompat>,
    pub attributes: HashMap<String, AttrValueCompat>,
    pub output_shape: Vec<usize>,
    pub needs_rescale: bool,
    pub n_bits: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorDataCompat {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub float_data: Vec<f64>,
    pub int_data: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub enum OpTypeCompat {
    Add,
    Div,
    Sub,
    Mul,
    Gemm,
    Conv,
    Relu,
    MaxPool,
    BatchNormalization,
    Max,
    Min,
    Cast,
    Clip,
    Exp,
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,
    Constant,
    Softmax,
    Sigmoid,
    Gelu,
    Tile,
    Gather,
    LayerNormalization,
    Resize,
    GridSample,
    Transpose,
    Concat,
    Slice,
    TopK,
    Shape,
    Log,
    Expand,
    ReduceMean,
}

#[derive(Debug, Clone, Deserialize)]
pub enum AttrValueCompat {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Tensor(TensorDataCompat),
}
