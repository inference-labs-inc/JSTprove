use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rmpv::Value;

use jstprove_remainder::onnx::graph::{LayerGraph, LayerNode, OpType};
use jstprove_remainder::onnx::parser::{self, AttrValue, ParsedModel};
use jstprove_remainder::onnx::quantizer::{self, QuantizedModel, ScaleConfig};
use jstprove_remainder::onnx::shape_inference;

use crate::circuit_functions::utils::onnx_model::{Architecture, Backend, CircuitParams, WANDB};
use crate::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};

pub struct ExpanderMetadata {
    pub circuit_params: CircuitParams,
    pub architecture: Architecture,
    pub wandb: WANDB,
}

pub fn generate_from_onnx(onnx_path: &Path) -> Result<ExpanderMetadata> {
    let parsed = parser::parse_onnx(onnx_path).context("parsing ONNX model")?;
    let graph = LayerGraph::from_parsed(&parsed).context("building layer graph")?;
    let config = ScaleConfig::default();
    let quantized = quantizer::quantize_model(graph, &config).context("quantizing model")?;
    let shapes = shape_inference::infer_all_shapes(&parsed, &quantized.graph)
        .context("inferring tensor shapes")?;

    let opset_version = parsed.opset_version as i16;

    let circuit_params = build_circuit_params(&parsed, &quantized, &config);
    let architecture = build_architecture(&parsed, &quantized, &shapes, opset_version);
    let wandb = build_wandb(&quantized, &shapes);

    Ok(ExpanderMetadata {
        circuit_params,
        architecture,
        wandb,
    })
}

fn build_circuit_params(
    parsed: &ParsedModel,
    quantized: &QuantizedModel,
    config: &ScaleConfig,
) -> CircuitParams {
    let initializer_names: std::collections::HashSet<&str> =
        parsed.initializers.keys().map(String::as_str).collect();

    let inputs: Vec<ONNXIO> = parsed
        .inputs
        .iter()
        .filter(|io| !initializer_names.contains(io.name.as_str()))
        .map(|io| ONNXIO {
            name: io.name.clone(),
            elem_type: io.elem_type as i16,
            shape: io
                .shape
                .iter()
                .map(|&d| if d <= 0 { 1 } else { d as usize })
                .collect(),
        })
        .collect();

    let outputs: Vec<ONNXIO> = parsed
        .outputs
        .iter()
        .map(|io| ONNXIO {
            name: io.name.clone(),
            elem_type: io.elem_type as i16,
            shape: io
                .shape
                .iter()
                .map(|&d| if d <= 0 { 1 } else { d as usize })
                .collect(),
        })
        .collect();

    let rescale_config: HashMap<String, bool> = quantized
        .graph
        .layers
        .iter()
        .filter(|l| l.needs_rescale)
        .map(|l| (l.name.clone(), true))
        .collect();

    CircuitParams {
        scale_base: config.base as u32,
        scale_exponent: config.exponent,
        rescale_config,
        inputs,
        outputs,
        freivalds_reps: 1,
        n_bits_config: quantized.n_bits_config.clone(),
        weights_as_inputs: false,
        backend: Backend::Expander,
    }
}

fn build_architecture(
    parsed: &ParsedModel,
    quantized: &QuantizedModel,
    all_shapes: &HashMap<String, Vec<usize>>,
    opset_version: i16,
) -> Architecture {
    let mut layers = Vec::new();
    let mut id = 0;

    for layer in &quantized.graph.layers {
        if layer.op_type == OpType::Constant {
            continue;
        }

        let shape = collect_layer_shapes(layer, all_shapes);
        let params = convert_attributes_to_params(layer, parsed);

        layers.push(ONNXLayer {
            id,
            name: layer.name.clone(),
            op_type: op_type_to_string(layer.op_type),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            shape,
            tensor: None,
            params,
            opset_version_number: opset_version,
        });
        id += 1;
    }

    Architecture {
        architecture: layers,
    }
}

fn build_wandb(quantized: &QuantizedModel, all_shapes: &HashMap<String, Vec<usize>>) -> WANDB {
    let mut layers = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut id = 0;

    for layer in &quantized.graph.layers {
        for (name, td) in &layer.weights {
            if seen.contains(name) {
                continue;
            }
            seen.insert(name.clone());

            let shape = td.shape();
            let tensor_data = td.as_i64_vec();
            let tensor_value = nest_flat_data(&tensor_data, &shape);

            let mut shape_map = HashMap::new();
            shape_map.insert(name.clone(), shape.clone());
            if let Some(s) = all_shapes.get(name) {
                shape_map.insert(name.clone(), s.clone());
            }

            layers.push(ONNXLayer {
                id,
                name: name.clone(),
                op_type: "Const".to_string(),
                inputs: vec![],
                outputs: vec![],
                shape: shape_map,
                tensor: Some(tensor_value),
                params: None,
                opset_version_number: -1,
            });
            id += 1;
        }
    }

    WANDB { w_and_b: layers }
}

fn collect_layer_shapes(
    layer: &LayerNode,
    all_shapes: &HashMap<String, Vec<usize>>,
) -> HashMap<String, Vec<usize>> {
    let mut shape_map = HashMap::new();

    for out_name in &layer.outputs {
        if let Some(shape) = all_shapes.get(out_name) {
            shape_map.insert(out_name.clone(), shape.clone());
        }
    }

    for in_name in &layer.inputs {
        if let Some(shape) = all_shapes.get(in_name) {
            shape_map.insert(in_name.clone(), shape.clone());
        }
    }

    shape_map
}

fn op_type_to_string(op: OpType) -> String {
    match op {
        OpType::Add => "Add",
        OpType::Sub => "Sub",
        OpType::Mul => "Mul",
        OpType::Gemm => "Gemm",
        OpType::Conv => "Conv",
        OpType::Relu => "ReLU",
        OpType::MaxPool => "MaxPool",
        OpType::BatchNormalization => "BatchNormalization",
        OpType::Max => "Max",
        OpType::Min => "Min",
        OpType::Clip => "Clip",
        OpType::Reshape => "Reshape",
        OpType::Flatten => "Flatten",
        OpType::Squeeze => "Squeeze",
        OpType::Unsqueeze => "Unsqueeze",
        OpType::Constant => "Constant",
    }
    .to_string()
}

fn convert_attributes_to_params(layer: &LayerNode, parsed: &ParsedModel) -> Option<Value> {
    let mut entries: Vec<(Value, Value)> = Vec::new();

    for (key, val) in &layer.attributes {
        let v = attr_value_to_msgpack(val);
        entries.push((Value::String(key.clone().into()), v));
    }

    attach_constant_inputs(layer, parsed, &mut entries);

    Some(Value::Map(entries))
}

fn attach_constant_inputs(
    layer: &LayerNode,
    parsed: &ParsedModel,
    entries: &mut Vec<(Value, Value)>,
) {
    for node in &parsed.nodes {
        if node.op_type != "Constant" {
            continue;
        }
        let const_output = match node.outputs.first() {
            Some(o) => o,
            None => continue,
        };
        if !layer.inputs.contains(const_output) {
            continue;
        }
        if let Some(AttrValue::Tensor(td)) = node.attributes.get("value") {
            let vals = if !td.float_data.is_empty() {
                Value::Array(td.float_data.iter().map(|&v| Value::from(v)).collect())
            } else {
                Value::Array(td.int_data.iter().map(|&v| Value::from(v)).collect())
            };
            entries.push((Value::String(const_output.clone().into()), vals));
        }
    }
}

fn nest_flat_data(data: &[i64], shape: &[usize]) -> Value {
    if shape.is_empty() || shape.iter().product::<usize>() == 0 {
        return Value::Array(data.iter().map(|&v| Value::from(v)).collect());
    }
    if shape.len() == 1 {
        return Value::Array(data.iter().map(|&v| Value::from(v)).collect());
    }
    let stride: usize = shape[1..].iter().product();
    Value::Array(
        (0..shape[0])
            .map(|i| nest_flat_data(&data[i * stride..(i + 1) * stride], &shape[1..]))
            .collect(),
    )
}

fn attr_value_to_msgpack(val: &AttrValue) -> Value {
    match val {
        AttrValue::Float(f) => Value::from(*f as f64),
        AttrValue::Int(i) => Value::from(*i),
        AttrValue::String(s) => Value::String(s.clone().into()),
        AttrValue::Floats(v) => Value::Array(v.iter().map(|&f| Value::from(f as f64)).collect()),
        AttrValue::Ints(v) => Value::Array(v.iter().map(|&i| Value::from(i)).collect()),
        AttrValue::Tensor(td) => {
            let vals = if !td.float_data.is_empty() {
                td.float_data.iter().map(|&v| Value::from(v)).collect()
            } else {
                td.int_data.iter().map(|&v| Value::from(v)).collect()
            };
            Value::Array(vals)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lenet_metadata_generation() {
        let model_path = Path::new("models/lenet.onnx");
        if !model_path.exists() {
            return;
        }

        let metadata = generate_from_onnx(model_path).unwrap();

        assert_eq!(metadata.circuit_params.scale_base, 2);
        assert_eq!(metadata.circuit_params.scale_exponent, 18);
        assert_eq!(metadata.circuit_params.backend, Backend::Expander);
        assert!(!metadata.circuit_params.weights_as_inputs);
        assert!(!metadata.circuit_params.inputs.is_empty());
        assert!(!metadata.circuit_params.outputs.is_empty());
        assert!(!metadata.architecture.architecture.is_empty());
        assert!(!metadata.wandb.w_and_b.is_empty());
        assert!(!metadata.circuit_params.n_bits_config.is_empty());
        assert!(!metadata.circuit_params.rescale_config.is_empty());

        for layer in &metadata.architecture.architecture {
            assert!(!layer.name.is_empty());
            assert!(!layer.op_type.is_empty());
            assert!(!layer.outputs.is_empty());
        }

        for wb in &metadata.wandb.w_and_b {
            assert!(wb.tensor.is_some());
            assert!(!wb.shape.is_empty());
        }
    }

    #[test]
    fn op_type_string_conversion() {
        assert_eq!(op_type_to_string(OpType::Conv), "Conv");
        assert_eq!(op_type_to_string(OpType::Gemm), "Gemm");
        assert_eq!(op_type_to_string(OpType::Relu), "ReLU");
        assert_eq!(op_type_to_string(OpType::Add), "Add");
        assert_eq!(op_type_to_string(OpType::Reshape), "Reshape");
    }
}
