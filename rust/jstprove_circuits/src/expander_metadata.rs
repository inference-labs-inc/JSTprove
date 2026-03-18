use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rmpv::Value;

use jstprove_onnx::graph::{LayerGraph, LayerNode, OpType};
use jstprove_onnx::parser::{self, AttrValue, ParsedModel};
use jstprove_onnx::quantizer::{self, QuantizedModel, ScaleConfig};
use jstprove_onnx::shape_inference;

use crate::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};
use crate::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};
use crate::proof_system::ProofSystem;

pub struct ExpanderMetadata {
    pub circuit_params: CircuitParams,
    pub architecture: Architecture,
    pub wandb: WANDB,
}

pub fn generate_from_onnx(onnx_path: &Path) -> Result<ExpanderMetadata> {
    generate_from_onnx_with_all_options(onnx_path, false, None, None)
}

pub fn generate_from_onnx_with_options(
    onnx_path: &Path,
    weights_as_inputs: bool,
) -> Result<ExpanderMetadata> {
    generate_from_onnx_with_all_options(onnx_path, weights_as_inputs, None, None)
}

/// # Errors
/// Returns an error if ONNX parsing, quantization, or shape inference fails,
/// or if the requested precision exceeds the field's capacity.
pub fn generate_from_onnx_for_field(
    onnx_path: &Path,
    n_bits: u32,
    target_precision: Option<u32>,
) -> Result<ExpanderMetadata> {
    generate_from_onnx_with_all_options(onnx_path, false, Some(n_bits), target_precision)
}

fn generate_from_onnx_with_all_options(
    onnx_path: &Path,
    weights_as_inputs: bool,
    n_bits: Option<u32>,
    target_precision: Option<u32>,
) -> Result<ExpanderMetadata> {
    let parsed = parser::parse_onnx(onnx_path).context("parsing ONNX model")?;
    let mut graph = LayerGraph::from_parsed(&parsed).context("building layer graph")?;

    let quantized = match (n_bits, target_precision) {
        (Some(nb), Some(digits)) => quantizer::quantize_model_for_precision(graph, digits, nb)
            .context("precision-targeted quantization")?,
        (Some(nb), None) => {
            let max_bound = quantizer::compute_max_bound(&mut graph)?;
            let config = ScaleConfig::adaptive(nb, max_bound);
            quantizer::quantize_model(graph, &config).context("field-aware quantization")?
        }
        (None, Some(_)) => anyhow::bail!("target_precision requires n_bits"),
        _ => {
            let config = ScaleConfig::default();
            quantizer::quantize_model(graph, &config).context("quantizing model")?
        }
    };
    let config = &quantized.scale_config;

    let shapes = shape_inference::infer_all_shapes(&parsed, &quantized.graph)
        .context("inferring tensor shapes")?;

    let opset_version = parsed.opset_version as i16;

    let circuit_params = build_circuit_params(&parsed, &quantized, config, weights_as_inputs);
    let architecture = build_architecture(&parsed, &quantized, &shapes, opset_version);
    let wandb = build_wandb(&quantized, &shapes).context("building weight/bias data")?;

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
    weights_as_inputs: bool,
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
        weights_as_inputs,
        proof_system: ProofSystem::Expander,
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

fn build_wandb(
    quantized: &QuantizedModel,
    all_shapes: &HashMap<String, Vec<usize>>,
) -> Result<WANDB> {
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
            let tensor_value = nest_flat_data(&tensor_data, &shape)
                .with_context(|| format!("nesting tensor '{name}'"))?;

            let mut shape_map = HashMap::new();
            let resolved_shape = all_shapes.get(name).cloned().unwrap_or(shape);
            shape_map.insert(name.clone(), resolved_shape);

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

    Ok(WANDB { w_and_b: layers })
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
        OpType::Div => "Div",
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

fn nest_flat_data(data: &[i64], shape: &[usize]) -> Result<Value> {
    let expected: usize = shape.iter().product();
    if !shape.is_empty() && expected != data.len() {
        anyhow::bail!(
            "nest_flat_data: shape product {expected} != data length {}",
            data.len()
        );
    }
    if shape.is_empty() || expected == 0 {
        return Ok(Value::Array(data.iter().map(|&v| Value::from(v)).collect()));
    }
    if shape.len() == 1 {
        return Ok(Value::Array(
            data[..shape[0]].iter().map(|&v| Value::from(v)).collect(),
        ));
    }
    let stride: usize = shape[1..].iter().product();
    let nested: Vec<Value> = (0..shape[0])
        .map(|i| {
            let start = i * stride;
            let end = start + stride;
            nest_flat_data(&data[start..end], &shape[1..])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Array(nested))
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
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );
        let model_path = model_path.as_path();

        let metadata = generate_from_onnx(model_path).unwrap();

        assert_eq!(metadata.circuit_params.scale_base, 2);
        assert_eq!(metadata.circuit_params.scale_exponent, 18);
        assert_eq!(metadata.circuit_params.proof_system, ProofSystem::Expander);
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
    fn lenet_metadata_generation_weights_as_inputs() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );

        let metadata = generate_from_onnx_with_options(model_path.as_path(), true).unwrap();

        assert!(metadata.circuit_params.weights_as_inputs);
        assert_eq!(metadata.circuit_params.scale_base, 2);
        assert_eq!(metadata.circuit_params.scale_exponent, 18);
        assert!(!metadata.circuit_params.inputs.is_empty());
        assert!(!metadata.circuit_params.outputs.is_empty());
    }

    #[test]
    fn bn254_adaptive_exponent_at_least_default() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );

        let parsed = parser::parse_onnx(&model_path).unwrap();
        let mut graph = LayerGraph::from_parsed(&parsed).unwrap();
        let max_bound = quantizer::compute_max_bound(&mut graph).unwrap();
        let expected_exp =
            ScaleConfig::max_safe_exponent(jstprove_onnx::quantizer::N_BITS_BN254, max_bound);

        let metadata = generate_from_onnx_for_field(
            model_path.as_path(),
            jstprove_onnx::quantizer::N_BITS_BN254,
            None,
        )
        .unwrap();

        assert_eq!(metadata.circuit_params.scale_base, 2);
        assert_eq!(metadata.circuit_params.scale_exponent, expected_exp);
        assert!(!metadata.circuit_params.n_bits_config.is_empty());
        assert!(!metadata.circuit_params.rescale_config.is_empty());
    }

    #[test]
    fn bn254_5_digit_precision() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );

        let metadata = generate_from_onnx_for_field(
            model_path.as_path(),
            jstprove_onnx::quantizer::N_BITS_BN254,
            Some(5),
        )
        .unwrap();

        assert_eq!(metadata.circuit_params.scale_base, 2);
        let expected_exp = ScaleConfig::exponent_for_digits(5);
        assert_eq!(metadata.circuit_params.scale_exponent, expected_exp);
    }

    #[test]
    fn bn254_8_digit_precision() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );

        let metadata = generate_from_onnx_for_field(
            model_path.as_path(),
            jstprove_onnx::quantizer::N_BITS_BN254,
            Some(8),
        )
        .unwrap();

        assert_eq!(metadata.circuit_params.scale_base, 2);
        let expected_exp = ScaleConfig::exponent_for_digits(8);
        assert_eq!(metadata.circuit_params.scale_exponent, expected_exp);
    }

    #[test]
    fn goldilocks_adaptive_exponent() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );

        let parsed = parser::parse_onnx(&model_path).unwrap();
        let mut graph = LayerGraph::from_parsed(&parsed).unwrap();
        let max_bound = quantizer::compute_max_bound(&mut graph).unwrap();
        let expected_exp =
            ScaleConfig::max_safe_exponent(jstprove_onnx::quantizer::N_BITS_GOLDILOCKS, max_bound);

        let metadata = generate_from_onnx_for_field(
            model_path.as_path(),
            jstprove_onnx::quantizer::N_BITS_GOLDILOCKS,
            None,
        )
        .unwrap();

        assert_eq!(metadata.circuit_params.scale_base, 2);
        assert_eq!(metadata.circuit_params.scale_exponent, expected_exp);
        assert!(!metadata.circuit_params.n_bits_config.is_empty());
    }

    #[test]
    fn precision_exceeds_goldilocks_capacity() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(
            model_path.exists(),
            "missing test fixture: {}",
            model_path.display()
        );

        let result = generate_from_onnx_for_field(
            model_path.as_path(),
            jstprove_onnx::quantizer::N_BITS_GOLDILOCKS,
            Some(9),
        );

        assert!(result.is_err());
        let msg = format!("{:#}", result.err().expect("expected an error"));
        assert!(
            msg.contains("decimal digits"),
            "error should mention decimal digits: {msg}"
        );
    }

    #[test]
    fn op_type_string_conversion() {
        assert_eq!(op_type_to_string(OpType::Conv), "Conv");
        assert_eq!(op_type_to_string(OpType::Gemm), "Gemm");
        assert_eq!(op_type_to_string(OpType::Relu), "ReLU");
        assert_eq!(op_type_to_string(OpType::Add), "Add");
        assert_eq!(op_type_to_string(OpType::Div), "Div");
        assert_eq!(op_type_to_string(OpType::Reshape), "Reshape");
    }
}
