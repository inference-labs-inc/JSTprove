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
use crate::proof_config::ProofConfig;
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
    let graph = LayerGraph::from_parsed(&parsed).context("building layer graph")?;
    generate_from_parsed(
        parsed,
        graph,
        weights_as_inputs,
        n_bits,
        target_precision,
        None,
        None,
    )
}

/// Generate metadata using externally-resolved tensor shapes instead of
/// running jstprove's internal shape inference. The caller (typically
/// dsperse via tract) provides a complete shape map that is trusted
/// as the single source of truth.
pub fn generate_from_onnx_with_shapes(
    onnx_path: &Path,
    precomputed_shapes: HashMap<String, Vec<usize>>,
) -> Result<ExpanderMetadata> {
    let parsed = parser::parse_onnx(onnx_path).context("parsing ONNX model")?;
    let graph = LayerGraph::from_parsed(&parsed).context("building layer graph")?;
    generate_from_parsed(
        parsed,
        graph,
        false,
        None,
        None,
        None,
        Some(precomputed_shapes),
    )
}

/// When `precomputed_max_bound` is `Some`, the caller must have already called
/// `quantizer::compute_max_bound` on `graph` (which applies `fold_all_batchnorms`).
/// Passing a stale value from an unfolded graph produces incorrect scale configs.
/// Pass `None` to let this function compute it on demand.
///
/// When `precomputed_shapes` is `Some`, the provided shape map is used directly
/// instead of running `infer_all_shapes`. This allows an external shape oracle
/// (e.g. tract) to serve as the single source of truth, eliminating dual
/// shape inference between jstprove and its callers.
fn generate_from_parsed(
    parsed: ParsedModel,
    mut graph: LayerGraph,
    weights_as_inputs: bool,
    n_bits: Option<u32>,
    target_precision: Option<u32>,
    precomputed_max_bound: Option<f64>,
    precomputed_shapes: Option<HashMap<String, Vec<usize>>>,
) -> Result<ExpanderMetadata> {
    let quantized = match (n_bits, target_precision) {
        (Some(nb), Some(digits)) => quantizer::quantize_model_for_precision(graph, digits, nb)
            .context("precision-targeted quantization")?,
        (Some(nb), None) => {
            let max_bound = match precomputed_max_bound {
                Some(mb) => mb,
                None => quantizer::compute_max_bound(&mut graph)?,
            };
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

    let shapes = match precomputed_shapes {
        Some(s) => {
            let mut missing = Vec::new();
            for layer in &quantized.graph.layers {
                for name in layer.inputs.iter().chain(layer.outputs.iter()) {
                    if !s.contains_key(name) && !parsed.initializers.contains_key(name) {
                        missing.push(name.clone());
                    }
                }
            }
            if !missing.is_empty() {
                missing.sort();
                missing.dedup();
                tracing::warn!(
                    count = missing.len(),
                    names = ?missing,
                    "precomputed shape map missing tensor entries; \
                     downstream metadata may be incomplete"
                );
            }
            s
        }
        None => shape_inference::infer_all_shapes(&parsed, &quantized.graph)
            .context("inferring tensor shapes")?,
    };

    let opset_version = parsed.opset_version as i16;

    let proof_config = n_bits.map(infer_proof_config_from_n_bits);
    let circuit_params =
        build_circuit_params(&parsed, &quantized, config, weights_as_inputs, proof_config);
    let architecture = build_architecture(&parsed, &quantized, &shapes, opset_version);
    let wandb = build_wandb(&quantized, &shapes).context("building weight/bias data")?;

    Ok(ExpanderMetadata {
        circuit_params,
        architecture,
        wandb,
    })
}

/// Pick a default proof config for a given quantization bit-width.
/// The mapping reflects the field tier required to represent values
/// at that precision; the PCS choice is the conventional default for
/// each field (Basefold for Goldilocks variants, Raw for BN254).
fn infer_proof_config_from_n_bits(n_bits: u32) -> ProofConfig {
    if n_bits <= jstprove_onnx::quantizer::N_BITS_GOLDILOCKS {
        ProofConfig::GoldilocksBasefold
    } else if n_bits <= jstprove_onnx::quantizer::N_BITS_GOLDILOCKS_EXT2 {
        ProofConfig::GoldilocksExt2Basefold
    } else {
        ProofConfig::Bn254Raw
    }
}

pub fn select_goldilocks_tier(
    max_bound: f64,
    target_precision: Option<u32>,
) -> Result<(ProofConfig, u32)> {
    use jstprove_onnx::quantizer::{
        MIN_USEFUL_EXPONENT, N_BITS_GOLDILOCKS, N_BITS_GOLDILOCKS_EXT2,
    };

    let base_exp = ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS, max_bound);
    let ext2_exp = ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS_EXT2, max_bound);

    match target_precision {
        Some(digits) => {
            let required_exp = ScaleConfig::exponent_for_digits(digits);
            if base_exp >= required_exp {
                Ok((ProofConfig::GoldilocksBasefold, N_BITS_GOLDILOCKS))
            } else if ext2_exp >= required_exp {
                Ok((ProofConfig::GoldilocksExt2Basefold, N_BITS_GOLDILOCKS_EXT2))
            } else {
                let max_digits_ext2 =
                    ScaleConfig::max_safe_digits(N_BITS_GOLDILOCKS_EXT2, max_bound);
                anyhow::bail!(
                    "requested {digits} decimal digits requires exponent={required_exp}, \
                     but Goldilocks base (31-bit) supports exponent up to {base_exp} and \
                     GoldilocksExt2 (63-bit) supports up to {ext2_exp} ({max_digits_ext2} digits) \
                     for this model's accumulation bound {max_bound:.2}"
                )
            }
        }
        None => {
            if base_exp >= MIN_USEFUL_EXPONENT {
                Ok((ProofConfig::GoldilocksBasefold, N_BITS_GOLDILOCKS))
            } else if ext2_exp >= MIN_USEFUL_EXPONENT {
                Ok((ProofConfig::GoldilocksExt2Basefold, N_BITS_GOLDILOCKS_EXT2))
            } else {
                anyhow::bail!(
                    "model accumulation bound {max_bound:.2} exceeds Goldilocks field family \
                     capacity: base (31-bit) max exponent={base_exp}, ext2 (63-bit) max \
                     exponent={ext2_exp}, minimum useful={MIN_USEFUL_EXPONENT}"
                )
            }
        }
    }
}

/// # Errors
/// Returns an error if ONNX parsing, quantization, or shape inference fails.
pub fn generate_from_onnx_goldilocks_auto(
    onnx_path: &Path,
    target_precision: Option<u32>,
) -> Result<ExpanderMetadata> {
    let parsed = parser::parse_onnx(onnx_path).context("parsing ONNX model")?;
    let mut graph = LayerGraph::from_parsed(&parsed).context("building layer graph")?;
    let max_bound = quantizer::compute_max_bound(&mut graph)?;

    let (proof_config, n_bits) = select_goldilocks_tier(max_bound, target_precision)?;

    tracing::debug!(
        %proof_config, n_bits, max_bound, "goldilocks auto-select"
    );

    generate_from_parsed(
        parsed,
        graph,
        false,
        Some(n_bits),
        target_precision,
        Some(max_bound),
        None,
    )
}

fn build_circuit_params(
    parsed: &ParsedModel,
    quantized: &QuantizedModel,
    config: &ScaleConfig,
    weights_as_inputs: bool,
    proof_config: Option<ProofConfig>,
) -> CircuitParams {
    use crate::proof_config::StampedProofConfig;
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
        proof_config: proof_config.map(StampedProofConfig::current),
        logup_chunk_bits: None,
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
        OpType::Cast => "Cast",
        OpType::Exp => "Exp",
        OpType::Sigmoid => "Sigmoid",
        OpType::Gelu => "Gelu",
        OpType::Softmax => "Softmax",
        OpType::Tile => "Tile",
        OpType::Gather => "Gather",
        OpType::LayerNormalization => "LayerNormalization",
        OpType::Resize => "Resize",
        OpType::GridSample => "GridSample",
        OpType::Transpose => "Transpose",
        OpType::Concat => "Concat",
        OpType::Slice => "Slice",
        OpType::TopK => "TopK",
        OpType::Shape => "Shape",
        OpType::Log => "Log",
        OpType::Expand => "Expand",
        OpType::ReduceMean => "ReduceMean",
        OpType::MatMul => "MatMul",
        OpType::AveragePool => "AveragePool",
        OpType::Pad => "Pad",
        OpType::Split => "Split",
        OpType::Where => "Where",
        OpType::Pow => "Pow",
        OpType::Sqrt => "Sqrt",
        OpType::Tanh => "Tanh",
        OpType::ReduceSum => "ReduceSum",
        OpType::Erf => "Erf",
        OpType::ConvTranspose => "ConvTranspose",
        OpType::LeakyRelu => "LeakyRelu",
        OpType::Identity => "Identity",
        OpType::Neg => "Neg",
        OpType::HardSwish => "HardSwish",
        OpType::GlobalAveragePool => "GlobalAveragePool",
        OpType::InstanceNormalization => "InstanceNormalization",
        OpType::GroupNormalization => "GroupNormalization",
        OpType::Not => "Not",
        OpType::And => "And",
        OpType::Equal => "Equal",
        OpType::Greater => "Greater",
        OpType::Less => "Less",
        OpType::ConstantOfShape => "ConstantOfShape",
        OpType::Sin => "Sin",
        OpType::Cos => "Cos",
        OpType::Range => "Range",
        OpType::ReduceMax => "ReduceMax",
        OpType::ScatterND => "ScatterND",
        OpType::GatherElements => "GatherElements",
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
    let mut attached: std::collections::HashSet<String> = std::collections::HashSet::new();

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
            attached.insert(const_output.clone());
        }
    }

    for input_name in &layer.inputs {
        if attached.contains(input_name) {
            continue;
        }
        if let Some(init) = parsed.initializers.get(input_name) {
            let vals = if !init.float_data.is_empty() {
                Value::Array(init.float_data.iter().map(|&v| Value::from(v)).collect())
            } else if !init.int_data.is_empty() {
                Value::Array(init.int_data.iter().map(|&v| Value::from(v)).collect())
            } else {
                continue;
            };
            entries.push((Value::String(input_name.clone().into()), vals));
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
    fn bn254_adaptive_exponent_matches_safe_bound() {
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
        assert_eq!(
            metadata.circuit_params.proof_config.map(|s| s.config),
            Some(ProofConfig::Bn254Raw)
        );
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
        assert_eq!(
            metadata.circuit_params.proof_config.map(|s| s.config),
            Some(ProofConfig::GoldilocksBasefold)
        );
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
    fn select_tier_base_sufficient_adaptive() {
        use jstprove_onnx::quantizer::{MIN_USEFUL_EXPONENT, N_BITS_GOLDILOCKS};

        // max_safe_exponent(31, 100) = (31 - 7 - 1) / 2 = 11 >= MIN_USEFUL (8)
        let max_bound = 100.0;
        assert!(
            ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS, max_bound) >= MIN_USEFUL_EXPONENT
        );

        let (config, n_bits) = select_goldilocks_tier(max_bound, None).unwrap();
        assert_eq!(config, ProofConfig::GoldilocksBasefold);
        assert_eq!(n_bits, N_BITS_GOLDILOCKS);
    }

    #[test]
    fn select_tier_promotes_to_ext2_adaptive() {
        use jstprove_onnx::quantizer::{
            MIN_USEFUL_EXPONENT, N_BITS_GOLDILOCKS, N_BITS_GOLDILOCKS_EXT2,
        };

        // max_safe_exponent(31, 2^20) = (31 - 20 - 1) / 2 = 5 < MIN_USEFUL (8)
        // max_safe_exponent(63, 2^20) = (63 - 20 - 1) / 2 = 21 >= MIN_USEFUL
        let max_bound = 2.0_f64.powi(20);
        assert!(ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS, max_bound) < MIN_USEFUL_EXPONENT);
        assert!(
            ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS_EXT2, max_bound)
                >= MIN_USEFUL_EXPONENT
        );

        let (config, n_bits) = select_goldilocks_tier(max_bound, None).unwrap();
        assert_eq!(config, ProofConfig::GoldilocksExt2Basefold);
        assert_eq!(n_bits, N_BITS_GOLDILOCKS_EXT2);
    }

    #[test]
    fn select_tier_promotes_to_ext2_with_target_precision() {
        use jstprove_onnx::quantizer::{N_BITS_GOLDILOCKS, N_BITS_GOLDILOCKS_EXT2};

        let max_bound = 100.0;
        let target_digits = 4;
        // exponent_for_digits(4) = ceil(4 * log2(10)) = 14
        // max_safe_exponent(31, 100) = 11 < 14 → base insufficient
        // max_safe_exponent(63, 100) = 27 >= 14 → ext2 sufficient
        let required = ScaleConfig::exponent_for_digits(target_digits);
        assert!(ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS, max_bound) < required);
        assert!(ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS_EXT2, max_bound) >= required);

        let (config, n_bits) = select_goldilocks_tier(max_bound, Some(target_digits)).unwrap();
        assert_eq!(config, ProofConfig::GoldilocksExt2Basefold);
        assert_eq!(n_bits, N_BITS_GOLDILOCKS_EXT2);
    }

    #[test]
    fn select_tier_errors_when_both_insufficient() {
        use jstprove_onnx::quantizer::{
            MIN_USEFUL_EXPONENT, N_BITS_GOLDILOCKS, N_BITS_GOLDILOCKS_EXT2,
        };

        // max_safe_exponent(31, 2^50) = 0, max_safe_exponent(63, 2^50) = 6
        // both below MIN_USEFUL_EXPONENT (8)
        let max_bound = 2.0_f64.powi(50);
        assert!(ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS, max_bound) < MIN_USEFUL_EXPONENT);
        assert!(
            ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS_EXT2, max_bound) < MIN_USEFUL_EXPONENT
        );

        let result = select_goldilocks_tier(max_bound, None);
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("accumulation bound"),
            "expected bound error: {msg}"
        );
    }

    #[test]
    fn select_tier_errors_precision_exceeds_ext2() {
        use jstprove_onnx::quantizer::N_BITS_GOLDILOCKS_EXT2;

        // exponent_for_digits(18) = ceil(18 * log2(10)) = 60
        // max_safe_exponent(63, 100) = 27 < 60 → ext2 insufficient
        let max_bound = 100.0;
        let target_digits = 18;
        let required = ScaleConfig::exponent_for_digits(target_digits);
        assert!(ScaleConfig::max_safe_exponent(N_BITS_GOLDILOCKS_EXT2, max_bound) < required);

        let result = select_goldilocks_tier(max_bound, Some(target_digits));
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("decimal digits"),
            "expected digits error: {msg}"
        );
    }

    #[test]
    fn op_type_string_conversion() {
        // Every OpType variant must round-trip through op_type_to_string.
        let cases: &[(OpType, &str)] = &[
            (OpType::Add, "Add"),
            (OpType::Div, "Div"),
            (OpType::Sub, "Sub"),
            (OpType::Mul, "Mul"),
            (OpType::Gemm, "Gemm"),
            (OpType::Conv, "Conv"),
            (OpType::Relu, "ReLU"),
            (OpType::MaxPool, "MaxPool"),
            (OpType::BatchNormalization, "BatchNormalization"),
            (OpType::Max, "Max"),
            (OpType::Min, "Min"),
            (OpType::Clip, "Clip"),
            (OpType::Reshape, "Reshape"),
            (OpType::Flatten, "Flatten"),
            (OpType::Squeeze, "Squeeze"),
            (OpType::Unsqueeze, "Unsqueeze"),
            (OpType::Constant, "Constant"),
            (OpType::Cast, "Cast"),
            (OpType::Exp, "Exp"),
            (OpType::Sigmoid, "Sigmoid"),
            (OpType::Gelu, "Gelu"),
            (OpType::Softmax, "Softmax"),
            (OpType::Tile, "Tile"),
            (OpType::Gather, "Gather"),
            (OpType::LayerNormalization, "LayerNormalization"),
            (OpType::Resize, "Resize"),
            (OpType::GridSample, "GridSample"),
            (OpType::Transpose, "Transpose"),
            (OpType::Concat, "Concat"),
            (OpType::Slice, "Slice"),
            (OpType::TopK, "TopK"),
            (OpType::Shape, "Shape"),
            (OpType::Log, "Log"),
            (OpType::Expand, "Expand"),
            (OpType::ReduceMean, "ReduceMean"),
            (OpType::MatMul, "MatMul"),
            (OpType::AveragePool, "AveragePool"),
            (OpType::Pad, "Pad"),
            (OpType::Split, "Split"),
            (OpType::Where, "Where"),
            (OpType::Pow, "Pow"),
            (OpType::Sqrt, "Sqrt"),
            (OpType::Tanh, "Tanh"),
            (OpType::ReduceSum, "ReduceSum"),
            (OpType::Erf, "Erf"),
            (OpType::ConvTranspose, "ConvTranspose"),
            (OpType::LeakyRelu, "LeakyRelu"),
            (OpType::Identity, "Identity"),
            (OpType::Neg, "Neg"),
            (OpType::Not, "Not"),
            (OpType::And, "And"),
            (OpType::Equal, "Equal"),
            (OpType::Greater, "Greater"),
            (OpType::Less, "Less"),
            (OpType::ConstantOfShape, "ConstantOfShape"),
            (OpType::Sin, "Sin"),
            (OpType::Cos, "Cos"),
            (OpType::Range, "Range"),
            (OpType::ReduceMax, "ReduceMax"),
            (OpType::ScatterND, "ScatterND"),
            (OpType::GatherElements, "GatherElements"),
        ];

        for &(op, expected_str) in cases {
            // Forward: op → string
            let s = op_type_to_string(op);
            assert_eq!(s, expected_str, "op_type_to_string({op:?}) mismatch");

            // Round-trip: string → op must recover the original variant
            let recovered =
                OpType::from_str(&s).unwrap_or_else(|e| panic!("from_str({s:?}) failed: {e}"));
            assert_eq!(
                recovered, op,
                "OpType::from_str({s:?}) round-trip mismatch for {op:?}"
            );
        }
    }
}
