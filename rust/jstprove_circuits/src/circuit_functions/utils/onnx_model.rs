use std::collections::HashMap;

use ndarray::ArrayD;
use rmpv::Value;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::circuit_functions::layers::LayerError;
use crate::circuit_functions::layers::LayerKind;
use crate::circuit_functions::utils::UtilsError;
use crate::circuit_functions::utils::build_layers::BuildLayerContext;
use crate::circuit_functions::utils::constants::VALUE;
use crate::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};
use crate::circuit_functions::utils::value_array::FromMsgpackValue;
use crate::circuit_functions::utils::value_array::{map_get, value_to_arrayd};
use crate::proof_config::StampedProofConfig;
use crate::proof_system::ProofSystem;

#[derive(Deserialize, Clone, Debug)]
pub struct Architecture {
    pub architecture: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct WANDB {
    pub w_and_b: Vec<ONNXLayer>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct CircuitParams {
    pub scale_base: u32,
    pub scale_exponent: u32,
    pub rescale_config: HashMap<String, bool>,
    pub inputs: Vec<ONNXIO>,
    pub outputs: Vec<ONNXIO>,
    #[serde(default = "default_freivalds_reps")]
    pub freivalds_reps: usize,
    #[serde(default)]
    pub n_bits_config: HashMap<String, usize>,
    #[serde(default)]
    pub weights_as_inputs: bool,
    #[serde(default, rename = "backend")]
    pub proof_system: ProofSystem,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_config: Option<StampedProofConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logup_chunk_bits: Option<usize>,
    /// Names of ONNX inputs that the model author has explicitly
    /// declared as **public** inputs of the holographic-GKR proof.
    /// Public inputs travel in the clear alongside the proof and
    /// are bound by the verifier directly. Inputs not listed here
    /// fall into one of two categories:
    ///
    /// * **WAI weights** (any name listed in [`WANDB`]) — committed
    ///   in the verifying key as part of the wiring polynomial,
    ///   never re-shipped per inference.
    /// * **Private activations** — committed in the per-inference
    ///   proof via the per-eval extension-field commitment, with
    ///   the validator cross-referencing the cleartext or hash they
    ///   expected.
    ///
    /// Empty by default for backwards compatibility with existing
    /// circuit bundles.
    #[serde(default)]
    pub public_inputs: Vec<String>,
}

impl CircuitParams {
    #[must_use]
    pub fn total_input_dims(&self) -> usize {
        self.inputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .sum()
    }

    #[must_use]
    pub fn total_output_dims(&self) -> usize {
        self.outputs
            .iter()
            .map(|obj| obj.shape.iter().product::<usize>())
            .sum()
    }

    #[must_use]
    pub fn effective_output_dims(&self) -> usize {
        let dims = self.total_output_dims();
        if dims == 0 && !self.outputs.is_empty() {
            self.outputs.len()
        } else {
            dims
        }
    }

    #[must_use]
    pub fn effective_input_dims(&self) -> usize {
        let dims = self.total_input_dims();
        if dims == 0 && !self.inputs.is_empty() {
            self.inputs.len()
        } else {
            dims
        }
    }

    pub fn disable_autotune(&mut self) {
        self.logup_chunk_bits = Some(crate::circuit_functions::gadgets::DEFAULT_LOGUP_CHUNK_BITS);
    }

    /// Partition the layer-0 input names into the three holographic-
    /// GKR semantic regions:
    ///
    /// * `weights`  — names listed in `wandb.w_and_b` (WAI weights
    ///   that go into the verifying key's wiring commitment)
    /// * `public`   — names listed in `self.public_inputs`
    ///   (cleartext per-inference public input)
    /// * `private`  — every other declared input (per-inference
    ///   private input committed in the proof's eval-time
    ///   commitment)
    ///
    /// Returns `(weights, public, private)` as three `Vec<String>`
    /// in the same name order they appear in `self.inputs` so the
    /// caller can preserve the layer-0 slot ordering.
    ///
    /// An input listed in both `wandb.w_and_b` and
    /// `self.public_inputs` is reported as a weight; weights take
    /// precedence because they are baked into the VK and shipping
    /// them as public inputs would defeat the purpose. The caller
    /// can detect this conflict by intersecting `wandb.w_and_b`
    /// names with `self.public_inputs` *before* calling this method,
    /// since the returned partitions are mutually exclusive
    /// (precedence removes overlaps from the result).
    /// # Errors
    /// Returns an error if any name in `self.public_inputs` does not
    /// appear in `self.inputs`.
    pub fn partition_input_names(&self, wandb: &WANDB) -> Result<InputNamePartition, String> {
        let input_name_set: std::collections::HashSet<&str> =
            self.inputs.iter().map(|io| io.name.as_str()).collect();
        for name in &self.public_inputs {
            if !input_name_set.contains(name.as_str()) {
                return Err(format!(
                    "public_inputs contains unknown input name: {name:?}"
                ));
            }
        }
        let weight_set: std::collections::HashSet<&str> =
            wandb.w_and_b.iter().map(|w| w.name.as_str()).collect();
        let public_set: std::collections::HashSet<&str> =
            self.public_inputs.iter().map(String::as_str).collect();
        let mut weights = Vec::new();
        let mut public = Vec::new();
        let mut private = Vec::new();
        for io in &self.inputs {
            let name = io.name.as_str();
            if weight_set.contains(name) {
                weights.push(io.name.clone());
            } else if public_set.contains(name) {
                public.push(io.name.clone());
            } else {
                private.push(io.name.clone());
            }
        }
        Ok(InputNamePartition {
            weights,
            public,
            private,
        })
    }
}

/// Result of [`CircuitParams::partition_input_names`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputNamePartition {
    pub weights: Vec<String>,
    pub public: Vec<String>,
    pub private: Vec<String>,
}

fn default_freivalds_reps() -> usize {
    1
}

#[derive(Deserialize, Clone)]
pub struct InputData {
    pub input: Value,
}

#[derive(Deserialize, Clone)]
pub struct OutputData {
    pub output: Value,
}

/// Fetches a weight or bias tensor from a map of ONNX layers and converts it to an `ArrayD<I>`.
///
/// # Arguments
/// - `w_and_b_map`: `HashMap` of layer names to ONNX layers.
/// - `weights_input`: Name of the tensor to fetch.
///
/// # Returns
/// A multi-dimensional array (`ArrayD<I>`) of the tensor values.
///
/// # Errors
/// - [`UtilsError::MissingTensor`] if the tensor is not found or missing its `value` field.
/// - [`UtilsError::ArrayConversionError`] if the tensor JSON cannot be converted to an `ArrayD<I>`.
pub fn get_w_or_b<
    I: DeserializeOwned + Clone + FromMsgpackValue + 'static,
    S: ::std::hash::BuildHasher,
>(
    w_and_b_map: &HashMap<String, &ONNXLayer, S>,
    weights_input: &String,
) -> Result<ArrayD<I>, UtilsError> {
    let weights_tensor_option = w_and_b_map
        .get(weights_input)
        .ok_or_else(|| UtilsError::MissingTensor {
            tensor: weights_input.clone(),
        })?
        .tensor
        .clone();

    match weights_tensor_option {
        Some(tensor_val) => {
            let inner_value = match &tensor_val {
                Value::Map(entries) if map_get(entries, VALUE).is_some() => map_get(entries, VALUE)
                    .cloned()
                    .ok_or_else(|| UtilsError::MissingTensor {
                        tensor: weights_input.clone(),
                    })?,
                _ => tensor_val.clone(),
            };

            value_to_arrayd(&inner_value).map_err(UtilsError::ArrayConversionError)
        }
        None => Err(UtilsError::MissingTensor {
            tensor: weights_input.clone(),
        }),
    }
}

/// Attempts to retrieve an optional weight or bias tensor from the layer context.
///
/// # Arguments
///
/// * `layer_context` - The context containing weight and bias initializers.
/// * `input` - The name of the weight or bias tensor to retrieve.
///
/// # Returns
///
/// * `Ok(Some(ArrayD<i64>))` if the tensor exists and is successfully retrieved.
/// * `Ok(None)` if the tensor is genuinely missing (optional).
///
/// # Errors
///
/// Returns `Err(UtilsError)` if the initializer is not missing, but some other error occurs in obtaining it.
/// e.g., invalid tensor shape, type mismatch, or other parsing/conversion errors.
pub fn get_optional_w_or_b(
    layer_context: &BuildLayerContext,
    input: &std::string::String,
) -> Result<Option<ArrayD<i64>>, UtilsError> {
    match get_w_or_b(layer_context.w_and_b_map, input) {
        Ok(arr) => Ok(Some(arr.into_dyn())),
        Err(UtilsError::MissingTensor { .. }) => Ok(None), // initializer genuinely missing
        Err(e) => Err(e),                                  // propogates other error
    }
}

#[must_use]
pub fn collect_all_shapes(layers: &[ONNXLayer], ios: &[ONNXIO]) -> HashMap<String, Vec<usize>> {
    let mut result = HashMap::new();

    // Merge from layers
    for layer in layers {
        for (key, shape) in &layer.shape {
            result.insert(key.clone(), shape.clone());
        }
    }

    // Merge from IOs
    for io in ios {
        result.insert(io.name.clone(), io.shape.clone());
    }

    result
}

#[must_use]
pub fn estimate_rescale_elements(params: &CircuitParams, architecture: &Architecture) -> usize {
    let shapes = collect_all_shapes(&architecture.architecture, &params.inputs);
    let mut total = 0usize;
    for layer in &architecture.architecture {
        let is_rescale = params
            .rescale_config
            .get(&layer.name)
            .copied()
            .unwrap_or(true);
        if !is_rescale {
            continue;
        }
        for output_name in &layer.outputs {
            if let Some(shape) = shapes.get(output_name) {
                let elems: usize = shape.iter().product();
                total = total.saturating_add(elems);
            }
        }
    }
    total
}

/// Extracts parameters and the expected input shape for an ONNX layer.
///
/// # Arguments
/// - `layer_context`: Context containing input shapes.
/// - `layer`: ONNX layer to extract information from.
///
/// # Returns
/// A tuple `(params, expected_shape)` where:
/// - `params` is the JSON object of layer parameters.
/// - `expected_shape` is a vector of input dimensions.
///
/// # Errors
/// - [`LayerError::MissingParameter`] if `params` is missing in the layer.
/// - [`LayerError::MissingInput`] if the layer has no input keys.
/// - [`LayerError::InvalidShape`] if the shape for the first input key is missing in `layer_context`.
pub fn extract_params_and_expected_shape(
    layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
) -> Result<(Value, Vec<usize>), LayerError> {
    let kind: LayerKind = layer.try_into()?;

    let params = layer
        .params
        .clone()
        .ok_or_else(|| LayerError::MissingParameter {
            layer: kind.clone(),
            param: "params".to_string(),
        })?;

    let key = layer
        .inputs
        .first()
        .ok_or_else(|| LayerError::MissingInput {
            layer: kind.clone(),
            name: layer.name.clone(),
        })?;

    let expected_shape = layer_context
        .shapes_map
        .get(key)
        .ok_or_else(|| LayerError::InvalidShape {
            layer: kind.clone(),
            msg: format!("Missing shape for input '{key}'"),
        })?
        .clone();

    Ok((params, expected_shape))
}

/// Extracts only the deserialized parameters from an ONNX layer, without
/// looking up the expected input shape from the shapes map.
///
/// # Errors
/// Returns `LayerError::MissingParameter` when the layer carries no params.
pub fn extract_params(
    layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
) -> Result<Value, LayerError> {
    let kind: LayerKind = layer.try_into()?;
    layer
        .params
        .clone()
        .ok_or_else(|| LayerError::MissingParameter {
            layer: kind,
            param: "params".to_string(),
        })
}

/// Retrieves a required parameter from a JSON `Value`.
///
/// # Arguments
/// - `layer_name`: Name of the layer for error context.
/// - `param_name`: Name of the parameter to fetch.
/// - `params`: JSON object containing layer parameters.
///
/// # Returns
/// The deserialized parameter of type `I`.
///
/// # Errors
/// - `UtilsError::MissingParam` if the parameter is absent.
/// - `UtilsError::ParseError` if the parameter exists but cannot be deserialized into `I`.
pub fn get_param<I: DeserializeOwned>(
    layer_name: &String,
    param_name: &str,
    params: &Value,
) -> Result<I, UtilsError> {
    let param_value = match params {
        Value::Map(entries) => map_get(entries, param_name),
        _ => None,
    }
    .ok_or_else(|| UtilsError::MissingParam {
        layer: layer_name.to_string(),
        param: param_name.to_string(),
    })?;

    rmpv::ext::from_value(param_value.clone()).map_err(|source| UtilsError::ParseError {
        layer: layer_name.to_string(),
        param: param_name.to_string(),
        source,
    })
}

/// Retrieves the input name at a given index from a list of input names.
///
/// # Arguments
/// - `inputs`: Slice of input names.
/// - `index`: Index to fetch.
/// - `layer`: Layer kind for error reporting.
/// - `param`: Parameter name for context in error messages.
///
/// # Returns
/// Reference to the input name string at the given index.
///
/// # Errors
/// - [`LayerError::MissingParameter`] if the index is out of bounds.
pub fn get_input_name<'a>(
    inputs: &'a [String],
    index: usize,
    layer: LayerKind,
    param: &str,
) -> Result<&'a String, LayerError> {
    inputs
        .get(index)
        .ok_or_else(|| LayerError::MissingParameter {
            layer,
            param: format!("{param} (index {index})"),
        })
}

#[must_use]
pub fn get_optional_input_name(inputs: &[String], index: usize) -> Option<&String> {
    inputs.get(index).filter(|s| !s.is_empty())
}

/// Retrieves a parameter from a JSON `Value`, returning a provided default if missing.
///
/// # Arguments
/// - `layer_name`: Name of the layer for error context.
/// - `param_name`: Name of the parameter to retrieve.
/// - `params`: JSON object containing layer parameters.
/// - `default`: Optional default value to use if parameter is missing.
///
/// # Returns
/// The deserialized parameter of type `I`.
///
/// # Errors
/// - [`UtilsError::MissingParam`] if the parameter is absent and no default is provided.
/// - [`UtilsError::ParseError`] if the parameter exists but cannot be deserialized into `I`.
pub fn get_param_or_default<I: DeserializeOwned + Clone>(
    layer_name: &str,
    param_name: &str,
    params: &Value,
    default: Option<&I>,
) -> Result<I, UtilsError> {
    let found = match params {
        Value::Map(entries) => map_get(entries, param_name).cloned(),
        _ => None,
    };
    match found {
        Some(param_value) => {
            rmpv::ext::from_value(param_value).map_err(|source| UtilsError::ParseError {
                layer: layer_name.to_string(),
                param: param_name.to_string(),
                source,
            })
        }
        None => match default {
            Some(d) => Ok(d.clone()),
            None => Err(UtilsError::MissingParam {
                layer: layer_name.to_string(),
                param: param_name.to_string(),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_params_weights_as_inputs_true() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [{"name": "input", "elem_type": 1, "shape": [1, 3, 224, 224]}],
            "outputs": [{"name": "output", "elem_type": 1, "shape": [1, 10]}],
            "weights_as_inputs": true
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert!(params.weights_as_inputs);
        assert_eq!(params.freivalds_reps, 1);
    }

    #[test]
    fn circuit_params_weights_as_inputs_defaults_false() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [{"name": "input", "elem_type": 1, "shape": [1, 3, 224, 224]}],
            "outputs": [{"name": "output", "elem_type": 1, "shape": [1, 10]}]
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert!(!params.weights_as_inputs);
    }

    #[test]
    fn circuit_params_weights_as_inputs_false_explicit() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": [],
            "weights_as_inputs": false
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert!(!params.weights_as_inputs);
    }

    fn make_io(name: &str) -> ONNXIO {
        ONNXIO {
            name: name.to_string(),
            elem_type: 1,
            shape: vec![1],
        }
    }

    fn make_params(input_names: &[&str], public_inputs: &[&str]) -> CircuitParams {
        CircuitParams {
            scale_base: 2,
            scale_exponent: 18,
            rescale_config: HashMap::new(),
            inputs: input_names.iter().map(|n| make_io(n)).collect(),
            outputs: vec![],
            freivalds_reps: 1,
            n_bits_config: HashMap::new(),
            weights_as_inputs: true,
            proof_system: ProofSystem::default(),
            proof_config: None,
            logup_chunk_bits: None,
            public_inputs: public_inputs.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn make_wandb(weight_names: &[&str]) -> WANDB {
        WANDB {
            w_and_b: weight_names
                .iter()
                .map(|n| ONNXLayer {
                    id: 0,
                    name: n.to_string(),
                    op_type: String::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    shape: HashMap::new(),
                    tensor: None,
                    params: None,
                    opset_version_number: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn public_inputs_default_empty() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": []
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert!(params.public_inputs.is_empty());
    }

    #[test]
    fn public_inputs_round_trip_through_serde() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [
                {"name": "x", "elem_type": 1, "shape": [1]},
                {"name": "y", "elem_type": 1, "shape": [1]}
            ],
            "outputs": [],
            "public_inputs": ["x"]
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.public_inputs, vec!["x".to_string()]);
        let round_tripped = serde_json::to_string(&params).unwrap();
        let again: CircuitParams = serde_json::from_str(&round_tripped).unwrap();
        assert_eq!(again.public_inputs, vec!["x".to_string()]);
    }

    #[test]
    fn partition_input_names_three_categories() {
        // weights:  "w1", "w2"
        // public:   "pub_input"
        // private:  "act"
        let params = make_params(&["w1", "act", "pub_input", "w2"], &["pub_input"]);
        let wandb = make_wandb(&["w1", "w2"]);
        let p = params.partition_input_names(&wandb).unwrap();
        assert_eq!(p.weights, vec!["w1".to_string(), "w2".to_string()]);
        assert_eq!(p.public, vec!["pub_input".to_string()]);
        assert_eq!(p.private, vec!["act".to_string()]);
    }

    #[test]
    fn partition_input_names_weight_takes_precedence_over_public() {
        // "x" is declared both as a WAI weight and as a public
        // input. Weights take precedence (weights are baked into
        // the VK and re-shipping them as public would defeat the
        // purpose).
        let params = make_params(&["x", "y"], &["x", "y"]);
        let wandb = make_wandb(&["x"]);
        let p = params.partition_input_names(&wandb).unwrap();
        assert_eq!(p.weights, vec!["x".to_string()]);
        assert_eq!(p.public, vec!["y".to_string()]);
        assert!(p.private.is_empty());
    }

    #[test]
    fn partition_input_names_with_no_public_decl_routes_unknown_to_private() {
        // Default behavior (empty public_inputs): every non-weight
        // input becomes private. Backwards compatible with existing
        // bundles.
        let params = make_params(&["w", "a", "b"], &[]);
        let wandb = make_wandb(&["w"]);
        let p = params.partition_input_names(&wandb).unwrap();
        assert_eq!(p.weights, vec!["w".to_string()]);
        assert!(p.public.is_empty());
        assert_eq!(p.private, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn total_input_dims_single_input() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [{"name": "x", "elem_type": 1, "shape": [1, 3, 224, 224]}],
            "outputs": []
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.total_input_dims(), 3 * 224 * 224);
    }

    #[test]
    fn total_input_dims_multiple_inputs() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [
                {"name": "a", "elem_type": 1, "shape": [2, 3]},
                {"name": "b", "elem_type": 1, "shape": [4, 5]}
            ],
            "outputs": []
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.total_input_dims(), 6 + 20);
    }

    #[test]
    fn total_output_dims_single_output() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": [{"name": "out", "elem_type": 1, "shape": [1, 10]}]
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.total_output_dims(), 10);
    }

    #[test]
    fn effective_input_dims_zero_shape_fallback() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [{"name": "in", "elem_type": 1, "shape": [0, 10]}],
            "outputs": []
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.total_input_dims(), 0);
        assert_eq!(params.effective_input_dims(), 1);
    }

    #[test]
    fn effective_output_dims_zero_shape_fallback() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": [{"name": "out", "elem_type": 1, "shape": [0, 10]}]
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.total_output_dims(), 0);
        assert_eq!(params.effective_output_dims(), 1);
    }

    #[test]
    fn backend_defaults_to_expander() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": []
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.proof_system, ProofSystem::Expander);
    }

    #[test]
    fn backend_remainder_round_trip() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": [],
            "backend": "remainder"
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.proof_system, ProofSystem::Remainder);
        assert!(params.proof_system.is_remainder());
        let bytes = rmp_serde::to_vec_named(&params).unwrap();
        let round_tripped: CircuitParams = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(round_tripped.proof_system, ProofSystem::Remainder);
    }

    #[test]
    fn backend_expander_explicit() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": [],
            "backend": "expander"
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.proof_system, ProofSystem::Expander);
        assert!(!params.proof_system.is_remainder());
    }

    #[test]
    fn backend_unknown_variant_rejected() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": [],
            "backend": "gkr"
        }"#;
        assert!(serde_json::from_str::<CircuitParams>(json).is_err());
    }

    #[test]
    fn total_dims_empty() {
        let json = r#"{
            "scale_base": 2,
            "scale_exponent": 18,
            "rescale_config": {},
            "inputs": [],
            "outputs": []
        }"#;
        let params: CircuitParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.total_input_dims(), 0);
        assert_eq!(params.total_output_dims(), 0);
    }
}
