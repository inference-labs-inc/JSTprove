/// Standard library imports
use std::collections::HashMap;

use ndarray::ArrayD;
use serde::Deserialize;
/// External crate imports
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::circuit_functions::layers::LayerError;
use crate::circuit_functions::layers::LayerKind;
use crate::circuit_functions::utils::UtilsError;
use crate::circuit_functions::utils::build_layers::BuildLayerContext;
use crate::circuit_functions::utils::constants::VALUE;
use crate::circuit_functions::utils::json_array::FromJsonNumber;
/// Internal crate imports
use crate::circuit_functions::utils::json_array::value_to_arrayd;
use crate::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};

#[derive(Deserialize, Clone, Debug)]
pub struct Architecture {
    pub architecture: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct WANDB {
    pub w_and_b: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CircuitParams {
    pub scale_base: u32,
    pub scale_exponent: u32,
    pub rescale_config: HashMap<String, bool>,
    pub inputs: Vec<ONNXIO>,
    pub outputs: Vec<ONNXIO>,
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
    I: DeserializeOwned + Clone + FromJsonNumber + 'static,
    S: ::std::hash::BuildHasher,
>(
    w_and_b_map: &HashMap<String, ONNXLayer, S>,
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
        Some(tensor_json) => {
            let inner_value = match &tensor_json {
                Value::Object(map) if map.contains_key(VALUE) => map
                    .get(VALUE)
                    .cloned()
                    .ok_or_else(|| UtilsError::MissingTensor {
                        tensor: weights_input.clone(),
                    })?,
                _ => tensor_json.clone(),
            };

            eprintln!(
                "Attempting to parse tensor for '{}': type = {}",
                weights_input,
                match &inner_value {
                    Value::Array(_) => "Array",
                    Value::Object(_) => "Object",
                    Value::Number(_) => "Number",
                    Value::String(_) => "String",
                    _ => "Other",
                }
            );
            value_to_arrayd(inner_value).map_err(UtilsError::ArrayConversionError)
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
    match get_w_or_b(&layer_context.w_and_b_map, input) {
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
    let param_value = params
        .get(param_name)
        .ok_or_else(|| UtilsError::MissingParam {
            layer: layer_name.to_string(),
            param: param_name.to_string(),
        })?;

    serde_json::from_value(param_value.clone()).map_err(|source| UtilsError::ParseError {
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
    match params.get(param_name) {
        Some(param_value) => {
            serde_json::from_value(param_value.clone()).map_err(|source| UtilsError::ParseError {
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
