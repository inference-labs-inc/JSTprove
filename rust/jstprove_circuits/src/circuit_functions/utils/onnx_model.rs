/// Standard library imports
use std::collections::HashMap;

/// External crate imports
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use ndarray::ArrayD;

use crate::circuit_functions::layers::LayerKind;
/// Internal crate imports
use crate::circuit_functions::utils::json_array::value_to_arrayd;
use crate::circuit_functions::utils::json_array::FromJsonNumber;
use crate::circuit_functions::utils::onnx_types::{ONNXIO, ONNXLayer};
use crate::circuit_functions::utils::UtilsError;
use crate::circuit_functions::layers::LayerError;

#[derive(Deserialize, Clone, Debug)]
pub struct Architecture{
    pub inputs: Vec<ONNXIO>,
    pub outputs: Vec<ONNXIO>,
    pub architecture: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct WANDB{
    pub w_and_b: Vec<ONNXLayer>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct CircuitParams{
    pub scale_base: u32,
    pub scaling: u32,
    pub rescale_config: HashMap<String, bool>
}

#[derive(Deserialize, Clone)]
pub struct InputData {
    pub input: Value,
}

#[derive(Deserialize, Clone)]
pub struct OutputData {
    pub output: Value,
}

pub fn get_w_or_b<I: DeserializeOwned + Clone + FromJsonNumber + 'static>(
    w_and_b_map: &HashMap<String, ONNXLayer>,
    weights_input: &String,
) -> Result<ArrayD<I>, UtilsError> {
    let weights_tensor_option = w_and_b_map
        .get(weights_input)
        .ok_or_else(|| UtilsError::MissingTensor { tensor: weights_input.clone() })?
        .tensor
        .clone();

    match weights_tensor_option {
        Some(tensor_json) => {
            let inner_value = match &tensor_json {
                Value::Object(map) if map.contains_key("value") => map.get("value").cloned().ok_or_else(|| {
                        UtilsError::MissingTensor { tensor: weights_input.clone() }
                    })?,
                _ => tensor_json.clone(),
            };
            

            eprintln!(
                "🔍 Attempting to parse tensor for '{}': type = {}",
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
        None => Err(UtilsError::MissingTensor { tensor: weights_input.clone() }),
    }
}

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

pub fn extract_params_and_expected_shape(
    layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    layer:  &crate::circuit_functions::utils::onnx_types::ONNXLayer
) -> Result<(Value, Vec<usize>), LayerError>{
    let kind: LayerKind = layer.try_into()?; 

    let params = layer.params.clone()
        .ok_or_else(|| LayerError::MissingParameter { layer: kind.clone(), param: "params".to_string() })?;
    
    let key = layer.inputs.first()
        .ok_or_else(|| LayerError::MissingInput { layer: kind.clone(), name: layer.name.clone() })?;
    
    let expected_shape = layer_context.shapes_map
        .get(key)
        .ok_or_else(||LayerError::InvalidShape { layer: kind.clone(), msg: format!("Missing shape for input '{}'", key) })?
        .clone();

    Ok((params, expected_shape))
}

pub fn get_param<I:DeserializeOwned>(layer_name: &String, param_name: &str, params: &Value) -> Result<I, UtilsError> {
    let param_value = params.get(param_name)
        .ok_or_else(|| UtilsError::MissingParam { 
            layer: layer_name.to_string(), 
            param: param_name.to_string() 
        })?;

    serde_json::from_value(param_value.clone())
        .map_err(|source| UtilsError::ParseError {
            layer: layer_name.to_string(),
            param: param_name.to_string(),
            source,
        })
}


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
        None => {
            match default {
                Some(d) => Ok(d.clone()),
                None => Err(UtilsError::MissingParam {
                    layer: layer_name.to_string(),
                    param: param_name.to_string(),
                }),
            }
        }
    }
}
