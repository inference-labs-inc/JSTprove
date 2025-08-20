/// Standard library imports
use std::collections::HashMap;

use ndarray::ArrayD;
/// External crate imports
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;

/// Internal crate imports
use crate::circuit_functions::utils::json_array::value_to_arrayd;
use crate::circuit_functions::utils::json_array::FromJsonNumber;
use crate::circuit_functions::utils::onnx_types::{ONNXLayer, ONNXIO};

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
    pub scale_exponent: u32,
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
) -> ArrayD<I> {
    let weights_tensor_option = match w_and_b_map.get(weights_input) {
        Some(tensor) => tensor.tensor.clone(),
        None => panic!(
            "🚨 ModelError - missing weights and biases: {}",
            weights_input
        ),
    };

    match weights_tensor_option {
        Some(tensor_json) => {
            // Unwrap "value" if tensor is an object with that key
            let inner_value = match &tensor_json {
                Value::Object(map) if map.contains_key("value") => {
                    map.get("value").cloned().unwrap()
                }
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
            return value_to_arrayd(inner_value).unwrap();
        }
        None => panic!(
            "🚨 ModelError - missing tensor in expected weights/bias: {}",
            weights_input
        ),
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
) -> (Value, Vec<usize>){
    let params = layer.params.clone()
        .ok_or_else(|| panic!("Missing Params for: {}", layer.name.clone())).unwrap();
    
    let key = layer.inputs.first()
        .ok_or_else(|| panic!("Missing input keys for: {}", layer.name.clone())).unwrap();
    
    let expected_shape = layer_context.shapes_map
        .get(key)
        .ok_or_else(|| panic!("Missing input shape for: {}", layer.name.clone())).unwrap()
        .clone();

    (params, expected_shape)
}

pub fn get_param<I: DeserializeOwned>(layer_name: &String, param_name: &str, params: &Value) -> I {
    match params.get(param_name){

        Some(param) => {
            let x = param.clone();
            serde_json::from_value(x.clone()).expect(&format!(
                "❌ Failed to parse param '{}': got value {}",
                param_name, x
            ))
        }
        None => panic!("ParametersError: {} is missing {}", layer_name, param_name),
    }
}

pub fn get_param_or_default<I: DeserializeOwned + Clone>(
    layer_name: &str,
    param_name: &str,
    params: &Value,
    default: Option<&I>,
) -> I {
    match params.get(param_name) {
        Some(param) => {
            let x = param.clone();
            match serde_json::from_value(x.clone()) {
                Ok(value) => value,
                Err(_) => {
                    eprintln!(
                        "⚠️ Warning: Failed to parse param '{}': got value {} — using default",
                        param_name, x
                    );
                    default.unwrap().clone()
                }
            }
        }
        None => {
            eprintln!(
                "⚠️ Warning: ParametersError: '{}' is missing '{}' — using default",
                layer_name, param_name
            );
            default.unwrap().clone()
        }
    }
}
