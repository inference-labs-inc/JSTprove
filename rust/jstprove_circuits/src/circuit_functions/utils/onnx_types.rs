use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize, Clone, Debug)]
pub struct ONNXIO {
    pub name: String,
    pub elem_type: i16,
    pub shape: Vec<usize>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ONNXLayer {
    pub id: usize,
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub shape: HashMap<String, Vec<usize>>,
    pub tensor: Option<Value>,
    pub params: Option<Value>,
    pub opset_version_number: i16,
}
