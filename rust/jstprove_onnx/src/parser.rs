use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use prost::Message;

use super::attribute_proto::AttributeType;
use super::tensor_proto::DataType;
use super::{AttributeProto, GraphProto, ModelProto, TensorProto};

pub struct ParsedModel {
    pub nodes: Vec<ParsedNode>,
    pub initializers: HashMap<String, TensorData>,
    pub inputs: Vec<IoSpec>,
    pub outputs: Vec<IoSpec>,
    pub opset_version: i64,
}

#[derive(Debug, Clone)]
pub struct ParsedNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, AttrValue>,
    pub domain: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AttrValue {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Tensor(TensorData),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorData {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub float_data: Vec<f64>,
    pub int_data: Vec<i64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IoSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub elem_type: i32,
}

pub fn parse_onnx(path: &Path) -> Result<ParsedModel> {
    let bytes = std::fs::read(path).context("reading ONNX file")?;
    let model = ModelProto::decode(bytes.as_slice()).context("decoding ONNX protobuf")?;

    let opset_version = model
        .opset_import
        .iter()
        .filter(|o| o.domain.as_deref().unwrap_or("").is_empty())
        .map(|o| o.version.unwrap_or(0))
        .max()
        .unwrap_or(0);

    let graph = model.graph.context("model has no graph")?;

    let initializers = parse_initializers(&graph)?;
    let nodes = parse_nodes(&graph)?;
    let inputs = parse_value_infos(&graph.input);
    let outputs = parse_value_infos(&graph.output);

    Ok(ParsedModel {
        nodes,
        initializers,
        inputs,
        outputs,
        opset_version,
    })
}

fn parse_initializers(graph: &GraphProto) -> Result<HashMap<String, TensorData>> {
    let mut map = HashMap::new();
    for init in &graph.initializer {
        let name = init.name.clone().unwrap_or_default();
        let td = extract_tensor_data(init)?;
        map.insert(name, td);
    }
    Ok(map)
}

fn extract_tensor_data(tensor: &TensorProto) -> Result<TensorData> {
    let name = tensor.name.clone().unwrap_or_default();
    let dims = tensor.dims.clone();
    let data_type = tensor.data_type.unwrap_or(0);
    let dt = DataType::try_from(data_type).unwrap_or(DataType::Undefined);

    let mut float_data = Vec::new();
    let mut int_data = Vec::new();

    if let Some(ref raw) = tensor.raw_data {
        match dt {
            DataType::Float => {
                for chunk in raw.chunks_exact(4) {
                    float_data.push(f32::from_le_bytes(chunk.try_into().unwrap()) as f64);
                }
            }
            DataType::Double => {
                for chunk in raw.chunks_exact(8) {
                    float_data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            DataType::Int32 => {
                for chunk in raw.chunks_exact(4) {
                    int_data.push(i32::from_le_bytes(chunk.try_into().unwrap()) as i64);
                }
            }
            DataType::Int64 => {
                for chunk in raw.chunks_exact(8) {
                    int_data.push(i64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            other => bail!("unsupported tensor data type in raw_data: {other:?}"),
        }
    } else {
        float_data.extend(tensor.float_data.iter().map(|f| *f as f64));
        float_data.extend(tensor.double_data.iter().copied());
        int_data.extend(tensor.int32_data.iter().map(|i| *i as i64));
        int_data.extend(tensor.int64_data.iter().copied());
    }

    Ok(TensorData {
        name,
        dims,
        data_type,
        float_data,
        int_data,
    })
}

fn parse_nodes(graph: &GraphProto) -> Result<Vec<ParsedNode>> {
    let mut result = Vec::new();
    for (idx, node) in graph.node.iter().enumerate() {
        let name = node.name.clone().unwrap_or_else(|| format!("node_{idx}"));
        let op_type = node.op_type.clone().unwrap_or_default();
        let domain = node.domain.clone().unwrap_or_default();

        let mut attributes = HashMap::new();
        for attr in &node.attribute {
            if let Some(ref attr_name) = attr.name {
                if let Some(val) = parse_attribute(attr)? {
                    attributes.insert(attr_name.clone(), val);
                }
            }
        }

        result.push(ParsedNode {
            name,
            op_type,
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            attributes,
            domain,
        });
    }
    Ok(result)
}

fn parse_attribute(attr: &AttributeProto) -> Result<Option<AttrValue>> {
    let attr_type = attr.r#type.and_then(|t| AttributeType::try_from(t).ok());

    match attr_type {
        Some(AttributeType::Float) => Ok(attr.f.map(AttrValue::Float)),
        Some(AttributeType::Int) => Ok(attr.i.map(AttrValue::Int)),
        Some(AttributeType::String) => Ok(attr
            .s
            .as_ref()
            .map(|s| AttrValue::String(String::from_utf8_lossy(s).to_string()))),
        Some(AttributeType::Floats) => Ok(Some(AttrValue::Floats(attr.floats.clone()))),
        Some(AttributeType::Ints) => Ok(Some(AttrValue::Ints(attr.ints.clone()))),
        Some(AttributeType::Tensor) => {
            if let Some(ref t) = attr.t {
                Ok(Some(AttrValue::Tensor(extract_tensor_data(t)?)))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

fn parse_value_infos(infos: &[super::ValueInfoProto]) -> Vec<IoSpec> {
    infos
        .iter()
        .filter_map(|vi| {
            let name = vi.name.clone().unwrap_or_default();
            let (shape, elem_type) = if let Some(ref tp) = vi.r#type {
                if let Some(ref value) = tp.value {
                    match value {
                        super::type_proto::Value::TensorType(tt) => {
                            let et = tt.elem_type.unwrap_or(0);
                            let shape = tt
                                .shape
                                .as_ref()
                                .map(|s| {
                                    s.dim
                                        .iter()
                                        .map(|d| {
                                            d.value
                                                .as_ref()
                                                .map(|v| match v {
                                                    super::tensor_shape_proto::dimension::Value::DimValue(dv) => *dv,
                                                    super::tensor_shape_proto::dimension::Value::DimParam(_) => -1,
                                                })
                                                .unwrap_or(-1)
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            (shape, et)
                        }
                        _ => (vec![], 0),
                    }
                } else {
                    (vec![], 0)
                }
            } else {
                (vec![], 0)
            };
            Some(IoSpec {
                name,
                shape,
                elem_type,
            })
        })
        .collect()
}

impl TensorData {
    pub fn total_elements(&self) -> usize {
        if self.dims.is_empty() {
            return 1;
        }
        self.dims.iter().map(|d| *d as usize).product()
    }

    pub fn as_f64_vec(&self) -> Vec<f64> {
        if !self.float_data.is_empty() {
            self.float_data.clone()
        } else {
            self.int_data.iter().map(|i| *i as f64).collect()
        }
    }

    pub fn as_i64_vec(&self) -> Vec<i64> {
        if !self.int_data.is_empty() {
            self.int_data.clone()
        } else {
            self.float_data.iter().map(|f| *f as i64).collect()
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().map(|d| *d as usize).collect()
    }
}
