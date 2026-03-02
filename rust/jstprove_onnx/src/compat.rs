use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use prost::Message;

use super::tensor_proto::DataType;
use super::{ModelProto, TensorProto};

const SUPPORTED_OPS: &[&str] = &[
    "Add",
    "Clip",
    "BatchNormalization",
    "Div",
    "Sub",
    "Mul",
    "Constant",
    "Flatten",
    "Gemm",
    "MaxPool",
    "Max",
    "Min",
    "Relu",
    "Reshape",
    "Conv",
    "Squeeze",
    "Unsqueeze",
];

pub fn is_compatible(path: &Path) -> Result<(bool, Vec<String>)> {
    if !path.exists() {
        return Ok((false, vec!["FILE_NOT_FOUND".to_string()]));
    }

    let bytes = std::fs::read(path).context("reading ONNX file")?;
    let model = match ModelProto::decode(bytes.as_slice()) {
        Ok(m) => m,
        Err(_) => return Ok((false, vec!["LOAD_ERROR".to_string()])),
    };

    let graph = match model.graph {
        Some(g) => g,
        None => return Ok((false, vec!["LOAD_ERROR".to_string()])),
    };

    let supported: HashSet<&str> = SUPPORTED_OPS.iter().copied().collect();
    let mut unsupported = Vec::new();

    for node in &graph.node {
        let op = node.op_type.as_deref().unwrap_or("");
        if !op.is_empty() && !supported.contains(op) {
            if !unsupported.contains(&op.to_string()) {
                unsupported.push(op.to_string());
            }
        }
    }

    if unsupported.is_empty() {
        Ok((true, vec![]))
    } else {
        Ok((false, unsupported))
    }
}

pub fn add_zero_bias_to_conv(path: &Path, output_path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).context("reading ONNX file")?;
    let mut model = ModelProto::decode(bytes.as_slice()).context("decoding ONNX protobuf")?;

    let graph = model.graph.as_mut().context("model has no graph")?;

    let mut new_initializers = Vec::new();

    for node in &mut graph.node {
        let op = node.op_type.as_deref().unwrap_or("");
        if op != "Conv" || node.input.len() != 2 {
            continue;
        }

        let weight_name = &node.input[1];
        let weight_init = graph
            .initializer
            .iter()
            .find(|i| i.name.as_deref() == Some(weight_name));

        let out_channels = match weight_init {
            Some(init) => {
                if init.dims.is_empty() {
                    continue;
                }
                init.dims[0] as usize
            }
            None => continue,
        };

        let node_id = node
            .name
            .clone()
            .or_else(|| node.output.first().cloned())
            .unwrap_or_else(|| weight_name.clone());
        let bias_name = format!("{node_id}_zero_bias");

        let data_type = weight_init.and_then(|i| i.data_type).unwrap_or(1);
        let raw_data = match DataType::try_from(data_type) {
            Ok(DataType::Double) => vec![0u8; out_channels * 8],
            _ => vec![0u8; out_channels * 4],
        };

        let bias_tensor = TensorProto {
            name: Some(bias_name.clone()),
            dims: vec![out_channels as i64],
            data_type: Some(data_type),
            raw_data: Some(raw_data),
            ..Default::default()
        };

        new_initializers.push(bias_tensor);
        node.input.push(bias_name);
    }

    graph.initializer.extend(new_initializers);

    let out_bytes = model.encode_to_vec();
    std::fs::write(output_path, out_bytes).context("writing modified ONNX file")?;
    Ok(())
}
