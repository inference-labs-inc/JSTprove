use prost::Message;

use jstprove_onnx::{
    tensor_shape_proto::{self, dimension},
    type_proto, AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};

/// An attribute value for a single-op ONNX node.
#[derive(Debug, Clone)]
pub enum AttrValue {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    String(Vec<u8>),
    /// A tensor value (for `Constant` node's `value` attribute, `ConstantOfShape`'s `value`, etc.)
    /// All data stored as INT64.
    Tensor {
        dims: Vec<i64>,
        data_type: i32,
        int64_data: Vec<i64>,
    },
}

/// A named node attribute.
#[derive(Debug, Clone)]
pub struct NodeAttr {
    pub name: &'static str,
    pub value: AttrValue,
}

/// A constant tensor baked into the ONNX model as a graph initializer.
/// Used for weights, biases, indices, and other compile-time constants.
#[derive(Debug, Clone)]
pub struct Initializer {
    pub name: &'static str,
    pub dims: Vec<i64>,
    pub data: Vec<i64>,
}

/// Build a minimal single-operator ONNX model and return it as serialized protobuf bytes.
///
/// # Arguments
/// * `op_type`      – ONNX op name, e.g. `"Relu"`, `"Add"`, `"Gemm"`
/// * `input_shapes` – `(name, dims, elem_type)` for each graph input.
///                    `elem_type` follows ONNX TensorProto.DataType: 1=FLOAT, 7=INT64.
///                    Initializer-only inputs (weights) do not need entries here.
/// * `output_name`  – name of the single output tensor
/// * `attrs`        – ONNX attributes for the node
/// * `initializers` – constant tensors baked into the graph (INT64 data)
/// * `output_shapes` – `(name, dims, elem_type)` for each graph output.
///                     **Must be specified** — JSTProve reads output shapes from the ONNX model
///                     and cannot infer them at circuit-build time without this information.
pub fn build_single_op_model(
    op_type: &str,
    input_shapes: &[(&str, &[i64], i32)],
    output_shapes: &[(&str, &[i64], i32)],
    attrs: &[NodeAttr],
    initializers: &[Initializer],
) -> anyhow::Result<Vec<u8>> {
    build_single_op_model_ordered(
        op_type,
        input_shapes,
        output_shapes,
        attrs,
        initializers,
        &[],
    )
}

/// Like `build_single_op_model` but with an explicit `node_input_order` slice that
/// overrides the default node-input ordering (dynamic inputs first, then initializers).
/// Pass `&[]` to use the default ordering.
///
/// This is needed for ops like ScatterND where ONNX requires `[data, indices, updates]`
/// but `indices` is a compile-time initializer and `updates` is a dynamic input.
pub fn build_single_op_model_ordered(
    op_type: &str,
    input_shapes: &[(&str, &[i64], i32)],
    output_shapes: &[(&str, &[i64], i32)],
    attrs: &[NodeAttr],
    initializers: &[Initializer],
    node_input_order: &[&str],
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        !output_shapes.is_empty(),
        "build_single_op_model: output_shapes must not be empty"
    );

    // Build graph inputs (ValueInfoProto for each dynamic input)
    let graph_inputs: Vec<ValueInfoProto> = input_shapes
        .iter()
        .map(|(name, dims, elem_type)| make_value_info(name, *dims, *elem_type))
        .collect();

    // Build graph outputs with explicit type/shape so JSTProve can read them
    let graph_outputs: Vec<ValueInfoProto> = output_shapes
        .iter()
        .map(|(name, dims, elem_type)| make_value_info(name, *dims, *elem_type))
        .collect();

    // Build node attributes
    let node_attrs: Vec<AttributeProto> = attrs.iter().map(make_attribute).collect();

    // Node inputs: use explicit ordering if provided, else dynamic inputs + initializer names
    let node_inputs: Vec<String> = if node_input_order.is_empty() {
        let mut v: Vec<String> = input_shapes.iter().map(|(n, _, _)| n.to_string()).collect();
        for init in initializers {
            v.push(init.name.to_string());
        }
        v
    } else {
        node_input_order.iter().map(|s| s.to_string()).collect()
    };

    // Node outputs = all declared outputs
    let node_outputs: Vec<String> = output_shapes
        .iter()
        .map(|(n, _, _)| n.to_string())
        .collect();

    // Build the single op node
    let node = NodeProto {
        op_type: Some(op_type.to_string()),
        domain: Some(String::new()),
        input: node_inputs,
        output: node_outputs,
        attribute: node_attrs,
        name: Some(format!("{op_type}_0")),
        doc_string: None,
        ..Default::default()
    };

    // Build initializer tensors
    let init_tensors: Vec<TensorProto> = initializers.iter().map(make_initializer).collect();

    let graph = GraphProto {
        name: Some("main".to_string()),
        node: vec![node],
        input: graph_inputs,
        output: graph_outputs,
        initializer: init_tensors,
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: Some(8),
        opset_import: vec![OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(17),
        }],
        producer_name: Some("jstprove-conformance".to_string()),
        graph: Some(graph),
        ..Default::default()
    };

    Ok(model.encode_to_vec())
}

fn make_value_info(name: &str, dims: &[i64], elem_type: i32) -> ValueInfoProto {
    let shape_dims: Vec<tensor_shape_proto::Dimension> = dims
        .iter()
        .map(|&d| tensor_shape_proto::Dimension {
            value: Some(dimension::Value::DimValue(d)),
            ..Default::default()
        })
        .collect();

    ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: Some(elem_type),
                shape: Some(TensorShapeProto { dim: shape_dims }),
            })),
            ..Default::default()
        }),
        doc_string: None,
        metadata_props: vec![],
    }
}

fn make_attribute(attr: &NodeAttr) -> AttributeProto {
    match &attr.value {
        AttrValue::Int(v) => AttributeProto {
            name: Some(attr.name.to_string()),
            r#type: Some(2), // INT
            i: Some(*v),
            ..Default::default()
        },
        AttrValue::Float(v) => AttributeProto {
            name: Some(attr.name.to_string()),
            r#type: Some(1), // FLOAT
            f: Some(*v),
            ..Default::default()
        },
        AttrValue::Ints(vs) => AttributeProto {
            name: Some(attr.name.to_string()),
            r#type: Some(7), // INTS
            ints: vs.clone(),
            ..Default::default()
        },
        AttrValue::Floats(vs) => AttributeProto {
            name: Some(attr.name.to_string()),
            r#type: Some(6), // FLOATS
            floats: vs.clone(),
            ..Default::default()
        },
        AttrValue::String(vs) => AttributeProto {
            name: Some(attr.name.to_string()),
            r#type: Some(3), // STRING
            s: Some(vs.clone()),
            ..Default::default()
        },
        AttrValue::Tensor {
            dims,
            data_type,
            int64_data,
        } => AttributeProto {
            name: Some(attr.name.to_string()),
            r#type: Some(4), // TENSOR
            t: Some(TensorProto {
                dims: dims.clone(),
                data_type: Some(*data_type),
                int64_data: int64_data.clone(),
                ..Default::default()
            }),
            ..Default::default()
        },
    }
}

fn make_initializer(init: &Initializer) -> TensorProto {
    TensorProto {
        name: Some(init.name.to_string()),
        dims: init.dims.clone(),
        data_type: Some(7), // INT64
        int64_data: init.data.clone(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_relu_model() {
        let bytes =
            build_single_op_model("Relu", &[("x", &[4], 7)], &[("y", &[4], 7)], &[], &[]).unwrap();
        assert!(!bytes.is_empty());
        // Should round-trip through prost decode
        let model = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = model.graph.unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type.as_deref(), Some("Relu"));
        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.output.len(), 1);
    }

    #[test]
    fn build_add_model() {
        let bytes = build_single_op_model(
            "Add",
            &[("a", &[3], 7), ("b", &[3], 7)],
            &[("c", &[3], 7)],
            &[],
            &[],
        )
        .unwrap();
        let model = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = model.graph.unwrap();
        assert_eq!(graph.node[0].input.len(), 2);
    }

    #[test]
    fn build_model_with_initializer() {
        let bytes = build_single_op_model(
            "Gather",
            &[("data", &[4, 3], 7)],
            &[("out", &[2, 3], 7)],
            &[NodeAttr {
                name: "axis",
                value: AttrValue::Int(0),
            }],
            &[Initializer {
                name: "indices",
                dims: vec![2],
                data: vec![0, 2],
            }],
        )
        .unwrap();
        let model = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = model.graph.unwrap();
        assert_eq!(graph.initializer.len(), 1);
        assert_eq!(graph.node[0].input, vec!["data", "indices"]);
    }
}
