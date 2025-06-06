use std::collections::{HashMap, HashSet};

use tract_core::{internal::DimLike, ops::cnn::Conv};
use tract_onnx::prelude::*;
use crate::layer_handlers::{extract_layer_params, layer_helpers::{is_relu_like, detect_einsum, EinsumType}, layer_ir::{LayerConstant, LayerIR, LayerKind, LayerParams, SerializableTensor}};

// use tract_core::ops::cnn::Conv;



pub fn analyze_model_internal<P: AsRef<std::path::Path>>(onnx_path: P) -> TractResult<Vec<LayerIR>> {
    let model = tract_onnx::onnx()
        .model_for_path(&onnx_path)?
        .into_typed()?; // Do not optimize yet

    let mut const_tensors: HashMap<usize, Tensor> = HashMap::new();
    let mut node_id_to_layer: HashMap<usize, LayerIR> = HashMap::new();
    // let mut output_edges: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut layer_ids = HashSet::new();

    // First pass: collect const tensors
    for node in model.nodes() {
        if node.op().name() == "Const" {
            if let Some(t) = node.outputs[0].fact.konst.as_ref().map(|arc| arc.as_ref().clone()) {
                const_tensors.insert(node.id, t);
            }
        }
    }

    // Second pass: build layers
    for node in model.nodes() {
        let op_name = node.op().name();
        let id = node.id;
        let name = node.name.clone();

        let inputs: Vec<usize> = node.inputs.iter().map(|i| i.node).collect();

        if op_name == "Const" || op_name == "Source" {
            continue; // skip weights and inputs
        }

        // Track output links (forward connectivity)
        let mut outputs: Vec<usize> = vec![];
        for output in &node.outputs {
            for succ in &output.successors {
                outputs.push(succ.node);
            }
        }

        // Try to infer shape
        let shape = node
            .outputs
            .get(0)
            .map(|o| o.fact.shape.iter().map(|d| d.to_usize().unwrap_or(0)).collect())
            .unwrap_or_default();
        // println!("id - {}, name - {}, {}, {:?}", id, name, op_name, shape);


        let constants: Vec<LayerConstant> = node.inputs
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, inlet)| {
                            const_tensors.get(&inlet.node).and_then(|t| {
                                SerializableTensor::from_tensor(t).map(|val| LayerConstant {
                                    input_index: idx,
                                    name: model.node(inlet.node).name.clone(),
                                    value: val,
                                })
                            })
                        })
                        .collect();

        if op_name.as_ref().eq("Max"){
            // println!("{:?}, {:?}, {:?}",weights, biases, inputs);
        }
        let params = extract_layer_params(node.op().as_typed().unwrap());

        // Map known ops to kind
        let kind = match op_name.as_ref() {
            "Conv" => LayerKind::Conv,
            "EinSum" => {
                    match &params {
                        Some(LayerParams::EinSum { equation }) => {
                            match detect_einsum(equation){
                                // TODO fix this approach. Dont want all the unknowns
                                EinsumType::MatMul => LayerKind::MatMul,
                                EinsumType::TransposedRHSMatMul => LayerKind::Unknown("MatMulTransposedRHS".into()),
                                EinsumType::TransposedLHSMatMul => LayerKind::Unknown("MatMulTransposedLHS".into()),
                                EinsumType::TransposedRHSLHSMatMul => LayerKind::Unknown("MatMulTransposedRHSLHS".into()),
                                _ => LayerKind::Unknown("EinSum".into())
                            }
                        }
                        _ => LayerKind::Unknown("EinSum".into()),
                    }
                }
            "Add" => LayerKind::Add,
            "Max" => {
                if is_relu_like(node, &const_tensors) {
                    LayerKind::Relu
                } else {
                    LayerKind::Unknown(op_name.to_string())
                }
            }
            "Reshape" => LayerKind::Reshape,
            _ => LayerKind::Unknown(op_name.to_string()),
        };

        let layer = LayerIR {
            id,
            name,
            kind,
            inputs,
            outputs,
            shape,
            constants,
            params
        };

        layer_ids.insert(id);
        node_id_to_layer.insert(id, layer);
    }

    // Third pass: remove non-layer edges (pointing to consts), clean output edges
    let mut result: Vec<LayerIR> = node_id_to_layer
        .into_iter()
        .map(|(id, mut layer)| {
            layer.inputs.retain(|i| layer_ids.contains(i));
            layer.outputs.retain(|i| layer_ids.contains(i));
            layer
        })
        .collect();

    // Not necessary but sorting by id for easier to read model
    result.sort_by_key(|k| k.id);

    Ok(result)
}


// fn get_w_and_b(){

// }

// fn get_architecture(){

// }
fn quantize_layer(){

}
fn quantize_model(){

}