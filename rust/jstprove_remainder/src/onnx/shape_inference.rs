use std::collections::HashMap;

use anyhow::{bail, Result};

use super::graph::{LayerGraph, LayerNode, OpType};
use super::parser::{AttrValue, ParsedModel, TensorData};

pub fn infer_all_shapes(
    model: &ParsedModel,
    graph: &LayerGraph,
) -> Result<HashMap<String, Vec<usize>>> {
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for init in model.initializers.values() {
        shapes.insert(init.name.clone(), init.shape());
    }

    for io in &model.inputs {
        let shape: Vec<usize> = io
            .shape
            .iter()
            .map(|&d| if d <= 0 { 1 } else { d as usize })
            .collect();
        shapes.insert(io.name.clone(), shape);
    }

    for io in &model.outputs {
        let shape: Vec<usize> = io
            .shape
            .iter()
            .map(|&d| if d <= 0 { 1 } else { d as usize })
            .collect();
        if !shape.is_empty() && shape.iter().all(|&d| d > 0) {
            shapes.insert(io.name.clone(), shape);
        }
    }

    for node in &model.nodes {
        if node.op_type == "Constant" {
            if let Some(AttrValue::Tensor(td)) = node.attributes.get("value") {
                let shape = td.shape();
                if let Some(out) = node.outputs.first() {
                    shapes.insert(out.clone(), shape);
                }
            }
            continue;
        }
    }

    for layer in graph.iter_topo() {
        let output_shapes = infer_layer_output_shape(layer, &shapes, &model.initializers)?;
        for (name, shape) in output_shapes {
            shapes.insert(name, shape);
        }
    }

    Ok(shapes)
}

fn get_shape<'a>(shapes: &'a HashMap<String, Vec<usize>>, name: &str) -> Option<&'a Vec<usize>> {
    shapes.get(name)
}

fn infer_layer_output_shape(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = layer.inputs.first().and_then(|n| get_shape(shapes, n));

    match layer.op_type {
        OpType::Conv => infer_conv(layer, shapes),
        OpType::Gemm => infer_gemm(layer, shapes),
        OpType::BatchNormalization => passthrough_shape(layer, input_shape),
        OpType::Relu | OpType::Max | OpType::Min | OpType::Clip => {
            passthrough_shape(layer, input_shape)
        }
        OpType::MaxPool => infer_maxpool(layer, input_shape),
        OpType::Add | OpType::Sub => infer_broadcast_binary(layer, shapes),
        OpType::Mul => infer_broadcast_binary(layer, shapes),
        OpType::Reshape => infer_reshape(layer, input_shape, initializers, shapes),
        OpType::Flatten => infer_flatten(layer, input_shape),
        OpType::Squeeze => infer_squeeze(layer, input_shape),
        OpType::Unsqueeze => infer_unsqueeze(layer, input_shape),
        OpType::Constant => Ok(vec![]),
    }
}

fn passthrough_shape(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: missing input shape", layer.name))?
        .clone();
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), shape.clone()))
        .collect())
}

fn infer_conv(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = layer
        .inputs
        .first()
        .and_then(|n| get_shape(shapes, n))
        .ok_or_else(|| anyhow::anyhow!("layer {}: Conv missing input shape", layer.name))?;

    let weight_shape = layer
        .inputs
        .get(1)
        .and_then(|n| get_shape(shapes, n))
        .ok_or_else(|| anyhow::anyhow!("layer {}: Conv missing weight shape", layer.name))?;

    let kernel_shape = layer
        .get_ints_attr("kernel_shape")
        .map(|v| v.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| {
            if weight_shape.len() >= 4 {
                weight_shape[2..].to_vec()
            } else {
                vec![1]
            }
        });

    let strides = layer
        .get_ints_attr("strides")
        .map(|v| v.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![1; kernel_shape.len()]);

    let pads = layer
        .get_ints_attr("pads")
        .map(|v| v.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![0; kernel_shape.len() * 2]);

    let dilations = layer
        .get_ints_attr("dilations")
        .map(|v| v.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![1; kernel_shape.len()]);

    let c_out = weight_shape[0];
    let spatial_dims = kernel_shape.len();

    let mut out_shape = vec![input_shape[0], c_out];
    for i in 0..spatial_dims {
        let in_dim = input_shape[2 + i];
        let pad = pads[i] + pads[spatial_dims + i];
        let effective_kernel = (kernel_shape[i] - 1) * dilations[i] + 1;
        let numerator = in_dim + pad;
        let out_dim = numerator.saturating_sub(effective_kernel) / strides[i] + 1;
        out_shape.push(out_dim);
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_gemm(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let a_shape = layer
        .inputs
        .first()
        .and_then(|n| get_shape(shapes, n))
        .ok_or_else(|| anyhow::anyhow!("layer {}: Gemm missing input A shape", layer.name))?;

    let b_shape = layer
        .inputs
        .get(1)
        .and_then(|n| get_shape(shapes, n))
        .ok_or_else(|| anyhow::anyhow!("layer {}: Gemm missing input B shape", layer.name))?;

    let trans_a = layer
        .get_int_attr("transA")
        .map(|v| v != 0)
        .unwrap_or(false);
    let trans_b = layer
        .get_int_attr("transB")
        .map(|v| v != 0)
        .unwrap_or(false);

    let m = if trans_a {
        a_shape.get(1).copied().unwrap_or(1)
    } else {
        a_shape.first().copied().unwrap_or(1)
    };
    let n = if trans_b {
        b_shape.first().copied().unwrap_or(1)
    } else {
        b_shape.get(1).copied().unwrap_or(1)
    };

    let out_shape = vec![m, n];
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_maxpool(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: MaxPool missing input shape", layer.name))?;

    let kernel_shape = layer
        .get_ints_attr("kernel_shape")
        .ok_or_else(|| anyhow::anyhow!("layer {}: MaxPool missing kernel_shape", layer.name))?;
    let kernel: Vec<usize> = kernel_shape.iter().map(|&d| d as usize).collect();

    let strides = layer
        .get_ints_attr("strides")
        .map(|v| v.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![1; kernel.len()]);

    let pads = layer
        .get_ints_attr("pads")
        .map(|v| v.iter().map(|&d| d as usize).collect::<Vec<_>>())
        .unwrap_or_else(|| vec![0; kernel.len() * 2]);

    let spatial_dims = kernel.len();
    let mut out_shape = vec![input_shape[0], input_shape[1]];
    for i in 0..spatial_dims {
        let in_dim = input_shape[2 + i];
        let pad = pads[i] + pads[spatial_dims + i];
        let total = in_dim + pad;
        let out_dim = total.saturating_sub(kernel[i]) / strides[i] + 1;
        out_shape.push(out_dim);
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let da = if i < max_rank - a.len() {
            1
        } else {
            a[i - (max_rank - a.len())]
        };
        let db = if i < max_rank - b.len() {
            1
        } else {
            b[i - (max_rank - b.len())]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            bail!("incompatible broadcast dimensions at axis {i}: {da} vs {db}");
        }
    }
    Ok(result)
}

fn infer_broadcast_binary(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let a = layer.inputs.first().and_then(|n| get_shape(shapes, n));
    let b = layer.inputs.get(1).and_then(|n| get_shape(shapes, n));

    let out_shape = match (a, b) {
        (Some(sa), Some(sb)) => {
            broadcast_shapes(sa, sb).map_err(|e| anyhow::anyhow!("layer {}: {e}", layer.name))?
        }
        (Some(s), None) | (None, Some(s)) => s.clone(),
        (None, None) => bail!("layer {}: binary op has no input shapes", layer.name),
    };

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_reshape(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Reshape missing input shape", layer.name))?;
    let input_size: usize = input_shape.iter().product();

    let shape_data = layer.inputs.get(1).and_then(|name| {
        initializers
            .get(name)
            .map(|td| td.as_i64_vec())
            .or_else(|| get_shape(shapes, name).map(|s| s.iter().map(|&d| d as i64).collect()))
    });

    let target_shape = if let Some(shape_vals) = shape_data {
        shape_vals
    } else if let Some(AttrValue::Ints(shape_attr)) = layer.attributes.get("shape") {
        shape_attr.clone()
    } else {
        bail!(
            "layer {}: Reshape target shape not found in initializers or attributes",
            layer.name
        );
    };

    let mut minus_one_idx: Option<usize> = None;
    let mut known_product: usize = 1;
    let mut minus_one_count: usize = 0;
    for (i, &d) in target_shape.iter().enumerate() {
        if d == -1 {
            minus_one_count += 1;
            minus_one_idx = Some(i);
        } else if d == 0 {
            let dim = input_shape.get(i).copied().unwrap_or(1);
            known_product *= dim;
        } else {
            known_product *= d as usize;
        }
    }

    if minus_one_count > 1 {
        bail!(
            "layer {}: Reshape target shape has multiple -1 dimensions ({minus_one_count})",
            layer.name
        );
    }

    let mut out_shape: Vec<usize> = target_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == -1 {
                0
            } else if d == 0 {
                input_shape.get(i).copied().unwrap_or(1)
            } else {
                d as usize
            }
        })
        .collect();

    if let Some(idx) = minus_one_idx {
        if known_product == 0 || input_size % known_product != 0 {
            bail!(
                "layer {}: Reshape incompatible sizes: input_size={input_size} known_product={known_product}",
                layer.name
            );
        }
        out_shape[idx] = input_size / known_product;
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_flatten(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Flatten missing input shape", layer.name))?;

    let axis = layer.get_int_attr("axis").unwrap_or(1) as usize;
    let axis = axis.min(input_shape.len());

    let dim0: usize = input_shape[..axis].iter().product::<usize>().max(1);
    let dim1: usize = input_shape[axis..].iter().product::<usize>().max(1);

    let out_shape = vec![dim0, dim1];
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_squeeze(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Squeeze missing input shape", layer.name))?;

    let axes: Vec<i64> = layer
        .get_ints_attr("axes")
        .map(|v| v.to_vec())
        .unwrap_or_default();

    let out_shape: Vec<usize> = if axes.is_empty() {
        input_shape.iter().copied().filter(|&d| d != 1).collect()
    } else {
        let rank = input_shape.len() as i64;
        let normalized: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + rank) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !normalized.contains(i))
            .map(|(_, &d)| d)
            .collect()
    };

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_unsqueeze(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Unsqueeze missing input shape", layer.name))?;

    let axes: Vec<i64> = layer
        .get_ints_attr("axes")
        .map(|v| v.to_vec())
        .unwrap_or_default();

    let new_rank = input_shape.len() + axes.len();
    let normalized: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + new_rank as i64) as usize
            } else {
                a as usize
            }
        })
        .collect();

    let mut out_shape = Vec::with_capacity(new_rank);
    let mut input_idx = 0;
    for i in 0..new_rank {
        if normalized.contains(&i) {
            out_shape.push(1);
        } else {
            out_shape.push(input_shape.get(input_idx).copied().unwrap_or(1));
            input_idx += 1;
        }
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_same_rank() {
        assert_eq!(
            broadcast_shapes(&[1, 3, 1], &[2, 1, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn broadcast_different_rank() {
        assert_eq!(
            broadcast_shapes(&[3, 1], &[1, 2, 1, 4]).unwrap(),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(broadcast_shapes(&[], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
    }
}
