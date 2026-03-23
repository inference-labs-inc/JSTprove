use std::collections::{HashMap, HashSet};

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
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .collect();
        shapes.insert(io.name.clone(), shape);
    }

    for io in &model.outputs {
        let shape: Vec<usize> = io
            .shape
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .collect();
        if !shape.is_empty() {
            shapes.insert(io.name.clone(), shape);
        }
    }

    let mut constant_tensors: HashMap<String, TensorData> = HashMap::new();
    for node in &model.nodes {
        if node.op_type == "Constant" {
            if let Some(AttrValue::Tensor(td)) = node.attributes.get("value") {
                if let Some(out) = node.outputs.first() {
                    shapes.insert(out.clone(), td.shape());
                    constant_tensors.insert(out.clone(), td.clone());
                }
            }
            continue;
        }
    }

    for layer in graph.iter_topo() {
        let output_shapes =
            infer_layer_output_shape(layer, &shapes, &model.initializers, &constant_tensors)?;
        for (name, shape) in output_shapes {
            shapes.insert(name, shape);
        }
        fold_constants(layer, &shapes, &model.initializers, &mut constant_tensors);
    }

    Ok(shapes)
}

fn lookup_constant<'a>(
    name: &str,
    initializers: &'a HashMap<String, TensorData>,
    constant_tensors: &'a HashMap<String, TensorData>,
) -> Option<&'a TensorData> {
    initializers
        .get(name)
        .or_else(|| constant_tensors.get(name))
}

fn fold_constants(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &mut HashMap<String, TensorData>,
) {
    let folded = match layer.op_type {
        OpType::Shape => fold_shape(layer, shapes),
        OpType::Gather => fold_gather(layer, initializers, constant_tensors),
        OpType::Unsqueeze => fold_unsqueeze(layer, initializers, constant_tensors),
        OpType::Squeeze => fold_squeeze(layer, initializers, constant_tensors),
        OpType::Concat => fold_concat(layer, initializers, constant_tensors),
        OpType::Slice => fold_slice(layer, initializers, constant_tensors),
        OpType::Cast => fold_cast(layer, initializers, constant_tensors),
        OpType::Reshape => fold_reshape(layer, initializers, constant_tensors),
        _ => None,
    };
    if let Some(td) = folded {
        for out_name in &layer.outputs {
            let mut entry = td.clone();
            entry.name = out_name.clone();
            constant_tensors.insert(out_name.clone(), entry);
        }
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn fold_shape(layer: &LayerNode, shapes: &HashMap<String, Vec<usize>>) -> Option<TensorData> {
    let input_name = layer.inputs.first()?;
    let input_shape = shapes.get(input_name.as_str())?;

    let rank = input_shape.len() as i64;
    let start_raw = layer.get_int_attr("start").unwrap_or(0);
    let end_raw = layer.get_int_attr("end").unwrap_or(rank);
    let start = if start_raw < 0 {
        (rank + start_raw).max(0) as usize
    } else {
        (start_raw as usize).min(input_shape.len())
    };
    let end = if end_raw < 0 {
        (rank + end_raw).max(0) as usize
    } else {
        (end_raw as usize).min(input_shape.len())
    };

    let dims: Vec<i64> = if end > start {
        input_shape[start..end].iter().map(|&d| d as i64).collect()
    } else {
        vec![]
    };

    Some(TensorData {
        name: String::new(),
        dims: vec![dims.len() as i64],
        data_type: 7, // INT64
        float_data: vec![],
        int_data: dims,
    })
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn fold_gather(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    let data_name = layer.inputs.first()?;
    let indices_name = layer.inputs.get(1)?;

    let data_td = lookup_constant(data_name, initializers, constant_tensors)?;
    let indices_td = lookup_constant(indices_name, initializers, constant_tensors)?;

    let data_shape = data_td.shape();
    if data_shape.is_empty() {
        return None;
    }

    let axis_raw = layer.get_int_attr("axis").unwrap_or(0);
    let rank = data_shape.len() as i64;
    let axis = if axis_raw < 0 {
        (axis_raw + rank) as usize
    } else {
        axis_raw as usize
    };
    if axis >= data_shape.len() {
        return None;
    }

    let data_vals = data_td.as_i64_vec();
    let indices_vals = indices_td.as_i64_vec();
    let indices_shape = indices_td.shape();
    let dim_size = data_shape[axis] as i64;

    if data_shape.len() == 1 {
        let gathered: Option<Vec<i64>> = indices_vals
            .iter()
            .map(|&idx| {
                let idx = if idx < 0 { idx + dim_size } else { idx };
                data_vals.get(idx as usize).copied()
            })
            .collect();
        let gathered = gathered?;

        let out_dims: Vec<i64> = indices_shape.iter().map(|&d| d as i64).collect();
        return Some(TensorData {
            name: String::new(),
            dims: out_dims,
            data_type: data_td.data_type,
            float_data: vec![],
            int_data: gathered,
        });
    }

    let outer_size: usize = data_shape[..axis].iter().product();
    let inner_size: usize = data_shape[axis + 1..].iter().product();
    let axis_size = data_shape[axis];
    let n_indices = indices_vals.len();

    let mut gathered = Vec::with_capacity(outer_size * n_indices * inner_size);
    for o in 0..outer_size {
        for &idx_raw in &indices_vals {
            let idx = if idx_raw < 0 {
                (idx_raw + dim_size) as usize
            } else {
                idx_raw as usize
            };
            if idx >= axis_size {
                return None;
            }
            let src_base = o * axis_size * inner_size + idx * inner_size;
            gathered.extend_from_slice(&data_vals[src_base..src_base + inner_size]);
        }
    }

    let mut out_dims: Vec<i64> = data_shape[..axis].iter().map(|&d| d as i64).collect();
    out_dims.extend(indices_shape.iter().map(|&d| d as i64));
    out_dims.extend(data_shape[axis + 1..].iter().map(|&d| d as i64));

    Some(TensorData {
        name: String::new(),
        dims: out_dims,
        data_type: data_td.data_type,
        float_data: vec![],
        int_data: gathered,
    })
}

fn fold_unsqueeze(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    let input_name = layer.inputs.first()?;
    let input_td = lookup_constant(input_name, initializers, constant_tensors)?;
    let input_shape = input_td.shape();

    let axes: Vec<i64> = match layer.get_ints_attr("axes").map(|v| v.to_vec()) {
        Some(v) => v,
        None => {
            let axes_name = layer.inputs.get(1)?;
            let axes_td = lookup_constant(axes_name, initializers, constant_tensors)?;
            axes_td.as_i64_vec()
        }
    };

    let new_rank = input_shape.len() + axes.len();
    let r = new_rank as i64;
    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    for &a in &axes {
        let n = if a < 0 { (a + r) as usize } else { a as usize };
        if n >= new_rank || !seen.insert(n) {
            return None;
        }
        normalized.push(n);
    }

    let mut out_dims: Vec<i64> = Vec::with_capacity(new_rank);
    let mut input_idx = 0;
    let in_dims: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
    for i in 0..new_rank {
        if seen.contains(&i) {
            out_dims.push(1);
        } else {
            out_dims.push(*in_dims.get(input_idx)?);
            input_idx += 1;
        }
    }

    Some(TensorData {
        name: String::new(),
        dims: out_dims,
        data_type: input_td.data_type,
        float_data: input_td.float_data.clone(),
        int_data: input_td.int_data.clone(),
    })
}

fn fold_squeeze(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    let input_name = layer.inputs.first()?;
    let input_td = lookup_constant(input_name, initializers, constant_tensors)?;
    let input_shape = input_td.shape();
    let rank = input_shape.len() as i64;

    let axes_opt: Option<Vec<i64>> =
        layer.get_ints_attr("axes").map(|v| v.to_vec()).or_else(|| {
            let axes_name = layer.inputs.get(1)?;
            let axes_td = lookup_constant(axes_name, initializers, constant_tensors)?;
            Some(axes_td.as_i64_vec())
        });

    let out_dims: Vec<i64> = if let Some(axes) = axes_opt {
        let mut squeeze_set = HashSet::new();
        for &a in &axes {
            let n = if a < 0 {
                (a + rank) as usize
            } else {
                a as usize
            };
            squeeze_set.insert(n);
        }
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !squeeze_set.contains(i))
            .map(|(_, &d)| d as i64)
            .collect()
    } else {
        input_shape
            .iter()
            .filter(|&&d| d != 1)
            .map(|&d| d as i64)
            .collect()
    };

    Some(TensorData {
        name: String::new(),
        dims: out_dims,
        data_type: input_td.data_type,
        float_data: input_td.float_data.clone(),
        int_data: input_td.int_data.clone(),
    })
}

#[allow(clippy::cast_possible_wrap)]
fn fold_concat(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    if layer.inputs.is_empty() {
        return None;
    }

    let mut all_inputs: Vec<&TensorData> = Vec::new();
    for name in &layer.inputs {
        let td = lookup_constant(name, initializers, constant_tensors)?;
        all_inputs.push(td);
    }

    let first = all_inputs[0];
    let first_shape = first.shape();
    let rank = first_shape.len();
    if rank == 0 {
        return None;
    }

    let raw_axis = layer.get_int_attr("axis").unwrap_or(0);
    let axis = if raw_axis < 0 {
        (raw_axis + rank as i64) as usize
    } else {
        raw_axis as usize
    };
    if axis >= rank {
        return None;
    }

    // Only fold 1-D shape tensors; multi-dimensional concat is left to the
    // normal shape-inference path which tracks shapes without needing values.
    if rank == 1 {
        let data_type = first.data_type;
        if all_inputs
            .iter()
            .any(|td| td.data_type != data_type || td.int_data.is_empty())
        {
            return None;
        }
        let mut concatenated = Vec::new();
        for td in &all_inputs {
            concatenated.extend_from_slice(&td.as_i64_vec());
        }
        let total_len = concatenated.len();
        return Some(TensorData {
            name: String::new(),
            dims: vec![total_len as i64],
            data_type,
            float_data: vec![],
            int_data: concatenated,
        });
    }

    None
}

#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn fold_slice(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    let input_name = layer.inputs.first()?;
    let input_td = lookup_constant(input_name, initializers, constant_tensors)?;
    let input_shape = input_td.shape();
    if input_shape.len() != 1 {
        return None;
    }

    let starts_name = layer.inputs.get(1)?;
    let ends_name = layer.inputs.get(2)?;
    let starts_td = lookup_constant(starts_name, initializers, constant_tensors)?;
    let ends_td = lookup_constant(ends_name, initializers, constant_tensors)?;

    let starts = starts_td.as_i64_vec();
    let ends = ends_td.as_i64_vec();
    let data = input_td.as_i64_vec();
    let dim = data.len() as i64;

    let axes: Vec<i64> = if let Some(axes_name) = layer.inputs.get(3).filter(|n| !n.is_empty()) {
        lookup_constant(axes_name, initializers, constant_tensors)?.as_i64_vec()
    } else {
        (0..starts.len() as i64).collect()
    };

    let steps: Vec<i64> = if let Some(steps_name) = layer.inputs.get(4).filter(|n| !n.is_empty()) {
        lookup_constant(steps_name, initializers, constant_tensors)?.as_i64_vec()
    } else {
        vec![1; starts.len()]
    };

    if axes.len() != 1 || axes[0] != 0 {
        return None;
    }

    let step = steps[0];
    if step == 0 {
        return None;
    }

    let mut start = starts[0];
    let mut end = ends[0];
    if start < 0 {
        start += dim;
    }
    if end < 0 {
        end += dim;
    }
    start = start.clamp(0, dim);
    end = end.clamp(0, dim);

    let sliced: Vec<i64> = if step > 0 {
        let mut v = Vec::new();
        let mut i = start;
        while i < end {
            v.push(data[i as usize]);
            i += step;
        }
        v
    } else {
        if dim == 0 {
            return Some(TensorData {
                name: String::new(),
                dims: vec![0],
                data_type: input_td.data_type,
                float_data: vec![],
                int_data: vec![],
            });
        }
        if start >= dim {
            start = dim.saturating_sub(1);
        }
        let mut v = Vec::new();
        let mut i = start;
        while i > end {
            v.push(data[i as usize]);
            i += step;
        }
        v
    };

    Some(TensorData {
        name: String::new(),
        dims: vec![sliced.len() as i64],
        data_type: input_td.data_type,
        float_data: vec![],
        int_data: sliced,
    })
}

fn fold_cast(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    let input_name = layer.inputs.first()?;
    let input_td = lookup_constant(input_name, initializers, constant_tensors)?;
    let to = layer.get_int_attr("to")?;

    Some(TensorData {
        name: String::new(),
        dims: input_td.dims.clone(),
        data_type: to as i32,
        float_data: input_td.float_data.clone(),
        int_data: input_td.int_data.clone(),
    })
}

fn fold_reshape(
    layer: &LayerNode,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Option<TensorData> {
    let input_name = layer.inputs.first()?;
    let input_td = lookup_constant(input_name, initializers, constant_tensors)?;
    let shape_name = layer.inputs.get(1)?;
    let shape_td = lookup_constant(shape_name, initializers, constant_tensors)?;
    let target = shape_td.as_i64_vec();

    let input_size = input_td.as_i64_vec().len();
    let mut out_dims = Vec::with_capacity(target.len());
    let mut minus_one_idx = None;
    let mut known_product: usize = 1;
    for (i, &d) in target.iter().enumerate() {
        if d == -1 {
            minus_one_idx = Some(i);
            out_dims.push(0i64);
        } else if d == 0 {
            let orig = input_td.shape();
            out_dims.push(*orig.get(i)? as i64);
            known_product *= orig[i];
        } else if d > 0 {
            out_dims.push(d);
            known_product *= d as usize;
        } else {
            return None;
        }
    }
    if let Some(idx) = minus_one_idx {
        if known_product == 0 {
            return None;
        }
        out_dims[idx] = (input_size / known_product) as i64;
    }

    Some(TensorData {
        name: String::new(),
        dims: out_dims,
        data_type: input_td.data_type,
        float_data: input_td.float_data.clone(),
        int_data: input_td.int_data.clone(),
    })
}

fn get_shape<'a>(shapes: &'a HashMap<String, Vec<usize>>, name: &str) -> Option<&'a Vec<usize>> {
    shapes.get(name)
}

fn nonneg_to_usize(vals: &[i64], attr: &str, layer_name: &str) -> Result<Vec<usize>> {
    vals.iter()
        .enumerate()
        .map(|(i, &d)| {
            if d < 0 {
                bail!("layer {layer_name}: {attr}[{i}] is negative ({d})");
            }
            Ok(d as usize)
        })
        .collect()
}

fn infer_layer_output_shape(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
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
        OpType::Add | OpType::Sub | OpType::Div => infer_broadcast_binary(layer, shapes),
        OpType::Mul => infer_broadcast_binary(layer, shapes),
        OpType::Reshape => infer_reshape(layer, input_shape, initializers, constant_tensors),
        OpType::Flatten => infer_flatten(layer, input_shape),
        OpType::Squeeze => infer_squeeze(layer, input_shape, initializers, constant_tensors),
        OpType::Unsqueeze => infer_unsqueeze(layer, input_shape, initializers, constant_tensors),
        OpType::Constant => Ok(vec![]),
        OpType::Cast | OpType::Exp | OpType::Softmax | OpType::Sigmoid | OpType::Gelu => {
            passthrough_shape(layer, input_shape)
        }
        OpType::LayerNormalization => {
            // Output[0] = Y: same shape as input.
            // Output[1] = Mean, Output[2] = InvStdDev: reduced shapes that
            // depend on the normalization axis. These are rare in practice and
            // would require axis-aware shape inference. Reject the multi-output
            // form rather than propagating incorrect shapes.
            if layer.outputs.len() > 1 {
                bail!(
                    "layer {}: LayerNormalization with more than one output (Mean/InvStdDev) \
                     is not supported by shape inference; use a single output (Y only)",
                    layer.name
                );
            }
            passthrough_shape(layer, input_shape)
        }
        OpType::Tile => infer_tile(layer, input_shape, initializers, constant_tensors),
        OpType::Gather => infer_gather(layer, shapes, initializers, constant_tensors),
        OpType::Resize => infer_resize(layer, input_shape, initializers, constant_tensors),
        OpType::GridSample => infer_gridsample(layer, shapes),
        OpType::Transpose => infer_transpose(layer, shapes),
        OpType::Concat => infer_concat(layer, shapes),
        OpType::Slice => infer_slice(layer, input_shape, initializers, constant_tensors),
        OpType::TopK => infer_topk(layer, input_shape, initializers, constant_tensors),
        OpType::Log => passthrough_shape(layer, input_shape),
        OpType::Expand => infer_expand(layer, shapes, initializers, constant_tensors),
        OpType::Shape => infer_shape_op(layer, input_shape),
        OpType::ReduceMean => infer_reduce(layer, input_shape, initializers, constant_tensors),
        OpType::MatMul => infer_matmul(layer, shapes),
        OpType::AveragePool => infer_maxpool(layer, input_shape),
        OpType::Pad => infer_pad(layer, input_shape, initializers, constant_tensors),
        OpType::Split => infer_split(layer, input_shape, initializers, constant_tensors),
        OpType::Where => infer_where(layer, shapes),
        OpType::Pow | OpType::Sqrt | OpType::Tanh | OpType::Erf => {
            passthrough_shape(layer, input_shape)
        }
        OpType::ReduceSum => infer_reduce(layer, input_shape, initializers, constant_tensors),
        OpType::ConvTranspose => infer_conv_transpose(layer, shapes),
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

    let kernel_shape = if let Some(v) = layer.get_ints_attr("kernel_shape") {
        nonneg_to_usize(v, "kernel_shape", &layer.name)?
    } else if weight_shape.len() >= 3 {
        weight_shape[2..].to_vec()
    } else {
        vec![1]
    };

    let strides = if let Some(v) = layer.get_ints_attr("strides") {
        nonneg_to_usize(v, "strides", &layer.name)?
    } else {
        vec![1; kernel_shape.len()]
    };

    let pads = if let Some(v) = layer.get_ints_attr("pads") {
        nonneg_to_usize(v, "pads", &layer.name)?
    } else {
        vec![0; kernel_shape.len() * 2]
    };

    let dilations = if let Some(v) = layer.get_ints_attr("dilations") {
        nonneg_to_usize(v, "dilations", &layer.name)?
    } else {
        vec![1; kernel_shape.len()]
    };

    if weight_shape.len() < 2 {
        bail!(
            "layer {}: Conv weight rank {} < 2",
            layer.name,
            weight_shape.len()
        );
    }
    let c_out = weight_shape[0];
    let spatial_dims = kernel_shape.len();

    if input_shape.len() < 2 + spatial_dims {
        bail!(
            "layer {}: Conv input rank {} too small for {spatial_dims} spatial dims",
            layer.name,
            input_shape.len()
        );
    }
    if pads.len() < 2 * spatial_dims
        || dilations.len() < spatial_dims
        || strides.len() < spatial_dims
    {
        bail!(
            "layer {}: Conv attribute length mismatch: pads={} dilations={} strides={} spatial_dims={spatial_dims}",
            layer.name,
            pads.len(),
            dilations.len(),
            strides.len()
        );
    }

    let mut out_shape = vec![input_shape[0], c_out];
    for i in 0..spatial_dims {
        let in_dim = input_shape[2 + i];
        let pad = pads[i] + pads[spatial_dims + i];
        if kernel_shape[i] == 0 {
            bail!("layer {}: Conv kernel_shape[{i}] is zero", layer.name);
        }
        let effective_kernel = (kernel_shape[i] - 1) * dilations[i] + 1;
        let padded = in_dim + pad;
        if strides[i] == 0 {
            bail!(
                "layer {}: Conv stride is zero at spatial dim {i}",
                layer.name
            );
        }
        if padded < effective_kernel {
            bail!(
                "layer {}: Conv padded input {padded} < effective kernel {effective_kernel} at spatial dim {i}",
                layer.name
            );
        }
        let out_dim = (padded - effective_kernel) / strides[i] + 1;
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

    if a_shape.len() != 2 {
        bail!(
            "layer {}: Gemm input A rank {} != 2 (batched Gemm not yet supported in circuit runtime)",
            layer.name,
            a_shape.len()
        );
    }
    if b_shape.len() != 2 {
        bail!(
            "layer {}: Gemm input B rank {} != 2 (batched Gemm not yet supported in circuit runtime)",
            layer.name,
            b_shape.len()
        );
    }

    let m = if trans_a { a_shape[1] } else { a_shape[0] };
    let k_a = if trans_a { a_shape[0] } else { a_shape[1] };
    let k_b = if trans_b { b_shape[1] } else { b_shape[0] };
    let n = if trans_b { b_shape[0] } else { b_shape[1] };

    if k_a != k_b {
        bail!(
            "layer {}: Gemm inner dimension mismatch: K_a={k_a} K_b={k_b}",
            layer.name
        );
    }

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

    let kernel_raw = layer
        .get_ints_attr("kernel_shape")
        .ok_or_else(|| anyhow::anyhow!("layer {}: MaxPool missing kernel_shape", layer.name))?;
    let kernel = nonneg_to_usize(kernel_raw, "kernel_shape", &layer.name)?;
    for (i, &k) in kernel.iter().enumerate() {
        if k == 0 {
            bail!("layer {}: MaxPool kernel_shape[{i}] is zero", layer.name);
        }
    }

    let strides = if let Some(v) = layer.get_ints_attr("strides") {
        nonneg_to_usize(v, "strides", &layer.name)?
    } else {
        vec![1; kernel.len()]
    };

    let pads = if let Some(v) = layer.get_ints_attr("pads") {
        nonneg_to_usize(v, "pads", &layer.name)?
    } else {
        vec![0; kernel.len() * 2]
    };

    let spatial_dims = kernel.len();

    if input_shape.len() < 2 + spatial_dims {
        bail!(
            "layer {}: MaxPool input rank {} too small for {spatial_dims} spatial dims",
            layer.name,
            input_shape.len()
        );
    }
    if pads.len() < 2 * spatial_dims || strides.len() < spatial_dims {
        bail!(
            "layer {}: MaxPool attribute length mismatch: pads={} strides={} spatial_dims={spatial_dims}",
            layer.name,
            pads.len(),
            strides.len()
        );
    }

    let mut out_shape = vec![input_shape[0], input_shape[1]];
    for i in 0..spatial_dims {
        let in_dim = input_shape[2 + i];
        let pad = pads[i] + pads[spatial_dims + i];
        let padded = in_dim + pad;
        if strides[i] == 0 {
            bail!(
                "layer {}: MaxPool stride is zero at spatial dim {i}",
                layer.name
            );
        }
        if padded < kernel[i] {
            bail!(
                "layer {}: MaxPool padded input {padded} < kernel {k} at spatial dim {i}",
                layer.name,
                k = kernel[i]
            );
        }
        let out_dim = (padded - kernel[i]) / strides[i] + 1;
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
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Reshape missing input shape", layer.name))?;
    let input_size: usize = input_shape.iter().product();

    let shape_data = layer.inputs.get(1).and_then(|name| {
        initializers
            .get(name)
            .or_else(|| constant_tensors.get(name))
            .map(|td| td.as_i64_vec())
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

    let allowzero = layer
        .get_int_attr("allowzero")
        .map(|v| v != 0)
        .unwrap_or(false);

    let mut minus_one_idx: Option<usize> = None;
    let mut known_product: usize = 1;
    let mut minus_one_count: usize = 0;
    let mut has_zero = false;
    for (i, &d) in target_shape.iter().enumerate() {
        if d == -1 {
            minus_one_count += 1;
            minus_one_idx = Some(i);
        } else if d == 0 {
            has_zero = true;
            if allowzero {
                known_product = 0;
            } else {
                if i >= input_shape.len() {
                    bail!(
                        "layer {}: Reshape d=0 at index {i} exceeds input rank {}",
                        layer.name,
                        input_shape.len()
                    );
                }
                known_product *= input_shape[i];
            }
        } else if d < -1 {
            bail!(
                "layer {}: Reshape invalid dimension value {d} at index {i}",
                layer.name
            );
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

    if allowzero && has_zero && minus_one_count > 0 {
        bail!(
            "layer {}: Reshape allowzero=1 with both 0 and -1 dims is invalid",
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
                if allowzero {
                    0
                } else {
                    input_shape[i]
                }
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
    } else if input_size != known_product {
        bail!(
            "layer {}: Reshape element count mismatch: input_size={input_size} target_product={known_product}",
            layer.name
        );
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

    let raw_axis = layer.get_int_attr("axis").unwrap_or(1);
    let rank = input_shape.len() as i64;
    if raw_axis < -rank || raw_axis > rank {
        bail!(
            "layer {}: Flatten axis {raw_axis} out of range for rank {rank}",
            layer.name
        );
    }
    let axis = if raw_axis < 0 {
        (raw_axis + rank) as usize
    } else {
        raw_axis as usize
    };

    let dim0: usize = input_shape[..axis].iter().product();
    let dim1: usize = input_shape[axis..].iter().product();

    let out_shape = vec![dim0, dim1];
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

enum AxesInput {
    Omitted,
    Resolved(Vec<i64>),
    Unresolved(String),
}

fn resolve_axes_from_input(
    layer: &LayerNode,
    input_idx: usize,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> AxesInput {
    let Some(name) = layer.inputs.get(input_idx) else {
        return AxesInput::Omitted;
    };
    if name.is_empty() {
        return AxesInput::Omitted;
    }
    match initializers
        .get(name)
        .or_else(|| constant_tensors.get(name))
    {
        Some(td) => AxesInput::Resolved(td.as_i64_vec()),
        None => AxesInput::Unresolved(name.clone()),
    }
}

fn infer_squeeze(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Squeeze missing input shape", layer.name))?;

    let axes_omitted;
    let axes: Vec<i64> = match layer.get_ints_attr("axes").map(|v| v.to_vec()) {
        Some(v) => {
            axes_omitted = false;
            v
        }
        None => match resolve_axes_from_input(layer, 1, initializers, constant_tensors) {
            AxesInput::Resolved(v) => {
                axes_omitted = false;
                v
            }
            AxesInput::Omitted => {
                axes_omitted = true;
                vec![]
            }
            AxesInput::Unresolved(name) => {
                bail!(
                    "layer {}: Squeeze axes input '{}' could not be resolved from initializers or constants",
                    layer.name,
                    name
                );
            }
        },
    };

    let out_shape: Vec<usize> = if axes.is_empty() && axes_omitted {
        input_shape.iter().copied().filter(|&d| d != 1).collect()
    } else if axes.is_empty() {
        input_shape.to_vec()
    } else {
        let rank = input_shape.len() as i64;
        let mut seen = std::collections::HashSet::new();
        let normalized: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < -rank || a >= rank {
                    bail!(
                        "layer {}: Squeeze axis {a} out of range for rank {rank}",
                        layer.name
                    );
                }
                let n = if a < 0 {
                    (a + rank) as usize
                } else {
                    a as usize
                };
                if !seen.insert(n) {
                    bail!("layer {}: Squeeze duplicate axis {n}", layer.name);
                }
                Ok(n)
            })
            .collect::<Result<Vec<_>>>()?;
        for &n in &normalized {
            if input_shape[n] != 1 {
                bail!(
                    "layer {}: Squeeze axis {n} has dimension {} (expected 1)",
                    layer.name,
                    input_shape[n]
                );
            }
        }
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !seen.contains(i))
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
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Unsqueeze missing input shape", layer.name))?;

    let axes: Vec<i64> = match layer.get_ints_attr("axes").map(|v| v.to_vec()) {
        Some(v) => v,
        None => match resolve_axes_from_input(layer, 1, initializers, constant_tensors) {
            AxesInput::Resolved(v) => v,
            AxesInput::Omitted => {
                bail!(
                    "layer {}: Unsqueeze requires axes (attribute or input tensor)",
                    layer.name
                );
            }
            AxesInput::Unresolved(name) => {
                bail!(
                    "layer {}: Unsqueeze axes input '{}' could not be resolved from initializers or constants",
                    layer.name,
                    name
                );
            }
        },
    };

    let new_rank = input_shape.len() + axes.len();
    let r = new_rank as i64;
    let mut seen = std::collections::HashSet::new();
    let normalized: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < -r || a >= r {
                bail!(
                    "layer {}: Unsqueeze axis {a} out of range for new rank {new_rank}",
                    layer.name
                );
            }
            let n = if a < 0 { (a + r) as usize } else { a as usize };
            if !seen.insert(n) {
                bail!("layer {}: Unsqueeze duplicate axis {n}", layer.name);
            }
            Ok(n)
        })
        .collect::<Result<Vec<_>>>()?;

    debug_assert_eq!(
        new_rank - normalized.len(),
        input_shape.len(),
        "Unsqueeze: non-axis positions {} != input rank {}",
        new_rank - normalized.len(),
        input_shape.len()
    );

    let mut out_shape = Vec::with_capacity(new_rank);
    let mut input_idx = 0;
    for i in 0..new_rank {
        if seen.contains(&i) {
            out_shape.push(1);
        } else {
            out_shape.push(input_shape[input_idx]);
            input_idx += 1;
        }
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_tile(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Tile missing input shape", layer.name))?;

    let repeats_data = layer
        .inputs
        .get(1)
        .and_then(|name| {
            initializers
                .get(name)
                .or_else(|| constant_tensors.get(name))
                .map(|td| td.as_i64_vec())
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: Tile repeats tensor not found in initializers or Constant nodes",
                layer.name
            )
        })?;

    if repeats_data.len() != input_shape.len() {
        bail!(
            "layer {}: Tile repeats length {} != input rank {}",
            layer.name,
            repeats_data.len(),
            input_shape.len()
        );
    }

    let out_shape: Vec<usize> = input_shape
        .iter()
        .zip(repeats_data.iter())
        .enumerate()
        .map(|(i, (&d, &r))| {
            if r < 1 {
                bail!(
                    "layer {}: Tile repeat[{i}] = {r} is less than 1",
                    layer.name
                );
            }
            let repeat_usize = usize::try_from(r).map_err(|_| {
                anyhow::anyhow!(
                    "layer {}: Tile repeat[{i}] = {r} cannot be converted to usize",
                    layer.name
                )
            })?;
            d.checked_mul(repeat_usize).ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {}: Tile output size overflow at axis {i}: {d} * {r}",
                    layer.name
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_gather(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let data_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer {}: Gather missing data input", layer.name))?;

    let indices_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: Gather missing indices input", layer.name))?;

    let data_shape = get_shape(shapes, data_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: Gather missing shape for data input '{data_name}'",
            layer.name
        )
    })?;

    let indices_shape = get_shape(shapes, indices_name)
        .cloned()
        .or_else(|| {
            initializers
                .get(indices_name)
                .or_else(|| constant_tensors.get(indices_name))
                .map(|td| td.shape())
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: Gather missing shape for indices input '{indices_name}'",
                layer.name
            )
        })?;

    let rank = data_shape.len();
    anyhow::ensure!(
        rank > 0,
        "layer {}: Gather data input must have rank >= 1",
        layer.name
    );

    let raw_axis = layer.get_int_attr("axis").unwrap_or(0);
    let axis = if raw_axis < 0 {
        let a = rank as i64 + raw_axis;
        anyhow::ensure!(
            a >= 0,
            "layer {}: Gather axis {} out of range for data rank {}",
            layer.name,
            raw_axis,
            rank
        );
        a as usize
    } else {
        let a = raw_axis as usize;
        anyhow::ensure!(
            a < rank,
            "layer {}: Gather axis {} out of range for data rank {}",
            layer.name,
            raw_axis,
            rank
        );
        a
    };

    // output_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]
    let mut out_shape = Vec::with_capacity(rank - 1 + indices_shape.len());
    out_shape.extend_from_slice(&data_shape[..axis]);
    out_shape.extend_from_slice(&indices_shape);
    out_shape.extend_from_slice(&data_shape[axis + 1..]);

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

fn infer_resize(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Resize missing input shape", layer.name))?;

    let rank = input_shape.len();

    let lookup = |name: &str| -> Option<&TensorData> {
        initializers
            .get(name)
            .or_else(|| constant_tensors.get(name))
    };

    // Prefer sizes (input[3]) which directly specifies output dimensions.
    if let Some(sizes_name) = layer.inputs.get(3).filter(|n| !n.is_empty()) {
        if let Some(td) = lookup(sizes_name) {
            let sizes = td.as_i64_vec();
            if sizes.len() == rank {
                let out_shape: Vec<usize> = sizes
                    .iter()
                    .map(|&s| {
                        if s < 0 {
                            bail!("layer {}: Resize sizes[i] = {s} is negative", layer.name);
                        }
                        Ok(s as usize)
                    })
                    .collect::<Result<_>>()?;
                return Ok(layer
                    .outputs
                    .iter()
                    .map(|o| (o.clone(), out_shape.clone()))
                    .collect());
            }
        }
    }

    // Fall back to scales (input[2]).
    if let Some(scales_name) = layer.inputs.get(2).filter(|n| !n.is_empty()) {
        if let Some(td) = lookup(scales_name) {
            let scales = &td.float_data;
            if scales.len() == rank {
                let out_shape: Vec<usize> = input_shape
                    .iter()
                    .zip(scales.iter())
                    .enumerate()
                    .map(|(i, (&d, &s))| {
                        if s <= 0.0 {
                            bail!("layer {}: Resize scales[{i}] = {s} must be > 0", layer.name);
                        }
                        Ok((d as f64 * s).floor() as usize)
                    })
                    .collect::<Result<_>>()?;
                return Ok(layer
                    .outputs
                    .iter()
                    .map(|o| (o.clone(), out_shape.clone()))
                    .collect());
            }
        }
    }

    bail!(
        "layer {}: Resize cannot determine output shape — neither sizes (input[3]) \
        nor scales (input[2]) are available as compile-time constants",
        layer.name
    )
}

/// Infer output shape for the ONNX `GridSample` operator.
///
/// Inputs: X [N, C, H_in, W_in], grid [N, H_out, W_out, 2]
/// Output: [N, C, H_out, W_out]
fn infer_gridsample(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let x_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer {}: GridSample missing X input", layer.name))?;
    let grid_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: GridSample missing grid input", layer.name))?;

    let x_shape = get_shape(shapes, x_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: GridSample missing shape for X '{x_name}'",
            layer.name
        )
    })?;
    let grid_shape = get_shape(shapes, grid_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: GridSample missing shape for grid '{grid_name}'",
            layer.name
        )
    })?;

    if x_shape.len() != 4 {
        bail!(
            "layer {}: GridSample X must be 4-D [N,C,H,W], got {}D",
            layer.name,
            x_shape.len()
        );
    }
    if grid_shape.len() != 4 || grid_shape[3] != 2 {
        bail!(
            "layer {}: GridSample grid must be [N,H_out,W_out,2], got {:?}",
            layer.name,
            grid_shape
        );
    }

    if grid_shape[0] != x_shape[0] {
        bail!(
            "layer {}: GridSample batch size mismatch: X batch = {}, grid batch = {}",
            layer.name,
            x_shape[0],
            grid_shape[0]
        );
    }

    // Output: [N, C, H_out, W_out]
    let out_shape = vec![x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]];
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shape for the ONNX `Transpose` operator.
///
/// If `perm` is provided it must be a permutation of `[0, rank)`.
/// If absent, the axes are reversed (numpy default).
fn infer_transpose(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer {}: Transpose missing input", layer.name))?;
    let input_shape = get_shape(shapes, input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: Transpose missing shape for input '{input_name}'",
            layer.name
        )
    })?;

    let rank = input_shape.len();
    let perm: Vec<usize> = if let Some(raw) = layer.get_ints_attr("perm") {
        raw.iter()
            .enumerate()
            .map(|(i, &a)| {
                let n = if a < 0 { rank as i64 + a } else { a };
                if n < 0 || n as usize >= rank {
                    bail!(
                        "layer {}: Transpose perm[{i}]={a} out of range for rank {rank}",
                        layer.name
                    );
                }
                Ok(n as usize)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        (0..rank).rev().collect()
    };

    if perm.len() != rank {
        bail!(
            "layer {}: Transpose perm length {} != input rank {}",
            layer.name,
            perm.len(),
            rank
        );
    }

    let mut seen = HashSet::with_capacity(perm.len());
    for &p in &perm {
        if !seen.insert(p) {
            bail!(
                "layer {}: Transpose perm contains duplicate axis {}",
                layer.name,
                p
            );
        }
    }

    let out_shape: Vec<usize> = perm.iter().map(|&p| input_shape[p]).collect();
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shape for the ONNX `Concat` operator.
///
/// All inputs must have the same rank. The output shape matches the inputs on
/// every axis except `axis`, where the output dimension equals the sum of all
/// input dimensions.
fn infer_concat(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let first_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer {}: Concat has no inputs", layer.name))?;
    let first_shape = get_shape(shapes, first_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: Concat missing shape for first input '{first_name}'",
            layer.name
        )
    })?;
    let rank = first_shape.len();

    let raw_axis = layer.get_int_attr("axis").unwrap_or(0);
    let axis = if raw_axis < 0 {
        let a = rank as i64 + raw_axis;
        if a < 0 {
            bail!(
                "layer {}: Concat axis {raw_axis} out of range for rank {rank}",
                layer.name
            );
        }
        a as usize
    } else {
        raw_axis as usize
    };

    if axis >= rank {
        bail!("layer {}: Concat axis {axis} >= rank {rank}", layer.name);
    }

    let mut out_shape = first_shape.clone();
    for name in layer.inputs.iter().skip(1) {
        let s = get_shape(shapes, name).ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: Concat missing shape for input '{name}'",
                layer.name
            )
        })?;
        if s.len() != rank {
            bail!(
                "layer {}: Concat input rank mismatch: {} vs {rank}",
                layer.name,
                s.len()
            );
        }

        for i in 0..rank {
            if i == axis {
                continue;
            }
            if s[i] != first_shape[i] {
                bail!(
                    "layer {}: Concat non-axis dimension mismatch for input '{}': axis {} has {} vs {} in first input",
                    layer.name,
                    name,
                    i,
                    s[i],
                    first_shape[i]
                );
            }
        }
        out_shape[axis] = out_shape[axis].checked_add(s[axis]).ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: Concat axis {} size overflow for input '{}': {} + {}",
                layer.name,
                axis,
                name,
                out_shape[axis],
                s[axis]
            )
        })?;
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shape for the ONNX `Slice` operator (opset >= 10).
///
/// Extracts a sub-tensor by selecting ranges `[start, end)` with an optional
/// `step` along each specified axis.  All of `starts`, `ends`, `axes`, and
/// `steps` must be compile-time constants (initializers or Constant nodes).
fn infer_slice(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Slice missing input shape", layer.name))?;
    let rank = input_shape.len();

    // Helper: look up a required i64 tensor from initializers or constant nodes.
    let get_i64 = |name: &str| -> Option<Vec<i64>> {
        initializers
            .get(name)
            .or_else(|| constant_tensors.get(name))
            .map(|td| td.as_i64_vec())
    };

    let starts_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: Slice missing starts input", layer.name))?;
    let starts_opt = get_i64(starts_name);

    let ends_name = layer
        .inputs
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("layer {}: Slice missing ends input", layer.name))?;
    let ends_opt = get_i64(ends_name);

    let passthrough = || {
        Ok(layer
            .outputs
            .iter()
            .map(|o| (o.clone(), input_shape.clone()))
            .collect())
    };

    if starts_opt.is_none() || ends_opt.is_none() {
        return passthrough();
    }

    let starts = starts_opt.unwrap();
    let ends = ends_opt.unwrap();

    if starts.len() != ends.len() {
        bail!(
            "layer {}: Slice starts and ends must have the same length (starts={}, ends={})",
            layer.name,
            starts.len(),
            ends.len()
        );
    }

    let axes_from_input: Option<Vec<i64>> = match layer.inputs.get(3).filter(|n| !n.is_empty()) {
        Some(n) => match get_i64(n) {
            Some(v) => Some(v),
            None => return passthrough(),
        },
        None => None,
    };

    let steps_from_input: Option<Vec<i64>> = match layer.inputs.get(4).filter(|n| !n.is_empty()) {
        Some(n) => match get_i64(n) {
            Some(v) => Some(v),
            None => return passthrough(),
        },
        None => None,
    };

    // Validate lengths when axes/steps are explicitly provided.
    if let Some(ref ax) = axes_from_input {
        if ax.len() != starts.len() {
            bail!(
                "layer {}: Slice axes length must match starts length (axes={}, starts={})",
                layer.name,
                ax.len(),
                starts.len()
            );
        }
    }
    if let Some(ref st) = steps_from_input {
        if st.len() != starts.len() {
            bail!(
                "layer {}: Slice steps length must match starts length (steps={}, starts={})",
                layer.name,
                st.len(),
                starts.len()
            );
        }
    }

    let axes: Vec<i64> = axes_from_input.unwrap_or_else(|| (0..starts.len() as i64).collect());
    let steps: Vec<i64> = steps_from_input.unwrap_or_else(|| vec![1i64; starts.len()]);

    let mut out_shape = input_shape.clone();
    for (i, &axis_raw) in axes.iter().enumerate() {
        let axis = if axis_raw < 0 {
            let a = rank as i64 + axis_raw;
            if a < 0 {
                bail!(
                    "layer {}: Slice axis {} out of range for rank {}",
                    layer.name,
                    axis_raw,
                    rank
                );
            }
            a as usize
        } else {
            axis_raw as usize
        };

        if axis >= rank {
            bail!(
                "layer {}: Slice axis {} out of range for rank {}",
                layer.name,
                axis_raw,
                rank
            );
        }

        let dim = input_shape[axis] as i64;

        let start = {
            let s = starts.get(i).copied().unwrap_or(0);
            let s = if s < 0 { dim + s } else { s };
            s.clamp(0, dim) as usize
        };

        let end = {
            let e = ends.get(i).copied().unwrap_or(dim);
            let e = if e < 0 { dim + e } else { e };
            e.clamp(0, dim) as usize
        };

        let step = steps.get(i).copied().unwrap_or(1);
        if step <= 0 {
            bail!(
                "layer {}: Slice step must be positive for axis {}, got {}",
                layer.name,
                axis,
                step
            );
        }
        let step = step as usize;

        out_shape[axis] = if start < end {
            (end - start).div_ceil(step)
        } else {
            0
        };
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shapes for the ONNX `TopK` operator (opset ≥ 10).
///
/// Inputs: data (input[0]), K (input[1] — must be a compile-time constant).
/// Attributes: `axis` (default −1), `largest` (default 1), `sorted` (default 1).
/// Both outputs (Values at output[0], Indices at output[1]) share the same shape:
/// `input_shape` with the `axis` dimension replaced by K.
fn infer_topk(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: TopK missing input shape", layer.name))?;
    let rank = input_shape.len();

    let axis_raw = layer.get_int_attr("axis").unwrap_or(-1);
    let axis = if axis_raw < 0 {
        let a = rank as i64 + axis_raw;
        if a < 0 {
            bail!(
                "layer {}: TopK axis {} out of range for rank {}",
                layer.name,
                axis_raw,
                rank
            );
        }
        a as usize
    } else {
        let a = axis_raw as usize;
        if a >= rank {
            bail!(
                "layer {}: TopK axis {} out of range for rank {}",
                layer.name,
                axis_raw,
                rank
            );
        }
        a
    };

    let largest = layer.get_int_attr("largest").unwrap_or(1);
    if largest != 1 {
        bail!(
            "layer {}: TopK only supports largest=1, got largest={}",
            layer.name,
            largest
        );
    }

    // K is input[1]: must be a compile-time constant initializer or Constant node.
    let k_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: TopK missing K input at index 1", layer.name))?;
    let k_td = initializers
        .get(k_name)
        .or_else(|| constant_tensors.get(k_name))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: TopK K tensor '{}' not found in initializers; \
                 K must be a compile-time constant",
                layer.name,
                k_name
            )
        })?;
    let k_vec = k_td.as_i64_vec();
    if k_vec.len() != 1 {
        bail!(
            "layer {}: TopK K tensor '{}' must be a scalar (1 element), got {} elements",
            layer.name,
            k_name,
            k_vec.len()
        );
    }
    let k = k_vec[0];
    if k <= 0 {
        bail!("layer {}: TopK K must be positive, got {}", layer.name, k);
    }
    let k = k as usize;
    if k > input_shape[axis] {
        bail!(
            "layer {}: TopK K={} exceeds axis dimension {} on axis {}",
            layer.name,
            k,
            input_shape[axis],
            axis
        );
    }

    let mut out_shape = input_shape.clone();
    out_shape[axis] = k;

    // TopK has two outputs: Values (output[0]) and Indices (output[1]).
    // Both share the same shape.
    let mut results = Vec::new();
    if let Some(name) = layer.outputs.first() {
        results.push((name.clone(), out_shape.clone()));
    }
    if let Some(name) = layer.outputs.get(1) {
        results.push((name.clone(), out_shape.clone()));
    }
    Ok(results)
}

fn infer_shape_op(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Shape missing input shape", layer.name))?;
    let rank = input_shape.len() as i64;
    let start_raw = layer.get_int_attr("start").unwrap_or(0);
    let end_raw = layer.get_int_attr("end").unwrap_or(rank);
    let start = if start_raw < 0 {
        (rank + start_raw).max(0) as usize
    } else {
        (start_raw as usize).min(input_shape.len())
    };
    let end = if end_raw < 0 {
        (rank + end_raw).max(0) as usize
    } else {
        (end_raw as usize).min(input_shape.len())
    };
    let len = end.saturating_sub(start);
    let output_shape = vec![len];
    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), output_shape.clone()))
        .collect())
}

fn infer_expand(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let shape_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: Expand missing shape input", layer.name))?;

    // Target shape must be a compile-time constant (initializer or Constant node).
    let shape_td = initializers
        .get(shape_name)
        .or_else(|| constant_tensors.get(shape_name))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "layer {}: Expand shape input '{}' is not a compile-time constant (initializer or Constant node)",
                layer.name,
                shape_name
            )
        })?;

    let target_shape: Vec<usize> = shape_td
        .as_i64_vec()
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d < 0 {
                bail!("layer {}: Expand shape[{i}] is negative ({d})", layer.name);
            }
            Ok(d as usize)
        })
        .collect::<Result<Vec<usize>>>()?;

    // Validate broadcast compatibility with input shape.
    if let Some(input_shape) = layer.inputs.first().and_then(|n| get_shape(shapes, n)) {
        let in_rank = input_shape.len();
        let out_rank = target_shape.len();
        if in_rank > out_rank {
            bail!(
                "layer {}: Expand input rank {} > target shape rank {} (cannot broadcast)",
                layer.name,
                in_rank,
                out_rank
            );
        }
        // Check broadcast compatibility from the right.
        let pad = out_rank - in_rank;
        for (i, &out_d) in target_shape.iter().enumerate() {
            if i < pad {
                continue; // prepended dim — input treated as size 1
            }
            let in_d = input_shape[i - pad];
            if in_d != 1 && in_d != out_d {
                bail!(
                    "layer {}: Expand broadcast mismatch at dim {i}: input {} != target {}",
                    layer.name,
                    in_d,
                    out_d
                );
            }
        }
    }

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), target_shape.clone()))
        .collect())
}

fn infer_reduce(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: ReduceMean missing input shape", layer.name))?;
    let rank = input_shape.len();
    let keepdims = layer.get_int_attr("keepdims").unwrap_or(1) != 0;

    // axes: from input[1] (opset 18+) → attribute (opset ≤ 17) → all axes.
    // If input[1] is present but is not a compile-time constant, bail: dynamic
    // axes are not representable at shape-inference time.
    let axes_input_name: Option<&str> = layer
        .inputs
        .get(1)
        .filter(|n| !n.is_empty())
        .map(|n| n.as_str());

    let axes_from_input: Option<Vec<i64>> = axes_input_name
        .and_then(|name| {
            initializers
                .get(name)
                .or_else(|| constant_tensors.get(name))
        })
        .map(|td| td.as_i64_vec());

    if axes_input_name.is_some() && axes_from_input.is_none() {
        bail!(
            "layer {}: ReduceMean has a dynamic axes input ('{}') that is not a \
             compile-time constant; dynamic axes are not supported in shape inference",
            layer.name,
            axes_input_name.unwrap_or("")
        );
    }

    let axes: Vec<usize> = if let Some(raw_axes) = axes_from_input
        .as_deref()
        .or_else(|| layer.get_ints_attr("axes"))
    {
        raw_axes
            .iter()
            .map(|&a| {
                let a = if a < 0 { a + rank as i64 } else { a };
                if a < 0 || a as usize >= rank {
                    bail!(
                        "layer {}: ReduceMean axis {} out of range for rank {}",
                        layer.name,
                        a,
                        rank
                    );
                }
                Ok(a as usize)
            })
            .collect::<Result<Vec<usize>>>()?
    } else {
        (0..rank).collect()
    };

    let output_shape: Vec<usize> = if keepdims {
        input_shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
            .collect()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if axes.contains(&i) { None } else { Some(d) })
            .collect()
    };

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), output_shape.clone()))
        .collect())
}

/// Infer output shape for ONNX `MatMul`.
///
/// Supports 2-D matrix multiplication: [M, K] @ [K, N] → [M, N].
/// For N-D inputs follows numpy broadcasting rules but we keep it simple
/// and only handle the 2-D case in circuit (reject higher-rank at build time).
fn infer_matmul(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let a_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer {}: MatMul missing input A", layer.name))?;
    let b_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: MatMul missing input B", layer.name))?;

    let a_shape = get_shape(shapes, a_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: MatMul missing shape for A '{a_name}'",
            layer.name
        )
    })?;
    let b_shape = get_shape(shapes, b_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: MatMul missing shape for B '{b_name}'",
            layer.name
        )
    })?;

    if a_shape.len() < 2 || b_shape.len() < 2 {
        bail!(
            "layer {}: MatMul inputs must be at least 2-D (A rank={}, B rank={})",
            layer.name,
            a_shape.len(),
            b_shape.len()
        );
    }

    // For 2D: [M, K] @ [K, N] → [M, N]
    // For ND: batch dims are broadcast, last two dims do matmul.
    let m = a_shape[a_shape.len() - 2];
    let k_a = a_shape[a_shape.len() - 1];
    let k_b = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];

    if k_a != k_b {
        bail!(
            "layer {}: MatMul inner dimension mismatch: A.K={k_a} B.K={k_b}",
            layer.name
        );
    }

    // Batch dimensions: broadcast leading dims.
    let a_batch = &a_shape[..a_shape.len() - 2];
    let b_batch = &b_shape[..b_shape.len() - 2];
    let batch_shape = if a_batch.is_empty() && b_batch.is_empty() {
        vec![]
    } else if a_batch.is_empty() {
        b_batch.to_vec()
    } else if b_batch.is_empty() {
        a_batch.to_vec()
    } else {
        broadcast_shapes(a_batch, b_batch).map_err(|e| {
            anyhow::anyhow!("layer {}: MatMul batch broadcast failed: {e}", layer.name)
        })?
    };

    let mut out_shape = batch_shape;
    out_shape.push(m);
    out_shape.push(n);

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shape for ONNX `Pad`.
///
/// Pads are either from attribute `pads` (opset < 11) or from `inputs[1]`
/// (opset ≥ 11, must be a compile-time constant).
/// `pads` is a flat vector [x1_begin, x2_begin, ..., x1_end, x2_end, ...].
fn infer_pad(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Pad missing input shape", layer.name))?;
    let rank = input_shape.len();

    // Try to get pads from inputs[1] (opset 11+) then from attribute.
    let pads: Vec<i64> = if let Some(pads_name) = layer.inputs.get(1).filter(|n| !n.is_empty()) {
        initializers
            .get(pads_name)
            .or_else(|| constant_tensors.get(pads_name))
            .map(|td| td.as_i64_vec())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "layer {}: Pad pads input '{}' is not a compile-time constant",
                    layer.name,
                    pads_name
                )
            })?
    } else if let Some(v) = layer.get_ints_attr("pads") {
        v.to_vec()
    } else {
        bail!("layer {}: Pad has no pads attribute or input", layer.name);
    };

    if pads.len() != 2 * rank {
        bail!(
            "layer {}: Pad pads length {} != 2 * rank {}",
            layer.name,
            pads.len(),
            rank
        );
    }

    let out_shape: Vec<usize> = (0..rank)
        .map(|i| {
            let begin = pads[i];
            let end = pads[rank + i];
            if begin < 0 || end < 0 {
                bail!(
                    "layer {}: Pad negative pad values are not supported",
                    layer.name
                );
            }
            Ok(input_shape[i] + begin as usize + end as usize)
        })
        .collect::<Result<Vec<usize>>>()?;

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shapes for ONNX `Split`.
///
/// Splits input along an axis into multiple outputs.
/// Split sizes come from `inputs[1]` (opset 13+) or the `split` attribute,
/// or default to equal splits.
fn infer_split(
    layer: &LayerNode,
    input_shape: Option<&Vec<usize>>,
    initializers: &HashMap<String, TensorData>,
    constant_tensors: &HashMap<String, TensorData>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_shape = input_shape
        .ok_or_else(|| anyhow::anyhow!("layer {}: Split missing input shape", layer.name))?;
    let rank = input_shape.len();
    let num_outputs = layer.outputs.len();

    let axis_raw = layer.get_int_attr("axis").unwrap_or(0);
    let axis = if axis_raw < 0 {
        let a = rank as i64 + axis_raw;
        if a < 0 {
            bail!(
                "layer {}: Split axis {} out of range for rank {}",
                layer.name,
                axis_raw,
                rank
            );
        }
        a as usize
    } else {
        let a = axis_raw as usize;
        if a >= rank {
            bail!(
                "layer {}: Split axis {} out of range for rank {}",
                layer.name,
                axis_raw,
                rank
            );
        }
        a
    };

    let axis_dim = input_shape[axis];

    // Try to get split sizes from inputs[1] or from attribute.
    let split_sizes: Vec<usize> =
        if let Some(split_name) = layer.inputs.get(1).filter(|n| !n.is_empty()) {
            let vals = initializers
                .get(split_name)
                .or_else(|| constant_tensors.get(split_name))
                .map(|td| td.as_i64_vec())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "layer {}: Split split input '{}' is not a compile-time constant",
                        layer.name,
                        split_name
                    )
                })?;
            vals.iter().map(|&v| v as usize).collect()
        } else if let Some(v) = layer.get_ints_attr("split") {
            v.iter().map(|&v| v as usize).collect()
        } else {
            // Equal split.
            if num_outputs == 0 {
                bail!("layer {}: Split has no outputs", layer.name);
            }
            if axis_dim % num_outputs != 0 {
                bail!(
                    "layer {}: Split axis dim {} not divisible by num_outputs {}",
                    layer.name,
                    axis_dim,
                    num_outputs
                );
            }
            vec![axis_dim / num_outputs; num_outputs]
        };

    if split_sizes.len() != num_outputs {
        bail!(
            "layer {}: Split split_sizes length {} != num_outputs {}",
            layer.name,
            split_sizes.len(),
            num_outputs
        );
    }

    let results: Result<Vec<_>> = layer
        .outputs
        .iter()
        .zip(split_sizes.iter())
        .map(|(out, &sz)| {
            let mut out_shape = input_shape.clone();
            out_shape[axis] = sz;
            Ok((out.clone(), out_shape))
        })
        .collect();
    results
}

/// Infer output shape for ONNX `Where`.
///
/// Output shape is the broadcast of all three inputs: condition, X, Y.
fn infer_where(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let cond_shape = layer.inputs.first().and_then(|n| get_shape(shapes, n));
    let x_shape = layer.inputs.get(1).and_then(|n| get_shape(shapes, n));
    let y_shape = layer.inputs.get(2).and_then(|n| get_shape(shapes, n));

    let out_shape = match (cond_shape, x_shape, y_shape) {
        (Some(c), Some(x), Some(y)) => {
            let cx = broadcast_shapes(c, x).map_err(|e| {
                anyhow::anyhow!("layer {}: Where broadcast cond/X: {e}", layer.name)
            })?;
            broadcast_shapes(&cx, y).map_err(|e| {
                anyhow::anyhow!("layer {}: Where broadcast cond-x/Y: {e}", layer.name)
            })?
        }
        (Some(s), _, _) | (_, Some(s), _) | (_, _, Some(s)) => s.clone(),
        (None, None, None) => bail!("layer {}: Where has no input shapes", layer.name),
    };

    Ok(layer
        .outputs
        .iter()
        .map(|o| (o.clone(), out_shape.clone()))
        .collect())
}

/// Infer output shape for ONNX `ConvTranspose`.
///
/// Formula: output_dim = stride * (input_dim - 1) + kernel_dim - 2*pad + output_padding
fn infer_conv_transpose(
    layer: &LayerNode,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<Vec<(String, Vec<usize>)>> {
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer {}: ConvTranspose missing input", layer.name))?;
    let weight_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("layer {}: ConvTranspose missing weight", layer.name))?;

    let input_shape = get_shape(shapes, input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: ConvTranspose missing shape for input '{input_name}'",
            layer.name
        )
    })?;
    let weight_shape = get_shape(shapes, weight_name).ok_or_else(|| {
        anyhow::anyhow!(
            "layer {}: ConvTranspose missing shape for weight '{weight_name}'",
            layer.name
        )
    })?;

    if weight_shape.len() < 2 {
        bail!(
            "layer {}: ConvTranspose weight rank {} < 2",
            layer.name,
            weight_shape.len()
        );
    }

    // Weight shape for ConvTranspose: [C_in, C_out/group, *kernel]
    let c_out_per_group = weight_shape[1];
    let group = layer.get_int_attr("group").unwrap_or(1) as usize;
    let c_out = c_out_per_group * group;

    let kernel_shape = if let Some(v) = layer.get_ints_attr("kernel_shape") {
        nonneg_to_usize(v, "kernel_shape", &layer.name)?
    } else if weight_shape.len() >= 3 {
        weight_shape[2..].to_vec()
    } else {
        vec![1]
    };

    let spatial_dims = kernel_shape.len();

    let strides = if let Some(v) = layer.get_ints_attr("strides") {
        nonneg_to_usize(v, "strides", &layer.name)?
    } else {
        vec![1; spatial_dims]
    };

    let pads = if let Some(v) = layer.get_ints_attr("pads") {
        nonneg_to_usize(v, "pads", &layer.name)?
    } else {
        vec![0; spatial_dims * 2]
    };

    let output_padding = if let Some(v) = layer.get_ints_attr("output_padding") {
        nonneg_to_usize(v, "output_padding", &layer.name)?
    } else {
        vec![0; spatial_dims]
    };

    let dilations = if let Some(v) = layer.get_ints_attr("dilations") {
        nonneg_to_usize(v, "dilations", &layer.name)?
    } else {
        vec![1; spatial_dims]
    };

    if input_shape.len() < 2 + spatial_dims {
        bail!(
            "layer {}: ConvTranspose input rank {} too small for {spatial_dims} spatial dims",
            layer.name,
            input_shape.len()
        );
    }

    // Check if output_shape attribute is provided (overrides computed shape).
    if let Some(out_shape_attr) = layer.get_ints_attr("output_shape") {
        let spatial: Vec<usize> = nonneg_to_usize(out_shape_attr, "output_shape", &layer.name)?;
        let mut out_shape = vec![input_shape[0], c_out];
        out_shape.extend_from_slice(&spatial);
        return Ok(layer
            .outputs
            .iter()
            .map(|o| (o.clone(), out_shape.clone()))
            .collect());
    }

    let mut out_shape = vec![input_shape[0], c_out];
    for i in 0..spatial_dims {
        let in_dim = input_shape[2 + i];
        let pad = if pads.len() >= 2 * spatial_dims {
            pads[i] + pads[spatial_dims + i]
        } else {
            0
        };
        let out_pad = output_padding.get(i).copied().unwrap_or(0);
        let d = dilations.get(i).copied().unwrap_or(1);
        let effective_kernel = (kernel_shape[i] - 1) * d + 1;
        let out_dim = strides[i] * (in_dim - 1) + effective_kernel - pad + out_pad;
        out_shape.push(out_dim);
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

    fn make_layer(
        op_type: OpType,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
        attrs: HashMap<String, AttrValue>,
    ) -> LayerNode {
        LayerNode {
            id: 0,
            name: "test_layer".to_string(),
            op_type,
            inputs: inputs.into_iter().map(String::from).collect(),
            outputs: outputs.into_iter().map(String::from).collect(),
            weights: HashMap::new(),
            attributes: attrs,
            output_shape: vec![],
            needs_rescale: false,
            n_bits: None,
        }
    }

    #[test]
    fn conv_1d_kernel_from_weights() {
        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), vec![1, 3, 10]);
        shapes.insert("w".to_string(), vec![8, 3, 3]);
        let layer = make_layer(OpType::Conv, vec!["x", "w"], vec!["y"], HashMap::new());
        let result = infer_conv(&layer, &shapes).unwrap();
        assert_eq!(result[0].1, vec![1, 8, 8]);
    }

    #[test]
    fn conv_1d_with_padding() {
        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), vec![1, 1, 5]);
        shapes.insert("w".to_string(), vec![1, 1, 3]);
        let mut attrs = HashMap::new();
        attrs.insert("pads".to_string(), AttrValue::Ints(vec![1, 1]));
        let layer = make_layer(OpType::Conv, vec!["x", "w"], vec!["y"], attrs);
        let result = infer_conv(&layer, &shapes).unwrap();
        assert_eq!(result[0].1, vec![1, 1, 5]);
    }

    #[test]
    fn reshape_allowzero_literal() {
        let input_shape = vec![0, 3, 4];
        let mut attrs = HashMap::new();
        attrs.insert("allowzero".to_string(), AttrValue::Int(1));
        let layer = make_layer(OpType::Reshape, vec!["x", "shape"], vec!["y"], attrs);
        let inits = HashMap::new();
        let mut consts = HashMap::new();
        consts.insert(
            "shape".to_string(),
            TensorData {
                name: "shape".to_string(),
                dims: vec![2],
                data_type: 7,
                float_data: vec![],
                int_data: vec![0, 12],
            },
        );
        let result = infer_reshape(&layer, Some(&input_shape), &inits, &consts).unwrap();
        assert_eq!(result[0].1, vec![0, 12]);
    }

    #[test]
    fn reshape_allowzero_with_minus_one_rejected() {
        let input_shape = vec![2, 3];
        let mut attrs = HashMap::new();
        attrs.insert("allowzero".to_string(), AttrValue::Int(1));
        let layer = make_layer(OpType::Reshape, vec!["x", "shape"], vec!["y"], attrs);
        let inits = HashMap::new();
        let mut consts = HashMap::new();
        consts.insert(
            "shape".to_string(),
            TensorData {
                name: "shape".to_string(),
                dims: vec![2],
                data_type: 7,
                float_data: vec![],
                int_data: vec![0, -1],
            },
        );
        assert!(infer_reshape(&layer, Some(&input_shape), &inits, &consts).is_err());
    }

    #[test]
    fn flatten_axis_zero() {
        let input_shape = vec![2, 3, 4];
        let layer = make_layer(OpType::Flatten, vec!["x"], vec!["y"], HashMap::new());
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttrValue::Int(0));
        let layer = LayerNode {
            attributes: attrs,
            ..layer
        };
        let result = infer_flatten(&layer, Some(&input_shape)).unwrap();
        assert_eq!(result[0].1, vec![1, 24]);
    }

    #[test]
    fn flatten_axis_eq_rank() {
        let input_shape = vec![2, 3, 4];
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttrValue::Int(3));
        let layer = make_layer(OpType::Flatten, vec!["x"], vec!["y"], attrs);
        let result = infer_flatten(&layer, Some(&input_shape)).unwrap();
        assert_eq!(result[0].1, vec![24, 1]);
    }

    #[test]
    fn flatten_negative_axis() {
        let input_shape = vec![2, 3, 4];
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttrValue::Int(-1));
        let layer = make_layer(OpType::Flatten, vec!["x"], vec!["y"], attrs);
        let result = infer_flatten(&layer, Some(&input_shape)).unwrap();
        assert_eq!(result[0].1, vec![6, 4]);
    }

    #[test]
    fn flatten_axis_out_of_range() {
        let input_shape = vec![2, 3];
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttrValue::Int(5));
        let layer = make_layer(OpType::Flatten, vec!["x"], vec!["y"], attrs);
        assert!(infer_flatten(&layer, Some(&input_shape)).is_err());
    }

    #[test]
    fn gemm_inner_dim_mismatch() {
        let mut shapes = HashMap::new();
        shapes.insert("a".to_string(), vec![3, 4]);
        shapes.insert("b".to_string(), vec![5, 6]);
        let layer = make_layer(OpType::Gemm, vec!["a", "b"], vec!["y"], HashMap::new());
        assert!(infer_gemm(&layer, &shapes).is_err());
    }

    #[test]
    fn squeeze_rejects_non_one_dim() {
        let input_shape = vec![2, 3, 1];
        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), AttrValue::Ints(vec![1]));
        let layer = make_layer(OpType::Squeeze, vec!["x"], vec!["y"], attrs);
        assert!(
            infer_squeeze(&layer, Some(&input_shape), &HashMap::new(), &HashMap::new()).is_err()
        );
    }

    #[test]
    fn unsqueeze_rejects_duplicate_axes() {
        let input_shape = vec![2, 3];
        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), AttrValue::Ints(vec![0, 0]));
        let layer = make_layer(OpType::Unsqueeze, vec!["x"], vec!["y"], attrs);
        assert!(
            infer_unsqueeze(&layer, Some(&input_shape), &HashMap::new(), &HashMap::new()).is_err()
        );
    }

    #[test]
    fn unsqueeze_reads_axes_from_input() {
        let input_shape = vec![1, 300, 64];
        let attrs = HashMap::new();
        let layer = make_layer(
            OpType::Unsqueeze,
            vec!["data", "axes_tensor"],
            vec!["out"],
            attrs,
        );
        let mut inits = HashMap::new();
        inits.insert(
            "axes_tensor".to_string(),
            TensorData {
                dims: vec![1],
                float_data: vec![],
                int_data: vec![3],
                data_type: 7,
                name: "axes_tensor".to_string(),
            },
        );
        let result = infer_unsqueeze(&layer, Some(&input_shape), &inits, &HashMap::new()).unwrap();
        assert_eq!(result[0].1, vec![1, 300, 64, 1]);
    }

    #[test]
    fn gemm_batched_3d_rejected() {
        let mut shapes = HashMap::new();
        shapes.insert("a".to_string(), vec![1, 300, 64]);
        shapes.insert("b".to_string(), vec![64, 64]);
        let layer = make_layer(OpType::Gemm, vec!["a", "b"], vec!["y"], HashMap::new());
        assert!(infer_gemm(&layer, &shapes).is_err());
    }

    #[test]
    fn gemm_2d_unchanged() {
        let mut shapes = HashMap::new();
        shapes.insert("a".to_string(), vec![4, 3]);
        shapes.insert("b".to_string(), vec![3, 5]);
        let layer = make_layer(OpType::Gemm, vec!["a", "b"], vec!["y"], HashMap::new());
        let result = infer_gemm(&layer, &shapes).unwrap();
        assert_eq!(result[0].1, vec![4, 5]);
    }

    #[test]
    fn slice_dynamic_bounds_passthrough() {
        let input_shape = vec![1, 300, 64, 8];
        let layer = make_layer(
            OpType::Slice,
            vec!["data", "dyn_starts", "dyn_ends"],
            vec!["out"],
            HashMap::new(),
        );
        let result =
            infer_slice(&layer, Some(&input_shape), &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(result[0].1, vec![1, 300, 64, 8]);
    }

    #[test]
    fn squeeze_reads_axes_from_input() {
        let input_shape = vec![1, 300, 1, 64];
        let attrs = HashMap::new();
        let layer = make_layer(
            OpType::Squeeze,
            vec!["data", "axes_tensor"],
            vec!["out"],
            attrs,
        );
        let mut inits = HashMap::new();
        inits.insert(
            "axes_tensor".to_string(),
            TensorData {
                dims: vec![1],
                float_data: vec![],
                int_data: vec![2],
                data_type: 7,
                name: "axes_tensor".to_string(),
            },
        );
        let result = infer_squeeze(&layer, Some(&input_shape), &inits, &HashMap::new()).unwrap();
        assert_eq!(result[0].1, vec![1, 300, 64]);
    }

    #[test]
    fn fold_shape_produces_dimension_values() {
        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), vec![1, 3, 224, 224]);
        let layer = make_layer(OpType::Shape, vec!["x"], vec!["x_shape"], HashMap::new());
        let result = fold_shape(&layer, &shapes).unwrap();
        assert_eq!(result.int_data, vec![1, 3, 224, 224]);
        assert_eq!(result.dims, vec![4]);
    }

    #[test]
    fn fold_gather_scalar_index_from_shape() {
        let mut constants = HashMap::new();
        constants.insert(
            "shape_out".to_string(),
            TensorData {
                name: "shape_out".to_string(),
                dims: vec![4],
                data_type: 7,
                float_data: vec![],
                int_data: vec![1, 3, 224, 224],
            },
        );
        constants.insert(
            "idx".to_string(),
            TensorData {
                name: "idx".to_string(),
                dims: vec![],
                data_type: 7,
                float_data: vec![],
                int_data: vec![2],
            },
        );
        let layer = make_layer(
            OpType::Gather,
            vec!["shape_out", "idx"],
            vec!["dim2"],
            HashMap::new(),
        );
        let result = fold_gather(&layer, &HashMap::new(), &constants).unwrap();
        assert_eq!(result.int_data, vec![224]);
    }

    #[test]
    fn fold_unsqueeze_wraps_scalar() {
        let mut constants = HashMap::new();
        constants.insert(
            "val".to_string(),
            TensorData {
                name: "val".to_string(),
                dims: vec![],
                data_type: 7,
                float_data: vec![],
                int_data: vec![42],
            },
        );
        constants.insert(
            "axes".to_string(),
            TensorData {
                name: "axes".to_string(),
                dims: vec![1],
                data_type: 7,
                float_data: vec![],
                int_data: vec![0],
            },
        );
        let layer = make_layer(
            OpType::Unsqueeze,
            vec!["val", "axes"],
            vec!["out"],
            HashMap::new(),
        );
        let result = fold_unsqueeze(&layer, &HashMap::new(), &constants).unwrap();
        assert_eq!(result.int_data, vec![42]);
        assert_eq!(result.dims, vec![1]);
    }

    #[test]
    fn fold_concat_1d_tensors() {
        let mut constants = HashMap::new();
        constants.insert(
            "a".to_string(),
            TensorData {
                name: "a".to_string(),
                dims: vec![1],
                data_type: 7,
                float_data: vec![],
                int_data: vec![1],
            },
        );
        constants.insert(
            "b".to_string(),
            TensorData {
                name: "b".to_string(),
                dims: vec![1],
                data_type: 7,
                float_data: vec![],
                int_data: vec![3],
            },
        );
        constants.insert(
            "c".to_string(),
            TensorData {
                name: "c".to_string(),
                dims: vec![1],
                data_type: 7,
                float_data: vec![],
                int_data: vec![224],
            },
        );
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttrValue::Int(0));
        let layer = make_layer(OpType::Concat, vec!["a", "b", "c"], vec!["out"], attrs);
        let result = fold_concat(&layer, &HashMap::new(), &constants).unwrap();
        assert_eq!(result.int_data, vec![1, 3, 224]);
        assert_eq!(result.dims, vec![3]);
    }

    #[test]
    fn fold_shape_gather_unsqueeze_concat_reshape_chain() {
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        shapes.insert("input".to_string(), vec![1, 512, 7, 7]);

        let mut constants: HashMap<String, TensorData> = HashMap::new();
        constants.insert(
            "idx0".to_string(),
            TensorData {
                name: "idx0".to_string(),
                dims: vec![],
                data_type: 7,
                float_data: vec![],
                int_data: vec![0],
            },
        );
        constants.insert(
            "minus_one".to_string(),
            TensorData {
                name: "minus_one".to_string(),
                dims: vec![1],
                data_type: 7,
                float_data: vec![],
                int_data: vec![-1],
            },
        );
        constants.insert(
            "unsqueeze_axes".to_string(),
            TensorData {
                name: "unsqueeze_axes".to_string(),
                dims: vec![1],
                data_type: 7,
                float_data: vec![],
                int_data: vec![0],
            },
        );

        let shape_layer = make_layer(
            OpType::Shape,
            vec!["input"],
            vec!["input_shape"],
            HashMap::new(),
        );
        let shape_shapes = infer_shape_op(&shape_layer, Some(&vec![1, 512, 7, 7])).unwrap();
        for (n, s) in &shape_shapes {
            shapes.insert(n.clone(), s.clone());
        }
        fold_constants(&shape_layer, &shapes, &HashMap::new(), &mut constants);
        assert_eq!(constants["input_shape"].int_data, vec![1, 512, 7, 7]);

        let gather_layer = make_layer(
            OpType::Gather,
            vec!["input_shape", "idx0"],
            vec!["batch_dim"],
            HashMap::new(),
        );
        fold_constants(&gather_layer, &shapes, &HashMap::new(), &mut constants);
        assert_eq!(constants["batch_dim"].int_data, vec![1]);

        let unsqueeze_layer = make_layer(
            OpType::Unsqueeze,
            vec!["batch_dim", "unsqueeze_axes"],
            vec!["batch_1d"],
            HashMap::new(),
        );
        fold_constants(&unsqueeze_layer, &shapes, &HashMap::new(), &mut constants);
        assert_eq!(constants["batch_1d"].int_data, vec![1]);
        assert_eq!(constants["batch_1d"].dims, vec![1]);

        let mut concat_attrs = HashMap::new();
        concat_attrs.insert("axis".to_string(), AttrValue::Int(0));
        let concat_layer = make_layer(
            OpType::Concat,
            vec!["batch_1d", "minus_one"],
            vec!["new_shape"],
            concat_attrs,
        );
        fold_constants(&concat_layer, &shapes, &HashMap::new(), &mut constants);
        assert_eq!(constants["new_shape"].int_data, vec![1, -1]);
        assert_eq!(constants["new_shape"].dims, vec![2]);
    }
}
