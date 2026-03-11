use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};

use crate::gadgets::rescale;
use crate::onnx::graph::OpType;
use crate::onnx::quantizer::QuantizedModel;
use crate::padding::{next_power_of_two, num_vars_for};
use crate::runner::circuit_builder::{
    delta_table_nv, pad_matrix, pad_to_size, transpose_matrix, SpatialInfo, RANGE_CHECK_CHUNK_BITS,
};

fn validate_input_size(model: &QuantizedModel, input_name: &str, input_len: usize) -> Result<()> {
    if let Some(expected_shape) = model.graph.input_shapes.get(input_name) {
        let expected_size = expected_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "input shape overflow for '{input_name}': dimensions {expected_shape:?} exceed usize",
                )
            })?;
        anyhow::ensure!(
            input_len == expected_size,
            "input size mismatch for '{input_name}': model expects shape {expected_shape:?} ({expected_size} elements) but received {input_len} elements",
        );
    }
    Ok(())
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct WitnessData {
    pub shreds: HashMap<String, Vec<i64>>,
    #[serde(default)]
    pub observed_n_bits: HashMap<String, usize>,
}

pub fn compute_multiplicities(values: &[i64], table_size: usize) -> Result<Vec<i64>> {
    let mut mults = vec![0i64; table_size];
    for (i, &v) in values.iter().enumerate() {
        anyhow::ensure!(
            v >= 0 && (v as usize) < table_size,
            "range check: value at index {i} is {v} which is outside table range [0, {table_size})"
        );
        mults[v as usize] += 1;
    }
    Ok(mults)
}

fn observed_n_bits_for_delta(max_val: u64, exponent: usize) -> usize {
    if max_val == 0 {
        return exponent + 1;
    }
    let bits_needed = 64 - max_val.leading_zeros() as usize;
    bits_needed + exponent
}

pub fn run(model_path: &Path, input_path: &Path, output_path: &Path, compress: bool) -> Result<()> {
    tracing::info!("loading model from {}", model_path.display());
    let model = super::compile::load_model(model_path)?;

    let quantized_input = load_and_quantize_input(input_path, model.scale_config.alpha)?;

    tracing::info!("computing witness for {} layers", model.graph.layers.len());
    let witness_data = compute_witness(&model, &quantized_input)?;

    let size = jstprove_io::serialize_to_file(&witness_data, output_path, compress)?;
    tracing::info!(
        "witness written to {} ({} shreds, {} observed_n_bits, {} bytes)",
        output_path.display(),
        witness_data.shreds.len(),
        witness_data.observed_n_bits.len(),
        size
    );
    Ok(())
}

pub fn load_and_quantize_input(input_path: &Path, alpha: i64) -> Result<Vec<i64>> {
    let raw = std::fs::read(input_path)?;
    let input_value: rmpv::Value = jstprove_io::deserialize_from_bytes(&raw)?;
    quantize_input_value(&input_value, alpha)
}

fn get_map_field<'a>(map: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
    map.as_map().and_then(|m| {
        m.iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v)
    })
}

pub fn quantize_input_value(input_value: &rmpv::Value, alpha: i64) -> Result<Vec<i64>> {
    let raw_input: Vec<f64> = get_map_field(input_value, "input")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("input msgpack must have an \"input\" array field"))?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_f64()
                .ok_or_else(|| anyhow::anyhow!("input[{i}] is not a number: {v}"))
        })
        .collect::<Result<Vec<f64>>>()?;

    Ok(raw_input
        .iter()
        .map(|&v| (v * alpha as f64).round() as i64)
        .collect())
}

pub fn load_witness(path: &Path) -> Result<WitnessData> {
    jstprove_io::deserialize_from_file(path)
}

// -------- Resize coordinate helpers --------

/// Convert a normalised grid coordinate in [-1, 1] to a continuous pixel
/// coordinate (GridSample convention).
fn gs_unnormalize(norm: f64, size: usize, align_corners: bool) -> f64 {
    if align_corners {
        (norm + 1.0) / 2.0 * (size.saturating_sub(1) as f64)
    } else {
        (norm + 1.0) / 2.0 * size as f64 - 0.5
    }
}

/// Nearest-neighbour GridSample: resolve a continuous pixel coordinate under
/// the given padding mode.  Returns `None` for zeros-padding out-of-bounds.
fn gs_apply_padding_nearest(
    x: f64,
    size: usize,
    padding_mode: &str,
    align_corners: bool,
) -> Option<usize> {
    let pixel = match padding_mode {
        "zeros" => {
            if x < -0.5 || x > size as f64 - 0.5 {
                return None;
            }
            (x + 0.5)
                .floor()
                .clamp(0.0, (size.saturating_sub(1)) as f64) as usize
        }
        "border" => (x + 0.5)
            .floor()
            .clamp(0.0, (size.saturating_sub(1)) as f64) as usize,
        "reflection" => {
            let reflected = gs_reflect(x, size, align_corners);
            (reflected + 0.5)
                .floor()
                .clamp(0.0, (size.saturating_sub(1)) as f64) as usize
        }
        _ => (x + 0.5)
            .floor()
            .clamp(0.0, (size.saturating_sub(1)) as f64) as usize,
    };
    Some(pixel)
}

/// Reflect a continuous pixel coordinate at the boundary for GridSample.
fn gs_reflect(x: f64, size: usize, align_corners: bool) -> f64 {
    if size <= 1 {
        return 0.0;
    }
    let (lo, range) = if align_corners {
        (0.0f64, (size - 1) as f64)
    } else {
        (-0.5f64, size as f64)
    };
    let period = 2.0 * range;
    let mut rel = (x - lo).rem_euclid(period);
    if rel > range {
        rel = period - rel;
    }
    (rel + lo).clamp(0.0, (size - 1) as f64)
}

fn unravel_index_witness(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for i in (0..shape.len()).rev() {
        if shape[i] > 0 {
            coords[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
    coords
}

fn ravel_index_witness(coords: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        flat += coords[i] * stride;
        stride *= shape[i];
    }
    flat
}

fn coord_to_input(out_idx: usize, in_size: usize, out_size: usize, mode: &str) -> f64 {
    let o = out_idx as f64;
    let in_f = in_size as f64;
    let out_f = out_size as f64;
    match mode {
        "half_pixel" => (o + 0.5) * in_f / out_f - 0.5,
        "asymmetric" => o * in_f / out_f,
        "align_corners" => {
            if out_size <= 1 {
                0.0
            } else {
                o * (in_f - 1.0) / (out_f - 1.0)
            }
        }
        "pytorch_half_pixel" => {
            if out_size > 1 {
                (o + 0.5) * in_f / out_f - 0.5
            } else {
                0.0
            }
        }
        "tf_half_pixel_for_nn" => (o + 0.5) * in_f / out_f,
        _ => o * in_f / out_f,
    }
}

fn nearest_round(x: f64, in_size: usize, mode: &str) -> usize {
    let idx: i64 = match mode {
        "round_prefer_floor" => {
            let f = x.floor();
            if x - f == 0.5 {
                f as i64
            } else {
                x.round() as i64
            }
        }
        "round_prefer_ceil" => {
            let f = x.floor();
            if x - f == 0.5 {
                x.ceil() as i64
            } else {
                x.round() as i64
            }
        }
        "floor" => x.floor() as i64,
        "ceil" => x.ceil() as i64,
        _ => x.floor() as i64,
    };
    idx.clamp(0, in_size as i64 - 1) as usize
}

/// Returns (floor_idx, ceil_idx, weight_floor, weight_ceil).
fn interp_corners(x: f64, in_size: usize) -> (usize, usize, f64, f64) {
    let xc = x.clamp(0.0, (in_size.saturating_sub(1)) as f64);
    let f = xc.floor() as usize;
    let c = (xc.ceil() as usize).min(in_size - 1);
    let frac = xc - xc.floor();
    (f, c, 1.0 - frac, frac)
}

fn layout_from_output_shape_runtime(
    output_shape: &[usize],
    input_layout: Option<&SpatialInfo>,
) -> Option<SpatialInfo> {
    let chw = if output_shape.len() == 3 {
        output_shape
    } else if output_shape.len() == 4 {
        &output_shape[1..]
    } else {
        return None;
    };

    match input_layout {
        Some(SpatialInfo::HWC { .. }) => {
            let c = chw[0];
            Some(SpatialInfo::HWC {
                h: chw[1],
                w: chw[2],
                c,
                stride_c: next_power_of_two(c),
            })
        }
        _ => Some(SpatialInfo::CHW {
            c: chw[0],
            h: chw[1],
            w: chw[2],
        }),
    }
}

fn transpose_layout_runtime(
    input_layout: Option<&SpatialInfo>,
    perm: &[usize],
    output_shape: &[usize],
) -> Option<SpatialInfo> {
    let layout = input_layout?;

    if output_shape.len() == 3 && perm.len() == 3 {
        match layout {
            SpatialInfo::CHW { c, h, w } => {
                let in_dims = [*c, *h, *w];
                if perm.iter().any(|&p| p >= 3) {
                    return None;
                }
                let out_dims = [in_dims[perm[0]], in_dims[perm[1]], in_dims[perm[2]]];
                Some(SpatialInfo::CHW {
                    c: out_dims[0],
                    h: out_dims[1],
                    w: out_dims[2],
                })
            }
            SpatialInfo::HWC { h, w, c, .. } => {
                let in_dims = [*h, *w, *c];
                if perm.iter().any(|&p| p >= 3) {
                    return None;
                }
                let out_dims = [in_dims[perm[0]], in_dims[perm[1]], in_dims[perm[2]]];
                Some(SpatialInfo::HWC {
                    h: out_dims[0],
                    w: out_dims[1],
                    c: out_dims[2],
                    stride_c: next_power_of_two(out_dims[2]),
                })
            }
        }
    } else if output_shape.len() == 4 && perm.len() == 4 {
        let find_out_axis = |src_axis: usize| perm.iter().position(|&p| p == src_axis);
        match layout {
            SpatialInfo::CHW { .. } => {
                let oc = find_out_axis(1)?;
                let oh = find_out_axis(2)?;
                let ow = find_out_axis(3)?;
                Some(SpatialInfo::CHW {
                    c: output_shape[oc],
                    h: output_shape[oh],
                    w: output_shape[ow],
                })
            }
            SpatialInfo::HWC { .. } => {
                let oh = find_out_axis(1)?;
                let ow = find_out_axis(2)?;
                let oc = find_out_axis(3)?;
                let c = output_shape[oc];
                Some(SpatialInfo::HWC {
                    h: output_shape[oh],
                    w: output_shape[ow],
                    c,
                    stride_c: next_power_of_two(c),
                })
            }
        }
    } else {
        None
    }
}

// -------- End Resize helpers --------

pub fn compute_witness(model: &QuantizedModel, quantized_input: &[i64]) -> Result<WitnessData> {
    anyhow::ensure!(
        model.scale_config.base == 2,
        "range check multiplicities require base == 2, got base = {}",
        model.scale_config.base
    );
    anyhow::ensure!(
        model.scale_config.exponent < 63,
        "scale_config.exponent {} is too large for range table construction",
        model.scale_config.exponent
    );

    let alpha = model.scale_config.alpha;
    let exponent = model.scale_config.exponent as usize;
    let offset = 1i64 << 30;

    let mut shreds: HashMap<String, Vec<i64>> = HashMap::new();
    let mut tensors: HashMap<String, Vec<i64>> = HashMap::new();
    let mut tensor_layouts: HashMap<String, SpatialInfo> = HashMap::new();
    let mut observed_n_bits: HashMap<String, usize> = HashMap::new();

    let input_name = model
        .graph
        .input_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no input names defined"))?
        .clone();

    let input_size = quantized_input.len();
    validate_input_size(model, &input_name, input_size)?;

    let input_padded_size = next_power_of_two(input_size);
    let input_padded = pad_to_size(quantized_input, input_padded_size);

    shreds.insert(input_name.clone(), input_padded.clone());
    tensors.insert(input_name.clone(), input_padded);

    for (name, shape) in &model.graph.input_shapes {
        if shape.len() == 3 {
            tensor_layouts.insert(
                name.clone(),
                SpatialInfo::CHW {
                    c: shape[0],
                    h: shape[1],
                    w: shape[2],
                },
            );
        }
    }

    anyhow::ensure!(
        model.graph.output_names.len() == 1,
        "expected exactly 1 output, found {} ({:?})",
        model.graph.output_names.len(),
        model.graph.output_names
    );
    let declared_output = &model.graph.output_names[0];
    let rescale_table_size = 1usize << model.scale_config.exponent;

    let tensor_shape_for = |tensor_name: &str| -> Option<Vec<usize>> {
        if let Some(shape) = model.graph.input_shapes.get(tensor_name) {
            if shape.len() == 3 {
                let mut with_batch = vec![1usize];
                with_batch.extend(shape.iter().copied());
                return Some(with_batch);
            }
            return Some(shape.clone());
        }
        model
            .graph
            .layers
            .iter()
            .find(|l| l.outputs.iter().any(|o| o == tensor_name))
            .map(|l| l.output_shape.clone())
    };

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} has no input", layer.name))?;
                let weight_tensor_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} has no weight", layer.name))?;
                let bias_tensor_name = layer.inputs.get(2);

                let input_data = tensors.get(input_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Gemm {} input {} not computed",
                        layer.name,
                        input_tensor_name
                    )
                })?;

                let weight_data = layer.weights.get(weight_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("Gemm {} missing weight {}", layer.name, weight_tensor_name)
                })?;

                let trans_a = layer
                    .get_int_attr("transA")
                    .map(|v| v != 0)
                    .unwrap_or(false);
                anyhow::ensure!(
                    !trans_a,
                    "Gemm {} has transA=1 which is not supported",
                    layer.name
                );
                let trans_b = layer
                    .get_int_attr("transB")
                    .map(|v| v != 0)
                    .unwrap_or(false);

                let w_shape = weight_data.shape();
                anyhow::ensure!(
                    w_shape.len() >= 2,
                    "Gemm {} weight has {} dims, need >= 2",
                    layer.name,
                    w_shape.len()
                );
                let (w_rows, w_cols) = (w_shape[0], w_shape[1]);
                let (k_dim, n_dim) = if trans_b {
                    (w_cols, w_rows)
                } else {
                    (w_rows, w_cols)
                };

                let k_padded = next_power_of_two(k_dim);
                let n_padded = next_power_of_two(n_dim);

                let w_transposed = if trans_b {
                    transpose_matrix(&weight_data.as_i64_vec(), w_rows, w_cols)
                } else {
                    weight_data.as_i64_vec()
                };
                let w_padded = pad_matrix(&w_transposed, k_dim, n_dim, k_padded, n_padded)?;

                let input_padded_for_mm = pad_to_size(input_data, k_padded);
                let mm = padded_matmul(&input_padded_for_mm, 1, k_padded, &w_padded, n_padded)?;

                let bias_padded = if let Some(bias_name) = bias_tensor_name {
                    if let Some(bias_data) = layer.weights.get(bias_name) {
                        let b = bias_data.as_i64_vec();
                        broadcast_bias(&b, b.len(), n_padded)
                    } else {
                        vec![0i64; n_padded]
                    }
                } else {
                    vec![0i64; n_padded]
                };

                let mm_with_bias: Vec<i64> = mm
                    .iter()
                    .zip(bias_padded.iter())
                    .map(|(&m, &b)| {
                        let sum = i128::from(m) + i128::from(b);
                        i64::try_from(sum).map_err(|_| {
                            anyhow::anyhow!(
                                "Gemm {} bias addition overflows i64: {} + {}",
                                layer.name,
                                m,
                                b
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<i64>>>()?;
                let (quotients, remainders) =
                    rescale::compute_rescale_array(&mm_with_bias, alpha, offset)?;

                shreds.insert(format!("{}_weight", layer.name), w_padded);
                shreds.insert(format!("{}_bias", layer.name), bias_padded);
                shreds.insert(format!("{}_q", layer.name), quotients.clone());
                shreds.insert(format!("{}_r", layer.name), remainders.clone());

                let r_max = remainders.iter().copied().max().unwrap_or(0);
                tracing::info!(
                    "Gemm {} rescale r_max={} table_size={}",
                    layer.name,
                    r_max,
                    rescale_table_size
                );
                if exponent > RANGE_CHECK_CHUNK_BITS {
                    let chunk_scale = 1i64 << RANGE_CHECK_CHUNK_BITS;
                    let chunk_table_size = 1usize << RANGE_CHECK_CHUNK_BITS;
                    let r_c0: Vec<i64> = remainders.iter().map(|&r| r % chunk_scale).collect();
                    let r_c1: Vec<i64> = remainders.iter().map(|&r| r / chunk_scale).collect();
                    shreds.insert(format!("{}_r_c0", layer.name), r_c0.clone());
                    shreds.insert(format!("{}_r_c1", layer.name), r_c1.clone());
                    shreds.insert(
                        format!("{}_r_c0_mults", layer.name),
                        compute_multiplicities(&r_c0, chunk_table_size)?,
                    );
                    shreds.insert(
                        format!("{}_r_c1_mults", layer.name),
                        compute_multiplicities(
                            &r_c1,
                            1usize << (exponent - RANGE_CHECK_CHUNK_BITS),
                        )?,
                    );
                } else {
                    shreds.insert(
                        format!("{}_r_mults", layer.name),
                        compute_multiplicities(&remainders, rescale_table_size)?,
                    );
                }

                for out in &layer.outputs {
                    tensors.insert(out.clone(), quotients.clone());
                }
            }
            OpType::Conv => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Conv {} has no input", layer.name))?;
                let weight_tensor_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Conv {} has no weight", layer.name))?;
                let bias_tensor_name = layer.inputs.get(2);

                let input_data = tensors.get(input_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Conv {} input {} not computed",
                        layer.name,
                        input_tensor_name
                    )
                })?;

                let weight_data = layer.weights.get(weight_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("Conv {} missing weight {}", layer.name, weight_tensor_name)
                })?;

                let input_layout = tensor_layouts
                    .get(input_tensor_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Conv {} input {} has no spatial layout",
                            layer.name,
                            input_tensor_name
                        )
                    })?
                    .clone();

                let w_shape = weight_data.shape();
                anyhow::ensure!(
                    w_shape.len() >= 4,
                    "Conv {} weight has {} dims, need >= 4",
                    layer.name,
                    w_shape.len()
                );
                let c_out = w_shape[0];
                let c_in = w_shape[1];
                let kh = w_shape[2];
                let kw = w_shape[3];

                let pads = layer.get_ints_attr("pads");
                let pad_top = pads.and_then(|p| p.first().copied()).unwrap_or(0) as usize;
                let pad_left = pads.and_then(|p| p.get(1).copied()).unwrap_or(0) as usize;
                let pad_bottom = pads.and_then(|p| p.get(2).copied()).unwrap_or(0) as usize;
                let pad_right = pads.and_then(|p| p.get(3).copied()).unwrap_or(0) as usize;

                let strides = layer.get_ints_attr("strides");
                let raw_stride_h = strides.and_then(|s| s.first().copied()).unwrap_or(1);
                let raw_stride_w = strides.and_then(|s| s.get(1).copied()).unwrap_or(1);
                anyhow::ensure!(
                    raw_stride_h > 0 && raw_stride_w > 0,
                    "Conv {} stride_h={} stride_w={} must be positive",
                    layer.name,
                    raw_stride_h,
                    raw_stride_w
                );
                let stride_h = raw_stride_h as usize;
                let stride_w = raw_stride_w as usize;

                let (input_ch, in_h, in_w) = input_layout.spatial_dims();
                anyhow::ensure!(
                    input_ch == c_in,
                    "Conv {}: weight c_in {} does not match input channels {}",
                    layer.name,
                    c_in,
                    input_ch
                );
                let padded_h = in_h + pad_top + pad_bottom;
                let padded_w = in_w + pad_left + pad_right;
                anyhow::ensure!(
                    padded_h >= kh && padded_w >= kw,
                    "Conv {}: padded input {}x{} smaller than kernel {}x{}",
                    layer.name,
                    padded_h,
                    padded_w,
                    kh,
                    kw
                );
                let out_h = (padded_h - kh) / stride_h + 1;
                let out_w = (padded_w - kw) / stride_w + 1;
                let patch_size = c_in
                    .checked_mul(kh)
                    .and_then(|v| v.checked_mul(kw))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Conv {} patch_size overflow: {} * {} * {}",
                            layer.name,
                            c_in,
                            kh,
                            kw
                        )
                    })?;
                let num_patches = out_h.checked_mul(out_w).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Conv {} num_patches overflow: {} * {}",
                        layer.name,
                        out_h,
                        out_w
                    )
                })?;

                let pad_patches = next_power_of_two(num_patches);
                let pad_psize = next_power_of_two(patch_size);
                let pad_cout = next_power_of_two(c_out);

                let im2col_size = pad_patches.checked_mul(pad_psize).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Conv {} im2col allocation overflow: {} * {}",
                        layer.name,
                        pad_patches,
                        pad_psize
                    )
                })?;
                let mut im2col_data = vec![0i64; im2col_size];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let patch = oh * out_w + ow;
                        for c in 0..c_in {
                            for kr in 0..kh {
                                for kc in 0..kw {
                                    let abs_h = oh * stride_h + kr;
                                    let abs_w = ow * stride_w + kc;
                                    if abs_h < pad_top || abs_w < pad_left {
                                        continue;
                                    }
                                    let ih = abs_h - pad_top;
                                    let iw = abs_w - pad_left;
                                    if ih >= in_h || iw >= in_w {
                                        continue;
                                    }
                                    let col = c * kh * kw + kr * kw + kc;
                                    let src = input_layout.index(c, ih, iw);
                                    im2col_data[patch * pad_psize + col] = input_data[src];
                                }
                            }
                        }
                    }
                }

                let kernel_t = transpose_matrix(&weight_data.as_i64_vec(), c_out, patch_size);
                let kernel_padded = pad_matrix(&kernel_t, patch_size, c_out, pad_psize, pad_cout)?;
                let mm = padded_matmul(
                    &im2col_data,
                    pad_patches,
                    pad_psize,
                    &kernel_padded,
                    pad_cout,
                )?;

                let result_size = pad_patches.checked_mul(pad_cout).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Conv {} result_size overflow: {} * {}",
                        layer.name,
                        pad_patches,
                        pad_cout
                    )
                })?;
                let bias_bc = if let Some(bias_name) = bias_tensor_name {
                    if let Some(bias_data) = layer.weights.get(bias_name) {
                        let b = bias_data.as_i64_vec();
                        (0..result_size)
                            .map(|i| {
                                let j = i % pad_cout;
                                if j < c_out && j < b.len() {
                                    b[j]
                                } else {
                                    0
                                }
                            })
                            .collect()
                    } else {
                        vec![0i64; result_size]
                    }
                } else {
                    vec![0i64; result_size]
                };

                let mm_with_bias: Vec<i64> = mm
                    .iter()
                    .zip(bias_bc.iter())
                    .map(|(&m, &b)| {
                        let sum = i128::from(m) + i128::from(b);
                        i64::try_from(sum).map_err(|_| {
                            anyhow::anyhow!(
                                "Conv {} bias addition overflows i64: {} + {}",
                                layer.name,
                                m,
                                b
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<i64>>>()?;
                let (quotients, remainders) =
                    rescale::compute_rescale_array(&mm_with_bias, alpha, offset)?;

                shreds.insert(format!("{}_weight", layer.name), kernel_padded);
                shreds.insert(format!("{}_bias", layer.name), bias_bc);
                shreds.insert(format!("{}_q", layer.name), quotients.clone());
                shreds.insert(format!("{}_r", layer.name), remainders.clone());

                let r_max = remainders.iter().copied().max().unwrap_or(0);
                tracing::info!(
                    "Conv {} rescale r_max={} table_size={}",
                    layer.name,
                    r_max,
                    rescale_table_size
                );
                if exponent > RANGE_CHECK_CHUNK_BITS {
                    let chunk_scale = 1i64 << RANGE_CHECK_CHUNK_BITS;
                    let chunk_table_size = 1usize << RANGE_CHECK_CHUNK_BITS;
                    let r_c0: Vec<i64> = remainders.iter().map(|&r| r % chunk_scale).collect();
                    let r_c1: Vec<i64> = remainders.iter().map(|&r| r / chunk_scale).collect();
                    shreds.insert(format!("{}_r_c0", layer.name), r_c0.clone());
                    shreds.insert(format!("{}_r_c1", layer.name), r_c1.clone());
                    shreds.insert(
                        format!("{}_r_c0_mults", layer.name),
                        compute_multiplicities(&r_c0, chunk_table_size)?,
                    );
                    shreds.insert(
                        format!("{}_r_c1_mults", layer.name),
                        compute_multiplicities(
                            &r_c1,
                            1usize << (exponent - RANGE_CHECK_CHUNK_BITS),
                        )?,
                    );
                } else {
                    shreds.insert(
                        format!("{}_r_mults", layer.name),
                        compute_multiplicities(&remainders, rescale_table_size)?,
                    );
                }

                for out in &layer.outputs {
                    tensors.insert(out.clone(), quotients.clone());
                    tensor_layouts.insert(
                        out.clone(),
                        SpatialInfo::HWC {
                            h: out_h,
                            w: out_w,
                            c: c_out,
                            stride_c: pad_cout,
                        },
                    );
                }
            }
            OpType::Relu => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Relu {} has no input", layer.name))?;
                let input_data = tensors.get(input_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Relu {} input {} not computed",
                        layer.name,
                        input_tensor_name
                    )
                })?;

                let nv = num_vars_for(input_data.len());
                let relu_out: Vec<i64> = input_data.iter().map(|&x| x.max(0)).collect();
                let delta_input: Vec<i64> = relu_out
                    .iter()
                    .zip(input_data.iter())
                    .map(|(&o, &x)| {
                        let diff = i128::from(o) - i128::from(x);
                        i64::try_from(diff).map_err(|_| {
                            anyhow::anyhow!("Relu {} delta overflow: {} - {}", layer.name, o, x)
                        })
                    })
                    .collect::<anyhow::Result<Vec<i64>>>()?;
                let delta_zero: Vec<i64> = relu_out.clone();

                let zero_vec = vec![0i64; 1 << nv];

                shreds.insert(format!("{}_zero", layer.name), zero_vec);
                shreds.insert(format!("{}_max", layer.name), relu_out.clone());
                shreds.insert(format!("{}_di", layer.name), delta_input.clone());
                shreds.insert(format!("{}_dz", layer.name), delta_zero.clone());

                {
                    let di_max = delta_input.iter().copied().max().unwrap_or(0) as u64;
                    let dz_max = delta_zero.iter().copied().max().unwrap_or(0) as u64;
                    let max_delta = di_max.max(dz_max);
                    let obs_n_bits = observed_n_bits_for_delta(max_delta, exponent);
                    let dnv = delta_table_nv(obs_n_bits, exponent);
                    let delta_table_size = 1usize << dnv;
                    tracing::info!(
                        "Relu {} observed_n_bits={} dnv={} table_size={} di_max={} dz_max={}",
                        layer.name,
                        obs_n_bits,
                        dnv,
                        delta_table_size,
                        di_max,
                        dz_max
                    );
                    observed_n_bits.insert(layer.name.clone(), obs_n_bits);
                    if dnv > RANGE_CHECK_CHUNK_BITS {
                        let chunk_scale = 1i64 << RANGE_CHECK_CHUNK_BITS;
                        let chunk_table_size = 1usize << RANGE_CHECK_CHUNK_BITS;
                        let hi_table_size = 1usize << (dnv - RANGE_CHECK_CHUNK_BITS);
                        let di_c0: Vec<i64> =
                            delta_input.iter().map(|&d| d % chunk_scale).collect();
                        let di_c1: Vec<i64> =
                            delta_input.iter().map(|&d| d / chunk_scale).collect();
                        let dz_c0: Vec<i64> = delta_zero.iter().map(|&d| d % chunk_scale).collect();
                        let dz_c1: Vec<i64> = delta_zero.iter().map(|&d| d / chunk_scale).collect();
                        shreds.insert(format!("{}_di_c0", layer.name), di_c0.clone());
                        shreds.insert(format!("{}_di_c1", layer.name), di_c1.clone());
                        shreds.insert(format!("{}_dz_c0", layer.name), dz_c0.clone());
                        shreds.insert(format!("{}_dz_c1", layer.name), dz_c1.clone());
                        shreds.insert(
                            format!("{}_di_c0_mults", layer.name),
                            compute_multiplicities(&di_c0, chunk_table_size)?,
                        );
                        shreds.insert(
                            format!("{}_di_c1_mults", layer.name),
                            compute_multiplicities(&di_c1, hi_table_size)?,
                        );
                        shreds.insert(
                            format!("{}_dz_c0_mults", layer.name),
                            compute_multiplicities(&dz_c0, chunk_table_size)?,
                        );
                        shreds.insert(
                            format!("{}_dz_c1_mults", layer.name),
                            compute_multiplicities(&dz_c1, hi_table_size)?,
                        );
                    } else {
                        shreds.insert(
                            format!("{}_di_mults", layer.name),
                            compute_multiplicities(&delta_input, delta_table_size)?,
                        );
                        shreds.insert(
                            format!("{}_dz_mults", layer.name),
                            compute_multiplicities(&delta_zero, delta_table_size)?,
                        );
                    }
                }

                let layout = tensor_layouts.get(input_tensor_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), relu_out.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::MaxPool => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} has no input", layer.name))?;
                let input_data = tensors.get(input_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "MaxPool {} input {} not computed",
                        layer.name,
                        input_tensor_name
                    )
                })?;

                let input_layout = tensor_layouts
                    .get(input_tensor_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MaxPool {} input {} has no spatial layout",
                            layer.name,
                            input_tensor_name
                        )
                    })?
                    .clone();

                let (c, in_h, in_w) = input_layout.spatial_dims();

                let kernel_shape = layer.get_ints_attr("kernel_shape").ok_or_else(|| {
                    anyhow::anyhow!("MaxPool {} missing kernel_shape", layer.name)
                })?;
                anyhow::ensure!(
                    kernel_shape.len() == 2,
                    "MaxPool {} requires exactly 2D kernel_shape, got {} dims",
                    layer.name,
                    kernel_shape.len()
                );
                anyhow::ensure!(
                    kernel_shape[0] > 0 && kernel_shape[1] > 0,
                    "MaxPool {} kernel_shape values must be positive, got [{}, {}]",
                    layer.name,
                    kernel_shape[0],
                    kernel_shape[1]
                );
                let pool_h = kernel_shape[0] as usize;
                let pool_w = kernel_shape[1] as usize;

                let strides = layer.get_ints_attr("strides");
                let raw_stride_h = strides.and_then(|s| s.first().copied()).unwrap_or(1);
                let raw_stride_w = strides.and_then(|s| s.get(1).copied()).unwrap_or(1);
                anyhow::ensure!(
                    raw_stride_h > 0 && raw_stride_w > 0,
                    "MaxPool {} stride_h={} stride_w={} must be positive",
                    layer.name,
                    raw_stride_h,
                    raw_stride_w
                );
                let stride_h = raw_stride_h as usize;
                let stride_w = raw_stride_w as usize;

                anyhow::ensure!(
                    in_h >= pool_h && in_w >= pool_w,
                    "MaxPool {}: input {}x{} smaller than kernel {}x{}",
                    layer.name,
                    in_h,
                    in_w,
                    pool_h,
                    pool_w
                );
                let pool_oh = (in_h - pool_h) / stride_h + 1;
                let pool_ow = (in_w - pool_w) / stride_w + 1;
                let window_size = pool_h.checked_mul(pool_w).ok_or_else(|| {
                    anyhow::anyhow!(
                        "MaxPool {} window_size overflow: {} * {}",
                        layer.name,
                        pool_h,
                        pool_w
                    )
                })?;
                let num_pool_out = pool_oh
                    .checked_mul(pool_ow)
                    .and_then(|v| v.checked_mul(c))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MaxPool {} num_pool_out overflow: {} * {} * {}",
                            layer.name,
                            pool_oh,
                            pool_ow,
                            c
                        )
                    })?;
                let pad_pool = next_power_of_two(num_pool_out);

                let mut max_values = vec![0i64; pad_pool];
                let mut window_elems: Vec<Vec<i64>> = vec![vec![0i64; pad_pool]; window_size];

                for ch in 0..c {
                    for poh in 0..pool_oh {
                        for pow in 0..pool_ow {
                            let dest_idx = (poh * pool_ow + pow) * c + ch;
                            let mut max_val = i64::MIN;
                            for ph in 0..pool_h {
                                for pw in 0..pool_w {
                                    let elem_pos = ph * pool_w + pw;
                                    let soh = poh * stride_h + ph;
                                    let sow = pow * stride_w + pw;
                                    let src_idx = input_layout.index(ch, soh, sow);
                                    let val = input_data[src_idx];
                                    window_elems[elem_pos][dest_idx] = val;
                                    max_val = max_val.max(val);
                                }
                            }
                            max_values[dest_idx] = max_val;
                        }
                    }
                }

                let deltas: Vec<Vec<i64>> = (0..window_size)
                    .map(|i| {
                        (0..pad_pool)
                            .map(|w| {
                                let diff =
                                    i128::from(max_values[w]) - i128::from(window_elems[i][w]);
                                i64::try_from(diff).map_err(|_| {
                                    anyhow::anyhow!(
                                        "MaxPool {} delta overflow: {} - {}",
                                        layer.name,
                                        max_values[w],
                                        window_elems[i][w]
                                    )
                                })
                            })
                            .collect::<anyhow::Result<Vec<i64>>>()
                    })
                    .collect::<anyhow::Result<Vec<Vec<i64>>>>()?;

                shreds.insert(format!("{}_max", layer.name), max_values.clone());
                for i in 0..window_size {
                    shreds.insert(format!("{}_d{}", layer.name, i), deltas[i].clone());
                }

                {
                    let max_delta = deltas
                        .iter()
                        .flat_map(|d| d.iter().copied())
                        .max()
                        .unwrap_or(0) as u64;
                    let obs_n_bits = observed_n_bits_for_delta(max_delta, exponent);
                    let dnv = delta_table_nv(obs_n_bits, exponent);
                    let dt_size = 1usize << dnv;
                    tracing::info!(
                        "MaxPool {} observed_n_bits={} dnv={} table_size={} max_delta={}",
                        layer.name,
                        obs_n_bits,
                        dnv,
                        dt_size,
                        max_delta
                    );
                    observed_n_bits.insert(layer.name.clone(), obs_n_bits);
                    if dnv > RANGE_CHECK_CHUNK_BITS {
                        let chunk_scale = 1i64 << RANGE_CHECK_CHUNK_BITS;
                        let chunk_table_size = 1usize << RANGE_CHECK_CHUNK_BITS;
                        let hi_table_size = 1usize << (dnv - RANGE_CHECK_CHUNK_BITS);
                        for i in 0..window_size {
                            let c0: Vec<i64> = deltas[i].iter().map(|&d| d % chunk_scale).collect();
                            let c1: Vec<i64> = deltas[i].iter().map(|&d| d / chunk_scale).collect();
                            shreds.insert(format!("{}_d{}_c0", layer.name, i), c0.clone());
                            shreds.insert(format!("{}_d{}_c1", layer.name, i), c1.clone());
                            shreds.insert(
                                format!("{}_d{}_c0_mults", layer.name, i),
                                compute_multiplicities(&c0, chunk_table_size)?,
                            );
                            shreds.insert(
                                format!("{}_d{}_c1_mults", layer.name, i),
                                compute_multiplicities(&c1, hi_table_size)?,
                            );
                        }
                    } else {
                        for i in 0..window_size {
                            shreds.insert(
                                format!("{}_d{}_mults", layer.name, i),
                                compute_multiplicities(&deltas[i], dt_size)?,
                            );
                        }
                    }
                }

                for out in &layer.outputs {
                    tensors.insert(out.clone(), max_values.clone());
                    tensor_layouts.insert(
                        out.clone(),
                        SpatialInfo::HWC {
                            h: pool_oh,
                            w: pool_ow,
                            c,
                            stride_c: c,
                        },
                    );
                }
            }
            OpType::BatchNormalization => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("BatchNorm {} has no input", layer.name))?;
                let mul_tensor_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing mul input", layer.name))?;
                let add_tensor_name = layer
                    .inputs
                    .get(2)
                    .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing add input", layer.name))?;

                let input_data = tensors.get(input_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "BatchNorm {} input {} not computed",
                        layer.name,
                        input_tensor_name
                    )
                })?;

                let mul_data = layer.weights.get(mul_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "BatchNorm {} missing weight '{}'",
                        layer.name,
                        mul_tensor_name
                    )
                })?;
                let add_data = layer.weights.get(add_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "BatchNorm {} missing weight '{}'",
                        layer.name,
                        add_tensor_name
                    )
                })?;

                let mul_per_ch = mul_data.as_i64_vec();
                let add_per_ch = add_data.as_i64_vec();
                let c = mul_per_ch.len();
                anyhow::ensure!(
                    add_per_ch.len() == c,
                    "BatchNorm {} mul/add length mismatch: {} vs {}",
                    layer.name,
                    c,
                    add_per_ch.len()
                );

                let input_layout = tensor_layouts.get(input_tensor_name).cloned();
                let padded_size = next_power_of_two(input_data.len());

                let mut mul_broadcast = vec![0i64; padded_size];
                let mut add_broadcast = vec![0i64; padded_size];

                match &input_layout {
                    Some(SpatialInfo::CHW { h, w, .. }) => {
                        let hw = h * w;
                        for ch in 0..c {
                            for s in 0..hw {
                                let idx = ch * hw + s;
                                if idx < padded_size {
                                    mul_broadcast[idx] = mul_per_ch[ch];
                                    add_broadcast[idx] = add_per_ch[ch];
                                }
                            }
                        }
                    }
                    Some(SpatialInfo::HWC { h, w, stride_c, .. }) => {
                        for row in 0..(h * w) {
                            for ch in 0..c {
                                let idx = row * stride_c + ch;
                                if idx < padded_size {
                                    mul_broadcast[idx] = mul_per_ch[ch];
                                    add_broadcast[idx] = add_per_ch[ch];
                                }
                            }
                        }
                    }
                    None => {
                        let spatial = if c > 0 && input_data.len() > c {
                            anyhow::ensure!(input_data.len() % c == 0,
                                "BatchNorm {} has no spatial layout and input len {} is not divisible by channels {}",
                                layer.name, input_data.len(), c);
                            input_data.len() / c
                        } else {
                            1
                        };
                        for ch in 0..c {
                            for s in 0..spatial {
                                let idx = ch * spatial + s;
                                if idx < padded_size {
                                    mul_broadcast[idx] = mul_per_ch[ch];
                                    add_broadcast[idx] = add_per_ch[ch];
                                }
                            }
                        }
                    }
                }

                let input_padded = pad_to_size(input_data, padded_size);
                let product: Vec<i64> = input_padded
                    .iter()
                    .zip(mul_broadcast.iter())
                    .map(|(&x, &m)| {
                        let prod = x as i128 * m as i128;
                        i64::try_from(prod).map_err(|_| {
                            anyhow::anyhow!(
                                "BatchNorm {} mul overflows i64: {} * {}",
                                layer.name,
                                x,
                                m
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<i64>>>()?;
                let with_add: Vec<i64> = product
                    .iter()
                    .zip(add_broadcast.iter())
                    .map(|(&p, &a)| {
                        let sum = p as i128 + a as i128;
                        i64::try_from(sum).map_err(|_| {
                            anyhow::anyhow!(
                                "BatchNorm {} add overflows i64: {} + {}",
                                layer.name,
                                p,
                                a
                            )
                        })
                    })
                    .collect::<anyhow::Result<Vec<i64>>>()?;

                let (quotients, remainders) =
                    rescale::compute_rescale_array(&with_add, alpha, offset)?;

                shreds.insert(format!("{}_mul", layer.name), mul_broadcast);
                shreds.insert(format!("{}_add", layer.name), add_broadcast);
                shreds.insert(format!("{}_q", layer.name), quotients.clone());
                shreds.insert(format!("{}_r", layer.name), remainders.clone());

                let r_max = remainders.iter().copied().max().unwrap_or(0);
                tracing::info!(
                    "BatchNorm {} rescale r_max={} table_size={}",
                    layer.name,
                    r_max,
                    rescale_table_size
                );
                if exponent > RANGE_CHECK_CHUNK_BITS {
                    let chunk_scale = 1i64 << RANGE_CHECK_CHUNK_BITS;
                    let chunk_table_size = 1usize << RANGE_CHECK_CHUNK_BITS;
                    let r_c0: Vec<i64> = remainders.iter().map(|&r| r % chunk_scale).collect();
                    let r_c1: Vec<i64> = remainders.iter().map(|&r| r / chunk_scale).collect();
                    shreds.insert(format!("{}_r_c0", layer.name), r_c0.clone());
                    shreds.insert(format!("{}_r_c1", layer.name), r_c1.clone());
                    shreds.insert(
                        format!("{}_r_c0_mults", layer.name),
                        compute_multiplicities(&r_c0, chunk_table_size)?,
                    );
                    shreds.insert(
                        format!("{}_r_c1_mults", layer.name),
                        compute_multiplicities(
                            &r_c1,
                            1usize << (exponent - RANGE_CHECK_CHUNK_BITS),
                        )?,
                    );
                } else {
                    shreds.insert(
                        format!("{}_r_mults", layer.name),
                        compute_multiplicities(&remainders, rescale_table_size)?,
                    );
                }

                for out in &layer.outputs {
                    tensors.insert(out.clone(), quotients.clone());
                    if let Some(ref layout) = input_layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Add | OpType::Sub => {
                let input_a_name = layer.inputs.first().ok_or_else(|| {
                    anyhow::anyhow!("{:?} {} has no first input", layer.op_type, layer.name)
                })?;
                let input_b_name = layer.inputs.get(1).ok_or_else(|| {
                    anyhow::anyhow!("{:?} {} has no second input", layer.op_type, layer.name)
                })?;

                let get_data = |name: &str| -> Result<Vec<i64>> {
                    if let Some(t) = tensors.get(name) {
                        Ok(t.clone())
                    } else if let Some(w) = layer.weights.get(name) {
                        Ok(w.as_i64_vec())
                    } else {
                        bail!(
                            "{:?} {} input {} not computed and not a weight",
                            layer.op_type,
                            layer.name,
                            name
                        )
                    }
                };

                let a_data = get_data(input_a_name)?;
                let b_data = get_data(input_b_name)?;

                let out_len = a_data.len().max(b_data.len());
                let padded_size = next_power_of_two(out_len);
                let a_padded = pad_to_size(&a_data, padded_size);
                let b_padded = pad_to_size(&b_data, padded_size);

                let result: Vec<i64> = if layer.op_type == OpType::Add {
                    a_padded
                        .iter()
                        .zip(b_padded.iter())
                        .map(|(&a, &b)| {
                            let sum = a as i128 + b as i128;
                            i64::try_from(sum).map_err(|_| {
                                anyhow::anyhow!("Add {} overflows i64: {} + {}", layer.name, a, b)
                            })
                        })
                        .collect::<anyhow::Result<Vec<i64>>>()?
                } else {
                    a_padded
                        .iter()
                        .zip(b_padded.iter())
                        .map(|(&a, &b)| {
                            let diff = a as i128 - b as i128;
                            i64::try_from(diff).map_err(|_| {
                                anyhow::anyhow!("Sub {} overflows i64: {} - {}", layer.name, a, b)
                            })
                        })
                        .collect::<anyhow::Result<Vec<i64>>>()?
                };

                if !tensors.contains_key(input_b_name) {
                    shreds.insert(format!("{}_{}", layer.name, input_b_name), b_padded);
                }
                if !tensors.contains_key(input_a_name) {
                    shreds.insert(format!("{}_{}", layer.name, input_a_name), a_padded);
                }
                shreds.insert(format!("{}_result", layer.name), result.clone());

                let layout = tensor_layouts
                    .get(input_a_name)
                    .or_else(|| tensor_layouts.get(input_b_name))
                    .cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), result.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Cast
            | OpType::Reshape
            | OpType::Flatten
            | OpType::Squeeze
            | OpType::Unsqueeze => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("shape op {} has no input", layer.name))?;
                let data = tensors
                    .get(input_tensor_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "shape op {} input {} not computed",
                            layer.name,
                            input_tensor_name
                        )
                    })?
                    .clone();
                let layout = tensor_layouts.get(input_tensor_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), data.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Exp => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Exp {} has no input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Exp {} input '{}' not computed", layer.name, input_name)
                    })?
                    .clone();

                let output_total: usize = layer.output_shape.iter().product();
                anyhow::ensure!(
                    input_data.len() >= output_total,
                    "Exp {} input too small: {} < {}",
                    layer.name,
                    input_data.len(),
                    output_total
                );

                let result: Vec<i64> = (0..output_total)
                    .map(|i| {
                        let x = input_data[i] as f64 / alpha as f64;
                        let y = x.exp();
                        (y * alpha as f64)
                            .round()
                            .clamp(i64::MIN as f64, i64::MAX as f64) as i64
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let out_name = format!("{}_out", layer.name);
                shreds.insert(out_name, padded.clone());

                let layout = tensor_layouts.get(input_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Sigmoid => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Sigmoid {} has no input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Sigmoid {} input '{}' not computed",
                            layer.name,
                            input_name
                        )
                    })?
                    .clone();

                let output_total: usize = layer.output_shape.iter().product();
                anyhow::ensure!(
                    input_data.len() >= output_total,
                    "Sigmoid {} input too small: {} < {}",
                    layer.name,
                    input_data.len(),
                    output_total
                );

                let result: Vec<i64> = (0..output_total)
                    .map(|i| {
                        let x = input_data[i] as f64 / alpha as f64;
                        let y = 1.0 / (1.0 + (-x).exp());
                        (y * alpha as f64)
                            .round()
                            .clamp(i64::MIN as f64, i64::MAX as f64) as i64
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let out_name = format!("{}_out", layer.name);
                shreds.insert(out_name, padded.clone());

                let layout = tensor_layouts.get(input_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Gelu => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Gelu {} has no input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Gelu {} input '{}' not computed", layer.name, input_name)
                    })?
                    .clone();

                let output_total: usize = layer.output_shape.iter().product();
                anyhow::ensure!(
                    input_data.len() >= output_total,
                    "Gelu {} input too small: {} < {}",
                    layer.name,
                    input_data.len(),
                    output_total
                );

                let result: Vec<i64> = (0..output_total)
                    .map(|i| {
                        let x = input_data[i] as f64 / alpha as f64;
                        let inner = 0.797_884_560_8 * (x + 0.044_715 * x * x * x);
                        let y = 0.5 * x * (1.0 + inner.tanh());
                        (y * alpha as f64)
                            .round()
                            .clamp(i64::MIN as f64, i64::MAX as f64) as i64
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let out_name = format!("{}_out", layer.name);
                shreds.insert(out_name, padded.clone());

                let layout = tensor_layouts.get(input_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Softmax => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Softmax {} has no input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Softmax {} input '{}' not computed",
                            layer.name,
                            input_name
                        )
                    })?
                    .clone();

                let input_shape = tensor_shape_for(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Softmax {} cannot resolve shape for input '{}'",
                        layer.name,
                        input_name
                    )
                })?;
                let rank = input_shape.len();
                let raw_axis = layer.get_int_attr("axis").unwrap_or(-1);
                let axis = if raw_axis < 0 {
                    let a = rank as i64 + raw_axis;
                    anyhow::ensure!(
                        a >= 0,
                        "Softmax {} axis {} out of range for rank {}",
                        layer.name,
                        raw_axis,
                        rank
                    );
                    a as usize
                } else {
                    raw_axis as usize
                };
                anyhow::ensure!(
                    axis < rank,
                    "Softmax {} axis {} out of range for rank {}",
                    layer.name,
                    raw_axis,
                    rank
                );

                let output_total: usize = layer.output_shape.iter().product();
                anyhow::ensure!(
                    input_data.len() >= output_total,
                    "Softmax {} input too small: {} < {}",
                    layer.name,
                    input_data.len(),
                    output_total
                );

                let axis_dim = input_shape[axis];
                let inner: usize = input_shape[axis + 1..].iter().product();
                let outer: usize = input_shape[..axis].iter().product();

                let mut result = vec![0i64; output_total];
                for o in 0..outer {
                    for inr in 0..inner {
                        let mut lane = Vec::with_capacity(axis_dim);
                        for k in 0..axis_dim {
                            let idx = (o * axis_dim + k) * inner + inr;
                            lane.push(input_data[idx] as f64 / alpha as f64);
                        }
                        let max_x = lane.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = lane.iter().map(|&x| (x - max_x).exp()).collect();
                        let denom: f64 = exps.iter().sum();
                        for (k, e) in exps.into_iter().enumerate() {
                            let idx = (o * axis_dim + k) * inner + inr;
                            result[idx] = ((e / denom) * alpha as f64)
                                .round()
                                .clamp(i64::MIN as f64, i64::MAX as f64)
                                as i64;
                        }
                    }
                }

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let out_name = format!("{}_out", layer.name);
                shreds.insert(out_name, padded.clone());

                let layout = tensor_layouts.get(input_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Tile => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Tile {} has no data input", layer.name))?;
                let repeats_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Tile {} has no repeats input", layer.name))?;

                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Tile {} input '{}' not computed", layer.name, input_name)
                    })?
                    .clone();
                let input_shape = tensor_shape_for(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Tile {} cannot resolve shape for input '{}'",
                        layer.name,
                        input_name
                    )
                })?;
                let repeats = layer
                    .weights
                    .get(repeats_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Tile {} repeats '{}' not found in weights",
                            layer.name,
                            repeats_name
                        )
                    })?
                    .as_i64_vec();

                anyhow::ensure!(
                    repeats.len() == input_shape.len(),
                    "Tile {} repeats rank {} does not match input rank {}",
                    layer.name,
                    repeats.len(),
                    input_shape.len()
                );

                let output_shape = &layer.output_shape;
                let output_total: usize = output_shape.iter().product();
                let result: Vec<i64> = (0..output_total)
                    .map(|out_flat| {
                        let out_coords = unravel_index_witness(out_flat, output_shape);
                        let in_coords: Vec<usize> = out_coords
                            .iter()
                            .enumerate()
                            .map(|(d, &c)| {
                                let dim = input_shape[d];
                                if dim == 0 {
                                    0
                                } else {
                                    c % dim
                                }
                            })
                            .collect();
                        let in_flat = ravel_index_witness(&in_coords, &input_shape);
                        input_data.get(in_flat).copied().unwrap_or(0)
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let out_name = format!("{}_out", layer.name);
                shreds.insert(out_name, padded.clone());

                let layout = tensor_layouts.get(input_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::TopK => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("TopK {} has no data input", layer.name))?;
                let k_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("TopK {} has no K input", layer.name))?;

                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("TopK {} input '{}' not computed", layer.name, input_name)
                    })?
                    .clone();
                let input_shape = tensor_shape_for(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "TopK {} cannot resolve shape for input '{}'",
                        layer.name,
                        input_name
                    )
                })?;

                let k_vec = layer
                    .weights
                    .get(k_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("TopK {} K '{}' not found in weights", layer.name, k_name)
                    })?
                    .as_i64_vec();
                let k_raw = *k_vec.first().ok_or_else(|| {
                    anyhow::anyhow!("TopK {} K tensor '{}' is empty", layer.name, k_name)
                })?;
                anyhow::ensure!(
                    k_raw > 0,
                    "TopK {} requires K > 0, got {}",
                    layer.name,
                    k_raw
                );
                let k = k_raw as usize;

                let rank = input_shape.len();
                let raw_axis = layer.get_int_attr("axis").unwrap_or(-1);
                let axis = if raw_axis < 0 {
                    let a = rank as i64 + raw_axis;
                    anyhow::ensure!(
                        a >= 0,
                        "TopK {} axis {} out of range for rank {}",
                        layer.name,
                        raw_axis,
                        rank
                    );
                    a as usize
                } else {
                    raw_axis as usize
                };
                anyhow::ensure!(
                    axis < rank,
                    "TopK {} axis {} out of range for rank {}",
                    layer.name,
                    raw_axis,
                    rank
                );

                let axis_dim = input_shape[axis];
                anyhow::ensure!(
                    k <= axis_dim,
                    "TopK {} K={} exceeds axis dimension {}",
                    layer.name,
                    k,
                    axis_dim
                );

                let output_shape = &layer.output_shape;
                let output_total: usize = output_shape.iter().product();
                let inner: usize = input_shape[axis + 1..].iter().product();
                let outer: usize = input_shape[..axis].iter().product();

                let mut values = vec![0i64; output_total];
                let mut indices = vec![0i64; output_total];
                for o in 0..outer {
                    for inr in 0..inner {
                        let mut lane: Vec<(i64, usize)> = (0..axis_dim)
                            .map(|i| {
                                let idx = (o * axis_dim + i) * inner + inr;
                                (input_data[idx], i)
                            })
                            .collect();

                        lane.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

                        for i in 0..k {
                            let out_idx = (o * k + i) * inner + inr;
                            values[out_idx] = lane[i].0;
                            indices[out_idx] = lane[i].1 as i64;
                        }
                    }
                }

                let padded_values = pad_to_size(&values, next_power_of_two(output_total));
                let values_shred = format!("{}_out", layer.name);
                shreds.insert(values_shred, padded_values.clone());

                if let Some(values_out) = layer.outputs.first() {
                    tensors.insert(values_out.clone(), padded_values.clone());
                    if let Some(layout) = tensor_layouts.get(input_name).cloned() {
                        tensor_layouts.insert(values_out.clone(), layout);
                    }
                }

                if let Some(indices_out) = layer.outputs.get(1) {
                    let padded_indices = pad_to_size(&indices, next_power_of_two(output_total));
                    let indices_shred = format!("{}_indices_out", layer.name);
                    shreds.insert(indices_shred, padded_indices.clone());
                    tensors.insert(indices_out.clone(), padded_indices);
                    if let Some(layout) = tensor_layouts.get(input_name).cloned() {
                        tensor_layouts.insert(indices_out.clone(), layout);
                    }
                }
            }
            OpType::LayerNormalization => {
                let x_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("LayerNorm {} has no input", layer.name))?;
                let gamma_name = layer.inputs.get(1).ok_or_else(|| {
                    anyhow::anyhow!("LayerNorm {} missing gamma input", layer.name)
                })?;

                let x_data = tensors
                    .get(x_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("LayerNorm {} input '{}' not computed", layer.name, x_name)
                    })?
                    .clone();

                let gamma_td = layer.weights.get(gamma_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "LayerNorm {} gamma '{}' not found in weights",
                        layer.name,
                        gamma_name
                    )
                })?;
                let gamma = gamma_td.as_i64_vec();

                let beta: Vec<i64> = if let Some(beta_name) = layer.inputs.get(2) {
                    let beta_td = layer.weights.get(beta_name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "LayerNorm {} beta '{}' not found in weights",
                            layer.name,
                            beta_name
                        )
                    })?;
                    beta_td.as_i64_vec()
                } else {
                    vec![0i64; gamma.len()]
                };

                let raw_axis = layer.get_int_attr("axis").unwrap_or(-1);
                let output_shape = &layer.output_shape;
                let rank = output_shape.len();
                let axis = if raw_axis < 0 {
                    let a = rank as i64 + raw_axis;
                    anyhow::ensure!(
                        a >= 0,
                        "LayerNorm {}: axis {} out of range for rank {}",
                        layer.name,
                        raw_axis,
                        rank
                    );
                    a as usize
                } else {
                    let a = raw_axis as usize;
                    anyhow::ensure!(
                        a < rank,
                        "LayerNorm {}: axis {} out of range for rank {}",
                        layer.name,
                        raw_axis,
                        rank
                    );
                    a
                };

                let outer_size: usize = output_shape[..axis].iter().product();
                let lane_size: usize = output_shape[axis..].iter().product();
                let total_size = outer_size * lane_size;

                anyhow::ensure!(
                    gamma.len() == lane_size,
                    "LayerNorm {}: gamma len {} != lane_size {}",
                    layer.name,
                    gamma.len(),
                    lane_size
                );
                anyhow::ensure!(
                    beta.len() == lane_size,
                    "LayerNorm {}: beta len {} != lane_size {}",
                    layer.name,
                    beta.len(),
                    lane_size
                );

                let scale_f64 = alpha as f64;
                let scale_sq = scale_f64 * scale_f64;
                const LN_EPSILON: f64 = 1e-5;

                let gamma_f64: Vec<f64> = gamma.iter().map(|&g| g as f64 / scale_f64).collect();
                let beta_f64: Vec<f64> = beta.iter().map(|&b| b as f64 / scale_sq).collect();

                let mut result: Vec<i64> = Vec::with_capacity(total_size);

                for outer_i in 0..outer_size {
                    let start = outer_i * lane_size;
                    let lane: Vec<f64> = x_data[start..start + lane_size]
                        .iter()
                        .map(|&v| v as f64 / scale_f64)
                        .collect();

                    let mean: f64 = lane.iter().sum::<f64>() / lane_size as f64;
                    let var: f64 =
                        lane.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / lane_size as f64;
                    let inv_std = 1.0 / (var + LN_EPSILON).sqrt();

                    for i in 0..lane_size {
                        let normalized = (lane[i] - mean) * inv_std;
                        let y_f = normalized * gamma_f64[i] + beta_f64[i];
                        let y_q = (y_f * scale_f64).round() as i64;
                        result.push(y_q);
                    }
                }

                let padded = pad_to_size(&result, next_power_of_two(total_size));
                let ln_out_name = format!("{}_out", layer.name);
                shreds.insert(ln_out_name, padded.clone());

                let layout = tensor_layouts.get(x_name).cloned();

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Gather => {
                let data_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Gather {} has no data input", layer.name))?;
                let indices_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Gather {} has no indices input", layer.name))?;

                let data_flat = tensors
                    .get(data_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Gather {} data '{}' not computed", layer.name, data_name)
                    })?
                    .clone();

                let indices_td = layer.weights.get(indices_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Gather {} indices '{}' not found in weights; \
                            only constant (initializer) indices are supported",
                        layer.name,
                        indices_name
                    )
                })?;
                let indices = indices_td.as_i64_vec();

                let axis = layer.get_int_attr("axis").unwrap_or(0);
                anyhow::ensure!(
                    axis == 0,
                    "Gather {}: only axis=0 is supported in the Remainder backend (got axis={})",
                    layer.name,
                    axis
                );

                let output_total: usize = layer.output_shape.iter().product();
                let slice_size = if indices.is_empty() {
                    0
                } else {
                    output_total / indices.len()
                };

                let mut result: Vec<i64> = Vec::with_capacity(output_total);
                for &idx in &indices {
                    let idx = usize::try_from(idx).map_err(|_| {
                        anyhow::anyhow!(
                            "Gather {}: negative index {} is not supported",
                            layer.name,
                            idx
                        )
                    })?;
                    let start = idx * slice_size;
                    let end = start + slice_size;
                    anyhow::ensure!(
                        end <= data_flat.len(),
                        "Gather {}: index {} out of bounds (data size {})",
                        layer.name,
                        idx,
                        data_flat.len()
                    );
                    result.extend_from_slice(&data_flat[start..end]);
                }

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let gather_out_name = format!("{}_out", layer.name);
                shreds.insert(gather_out_name, padded.clone());

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                }
            }
            OpType::Resize => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Resize {} has no input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Resize {} input '{}' not computed", layer.name, input_name)
                    })?
                    .clone();

                let output_shape = &layer.output_shape;
                let output_total: usize = output_shape.iter().product();

                let input_shape = tensor_shape_for(input_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Resize {}: unable to resolve producer/input shape for '{}'",
                        layer.name,
                        input_name
                    )
                })?;
                anyhow::ensure!(
                    input_shape.len() == output_shape.len(),
                    "Resize {}: input shape rank {} does not match output rank {} (input_shape={:?}, output_shape={:?})",
                    layer.name,
                    input_shape.len(),
                    output_shape.len(),
                    input_shape,
                    output_shape
                );

                if let Some(sizes_name) = layer.inputs.get(3).filter(|n| !n.is_empty()) {
                    let sizes_td = layer.weights.get(sizes_name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Resize {}: sizes tensor '{}' not found in layer weights",
                            layer.name,
                            sizes_name
                        )
                    })?;
                    let sizes = sizes_td.as_i64_vec();
                    anyhow::ensure!(
                        sizes.len() == output_shape.len(),
                        "Resize {}: sizes tensor '{}' length {} != output rank {}",
                        layer.name,
                        sizes_name,
                        sizes.len(),
                        output_shape.len()
                    );
                    for (i, (&sz, &out_d)) in sizes.iter().zip(output_shape.iter()).enumerate() {
                        anyhow::ensure!(
                            sz >= 0,
                            "Resize {}: sizes tensor '{}' has negative dimension {} at axis {}",
                            layer.name,
                            sizes_name,
                            sz,
                            i
                        );
                        anyhow::ensure!(
                            sz as usize == out_d,
                            "Resize {}: sizes tensor '{}' axis {} mismatch: {} vs output {}",
                            layer.name,
                            sizes_name,
                            i,
                            sz,
                            out_d
                        );
                    }
                } else {
                    let scales_name = layer.inputs.get(2).filter(|n| !n.is_empty()).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Resize {}: missing scales input (input[2]); cannot validate resize parameters",
                            layer.name
                        )
                    })?;
                    let scales_td = layer.weights.get(scales_name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Resize {}: scales tensor '{}' not found in layer weights",
                            layer.name,
                            scales_name
                        )
                    })?;
                    let scales = &scales_td.float_data;
                    anyhow::ensure!(
                        scales.len() == output_shape.len(),
                        "Resize {}: scales tensor '{}' length {} != output rank {}",
                        layer.name,
                        scales_name,
                        scales.len(),
                        output_shape.len()
                    );
                }

                let mode = layer.get_string_attr("mode").unwrap_or("nearest");
                let coord_mode = layer
                    .get_string_attr("coordinate_transformation_mode")
                    .unwrap_or("half_pixel");
                let nearest_mode = layer
                    .get_string_attr("nearest_mode")
                    .unwrap_or("round_prefer_floor");

                let result: Vec<i64> = if mode == "nearest" {
                    (0..output_total)
                        .map(|out_flat| {
                            let out_coords = unravel_index_witness(out_flat, output_shape);
                            let mut in_coords = vec![0usize; output_shape.len()];
                            for d in 0..output_shape.len() {
                                let x = coord_to_input(
                                    out_coords[d],
                                    input_shape[d],
                                    output_shape[d],
                                    coord_mode,
                                );
                                in_coords[d] = nearest_round(x, input_shape[d], nearest_mode);
                            }
                            let in_flat = ravel_index_witness(&in_coords, &input_shape);
                            input_data.get(in_flat).copied().unwrap_or(0)
                        })
                        .collect()
                } else {
                    // Linear / bilinear interpolation.
                    let alpha = model.scale_config.alpha;
                    let resize_dims: Vec<usize> = (0..output_shape.len())
                        .filter(|&d| input_shape[d] != output_shape[d])
                        .collect();
                    (0..output_total)
                        .map(|out_flat| {
                            let out_coords = unravel_index_witness(out_flat, output_shape);
                            let dim_info: Vec<(usize, usize, f64, f64)> = resize_dims
                                .iter()
                                .map(|&d| {
                                    let x = coord_to_input(
                                        out_coords[d],
                                        input_shape[d],
                                        output_shape[d],
                                        coord_mode,
                                    );
                                    interp_corners(x, input_shape[d])
                                })
                                .collect();

                            let n_corners = 1usize << resize_dims.len();
                            let mut sum_i128: i128 = 0;
                            for mask in 0..n_corners {
                                let mut in_coords = out_coords.clone();
                                let mut w = 1.0f64;
                                for (i, &d) in resize_dims.iter().enumerate() {
                                    let (f_idx, c_idx, w_f, w_c) = dim_info[i];
                                    if mask & (1 << i) == 0 {
                                        in_coords[d] = f_idx;
                                        w *= w_f;
                                    } else {
                                        in_coords[d] = c_idx;
                                        w *= w_c;
                                    }
                                }
                                let in_flat = ravel_index_witness(&in_coords, &input_shape);
                                let x_q = input_data.get(in_flat).copied().unwrap_or(0);
                                let w_q = (w * alpha as f64).round() as i128;
                                sum_i128 += x_q as i128 * w_q;
                            }
                            let half = alpha as i128 / 2;
                            let y = (sum_i128 + half) / alpha as i128;
                            y.clamp(i64::MIN as i128, i64::MAX as i128) as i64
                        })
                        .collect()
                };

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let resize_out_name = format!("{}_out", layer.name);
                shreds.insert(resize_out_name, padded.clone());

                let input_layout = tensor_layouts.get(input_name);
                let output_layout = layout_from_output_shape_runtime(output_shape, input_layout);

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = output_layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::GridSample => {
                // Inputs: X [N, C, H_in, W_in], grid [N, H_out, W_out, 2] (constant).
                let x_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("GridSample {} has no X input", layer.name))?;
                let x_data = tensors
                    .get(x_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("GridSample {} input '{}' not computed", layer.name, x_name)
                    })?
                    .clone();

                let grid_name = layer.inputs.get(1).ok_or_else(|| {
                    anyhow::anyhow!("GridSample {} has no grid input", layer.name)
                })?;
                let grid_td = layer.weights.get(grid_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "GridSample {}: grid '{}' must be a compile-time constant (initializer)",
                        layer.name,
                        grid_name
                    )
                })?;
                // Grid is quantised at α¹; int_data holds round(grid_f * alpha).
                let grid_flat = grid_td.as_i64_vec();

                let output_shape = &layer.output_shape;
                anyhow::ensure!(
                    output_shape.len() == 4,
                    "GridSample {} output_shape must be 4-D, got {:?}",
                    layer.name,
                    output_shape
                );
                let [n, c, h_out, w_out] = [
                    output_shape[0],
                    output_shape[1],
                    output_shape[2],
                    output_shape[3],
                ];
                let output_total = n * c * h_out * w_out;

                let alpha = model.scale_config.alpha as f64;

                let mode = layer.get_string_attr("mode").unwrap_or("bilinear");
                let padding_mode = layer.get_string_attr("padding_mode").unwrap_or("zeros");
                let align_corners = layer.get_int_attr("align_corners").unwrap_or(0) != 0;

                let x_shape = tensor_shape_for(x_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "GridSample {}: unable to resolve shape for input '{}'",
                        layer.name,
                        x_name
                    )
                })?;
                anyhow::ensure!(
                    x_shape.len() == 4,
                    "GridSample {}: input '{}' shape must be 4-D [N,C,H,W], got {:?}",
                    layer.name,
                    x_name,
                    x_shape
                );
                let n_in = x_shape[0];
                let c_in = x_shape[1];
                let h_in = x_shape[2];
                let w_in = x_shape[3];
                anyhow::ensure!(
                    n_in == n && c_in == c,
                    "GridSample {}: output shape {:?} is inconsistent with input shape {:?}",
                    layer.name,
                    output_shape,
                    x_shape
                );

                let grid_dims = grid_td.shape();
                anyhow::ensure!(
                    grid_dims.len() == 4 && grid_dims[3] == 2,
                    "GridSample {}: grid must be [N,H_out,W_out,2], got {:?}",
                    layer.name,
                    grid_dims
                );

                let result: Vec<i64> = if mode == "nearest" {
                    (0..output_total)
                        .map(|out_flat| {
                            // Unravel (n_i, c_i, h_i, w_i) from out_flat.
                            let w_i = out_flat % w_out;
                            let h_i = (out_flat / w_out) % h_out;
                            let c_i = (out_flat / (w_out * h_out)) % c;
                            let n_i = out_flat / (w_out * h_out * c);

                            let sp = n_i * h_out * w_out + h_i * w_out + w_i;
                            let x_norm = grid_flat[sp * 2] as f64 / alpha;
                            let y_norm = grid_flat[sp * 2 + 1] as f64 / alpha;

                            let x_cont = gs_unnormalize(x_norm, w_in, align_corners);
                            let y_cont = gs_unnormalize(y_norm, h_in, align_corners);

                            let (y_px, x_px) = match (
                                gs_apply_padding_nearest(y_cont, h_in, padding_mode, align_corners),
                                gs_apply_padding_nearest(x_cont, w_in, padding_mode, align_corners),
                            ) {
                                (Some(y), Some(x)) => (y, x),
                                _ => return 0, // zeros padding, OOB
                            };

                            let in_flat =
                                n_i * c * h_in * w_in + c_i * h_in * w_in + y_px * w_in + x_px;
                            x_data.get(in_flat).copied().unwrap_or(0)
                        })
                        .collect()
                } else {
                    // Bilinear interpolation.
                    let alpha_i = model.scale_config.alpha;
                    (0..output_total)
                        .map(|out_flat| {
                            let w_i = out_flat % w_out;
                            let h_i = (out_flat / w_out) % h_out;
                            let c_i = (out_flat / (w_out * h_out)) % c;
                            let n_i = out_flat / (w_out * h_out * c);

                            let sp = n_i * h_out * w_out + h_i * w_out + w_i;
                            let x_norm = grid_flat[sp * 2] as f64 / alpha;
                            let y_norm = grid_flat[sp * 2 + 1] as f64 / alpha;

                            let x_cont = gs_unnormalize(x_norm, w_in, align_corners);
                            let y_cont = gs_unnormalize(y_norm, h_in, align_corners);

                            let (x_adj, y_adj) = if padding_mode == "reflection" {
                                (
                                    gs_reflect(x_cont, w_in, align_corners),
                                    gs_reflect(y_cont, h_in, align_corners),
                                )
                            } else {
                                (x_cont, y_cont)
                            };

                            let (h_fl, h_ce, wh_fl, wh_ce) = interp_corners(y_adj, h_in);
                            let (w_fl, w_ce, ww_fl, ww_ce) = interp_corners(x_adj, w_in);

                            // 4 corners: (h_fl,w_fl),(h_fl,w_ce),(h_ce,w_fl),(h_ce,w_ce)
                            let corners = [
                                (h_fl, w_fl, wh_fl * ww_fl),
                                (h_fl, w_ce, wh_fl * ww_ce),
                                (h_ce, w_fl, wh_ce * ww_fl),
                                (h_ce, w_ce, wh_ce * ww_ce),
                            ];

                            let mut sum_i128: i128 = 0;
                            for (ch, cw, wf) in corners {
                                // Check OOB for zeros padding.
                                let valid = padding_mode != "zeros"
                                    || (y_adj >= -0.5
                                        && y_adj <= h_in as f64 - 0.5
                                        && x_adj >= -0.5
                                        && x_adj <= w_in as f64 - 0.5
                                        && ch < h_in
                                        && cw < w_in);
                                if !valid {
                                    continue;
                                }
                                let in_flat =
                                    n_i * c * h_in * w_in + c_i * h_in * w_in + ch * w_in + cw;
                                let x_q = x_data.get(in_flat).copied().unwrap_or(0);
                                let w_q = (wf * alpha_i as f64).round() as i128;
                                sum_i128 += x_q as i128 * w_q;
                            }
                            let half = alpha_i as i128 / 2;
                            ((sum_i128 + half) / alpha_i as i128)
                                .clamp(i64::MIN as i128, i64::MAX as i128)
                                as i64
                        })
                        .collect()
                };

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let gridsample_out_name = format!("{}_out", layer.name);
                shreds.insert(gridsample_out_name, padded.clone());

                let input_layout = tensor_layouts.get(x_name);
                let output_layout = layout_from_output_shape_runtime(output_shape, input_layout);

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = output_layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Transpose => {
                // Transpose reorders elements according to the `perm` attribute.
                // Like Resize (nearest) this is a structural operation computed
                // by the prover and supplied as a committed shred.
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Transpose {} has no input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Transpose {} input '{}' not computed",
                            layer.name,
                            input_name
                        )
                    })?
                    .clone();

                let output_shape = &layer.output_shape;
                let output_total: usize = output_shape.iter().product();

                // Recover input shape from the output shape + perm.
                // perm[i] = axis of the input that maps to output axis i.
                let rank = output_shape.len();
                let perm: Vec<usize> = if let Some(raw) = layer.get_ints_attr("perm") {
                    raw.iter()
                        .map(|&a| if a < 0 { rank as i64 + a } else { a } as usize)
                        .collect()
                } else {
                    (0..rank).rev().collect()
                };

                // Reconstruct input_shape: input_shape[perm[i]] = output_shape[i]
                let mut input_shape = vec![0usize; rank];
                for (i, &p) in perm.iter().enumerate() {
                    input_shape[p] = output_shape[i];
                }

                // Build inverse permutation: inv_perm[p] = i  (output axis i came from input axis p).
                let mut inv_perm = vec![0usize; rank];
                for (i, &p) in perm.iter().enumerate() {
                    inv_perm[p] = i;
                }

                let result: Vec<i64> = (0..output_total)
                    .map(|out_flat| {
                        let out_coords = unravel_index_witness(out_flat, output_shape);
                        // in_coords[p] = out_coords[inv_perm[p]]
                        let in_coords: Vec<usize> =
                            (0..rank).map(|p| out_coords[inv_perm[p]]).collect();
                        let in_flat = ravel_index_witness(&in_coords, &input_shape);
                        input_data.get(in_flat).copied().unwrap_or(0)
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let transpose_out_name = format!("{}_out", layer.name);
                shreds.insert(transpose_out_name, padded.clone());

                let input_layout = tensor_layouts.get(input_name);
                let output_layout = transpose_layout_runtime(input_layout, &perm, output_shape);

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = output_layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    } else {
                        tensor_layouts.remove(out);
                    }
                }
            }
            OpType::Slice => {
                // Slice extracts a sub-tensor — purely structural, committed shred.
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Slice {} has no data input", layer.name))?;
                let input_data = tensors
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("Slice {} input '{}' not computed", layer.name, input_name)
                    })?
                    .clone();

                // Resolve input shape from the producing layer.
                let input_shape: Vec<usize> = model
                    .graph
                    .layers
                    .iter()
                    .find(|l| l.outputs.first().map(String::as_str) == Some(input_name.as_str()))
                    .map(|l| l.output_shape.clone())
                    .or_else(|| model.graph.input_shapes.get(input_name).cloned())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Slice {} cannot determine input shape for '{}'",
                            layer.name,
                            input_name
                        )
                    })?;

                let output_shape = &layer.output_shape;
                let output_total: usize = output_shape.iter().product();
                let rank = input_shape.len();

                // Read starts (required, input[1]).
                let starts_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Slice {} missing starts input", layer.name))?;
                let starts = layer
                    .weights
                    .get(starts_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Slice {} starts '{}' not found in weights",
                            layer.name,
                            starts_name
                        )
                    })?
                    .as_i64_vec();

                // Read ends (required, input[2]).
                let ends_name = layer
                    .inputs
                    .get(2)
                    .ok_or_else(|| anyhow::anyhow!("Slice {} missing ends input", layer.name))?;
                let _ends = layer
                    .weights
                    .get(ends_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Slice {} ends '{}' not found in weights",
                            layer.name,
                            ends_name
                        )
                    })?
                    .as_i64_vec();

                // Read axes (optional, input[3]).
                let axes: Vec<i64> = layer
                    .inputs
                    .get(3)
                    .filter(|n| !n.is_empty())
                    .and_then(|n| layer.weights.get(n))
                    .map(|td| td.as_i64_vec())
                    .unwrap_or_else(|| (0..starts.len() as i64).collect());

                // Read steps (optional, input[4]).
                let steps: Vec<i64> = layer
                    .inputs
                    .get(4)
                    .filter(|n| !n.is_empty())
                    .and_then(|n| layer.weights.get(n))
                    .map(|td| td.as_i64_vec())
                    .unwrap_or_else(|| vec![1i64; starts.len()]);

                // Build per-axis start/step.
                let mut axis_start = vec![0usize; rank];
                let mut axis_step = vec![1usize; rank];
                for (i, &ax_raw) in axes.iter().enumerate() {
                    let ax = if ax_raw < 0 {
                        (rank as i64 + ax_raw) as usize
                    } else {
                        ax_raw as usize
                    };
                    let dim = input_shape[ax] as i64;
                    let s = starts.get(i).copied().unwrap_or(0);
                    let s = if s < 0 { dim + s } else { s };
                    axis_start[ax] = s.clamp(0, dim) as usize;
                    let step = steps.get(i).copied().unwrap_or(1);
                    anyhow::ensure!(
                        step > 0,
                        "Slice {}: unsupported non-positive step {} for axis {} (starts={:?}, steps={:?})",
                        layer.name,
                        step,
                        ax,
                        starts,
                        steps
                    );
                    axis_step[ax] = step as usize;
                }

                let result: Vec<i64> = (0..output_total)
                    .map(|out_flat| {
                        let out_coords = unravel_index_witness(out_flat, output_shape);
                        let in_coords: Vec<usize> = (0..rank)
                            .map(|ax| axis_start[ax] + out_coords[ax] * axis_step[ax])
                            .collect();
                        let in_flat = ravel_index_witness(&in_coords, &input_shape);
                        input_data.get(in_flat).copied().unwrap_or(0)
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let slice_out_name = format!("{}_out", layer.name);
                shreds.insert(slice_out_name, padded.clone());

                let input_layout = tensor_layouts.get(input_name);
                let output_layout = layout_from_output_shape_runtime(output_shape, input_layout);

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    if let Some(ref layout) = output_layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::Concat => {
                // Concat joins inputs along axis — purely structural, committed shred.
                let output_shape = &layer.output_shape;
                let output_total: usize = output_shape.iter().product();
                let rank = output_shape.len();

                let raw_axis = layer.get_int_attr("axis").unwrap_or(0);
                let axis = if raw_axis < 0 {
                    (rank as i64 + raw_axis) as usize
                } else {
                    raw_axis as usize
                };

                // Collect input data and shapes.
                // Each input's actual shape is resolved by finding the producing layer
                // in the graph (compile.rs stores the first-output shape in layer.output_shape).
                let mut input_datas: Vec<Vec<i64>> = Vec::new();
                let mut input_shapes: Vec<Vec<usize>> = Vec::new();

                for input_name in &layer.inputs {
                    let data = tensors
                        .get(input_name)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Concat {} input '{}' not computed",
                                layer.name,
                                input_name
                            )
                        })?
                        .clone();

                    // Resolve actual shape from the producing layer's recorded output_shape.
                    let in_shape = model
                        .graph
                        .layers
                        .iter()
                        .find(|l| {
                            l.outputs.first().map(String::as_str) == Some(input_name.as_str())
                        })
                        .map(|l| l.output_shape.clone())
                        .or_else(|| model.graph.input_shapes.get(input_name).cloned())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Concat {}: could not resolve shape for input '{}' on axis {}",
                                layer.name,
                                input_name,
                                axis
                            )
                        })?;

                    input_shapes.push(in_shape);
                    input_datas.push(data);
                }

                // Build cumulative axis offsets.
                let mut axis_offsets = vec![0usize; input_shapes.len() + 1];
                for (i, s) in input_shapes.iter().enumerate() {
                    axis_offsets[i + 1] = axis_offsets[i] + s.get(axis).copied().unwrap_or(0);
                }

                let result: Vec<i64> = (0..output_total)
                    .map(|out_flat| {
                        let out_coords = unravel_index_witness(out_flat, output_shape);
                        let ax_coord = out_coords[axis];
                        let input_idx = axis_offsets
                            .partition_point(|&o| o <= ax_coord)
                            .saturating_sub(1)
                            .min(input_datas.len() - 1);
                        let local_ax = ax_coord - axis_offsets[input_idx];
                        let mut in_coords = out_coords.clone();
                        in_coords[axis] = local_ax;
                        let in_flat = ravel_index_witness(&in_coords, &input_shapes[input_idx]);
                        input_datas[input_idx].get(in_flat).copied().unwrap_or(0)
                    })
                    .collect();

                let padded = pad_to_size(&result, next_power_of_two(output_total));
                let concat_out_name = format!("{}_out", layer.name);
                shreds.insert(concat_out_name, padded.clone());

                for out in &layer.outputs {
                    tensors.insert(out.clone(), padded.clone());
                    tensor_layouts.remove(out);
                }
            }
            other => {
                bail!(
                    "witness: unsupported op type {:?} in layer {}",
                    other,
                    layer.name
                );
            }
        }
    }

    let final_output = tensors
        .get(declared_output)
        .ok_or_else(|| anyhow::anyhow!("declared output '{declared_output}' not computed"))?;
    shreds.insert("expected_output".to_string(), final_output.clone());

    let rc_plan = compute_range_check_plan_with_overrides(model, &observed_n_bits)?;

    let mut grouped: std::collections::BTreeMap<(usize, usize), Vec<String>> =
        std::collections::BTreeMap::new();
    for (&table_nv, shred_names) in &rc_plan {
        let mut by_nv: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();
        for name in shred_names {
            let nv = shreds
                .get(name)
                .map(|s| num_vars_for(s.len()))
                .ok_or_else(|| {
                    anyhow::anyhow!("range check shred '{name}' not found in witness shreds")
                })?;
            by_nv.entry(nv).or_default().push(name.clone());
        }
        for (node_nv, names) in by_nv {
            grouped
                .entry((table_nv, node_nv))
                .or_default()
                .extend(names);
        }
    }

    for (&(table_nv, node_nv), shred_names) in &grouped {
        anyhow::ensure!(
            table_nv < 63,
            "table_nv {table_nv} is too large for range table construction"
        );
        let table_shred_name = format!("range_table_{table_nv}");
        if !shreds.contains_key(&table_shred_name) {
            let table_data: Vec<i64> = (0..(1i64 << table_nv)).collect();
            shreds.insert(table_shred_name, table_data);
        }

        let real_count = shred_names.len();
        let target_count = real_count.next_power_of_two();
        let dummy_count = target_count - real_count;
        let dummy_size = 1usize << node_nv;
        for i in 0..dummy_count {
            let dummy_name = format!("range_dummy_t{table_nv}_n{node_nv}_{i}");
            let dummy_data = vec![0i64; dummy_size];
            let dummy_mults = compute_multiplicities(&dummy_data, 1 << table_nv)?;
            shreds.insert(format!("{dummy_name}_mults"), dummy_mults);
            shreds.insert(dummy_name, dummy_data);
        }
    }

    Ok(WitnessData {
        shreds,
        observed_n_bits,
    })
}

fn compute_range_check_plan_with_overrides(
    model: &QuantizedModel,
    observed_n_bits: &HashMap<String, usize>,
) -> Result<std::collections::BTreeMap<usize, Vec<String>>> {
    let exponent = model.scale_config.exponent as usize;
    let mut plan: std::collections::BTreeMap<usize, Vec<String>> =
        std::collections::BTreeMap::new();

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm | OpType::Conv | OpType::BatchNormalization => {
                if exponent <= RANGE_CHECK_CHUNK_BITS {
                    plan.entry(exponent)
                        .or_default()
                        .push(format!("{}_r", layer.name));
                } else {
                    plan.entry(RANGE_CHECK_CHUNK_BITS)
                        .or_default()
                        .push(format!("{}_r_c0", layer.name));
                    plan.entry(exponent - RANGE_CHECK_CHUNK_BITS)
                        .or_default()
                        .push(format!("{}_r_c1", layer.name));
                }
            }
            OpType::Relu => {
                let n_bits = observed_n_bits.get(&layer.name).copied().or(layer.n_bits);
                if let Some(n_bits) = n_bits {
                    let dnv = delta_table_nv(n_bits, exponent);
                    if dnv > RANGE_CHECK_CHUNK_BITS {
                        plan.entry(RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_di_c0", layer.name));
                        plan.entry(RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_dz_c0", layer.name));
                        plan.entry(dnv - RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_di_c1", layer.name));
                        plan.entry(dnv - RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_dz_c1", layer.name));
                    } else {
                        let entry = plan.entry(dnv).or_default();
                        entry.push(format!("{}_di", layer.name));
                        entry.push(format!("{}_dz", layer.name));
                    }
                }
            }
            OpType::MaxPool => {
                let n_bits = observed_n_bits.get(&layer.name).copied().or(layer.n_bits);
                if let Some(n_bits) = n_bits {
                    let dnv = delta_table_nv(n_bits, exponent);
                    let kernel_shape = layer.get_ints_attr("kernel_shape").ok_or_else(|| {
                        anyhow::anyhow!("MaxPool {} missing kernel_shape attribute", layer.name)
                    })?;
                    anyhow::ensure!(
                        kernel_shape.len() == 2,
                        "MaxPool {} requires exactly 2D kernel_shape, got {} dims",
                        layer.name,
                        kernel_shape.len()
                    );
                    anyhow::ensure!(
                        kernel_shape[0] > 0 && kernel_shape[1] > 0,
                        "MaxPool {} kernel_shape values must be positive, got [{}, {}]",
                        layer.name,
                        kernel_shape[0],
                        kernel_shape[1]
                    );
                    let window_size = (kernel_shape[0] as usize)
                        .checked_mul(kernel_shape[1] as usize)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "MaxPool {} kernel window_size overflow: {} * {}",
                                layer.name,
                                kernel_shape[0],
                                kernel_shape[1]
                            )
                        })?;
                    if dnv > RANGE_CHECK_CHUNK_BITS {
                        for i in 0..window_size {
                            plan.entry(RANGE_CHECK_CHUNK_BITS)
                                .or_default()
                                .push(format!("{}_d{}_c0", layer.name, i));
                            plan.entry(dnv - RANGE_CHECK_CHUNK_BITS)
                                .or_default()
                                .push(format!("{}_d{}_c1", layer.name, i));
                        }
                    } else {
                        let entry = plan.entry(dnv).or_default();
                        for i in 0..window_size {
                            entry.push(format!("{}_d{}", layer.name, i));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(plan)
}

fn padded_matmul(
    a: &[i64],
    a_rows: usize,
    a_cols: usize,
    b: &[i64],
    b_cols: usize,
) -> anyhow::Result<Vec<i64>> {
    let alloc_size = a_rows.checked_mul(b_cols).ok_or_else(|| {
        anyhow::anyhow!("padded_matmul: a_rows * b_cols overflow: {a_rows}x{b_cols}")
    })?;
    let mut out = vec![0i64; alloc_size];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0i128;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] as i128 * b[k * b_cols + j] as i128;
            }
            out[i * b_cols + j] = i64::try_from(sum).map_err(|_| {
                anyhow::anyhow!("matmul accumulator overflows i64 at [{i}][{j}]: {sum}")
            })?;
        }
    }
    Ok(out)
}

fn broadcast_bias(bias: &[i64], bias_len: usize, total: usize) -> Vec<i64> {
    let mut out = vec![0i64; total];
    for j in 0..bias_len.min(total) {
        out[j] = bias[j];
    }
    out
}

pub fn prepare_public_shreds(
    model: &QuantizedModel,
    quantized_input: &[i64],
    expected_output: &[i64],
    observed_n_bits: &HashMap<String, usize>,
) -> Result<HashMap<String, Vec<i64>>> {
    anyhow::ensure!(
        model.scale_config.base == 2,
        "range check tables require base == 2, got base = {}",
        model.scale_config.base
    );
    anyhow::ensure!(
        model.scale_config.exponent < 63,
        "scale_config.exponent {} is too large for range table construction",
        model.scale_config.exponent
    );

    let mut shreds: HashMap<String, Vec<i64>> = HashMap::new();
    let mut tensor_sizes: HashMap<String, usize> = HashMap::new();
    let mut tensor_layouts: HashMap<String, SpatialInfo> = HashMap::new();

    let input_name = model
        .graph
        .input_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no input names defined"))?
        .clone();

    validate_input_size(model, &input_name, quantized_input.len())?;

    let input_padded_size = next_power_of_two(quantized_input.len());
    shreds.insert(
        input_name.clone(),
        pad_to_size(quantized_input, input_padded_size),
    );
    tensor_sizes.insert(input_name.clone(), input_padded_size);

    for (name, shape) in &model.graph.input_shapes {
        if shape.len() == 3 {
            tensor_layouts.insert(
                name.clone(),
                SpatialInfo::CHW {
                    c: shape[0],
                    h: shape[1],
                    w: shape[2],
                },
            );
        }
    }

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm => {
                let weight_tensor_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} has no weight", layer.name))?;
                let bias_tensor_name = layer.inputs.get(2);
                let weight_data = layer.weights.get(weight_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("Gemm {} missing weight {}", layer.name, weight_tensor_name)
                })?;
                let trans_b = layer
                    .get_int_attr("transB")
                    .map(|v| v != 0)
                    .unwrap_or(false);
                let w_shape = weight_data.shape();
                anyhow::ensure!(
                    w_shape.len() >= 2,
                    "Gemm {} weight has {} dims, need >= 2",
                    layer.name,
                    w_shape.len()
                );
                let (w_rows, w_cols) = (w_shape[0], w_shape[1]);
                let (k_dim, n_dim) = if trans_b {
                    (w_cols, w_rows)
                } else {
                    (w_rows, w_cols)
                };
                let k_padded = next_power_of_two(k_dim);
                let n_padded = next_power_of_two(n_dim);

                let w_transposed = if trans_b {
                    transpose_matrix(&weight_data.as_i64_vec(), w_rows, w_cols)
                } else {
                    weight_data.as_i64_vec()
                };
                shreds.insert(
                    format!("{}_weight", layer.name),
                    pad_matrix(&w_transposed, k_dim, n_dim, k_padded, n_padded)?,
                );

                let bias_padded = if let Some(bias_name) = bias_tensor_name {
                    if let Some(bias_data) = layer.weights.get(bias_name) {
                        let b = bias_data.as_i64_vec();
                        broadcast_bias(&b, b.len(), n_padded)
                    } else {
                        vec![0i64; n_padded]
                    }
                } else {
                    vec![0i64; n_padded]
                };
                shreds.insert(format!("{}_bias", layer.name), bias_padded);

                let out_size = n_padded;
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), out_size);
                }
            }
            OpType::Conv => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Conv {} has no input", layer.name))?;
                let weight_tensor_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Conv {} has no weight", layer.name))?;
                let bias_tensor_name = layer.inputs.get(2);
                let weight_data = layer.weights.get(weight_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!("Conv {} missing weight {}", layer.name, weight_tensor_name)
                })?;
                let input_layout = tensor_layouts
                    .get(input_tensor_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Conv {} input {} has no spatial layout",
                            layer.name,
                            input_tensor_name
                        )
                    })?
                    .clone();

                let w_shape = weight_data.shape();
                anyhow::ensure!(
                    w_shape.len() >= 4,
                    "Conv {} weight has {} dims, need >= 4",
                    layer.name,
                    w_shape.len()
                );
                let c_out = w_shape[0];
                let c_in = w_shape[1];
                let kh = w_shape[2];
                let kw = w_shape[3];
                let pads = layer.get_ints_attr("pads");
                let pad_top = pads.and_then(|p| p.first().copied()).unwrap_or(0) as usize;
                let pad_left = pads.and_then(|p| p.get(1).copied()).unwrap_or(0) as usize;
                let pad_bottom = pads.and_then(|p| p.get(2).copied()).unwrap_or(0) as usize;
                let pad_right = pads.and_then(|p| p.get(3).copied()).unwrap_or(0) as usize;
                let strides = layer.get_ints_attr("strides");
                let raw_stride_h = strides.and_then(|s| s.first().copied()).unwrap_or(1);
                let raw_stride_w = strides.and_then(|s| s.get(1).copied()).unwrap_or(1);
                anyhow::ensure!(
                    raw_stride_h > 0 && raw_stride_w > 0,
                    "Conv {} stride_h={} stride_w={} must be positive",
                    layer.name,
                    raw_stride_h,
                    raw_stride_w
                );
                let stride_h = raw_stride_h as usize;
                let stride_w = raw_stride_w as usize;
                let (input_ch, in_h, in_w) = input_layout.spatial_dims();
                anyhow::ensure!(
                    input_ch == c_in,
                    "Conv {}: weight c_in {} does not match input channels {}",
                    layer.name,
                    c_in,
                    input_ch
                );
                let padded_h = in_h + pad_top + pad_bottom;
                let padded_w = in_w + pad_left + pad_right;
                anyhow::ensure!(
                    padded_h >= kh && padded_w >= kw,
                    "Conv {}: padded input {}x{} smaller than kernel {}x{}",
                    layer.name,
                    padded_h,
                    padded_w,
                    kh,
                    kw
                );
                let out_h = (padded_h - kh) / stride_h + 1;
                let out_w = (padded_w - kw) / stride_w + 1;
                let patch_size = c_in
                    .checked_mul(kh)
                    .and_then(|v| v.checked_mul(kw))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Conv {} patch_size overflow: {} * {} * {}",
                            layer.name,
                            c_in,
                            kh,
                            kw
                        )
                    })?;
                let num_patches = out_h.checked_mul(out_w).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Conv {} num_patches overflow: {} * {}",
                        layer.name,
                        out_h,
                        out_w
                    )
                })?;
                let pad_psize = next_power_of_two(patch_size);
                let pad_cout = next_power_of_two(c_out);
                let pad_patches = next_power_of_two(num_patches);
                let result_size = pad_patches.checked_mul(pad_cout).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Conv {} result_size overflow: {} * {}",
                        layer.name,
                        pad_patches,
                        pad_cout
                    )
                })?;

                let kernel_t = transpose_matrix(&weight_data.as_i64_vec(), c_out, patch_size);
                shreds.insert(
                    format!("{}_weight", layer.name),
                    pad_matrix(&kernel_t, patch_size, c_out, pad_psize, pad_cout)?,
                );

                let bias_bc = if let Some(bias_name) = bias_tensor_name {
                    if let Some(bias_data) = layer.weights.get(bias_name) {
                        let b = bias_data.as_i64_vec();
                        (0..result_size)
                            .map(|i| {
                                let j = i % pad_cout;
                                if j < c_out && j < b.len() {
                                    b[j]
                                } else {
                                    0
                                }
                            })
                            .collect()
                    } else {
                        vec![0i64; result_size]
                    }
                } else {
                    vec![0i64; result_size]
                };
                shreds.insert(format!("{}_bias", layer.name), bias_bc);

                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), result_size);
                    tensor_layouts.insert(
                        out_name.clone(),
                        SpatialInfo::HWC {
                            h: out_h,
                            w: out_w,
                            c: c_out,
                            stride_c: pad_cout,
                        },
                    );
                }
            }
            OpType::Relu => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Relu {} has no input", layer.name))?;
                let sz = tensor_sizes
                    .get(input_tensor_name)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Relu {} input {} not found in tensor_sizes",
                            layer.name,
                            input_tensor_name
                        )
                    })?;
                let nv = num_vars_for(sz);
                shreds.insert(format!("{}_zero", layer.name), vec![0i64; 1 << nv]);

                let layout = tensor_layouts.get(input_tensor_name).cloned();
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out_name.clone(), layout.clone());
                    }
                }
            }
            OpType::MaxPool => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} has no input", layer.name))?;
                let input_layout = tensor_layouts
                    .get(input_tensor_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MaxPool {} input {} has no spatial layout",
                            layer.name,
                            input_tensor_name
                        )
                    })?
                    .clone();
                let (c, in_h, in_w) = input_layout.spatial_dims();
                let kernel_shape = layer.get_ints_attr("kernel_shape").ok_or_else(|| {
                    anyhow::anyhow!("MaxPool {} missing kernel_shape", layer.name)
                })?;
                anyhow::ensure!(
                    kernel_shape.len() == 2,
                    "MaxPool {} requires exactly 2D kernel_shape, got {} dims",
                    layer.name,
                    kernel_shape.len()
                );
                anyhow::ensure!(
                    kernel_shape[0] > 0 && kernel_shape[1] > 0,
                    "MaxPool {} kernel_shape values must be positive, got [{}, {}]",
                    layer.name,
                    kernel_shape[0],
                    kernel_shape[1]
                );
                let pool_h = kernel_shape[0] as usize;
                let pool_w = kernel_shape[1] as usize;
                let strides = layer.get_ints_attr("strides");
                let raw_stride_h = strides.and_then(|s| s.first().copied()).unwrap_or(1);
                let raw_stride_w = strides.and_then(|s| s.get(1).copied()).unwrap_or(1);
                anyhow::ensure!(
                    raw_stride_h > 0 && raw_stride_w > 0,
                    "MaxPool {} stride_h={} stride_w={} must be positive",
                    layer.name,
                    raw_stride_h,
                    raw_stride_w
                );
                let stride_h = raw_stride_h as usize;
                let stride_w = raw_stride_w as usize;
                anyhow::ensure!(
                    in_h >= pool_h && in_w >= pool_w,
                    "MaxPool {}: input {}x{} smaller than kernel {}x{}",
                    layer.name,
                    in_h,
                    in_w,
                    pool_h,
                    pool_w
                );
                let pool_oh = (in_h - pool_h) / stride_h + 1;
                let pool_ow = (in_w - pool_w) / stride_w + 1;
                let num_pool_out = pool_oh
                    .checked_mul(pool_ow)
                    .and_then(|v| v.checked_mul(c))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "MaxPool {} num_pool_out overflow: {} * {} * {}",
                            layer.name,
                            pool_oh,
                            pool_ow,
                            c
                        )
                    })?;
                let pad_pool = next_power_of_two(num_pool_out);

                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), pad_pool);
                    tensor_layouts.insert(
                        out_name.clone(),
                        SpatialInfo::HWC {
                            h: pool_oh,
                            w: pool_ow,
                            c,
                            stride_c: c,
                        },
                    );
                }
            }
            OpType::BatchNormalization => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("BatchNorm {} has no input", layer.name))?;
                let mul_tensor_name = layer
                    .inputs
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing mul input", layer.name))?;
                let add_tensor_name = layer
                    .inputs
                    .get(2)
                    .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing add input", layer.name))?;
                let mul_data = layer.weights.get(mul_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "BatchNorm {} missing weight '{}'",
                        layer.name,
                        mul_tensor_name
                    )
                })?;
                let add_data = layer.weights.get(add_tensor_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "BatchNorm {} missing weight '{}'",
                        layer.name,
                        add_tensor_name
                    )
                })?;
                let mul_per_ch = mul_data.as_i64_vec();
                let add_per_ch = add_data.as_i64_vec();
                let ch = mul_per_ch.len();
                anyhow::ensure!(
                    add_per_ch.len() == ch,
                    "BatchNorm {} mul/add length mismatch: {} vs {}",
                    layer.name,
                    ch,
                    add_per_ch.len()
                );
                let sz = tensor_sizes
                    .get(input_tensor_name)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "BatchNorm {} input {} not found in tensor_sizes",
                            layer.name,
                            input_tensor_name
                        )
                    })?;
                let padded_size = next_power_of_two(sz);

                let mut mul_broadcast = vec![0i64; padded_size];
                let mut add_broadcast = vec![0i64; padded_size];
                let input_layout = tensor_layouts.get(input_tensor_name).cloned();

                match &input_layout {
                    Some(SpatialInfo::CHW { h, w, .. }) => {
                        let hw = h * w;
                        for c in 0..ch {
                            for s in 0..hw {
                                let idx = c * hw + s;
                                if idx < padded_size {
                                    mul_broadcast[idx] = mul_per_ch[c];
                                    add_broadcast[idx] = add_per_ch[c];
                                }
                            }
                        }
                    }
                    Some(SpatialInfo::HWC { h, w, stride_c, .. }) => {
                        for row in 0..(h * w) {
                            for c in 0..ch {
                                let idx = row * stride_c + c;
                                if idx < padded_size {
                                    mul_broadcast[idx] = mul_per_ch[c];
                                    add_broadcast[idx] = add_per_ch[c];
                                }
                            }
                        }
                    }
                    None => {
                        let spatial = if ch > 0 && sz > ch {
                            anyhow::ensure!(sz % ch == 0,
                                "BatchNorm {} has no spatial layout and size {} is not divisible by channels {}",
                                layer.name, sz, ch);
                            sz / ch
                        } else {
                            1
                        };
                        for c in 0..ch {
                            for s in 0..spatial {
                                let idx = c * spatial + s;
                                if idx < padded_size {
                                    mul_broadcast[idx] = mul_per_ch[c];
                                    add_broadcast[idx] = add_per_ch[c];
                                }
                            }
                        }
                    }
                }

                shreds.insert(format!("{}_mul", layer.name), mul_broadcast);
                shreds.insert(format!("{}_add", layer.name), add_broadcast);

                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), padded_size);
                    if let Some(ref layout) = input_layout {
                        tensor_layouts.insert(out_name.clone(), layout.clone());
                    }
                }
            }
            OpType::Add | OpType::Sub => {
                let input_a_name = layer.inputs.first().ok_or_else(|| {
                    anyhow::anyhow!("{:?} {} has no first input", layer.op_type, layer.name)
                })?;
                let input_b_name = layer.inputs.get(1).ok_or_else(|| {
                    anyhow::anyhow!("{:?} {} has no second input", layer.op_type, layer.name)
                })?;

                let a_is_tensor = tensor_sizes.contains_key(input_a_name);
                let b_is_tensor = tensor_sizes.contains_key(input_b_name);

                let a_sz = if a_is_tensor {
                    tensor_sizes.get(input_a_name).copied().unwrap_or(1)
                } else {
                    layer
                        .weights
                        .get(input_a_name)
                        .map(|w| w.as_i64_vec().len())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "{:?} {} missing weight for input {}",
                                layer.op_type,
                                layer.name,
                                input_a_name
                            )
                        })?
                };
                let b_sz = if b_is_tensor {
                    tensor_sizes.get(input_b_name).copied().unwrap_or(1)
                } else {
                    layer
                        .weights
                        .get(input_b_name)
                        .map(|w| w.as_i64_vec().len())
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "{:?} {} missing weight for input {}",
                                layer.op_type,
                                layer.name,
                                input_b_name
                            )
                        })?
                };
                let out_sz = next_power_of_two(a_sz.max(b_sz));

                if !b_is_tensor {
                    if let Some(w) = layer.weights.get(input_b_name) {
                        let data = w.as_i64_vec();
                        shreds.insert(
                            format!("{}_{}", layer.name, input_b_name),
                            pad_to_size(&data, out_sz),
                        );
                    }
                }
                if !a_is_tensor {
                    if let Some(w) = layer.weights.get(input_a_name) {
                        let data = w.as_i64_vec();
                        shreds.insert(
                            format!("{}_{}", layer.name, input_a_name),
                            pad_to_size(&data, out_sz),
                        );
                    }
                }

                let layout = tensor_layouts
                    .get(input_a_name)
                    .or_else(|| tensor_layouts.get(input_b_name))
                    .cloned();
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), out_sz);
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out_name.clone(), layout.clone());
                    }
                }
            }
            OpType::Cast
            | OpType::Reshape
            | OpType::Flatten
            | OpType::Squeeze
            | OpType::Unsqueeze => {
                let input_tensor_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("shape op {} has no input", layer.name))?;
                let sz = tensor_sizes
                    .get(input_tensor_name)
                    .copied()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{:?} {} input {} not found in tensor_sizes",
                            layer.op_type,
                            layer.name,
                            input_tensor_name
                        )
                    })?;
                let layout = tensor_layouts.get(input_tensor_name).cloned();
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out_name.clone(), layout.clone());
                    }
                }
            }
            OpType::Exp
            | OpType::Sigmoid
            | OpType::Gelu
            | OpType::Softmax
            | OpType::Tile
            | OpType::TopK => {
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);

                let input_name = layer.inputs.first().ok_or_else(|| {
                    anyhow::anyhow!("{:?} {} has no input", layer.op_type, layer.name)
                })?;
                let layout = tensor_layouts.get(input_name).cloned();

                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out_name.clone(), layout.clone());
                    }
                }
            }
            OpType::Gather => {
                // The gather output is a committed shred whose values are computed
                // by the prover (compute_witness). Here we only track the output
                // size so that downstream ops can derive their own sizes correctly.
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            OpType::LayerNormalization => {
                // LayerNorm output is a committed shred computed by the prover.
                // Output shape is the same as the input shape (passthrough shape).
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            OpType::Resize => {
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            OpType::GridSample => {
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            OpType::Transpose => {
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            OpType::Concat => {
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            OpType::Slice => {
                let out_total: usize = layer.output_shape.iter().product();
                let sz = next_power_of_two(out_total);
                for out_name in &layer.outputs {
                    tensor_sizes.insert(out_name.clone(), sz);
                }
            }
            other => {
                bail!(
                    "prepare_public_shreds: unsupported op type {:?} in layer {}",
                    other,
                    layer.name
                );
            }
        }
    }

    let out_padded_size = next_power_of_two(expected_output.len());
    shreds.insert(
        "expected_output".to_string(),
        pad_to_size(expected_output, out_padded_size),
    );

    let rc_plan = compute_range_check_plan_with_overrides(model, observed_n_bits)?;
    for (&table_nv, _) in &rc_plan {
        anyhow::ensure!(
            table_nv < 63,
            "table_nv {table_nv} is too large for range table construction"
        );
        let table_shred_name = format!("range_table_{table_nv}");
        if !shreds.contains_key(&table_shred_name) {
            let table_data: Vec<i64> = (0..(1i64 << table_nv)).collect();
            shreds.insert(table_shred_name, table_data);
        }
    }

    Ok(shreds)
}
