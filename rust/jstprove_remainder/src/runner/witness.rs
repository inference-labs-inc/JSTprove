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
            OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
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
            OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
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
