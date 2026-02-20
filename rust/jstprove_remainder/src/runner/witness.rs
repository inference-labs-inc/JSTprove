use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};

use crate::gadgets::rescale;
use crate::onnx::graph::OpType;
use crate::onnx::quantizer::QuantizedModel;
use crate::padding::{next_power_of_two, num_vars_for};
use crate::runner::circuit_builder::{transpose_matrix, pad_matrix, pad_to_size, SpatialInfo};

use super::serialization;

pub fn run(model_path: &Path, input_path: &Path, output_path: &Path, compress: bool) -> Result<()> {
    tracing::info!("loading model from {}", model_path.display());
    let model = super::compile::load_model(model_path)?;

    let quantized_input = load_and_quantize_input(input_path, model.scale_config.alpha)?;

    tracing::info!("computing witness for {} layers", model.graph.layers.len());
    let witness = compute_witness(&model, &quantized_input)?;

    let size = serialization::serialize_to_file(&witness, output_path, compress)?;
    tracing::info!(
        "witness written to {} ({} shreds, {} bytes)",
        output_path.display(),
        witness.len(),
        size
    );
    Ok(())
}

pub fn load_and_quantize_input(input_path: &Path, alpha: i64) -> Result<Vec<i64>> {
    let input_json: serde_json::Value = serde_json::from_reader(std::fs::File::open(input_path)?)?;
    quantize_input_json(&input_json, alpha)
}

pub fn quantize_input_json(input_json: &serde_json::Value, alpha: i64) -> Result<Vec<i64>> {
    let raw_input: Vec<f64> = input_json.get("input")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("input JSON must have an \"input\" array field"))?
        .iter()
        .enumerate()
        .map(|(i, v)| v.as_f64().ok_or_else(|| anyhow::anyhow!("input[{}] is not a number: {}", i, v)))
        .collect::<Result<Vec<f64>>>()?;

    Ok(raw_input.iter()
        .map(|&v| (v * alpha as f64).round() as i64)
        .collect())
}

pub fn load_witness(path: &Path) -> Result<HashMap<String, Vec<i64>>> {
    serialization::deserialize_from_file(path)
}

pub fn compute_witness(model: &QuantizedModel, quantized_input: &[i64]) -> Result<HashMap<String, Vec<i64>>> {
    let alpha = model.scale_config.alpha;
    let offset = 1i64 << 30;

    let mut shreds: HashMap<String, Vec<i64>> = HashMap::new();
    let mut tensors: HashMap<String, Vec<i64>> = HashMap::new();
    let mut tensor_layouts: HashMap<String, SpatialInfo> = HashMap::new();

    let input_name = model.graph.input_names.first()
        .ok_or_else(|| anyhow::anyhow!("model has no input names defined"))?
        .clone();

    let input_size = quantized_input.len();
    let input_padded_size = next_power_of_two(input_size);
    let input_padded = pad_to_size(quantized_input, input_padded_size);

    shreds.insert(input_name.clone(), input_padded.clone());
    tensors.insert(input_name.clone(), input_padded);

    for (name, shape) in &model.graph.input_shapes {
        if shape.len() == 3 {
            tensor_layouts.insert(
                name.clone(),
                SpatialInfo::CHW { c: shape[0], h: shape[1], w: shape[2] },
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

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm => {
                let input_tensor_name = layer.inputs.first()
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} has no input", layer.name))?;
                let weight_tensor_name = layer.inputs.get(1)
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} has no weight", layer.name))?;
                let bias_tensor_name = layer.inputs.get(2);

                let input_data = tensors.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} input {} not computed", layer.name, input_tensor_name))?;

                let weight_data = layer.weights.get(weight_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("Gemm {} missing weight {}", layer.name, weight_tensor_name))?;

                let trans_a = layer.get_int_attr("transA").map(|v| v != 0).unwrap_or(false);
                anyhow::ensure!(!trans_a, "Gemm {} has transA=1 which is not supported", layer.name);
                let trans_b = layer.get_int_attr("transB").map(|v| v != 0).unwrap_or(false);

                let w_shape = weight_data.shape();
                anyhow::ensure!(w_shape.len() >= 2, "Gemm {} weight has {} dims, need >= 2", layer.name, w_shape.len());
                let (w_rows, w_cols) = (w_shape[0], w_shape[1]);
                let (k_dim, n_dim) = if trans_b { (w_cols, w_rows) } else { (w_rows, w_cols) };

                let k_padded = next_power_of_two(k_dim);
                let n_padded = next_power_of_two(n_dim);

                let w_transposed = if trans_b {
                    transpose_matrix(&weight_data.as_i64_vec(), w_rows, w_cols)
                } else {
                    weight_data.as_i64_vec()
                };
                let w_padded = pad_matrix(&w_transposed, k_dim, n_dim, k_padded, n_padded);

                let input_padded_for_mm = pad_to_size(input_data, k_padded);
                let mm = padded_matmul(&input_padded_for_mm, 1, k_padded, &w_padded, n_padded);

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

                let mm_with_bias: Vec<i64> = mm.iter().zip(bias_padded.iter()).map(|(m, b)| m + b).collect();
                let (quotients, remainders) = rescale::compute_rescale_array(&mm_with_bias, alpha, offset);

                shreds.insert(format!("{}_weight", layer.name), w_padded);
                shreds.insert(format!("{}_bias", layer.name), bias_padded);
                shreds.insert(format!("{}_q", layer.name), quotients.clone());
                shreds.insert(format!("{}_r", layer.name), remainders);

                for out in &layer.outputs {
                    tensors.insert(out.clone(), quotients.clone());
                }
            }
            OpType::Conv => {
                let input_tensor_name = layer.inputs.first()
                    .ok_or_else(|| anyhow::anyhow!("Conv {} has no input", layer.name))?;
                let weight_tensor_name = layer.inputs.get(1)
                    .ok_or_else(|| anyhow::anyhow!("Conv {} has no weight", layer.name))?;
                let bias_tensor_name = layer.inputs.get(2);

                let input_data = tensors.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("Conv {} input {} not computed", layer.name, input_tensor_name))?;

                let weight_data = layer.weights.get(weight_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("Conv {} missing weight {}", layer.name, weight_tensor_name))?;

                let input_layout = tensor_layouts.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("Conv {} input {} has no spatial layout", layer.name, input_tensor_name))?
                    .clone();

                let w_shape = weight_data.shape();
                anyhow::ensure!(w_shape.len() >= 4, "Conv {} weight has {} dims, need >= 4", layer.name, w_shape.len());
                let c_out = w_shape[0];
                let c_in = w_shape[1];
                let kh = w_shape[2];
                let kw = w_shape[3];

                if let Some(pads) = layer.get_ints_attr("pads") {
                    anyhow::ensure!(pads.iter().all(|&p| p == 0), "Conv {} has non-zero pads {:?} which is not supported", layer.name, pads);
                }

                let strides = layer.get_ints_attr("strides");
                let stride_h = strides.and_then(|s| s.first()).map(|&v| v as usize).unwrap_or(1);
                let stride_w = strides.and_then(|s| s.get(1)).map(|&v| v as usize).unwrap_or(1);

                let (input_ch, in_h, in_w) = input_layout.spatial_dims();
                anyhow::ensure!(input_ch == c_in, "Conv {}: weight c_in {} does not match input channels {}", layer.name, c_in, input_ch);
                anyhow::ensure!(in_h >= kh && in_w >= kw, "Conv {}: input {}x{} smaller than kernel {}x{}", layer.name, in_h, in_w, kh, kw);
                let out_h = (in_h - kh) / stride_h + 1;
                let out_w = (in_w - kw) / stride_w + 1;
                let patch_size = c_in * kh * kw;
                let num_patches = out_h * out_w;

                let pad_patches = next_power_of_two(num_patches);
                let pad_psize = next_power_of_two(patch_size);
                let pad_cout = next_power_of_two(c_out);

                let mut im2col_data = vec![0i64; pad_patches * pad_psize];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let patch = oh * out_w + ow;
                        for c in 0..c_in {
                            for kr in 0..kh {
                                for kc in 0..kw {
                                    let col = c * kh * kw + kr * kw + kc;
                                    let ih = oh * stride_h + kr;
                                    let iw = ow * stride_w + kc;
                                    let src = input_layout.index(c, ih, iw);
                                    im2col_data[patch * pad_psize + col] = input_data[src];
                                }
                            }
                        }
                    }
                }

                let kernel_t = transpose_matrix(&weight_data.as_i64_vec(), c_out, patch_size);
                let kernel_padded = pad_matrix(&kernel_t, patch_size, c_out, pad_psize, pad_cout);
                let mm = padded_matmul(&im2col_data, pad_patches, pad_psize, &kernel_padded, pad_cout);

                let result_size = pad_patches * pad_cout;
                let bias_bc = if let Some(bias_name) = bias_tensor_name {
                    if let Some(bias_data) = layer.weights.get(bias_name) {
                        let b = bias_data.as_i64_vec();
                        (0..result_size)
                            .map(|i| {
                                let j = i % pad_cout;
                                if j < c_out && j < b.len() { b[j] } else { 0 }
                            })
                            .collect()
                    } else {
                        vec![0i64; result_size]
                    }
                } else {
                    vec![0i64; result_size]
                };

                let mm_with_bias: Vec<i64> = mm.iter().zip(bias_bc.iter()).map(|(m, b)| m + b).collect();
                let (quotients, remainders) = rescale::compute_rescale_array(&mm_with_bias, alpha, offset);

                shreds.insert(format!("{}_weight", layer.name), kernel_padded);
                shreds.insert(format!("{}_bias", layer.name), bias_bc);
                shreds.insert(format!("{}_q", layer.name), quotients.clone());
                shreds.insert(format!("{}_r", layer.name), remainders);

                for out in &layer.outputs {
                    tensors.insert(out.clone(), quotients.clone());
                    tensor_layouts.insert(out.clone(), SpatialInfo::HWC {
                        h: out_h, w: out_w, c: c_out, stride_c: pad_cout,
                    });
                }
            }
            OpType::Relu => {
                let input_tensor_name = layer.inputs.first()
                    .ok_or_else(|| anyhow::anyhow!("Relu {} has no input", layer.name))?;
                let input_data = tensors.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("Relu {} input {} not computed", layer.name, input_tensor_name))?;

                let nv = num_vars_for(input_data.len());
                let relu_out: Vec<i64> = input_data.iter().map(|&x| x.max(0)).collect();
                let delta_input: Vec<i64> = relu_out.iter().zip(input_data.iter()).map(|(o, x)| o - x).collect();
                let delta_zero: Vec<i64> = relu_out.clone();

                let zero_vec = vec![0i64; 1 << nv];

                shreds.insert(format!("{}_zero", layer.name), zero_vec);
                shreds.insert(format!("{}_max", layer.name), relu_out.clone());
                shreds.insert(format!("{}_di", layer.name), delta_input);
                shreds.insert(format!("{}_dz", layer.name), delta_zero);

                let layout = tensor_layouts.get(input_tensor_name).cloned();
                for out in &layer.outputs {
                    tensors.insert(out.clone(), relu_out.clone());
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            OpType::MaxPool => {
                let input_tensor_name = layer.inputs.first()
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} has no input", layer.name))?;
                let input_data = tensors.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} input {} not computed", layer.name, input_tensor_name))?;

                let input_layout = tensor_layouts.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} input {} has no spatial layout", layer.name, input_tensor_name))?
                    .clone();

                let (c, in_h, in_w) = input_layout.spatial_dims();

                let kernel_shape = layer.get_ints_attr("kernel_shape")
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} missing kernel_shape", layer.name))?;
                let pool_h = kernel_shape.first().map(|&v| v as usize)
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} kernel_shape empty", layer.name))?;
                let pool_w = kernel_shape.get(1).map(|&v| v as usize)
                    .ok_or_else(|| anyhow::anyhow!("MaxPool {} kernel_shape has < 2 dims", layer.name))?;

                let strides = layer.get_ints_attr("strides");
                let stride_h = strides.and_then(|s| s.first()).map(|&v| v as usize).unwrap_or(pool_h);
                let stride_w = strides.and_then(|s| s.get(1)).map(|&v| v as usize).unwrap_or(pool_w);

                anyhow::ensure!(in_h >= pool_h && in_w >= pool_w, "MaxPool {}: input {}x{} smaller than kernel {}x{}", layer.name, in_h, in_w, pool_h, pool_w);
                let pool_oh = (in_h - pool_h) / stride_h + 1;
                let pool_ow = (in_w - pool_w) / stride_w + 1;
                let window_size = pool_h * pool_w;
                let num_pool_out = pool_oh * pool_ow * c;
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
                                    if val > max_val { max_val = val; }
                                }
                            }
                            max_values[dest_idx] = max_val;
                        }
                    }
                }

                let deltas: Vec<Vec<i64>> = (0..window_size)
                    .map(|i| (0..pad_pool).map(|w| max_values[w] - window_elems[i][w]).collect())
                    .collect();

                shreds.insert(format!("{}_max", layer.name), max_values.clone());
                for i in 0..window_size {
                    shreds.insert(format!("{}_d{}", layer.name, i), deltas[i].clone());
                }

                for out in &layer.outputs {
                    tensors.insert(out.clone(), max_values.clone());
                    tensor_layouts.insert(out.clone(), SpatialInfo::HWC {
                        h: pool_oh, w: pool_ow, c, stride_c: c,
                    });
                }
            }
            OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
                let input_tensor_name = layer.inputs.first()
                    .ok_or_else(|| anyhow::anyhow!("shape op {} has no input", layer.name))?;
                let data = tensors.get(input_tensor_name)
                    .ok_or_else(|| anyhow::anyhow!("shape op {} input {} not computed", layer.name, input_tensor_name))?
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
                bail!("witness: unsupported op type {:?} in layer {}", other, layer.name);
            }
        }

    }

    let final_output = tensors.get(declared_output)
        .ok_or_else(|| anyhow::anyhow!("declared output '{}' not computed", declared_output))?;
    shreds.insert("expected_output".to_string(), final_output.clone());

    Ok(shreds)
}

fn padded_matmul(a: &[i64], a_rows: usize, a_cols: usize, b: &[i64], b_cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0i128;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] as i128 * b[k * b_cols + j] as i128;
            }
            out[i * b_cols + j] = i64::try_from(sum).expect("matmul accumulator overflows i64");
        }
    }
    out
}

fn broadcast_bias(bias: &[i64], bias_len: usize, total: usize) -> Vec<i64> {
    let mut out = vec![0i64; total];
    for j in 0..bias_len.min(total) {
        out[j] = bias[j];
    }
    out
}
