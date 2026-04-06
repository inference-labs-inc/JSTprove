use std::cmp::max;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context as _, Result};
use ndarray::{Array4, ArrayD};

use expander_compiler::frontend::{API, Config, GoldilocksConfig, Variable};
use expander_compiler::zkcuda::context::{Context, DeviceMemoryHandle};
use expander_compiler::zkcuda::kernel::{IOVecSpec, KernelPrimitive, compile_with_spec_and_shapes};
use expander_compiler::zkcuda::proving_system::{Expander, ProvingSystem};
use expander_compiler::zkcuda::shape::Reshape;

use crate::circuit_functions::gadgets::LogupRangeCheckContext;
use crate::circuit_functions::hints::build_logup_hint_registry;
use crate::circuit_functions::layers::LayerKind;
use crate::circuit_functions::utils::build_layers::{BuildLayerContext, default_n_bits_for_config};
use crate::circuit_functions::utils::graph_pattern_matching::{
    PatternMatcher, PatternRegistry, optimization_skip_layers,
};
use crate::circuit_functions::utils::onnx_model::{
    Architecture, CircuitParams, WANDB, collect_all_shapes, get_w_or_b,
};
use crate::circuit_functions::utils::onnx_types::ONNXLayer;
use crate::circuit_functions::utils::tensor_ops::convert_val_to_field_element;
use crate::expander_metadata;

use jstprove_onnx::quantizer::N_BITS_GOLDILOCKS;

use expander_compiler::circuit::config::CircuitField;

fn cap_n_bits_config(params: &mut CircuitParams, n_bits_field: u32) {
    let max_n_bits = 63_usize;
    for nb in params.n_bits_config.values_mut() {
        if *nb > max_n_bits {
            *nb = n_bits_field as usize;
        }
    }
}

#[derive(Clone, Debug)]
enum NativeLayerOp {
    Conv2D {
        weights: ArrayD<i64>,
        bias: Option<ArrayD<i64>>,
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
        dilation: Vec<usize>,
        scaling_exponent: usize,
        is_rescale: bool,
        apply_relu: bool,
    },
    Gemm {
        weights: ArrayD<i64>,
        bias: ArrayD<i64>,
        transa: bool,
        transb: bool,
        scaling_exponent: usize,
        is_rescale: bool,
        apply_relu: bool,
    },
    Relu,
    MaxPool2D {
        kernel_shape: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
        dilation: Vec<usize>,
    },
    Flatten,
}

struct LayerDescriptor {
    name: String,
    output_activation_len: usize,
    output_shape: Vec<usize>,
    weight_entries: Vec<(String, usize)>,
    native_op: NativeLayerOp,
}

struct CompiledPipeline<C: Config> {
    kernels: Vec<KernelPrimitive<C>>,
    layer_descriptors: Vec<LayerDescriptor>,
    input_len: usize,
}

pub struct PipelineResult {
    pub compile_ms: f64,
    pub execution_ms: f64,
    pub prove_ms: f64,
    pub verify_ms: f64,
    pub num_kernels: usize,
    pub verified: bool,
}

/// # Errors
/// Returns an error if ONNX parsing, kernel compilation, or proving fails.
pub fn run_pipeline_goldilocks(onnx_path: &Path, input_data: &[f64]) -> Result<PipelineResult> {
    run_pipeline::<GoldilocksConfig>(onnx_path, input_data, N_BITS_GOLDILOCKS)
}

fn native_forward_i64(op: &NativeLayerOp, input: &[i64], input_shape: &[usize]) -> Vec<i64> {
    match op {
        NativeLayerOp::Conv2D {
            weights,
            bias,
            kernel_shape,
            strides,
            pads,
            dilation,
            scaling_exponent,
            is_rescale,
            apply_relu,
        } => native_conv2d(
            input,
            input_shape,
            weights,
            bias.as_ref(),
            kernel_shape,
            strides,
            pads,
            dilation,
            *scaling_exponent,
            *is_rescale,
            *apply_relu,
        ),
        NativeLayerOp::Gemm {
            weights,
            bias,
            transa,
            transb,
            scaling_exponent,
            is_rescale,
            apply_relu,
        } => native_gemm(
            input,
            input_shape,
            weights,
            bias,
            *transa,
            *transb,
            *scaling_exponent,
            *is_rescale,
            *apply_relu,
        ),
        NativeLayerOp::Relu => input.iter().map(|&v| max(0, v)).collect(),
        NativeLayerOp::MaxPool2D {
            kernel_shape,
            strides,
            pads,
            dilation,
        } => native_maxpool2d(input, input_shape, kernel_shape, strides, pads, dilation),
        NativeLayerOp::Flatten => input.to_vec(),
    }
}

#[allow(clippy::cast_possible_truncation)]
fn native_rescale(value: i128, scaling_exponent: usize, is_rescale: bool, apply_relu: bool) -> i64 {
    let mut v = value;
    if is_rescale {
        let alpha = 1i128 << scaling_exponent;
        v = v.div_euclid(alpha);
    }
    #[allow(clippy::cast_possible_truncation)]
    let result = v as i64;
    if apply_relu { max(0, result) } else { result }
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn native_conv2d(
    input: &[i64],
    input_shape: &[usize],
    weights: &ArrayD<i64>,
    bias: Option<&ArrayD<i64>>,
    kernel_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilation: &[usize],
    scaling_exponent: usize,
    is_rescale: bool,
    apply_relu: bool,
) -> Vec<i64> {
    let n = input_shape[0];
    let c_in = input_shape[1];
    let h_in = input_shape[2];
    let w_in = input_shape[3];
    let input_arr = Array4::from_shape_vec((n, c_in, h_in, w_in), input.to_vec())
        .expect("native_conv2d: input reshape");

    let w_shape = weights.shape();
    let c_out = w_shape[0];
    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let sh = strides[0];
    let sw = strides[1];
    let dh = dilation[0];
    let dw = dilation[1];
    let pad_top = pads[0];
    let pad_left = pads[1];

    let h_out = (h_in + pads[0] + pads[2] - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + pads[1] + pads[3] - dw * (kw - 1) - 1) / sw + 1;

    let weights_4d = weights
        .clone()
        .into_shape_with_order((c_out, c_in, kh, kw))
        .expect("native_conv2d: weights reshape");

    let mut output = Vec::with_capacity(n * c_out * h_out * w_out);
    for b in 0..n {
        for oc in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc: i128 = 0;
                    for ic in 0..c_in {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = (oh * sh + khi * dh) as isize - pad_top as isize;
                                let iw = (ow * sw + kwi * dw) as isize - pad_left as isize;
                                if ih >= 0
                                    && (ih as usize) < h_in
                                    && iw >= 0
                                    && (iw as usize) < w_in
                                {
                                    let iv = input_arr[[b, ic, ih as usize, iw as usize]];
                                    let wv = weights_4d[[oc, ic, khi, kwi]];
                                    acc += i128::from(iv) * i128::from(wv);
                                }
                            }
                        }
                    }
                    if let Some(b_arr) = bias {
                        acc += i128::from(b_arr[oc]);
                    }
                    output.push(native_rescale(
                        acc,
                        scaling_exponent,
                        is_rescale,
                        apply_relu,
                    ));
                }
            }
        }
    }
    output
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::fn_params_excessive_bools,
    clippy::similar_names
)]
fn native_gemm(
    input: &[i64],
    input_shape: &[usize],
    weights: &ArrayD<i64>,
    bias: &ArrayD<i64>,
    transa: bool,
    transb: bool,
    scaling_exponent: usize,
    is_rescale: bool,
    apply_relu: bool,
) -> Vec<i64> {
    let (m, k) = if input_shape.len() >= 2 {
        (input_shape[0], input_shape[1])
    } else {
        (1, input_shape[0])
    };
    let (m_a, k_a) = if transa { (k, m) } else { (m, k) };
    let _ = (m_a, k_a);

    let w_shape = weights.shape();
    let (k_w, n_w) = (w_shape[0], w_shape[1]);
    let (k_b, n_b) = if transb { (n_w, k_w) } else { (k_w, n_w) };

    let k_eff = if transa { m } else { k };
    assert_eq!(k_eff, k_b, "native_gemm: inner dimension mismatch");

    let weights_flat: Vec<i64> = weights.iter().copied().collect();

    let mut output = Vec::with_capacity(m * n_b);
    let (m_eff, _) = if transa { (k, m) } else { (m, k) };

    for i in 0..m_eff {
        for j in 0..n_b {
            let mut acc: i128 = 0;
            for p in 0..k_eff {
                let a_val = if transa {
                    input[p * m + i]
                } else {
                    input[i * k + p]
                };
                let b_val = if transb {
                    weights_flat[j * n_w + p]
                } else {
                    weights_flat[p * n_w + j]
                };
                acc += i128::from(a_val) * i128::from(b_val);
            }
            let bias_val = if bias.is_empty() {
                0
            } else if bias.len() == n_b {
                i128::from(bias[j])
            } else {
                i128::from(bias[[i, j]])
            };
            acc += bias_val;
            output.push(native_rescale(
                acc,
                scaling_exponent,
                is_rescale,
                apply_relu,
            ));
        }
    }
    output
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn native_maxpool2d(
    input: &[i64],
    input_shape: &[usize],
    kernel_shape: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilation: &[usize],
) -> Vec<i64> {
    let n = input_shape[0];
    let c = input_shape[1];
    let h_in = input_shape[2];
    let w_in = input_shape[3];
    let input_arr = Array4::from_shape_vec((n, c, h_in, w_in), input.to_vec())
        .expect("native_maxpool2d: input reshape");

    let kh = kernel_shape[0];
    let kw = kernel_shape[1];
    let sh = strides[0];
    let sw = strides[1];
    let dh = dilation[0];
    let dw = dilation[1];
    let pad_top = pads[0];
    let pad_left = pads[1];

    let h_out = (h_in + pads[0] + pads[2] - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + pads[1] + pads[3] - dw * (kw - 1) - 1) / sw + 1;

    let mut output = Vec::with_capacity(n * c * h_out * w_out);
    for b in 0..n {
        for ch in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = i64::MIN;
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let ih = (oh * sh + khi * dh) as isize - pad_top as isize;
                            let iw = (ow * sw + kwi * dw) as isize - pad_left as isize;
                            if ih >= 0 && (ih as usize) < h_in && iw >= 0 && (iw as usize) < w_in {
                                let v = input_arr[[b, ch, ih as usize, iw as usize]];
                                if v > max_val {
                                    max_val = v;
                                }
                            }
                        }
                    }
                    output.push(max_val);
                }
            }
        }
    }
    output
}

fn run_pipeline<C: Config>(
    onnx_path: &Path,
    input_data: &[f64],
    n_bits: u32,
) -> Result<PipelineResult>
where
    Expander<C>: ProvingSystem<C>,
{
    let metadata = expander_metadata::generate_from_onnx_for_field(onnx_path, n_bits, None)
        .context("generating ONNX metadata")?;

    let mut params = metadata.circuit_params.clone();
    let architecture = &metadata.architecture;
    let wandb = &metadata.wandb;

    cap_n_bits_config(&mut params, n_bits);

    let t = Instant::now();
    let pipeline = compile_kernels::<C>(&params, architecture, wandb)?;
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

    let num_kernels = pipeline.kernels.len();

    let t = Instant::now();
    let hint_registry = build_logup_hint_registry::<CircuitField<C>>();
    let mut ctx: Context<C, _> = Context::new(hint_registry);

    anyhow::ensure!(
        input_data.len() == pipeline.input_len,
        "input_data length ({}) does not match model input length ({})",
        input_data.len(),
        pipeline.input_len
    );
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let alpha = f64::from(params.scale_base).powi(params.scale_exponent as i32);

    #[allow(clippy::cast_possible_truncation)]
    let input_i64: Vec<i64> = input_data.iter().map(|&v| (v * alpha) as i64).collect();
    let input_vals: Vec<CircuitField<C>> = input_i64
        .iter()
        .map(|&v| convert_val_to_field_element::<C>(v))
        .collect();
    let mut activations: DeviceMemoryHandle = ctx.copy_to_device(&input_vals);
    activations = activations.reshape(&[1, pipeline.input_len]);

    let first_input_shape: Vec<usize> = params
        .inputs
        .first()
        .map_or_else(|| vec![pipeline.input_len], |io| io.shape.clone());
    let mut current_i64: Vec<i64> = input_i64;
    let mut current_shape: Vec<usize> = first_input_shape;

    for (i, (kernel, desc)) in pipeline
        .kernels
        .iter()
        .zip(pipeline.layer_descriptors.iter())
        .enumerate()
    {
        let output_i64 = native_forward_i64(&desc.native_op, &current_i64, &current_shape);

        let output_field: Vec<CircuitField<C>> = output_i64
            .iter()
            .map(|&v| convert_val_to_field_element::<C>(v))
            .collect();

        let mut io_handles: Vec<_> = vec![activations.clone()];

        for (_wname, wlen) in &desc.weight_entries {
            let wvals = vec![CircuitField::<C>::default(); *wlen];
            let whandle = ctx.copy_to_device(&wvals);
            io_handles.push(whandle.reshape(&[1, *wlen]));
        }

        let out_handle_dev = ctx.copy_to_device(&output_field);
        io_handles.push(out_handle_dev.reshape(&[1, desc.output_activation_len]));

        let io_slice: &mut [_] = &mut io_handles;
        ctx.register_kernel_io(kernel, 1, io_slice)
            .map_err(|e| anyhow::anyhow!("kernel {i} ({}) register failed: {e}", desc.name))?;

        activations = io_slice
            .last()
            .cloned()
            .unwrap_or(None)
            .reshape(&[1, desc.output_activation_len]);

        current_i64 = output_i64;
        current_shape.clone_from(&desc.output_shape);

        tracing::info!(kernel = i, name = %desc.name, "kernel registered");
    }

    let computation_graph = ctx
        .compile_computation_graph()
        .map_err(|e| anyhow::anyhow!("compile_computation_graph: {e}"))?;
    ctx.solve_witness()
        .map_err(|e| anyhow::anyhow!("solve_witness: {e}"))?;
    let execution_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (prover_setup, verifier_setup) =
        <Expander<C> as ProvingSystem<C>>::setup(&computation_graph);
    let proof = <Expander<C> as ProvingSystem<C>>::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let verified =
        <Expander<C> as ProvingSystem<C>>::verify(&verifier_setup, &computation_graph, &proof);
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    Ok(PipelineResult {
        compile_ms,
        execution_ms,
        prove_ms,
        verify_ms,
        num_kernels,
        verified,
    })
}

fn extract_onnx_param_vec<T: serde::de::DeserializeOwned + Clone>(
    params: &rmpv::Value,
    key: &str,
    default: &[T],
) -> Vec<T> {
    use crate::circuit_functions::utils::onnx_model::get_param_or_default;
    get_param_or_default::<Vec<T>>("", key, params, Some(&default.to_vec())).unwrap_or_default()
}

fn extract_onnx_param_scalar<T: serde::de::DeserializeOwned + Clone>(
    params: &rmpv::Value,
    key: &str,
    default: T,
) -> T {
    use crate::circuit_functions::utils::onnx_model::get_param_or_default;
    get_param_or_default::<T>("", key, params, Some(&default)).unwrap_or(default)
}

#[allow(clippy::too_many_lines)]
fn compile_kernels<C: Config>(
    params: &CircuitParams,
    architecture: &Architecture,
    wandb: &WANDB,
) -> Result<CompiledPipeline<C>> {
    let w_and_b_map: HashMap<String, &ONNXLayer> = wandb
        .w_and_b
        .iter()
        .map(|layer| (layer.name.clone(), layer))
        .collect();

    let inputs = &params.inputs;
    let shapes_map: HashMap<String, Vec<usize>> =
        collect_all_shapes(&architecture.architecture, inputs);

    let layer_context = BuildLayerContext {
        w_and_b_map: &w_and_b_map,
        shapes_map: &shapes_map,
        n_bits_config: &params.n_bits_config,
        default_n_bits: default_n_bits_for_config::<C>(),
        weights_as_inputs: false,
    };

    let matcher = PatternMatcher::new();
    let opt_patterns_by_layername = matcher.run(&architecture.architecture)?;

    let mut skip_next_layer: HashMap<String, bool> = HashMap::new();
    let mut kernels: Vec<KernelPrimitive<C>> = Vec::new();
    let mut descriptors: Vec<LayerDescriptor> = Vec::new();

    let input_len: usize = params
        .inputs
        .iter()
        .map(|io| io.shape.iter().product::<usize>())
        .sum();
    let mut prev_output_len = input_len;

    let first_input_shape: Vec<usize> = params
        .inputs
        .first()
        .map_or_else(|| vec![input_len], |io| io.shape.clone());
    let mut prev_output_shape = first_input_shape;

    for (i, original_layer) in architecture.architecture.iter().enumerate() {
        let mut layer = original_layer.clone();
        if *skip_next_layer.get(&layer.name).unwrap_or(&false) {
            continue;
        }

        let outputs = layer.outputs.clone();
        let optimization_pattern_match = opt_patterns_by_layername.get(&layer.name);
        let (optimization_pattern, outputs, layers_to_skip) =
            match optimization_skip_layers(optimization_pattern_match, &outputs).unwrap_or(None) {
                Some(opt) => opt,
                None => (PatternRegistry::None, outputs.clone(), vec![]),
            };
        for item in layers_to_skip {
            skip_next_layer.insert(item.to_string(), true);
        }
        layer.outputs = outputs;

        let is_rescale = *params.rescale_config.get(&layer.name).unwrap_or(&true);

        let layer_kind = LayerKind::try_from(layer.op_type.as_str())
            .map_err(|_| anyhow::anyhow!("unsupported layer: {}", layer.op_type))?;

        let act_in_len = prev_output_len;

        let mut weight_entries: Vec<(String, usize)> = Vec::new();
        let mut skip_inputs: Vec<&str> = Vec::new();
        for name in layer.inputs.iter().skip(1) {
            if let Some(wb) = w_and_b_map.get(name.as_str()) {
                let len: usize = wb
                    .shape
                    .get(name.as_str())
                    .map_or(1, |s| s.iter().product());
                weight_entries.push((name.clone(), len));
            } else {
                skip_inputs.push(name);
            }
        }
        anyhow::ensure!(
            skip_inputs.is_empty(),
            "layer '{}' has non-initializer secondary inputs {:?} \
             (skip connections are not yet supported in the zkCUDA pipeline)",
            layer.name,
            skip_inputs
        );

        let act_out_shape: Vec<usize> = layer
            .outputs
            .first()
            .and_then(|oname| shapes_map.get(oname))
            .cloned()
            .unwrap_or_else(|| vec![act_in_len]);
        let act_out_len: usize = act_out_shape.iter().product();

        let mut io_specs = vec![IOVecSpec {
            len: act_in_len,
            is_input: true,
            is_output: false,
        }];
        let mut io_shapes: Vec<Vec<usize>> = vec![vec![act_in_len]];

        for (_wname, wlen) in &weight_entries {
            io_specs.push(IOVecSpec {
                len: *wlen,
                is_input: true,
                is_output: false,
            });
            io_shapes.push(vec![*wlen]);
        }

        io_specs.push(IOVecSpec {
            len: act_out_len,
            is_input: false,
            is_output: true,
        });
        io_shapes.push(vec![act_out_len]);

        let layer_clone = layer.clone();
        let params_clone = params.clone();
        let layer_name = layer.name.clone();

        let builder = layer_kind.builder::<C, API<C>>();
        let built_layer = builder(
            &layer_clone,
            &params_clone,
            optimization_pattern.clone(),
            is_rescale,
            i,
            &layer_context,
        )
        .map_err(|e| anyhow::anyhow!("building layer {layer_name}: {e}"))?;

        let input_activation_name = layer
            .inputs
            .first()
            .cloned()
            .unwrap_or_else(|| "input".into());

        let chunk_bits = params
            .logup_chunk_bits
            .unwrap_or(crate::circuit_functions::gadgets::DEFAULT_LOGUP_CHUNK_BITS);

        let act_in_tensor_shape: Vec<usize> = shapes_map
            .get(&input_activation_name)
            .cloned()
            .unwrap_or_else(|| vec![act_in_len]);

        let weight_shapes: Vec<(String, Vec<usize>)> = weight_entries
            .iter()
            .map(|(wname, wlen)| {
                let shape = w_and_b_map
                    .get(wname.as_str())
                    .and_then(|wb| wb.shape.get(wname.as_str()))
                    .cloned()
                    .unwrap_or_else(|| vec![*wlen]);
                (wname.clone(), shape)
            })
            .collect();

        let kernel = compile_with_spec_and_shapes::<C, _>(
            |api: &mut API<C>, io_vars: &mut Vec<Vec<Variable>>| {
                let act_in = &io_vars[0];
                let act_in_arr =
                    ArrayD::from_shape_vec(ndarray::IxDyn(&act_in_tensor_shape), act_in.clone())
                        .unwrap_or_else(|e| panic!(
                            "reshape input activations for layer '{}' with shape {:?} and {} elements: {e}",
                            layer_name, act_in_tensor_shape, act_in.len()
                        ));

                let mut tensor_map: HashMap<String, ArrayD<Variable>> = HashMap::new();
                tensor_map.insert(input_activation_name.clone(), act_in_arr);

                for (idx, (wname, wshape)) in weight_shapes.iter().enumerate() {
                    let wvars = &io_vars[1 + idx];
                    let warr = ArrayD::from_shape_vec(ndarray::IxDyn(wshape), wvars.clone())
                        .unwrap_or_else(|e| panic!(
                            "reshape weights '{}' for layer '{}' with shape {:?} and {} elements: {e}",
                            wname, layer_name, wshape, wvars.len()
                        ));
                    tensor_map.insert(wname.clone(), warr);
                }

                let mut logup_ctx = LogupRangeCheckContext::new(chunk_bits);
                logup_ctx.init::<C, API<C>>(api);

                let (_, output_arr) = built_layer
                    .apply(api, &mut logup_ctx, &tensor_map)
                    .unwrap_or_else(|e| panic!(
                        "apply failed for layer '{layer_name}': {e}"
                    ));

                logup_ctx.finalize::<C, API<C>>(api);

                let out_flat: Vec<Variable> = output_arr.iter().copied().collect();
                let out_idx = io_vars.len() - 1;
                io_vars[out_idx] = out_flat;
            },
            &io_specs,
            &io_shapes,
        )
        .map_err(|e| anyhow::anyhow!("compile kernel for layer {layer_name}: {e}"))?;

        let native_op = build_native_op(
            &layer,
            &layer_kind,
            &optimization_pattern,
            is_rescale,
            params,
            &w_and_b_map,
            &prev_output_shape,
            &layer_context,
        )?;

        tracing::info!(
            layer = i,
            name = %layer.name,
            act_in = act_in_len,
            act_out = act_out_len,
            weights = weight_entries.len(),
            "compiled kernel"
        );

        prev_output_len = act_out_len;
        prev_output_shape.clone_from(&act_out_shape);

        descriptors.push(LayerDescriptor {
            name: layer.name.clone(),
            output_activation_len: act_out_len,
            output_shape: act_out_shape,
            weight_entries,
            native_op,
        });
        kernels.push(kernel);
    }

    Ok(CompiledPipeline {
        kernels,
        layer_descriptors: descriptors,
        input_len,
    })
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
fn build_native_op(
    layer: &ONNXLayer,
    _layer_kind: &LayerKind,
    optimization_pattern: &PatternRegistry,
    is_rescale: bool,
    params: &CircuitParams,
    w_and_b_map: &HashMap<String, &ONNXLayer>,
    _input_shape: &[usize],
    _layer_context: &BuildLayerContext,
) -> Result<NativeLayerOp> {
    use crate::circuit_functions::utils::onnx_model::extract_params;

    let op_type = layer.op_type.as_str();
    match op_type {
        "Conv" => {
            let lp = extract_params(layer).map_err(|e| anyhow::anyhow!("{e}"))?;
            let kernel_shape: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "kernel_shape", &[])
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let strides: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "strides", &[1, 1])
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let spatial_rank = kernel_shape.len();
            let default_pads: Vec<i64> = vec![0; 2 * spatial_rank];
            let pads: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "pads", &default_pads)
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let dilation: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "dilations", &[1, 1])
                .into_iter()
                .map(|v| v as usize)
                .collect();

            let w_name = layer
                .inputs
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("Conv layer missing weight input"))?;
            let weights: ArrayD<i64> = get_w_or_b(w_and_b_map, w_name)
                .map_err(|e| anyhow::anyhow!("Conv weights: {e}"))?;
            let bias: Option<ArrayD<i64>> = layer
                .inputs
                .get(2)
                .and_then(|b_name| get_w_or_b::<i64, _>(w_and_b_map, b_name).ok());

            let apply_relu = matches!(optimization_pattern, PatternRegistry::ConvRelu);

            Ok(NativeLayerOp::Conv2D {
                weights,
                bias,
                kernel_shape,
                strides,
                pads,
                dilation,
                scaling_exponent: params.scale_exponent as usize,
                is_rescale,
                apply_relu,
            })
        }
        "Gemm" => {
            let lp = extract_params(layer).map_err(|e| anyhow::anyhow!("{e}"))?;
            let transa: usize = extract_onnx_param_scalar(&lp, "transA", 0);
            let transb: usize = extract_onnx_param_scalar(&lp, "transB", 0);

            let w_name = layer
                .inputs
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("Gemm layer missing weight input"))?;
            let weights: ArrayD<i64> = get_w_or_b(w_and_b_map, w_name)
                .map_err(|e| anyhow::anyhow!("Gemm weights: {e}"))?;
            let b_name = layer
                .inputs
                .get(2)
                .ok_or_else(|| anyhow::anyhow!("Gemm layer missing bias input"))?;
            let bias: ArrayD<i64> =
                get_w_or_b(w_and_b_map, b_name).map_err(|e| anyhow::anyhow!("Gemm bias: {e}"))?;

            let apply_relu = matches!(optimization_pattern, PatternRegistry::GemmRelu);

            Ok(NativeLayerOp::Gemm {
                weights,
                bias,
                transa: transa != 0,
                transb: transb != 0,
                scaling_exponent: params.scale_exponent as usize,
                is_rescale,
                apply_relu,
            })
        }
        "Relu" | "ReLU" => Ok(NativeLayerOp::Relu),
        "MaxPool" => {
            let lp = extract_params(layer).map_err(|e| anyhow::anyhow!("{e}"))?;
            let kernel_shape: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "kernel_shape", &[])
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let strides: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "strides", &[1, 1])
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let spatial_rank = kernel_shape.len();
            let default_pads: Vec<i64> = vec![0; 2 * spatial_rank];
            let pads: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "pads", &default_pads)
                .into_iter()
                .map(|v| v as usize)
                .collect();
            let dilation: Vec<usize> = extract_onnx_param_vec::<i64>(&lp, "dilations", &[1, 1])
                .into_iter()
                .map(|v| v as usize)
                .collect();

            Ok(NativeLayerOp::MaxPool2D {
                kernel_shape,
                strides,
                pads,
                dilation,
            })
        }
        "Flatten" => Ok(NativeLayerOp::Flatten),
        _ => anyhow::bail!(
            "native forward pass not implemented for layer type '{op_type}'; \
             falling back to call_kernel would be needed"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn lenet_zkcuda_pipeline() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models/lenet.onnx");
        assert!(model_path.exists(), "lenet.onnx not found");

        let num_act: usize = 3072;
        let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

        let result = run_pipeline_goldilocks(&model_path, &activations).unwrap();

        println!("kernels:  {}", result.num_kernels);
        println!("compile:  {:.2}s", result.compile_ms / 1000.0);
        println!("execution: {:.1}ms", result.execution_ms);
        println!("prove:    {:.1}ms", result.prove_ms);
        println!("verify:   {:.1}ms", result.verify_ms);
        println!("verified: {}", result.verified);
        assert!(result.verified);
    }

    #[test]
    #[ignore]
    fn vgg16_zkcuda_pipeline() {
        let model_path = Path::new("/tmp/vgg16_dsperse/model.onnx");
        if !model_path.exists() {
            eprintln!("VGG-16 model not found at /tmp/vgg16_dsperse/model.onnx, skipping");
            return;
        }

        let num_act: usize = 224 * 224 * 3;
        let activations: Vec<f64> = (0..num_act).map(|i| i as f64 / num_act as f64).collect();

        let result = run_pipeline_goldilocks(model_path, &activations).unwrap();

        println!("kernels:  {}", result.num_kernels);
        println!("compile:  {:.2}s", result.compile_ms / 1000.0);
        println!("execution: {:.1}ms", result.execution_ms);
        println!("prove:    {:.1}ms", result.prove_ms);
        println!("verify:   {:.1}ms", result.verify_ms);
        println!("verified: {}", result.verified);
        assert!(result.verified);
    }
}
