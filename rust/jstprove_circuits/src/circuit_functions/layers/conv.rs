/// Standard library imports
use std::{
    cmp::{max, min},
    collections::HashMap,
};

/// External crate imports
use ndarray::{ArrayD, s};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
    utils::{
        UtilsError,
        constants::{BIAS, DILATION, GROUP, INPUT, KERNEL_SHAPE, PADS, STRIDES, WEIGHTS},
        graph_pattern_matching::PatternRegistry,
        onnx_model::{
            extract_params_and_expected_shape, get_input_name, get_param, get_param_or_default,
            get_w_or_b,
        },
        typecasting::{AsI32, AsUsize, UsizeAsU32, i32_to_usize},
    },
};

use crate::circuit_functions::{
    layers::layer_ops::LayerOp,
    utils::{quantization::rescale_array, tensor_ops::load_array_constants_or_get_inputs},
};

// -------- Struct --------

#[derive(Debug)]
#[allow(dead_code)]
pub struct ConvLayer {
    name: String,
    index: usize,
    weights: Option<ArrayD<i64>>,
    bias: Option<ArrayD<i64>>,
    strides: Vec<u32>,
    kernel_shape: Vec<u32>,
    group: Vec<u32>,
    dilation: Vec<u32>,
    pads: Vec<u32>,
    input_shape: Vec<usize>,
    scaling: u64,
    optimization_pattern: PatternRegistry,
    v_plus_one: usize,
    is_rescale: bool,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let is_relu = matches!(self.optimization_pattern, PatternRegistry::ConvRelu);

        let input_name = get_input_name(&self.inputs, 0, LayerKind::Conv, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Conv,
                name: input_name.clone(),
            })?
            .clone();

        let w_name = get_input_name(&self.inputs, 1, LayerKind::Conv, WEIGHTS)?;
        let weights = load_array_constants_or_get_inputs(
            api,
            &input,
            w_name,
            &self.weights,
            LayerKind::Conv,
        )?;

        let b_name = get_input_name(&self.inputs, 2, LayerKind::Conv, BIAS)?;
        let bias =
            load_array_constants_or_get_inputs(api, &input, b_name, &self.bias, LayerKind::Conv)?;

        let in_shape = layer_input
            .shape()
            .iter()
            .map(|&x| x.as_u32())
            .collect::<Result<Vec<u32>, UtilsError>>()?;

        let out = conv_4d_run(
            api,
            &layer_input,
            &weights,
            &bias,
            &Conv2DParams {
                dilations: self.dilation.clone(),
                kernel_shape: self.kernel_shape.clone(),
                pads: self.pads.clone(),
                strides: self.strides.clone(),
                input_shape: in_shape,
                groups: self.group.clone(),
            },
            &ConvQuantizationParams {
                scaling: self.scaling,
                quantized: self.is_rescale,
                v_plus_one: self.v_plus_one,
                is_relu,
            },
        )?;

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Conv,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;
        let w_name = get_input_name(&layer.inputs, 1, LayerKind::Conv, WEIGHTS)?;
        let b_name = get_input_name(&layer.inputs, 2, LayerKind::Conv, BIAS)?;

        let (weights, bias) = if layer_context.weights_as_inputs {
            (None, None)
        } else {
            let w = get_w_or_b(&layer_context.w_and_b_map, w_name).map_err(|e| {
                LayerError::MissingParameter {
                    layer: LayerKind::Conv,
                    param: format!("weights (W), requested={w_name}: {e}"),
                }
            })?;
            let b = get_w_or_b(&layer_context.w_and_b_map, b_name).map_err(|e| {
                LayerError::MissingParameter {
                    layer: LayerKind::Conv,
                    param: format!("bias (B), requested={b_name}: {e}"),
                }
            })?;
            (Some(w), Some(b))
        };

        let kernel_shape: Vec<u32> = get_param(&layer.name, KERNEL_SHAPE, &params)?;
        let spatial_rank = kernel_shape.len();

        let default_pads: Vec<u32> = vec![0; 2 * spatial_rank];

        let conv = Self {
            name: layer.name.clone(),
            index,
            weights,
            bias,
            strides: get_param(&layer.name, STRIDES, &params)?,
            kernel_shape,
            group: vec![get_param_or_default(&layer.name, GROUP, &params, Some(&1))?],
            dilation: get_param(&layer.name, DILATION, &params)?,
            pads: get_param_or_default(&layer.name, PADS, &params, Some(&default_pads))?,
            input_shape: expected_shape.clone(),
            scaling: circuit_params.scale_exponent.into(),
            optimization_pattern,
            v_plus_one: layer_context.n_bits_for(&layer.name),
            is_rescale,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };
        Ok(Box::new(conv))
    }
}

pub struct Conv2DParams {
    dilations: Vec<u32>,
    kernel_shape: Vec<u32>,
    pads: Vec<u32>,
    strides: Vec<u32>,
    input_shape: Vec<u32>,
    groups: Vec<u32>,
}

pub struct ConvQuantizationParams {
    scaling: u64,
    quantized: bool,
    v_plus_one: usize,
    is_relu: bool,
}

/// Fills in default convolution parameters if any are missing.
///
/// Ensures that dilations, kernel shape, pads, and strides have valid default
/// values based on the input tensor shape. Leaves provided parameters unchanged.
///
/// # Arguments
/// - `conv_params`: Original convolution parameters, possibly with missing fields.
///
/// # Returns
/// A new `Conv2DParams` struct with all fields fully populated.
///
/// # Errors
/// - [`LayerError::InvalidShape`] if input shape has fewer than 3 dimensions.
pub fn set_default_params(conv_params: &Conv2DParams) -> Result<Conv2DParams, LayerError> {
    let input_shape = &conv_params.input_shape;
    let dilations = &conv_params.dilations;
    let kernel_shape = &conv_params.kernel_shape;
    let pads = &conv_params.pads;
    let strides = &conv_params.strides;

    if input_shape.len() < 3 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Conv,
            msg: format!("input_shape must be at least 3, got {input_shape:?}"),
        });
    }

    // If dilations is empty, fill it with 1s of the appropriate length
    let mut dilations_out = dilations.to_owned();
    let mut kernel_shape_out = kernel_shape.to_owned();
    let mut pads_out = pads.to_owned();
    let mut strides_out = strides.to_owned();

    if dilations.is_empty() {
        dilations_out = vec![1; input_shape[2..].len()];
    }

    // If kernel_shape is empty, fill it with W.shape()[2..]
    if kernel_shape.is_empty() {
        kernel_shape_out = input_shape[2..].to_vec();
    }

    // If pads is empty, fill it with 0s, twice the length of X.shape()[2..]
    if pads.is_empty() {
        let shape_len = input_shape[2..].len();
        pads_out = vec![0; shape_len * 2];
    }

    // If strides is empty, fill it with 1s of the appropriate length
    if strides.is_empty() {
        strides_out = vec![1; input_shape[2..].len()];
    }
    Ok(Conv2DParams {
        dilations: dilations_out,
        kernel_shape: kernel_shape_out,
        pads: pads_out,
        strides: strides_out,
        input_shape: conv_params.input_shape.clone(),
        groups: conv_params.groups.clone(),
    })
}

/// Validates convolution configuration for unsupported shapes or parameters.
///
/// Checks for input length, group convolution constraints, and dilation support.
///
/// # Arguments
/// - `input_shape`: Input tensor shape (NCHW).
/// - `group`: Number of groups for group convolution.
/// - `dilations`: Dilation values per dimension.
///
/// # Errors
/// - [`LayerError::InvalidShape`] if input shape or channels are incompatible.
/// - [`LayerError::UnsupportedConfig`] if group > 1, unsupported dilations, or unsupported input length.
pub fn not_yet_implemented_conv(
    input_shape: &[u32],
    group: &[u32],
    dilations: &Vec<u32>,
) -> Result<(), CircuitError> {
    if input_shape[1] != input_shape[1] * group[0] || input_shape[0] % group[0] != 0 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Conv,
            msg: "shape inconsistencies (channels or batch vs group)".into(),
        }
        .into());
    }
    if group[0] > 1 {
        return Err(LayerError::UnsupportedConfig {
            layer: LayerKind::Conv,
            msg: "Not yet implemented for group > 1".into(),
        }
        .into());
    }
    if (dilations.first().copied().unwrap_or(1) != 1)
        || (dilations.iter().min() != dilations.iter().max())
    {
        return Err(LayerError::UnsupportedConfig {
            layer: LayerKind::Conv,
            msg: format!("unsupported dilation {dilations:?}"),
        }
        .into());
    }
    if input_shape.len() <= 3 {
        return Err(LayerError::UnsupportedConfig {
            layer: LayerKind::Conv,
            msg: "Input shape length 3 or less not yet implemented".into(),
        }
        .into());
    }
    if input_shape.len() >= 5 {
        return Err(LayerError::UnsupportedConfig {
            layer: LayerKind::Conv,
            msg: "Input shape length 5 or greater not yet implemented".into(),
        }
        .into());
    }
    Ok(())
}

/// Setup the initial array for convolution. Incorporates bias
fn conv_shape_4_setup_res<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    bias: &ArrayD<Variable>,
    shape_0: usize,
    shape_1: usize,
    shape_2: usize,
    shape_3: usize,
) -> ArrayD<Variable> {
    let shape = vec![shape_0, shape_1, shape_2, shape_3];

    if bias.is_empty() {
        // Create array filled with zeros
        let zero = api.constant(0);
        ArrayD::from_elem(shape, zero)
    } else {
        // Create array filled with bias values
        let mut res = ArrayD::from_elem(shape, bias[[0]]);
        if !bias.is_empty() {
            // Broadcast bias values across the second dimension (j)
            for j in 0..bias.shape()[0] {
                res.slice_mut(s![.., j, .., ..]).fill(bias[[j]]); // Assign bias[j] to all relevant elements
            }
        }
        res
    }
}

fn get_u(input: &[u32], idx: usize, what: &str) -> Result<usize, CircuitError> {
    input.get(idx).copied().map(|v| v as usize).ok_or_else(|| {
        LayerError::InvalidParameterValue {
            layer: LayerKind::Conv,
            layer_name: "Conv".into(),
            param_name: what.into(),
            value: format!("missing index {idx}"),
        }
        .into()
    })
}

fn have_matching_shapes(x: &ArrayD<Variable>, y: &ArrayD<Variable>) -> bool {
    x.shape() == y.shape()
}

fn validate_conv_params(params: &Conv2DParams) -> Result<(), CircuitError> {
    if params.pads.len() < 4 {
        return Err(LayerError::InvalidParameterValue {
            layer: LayerKind::Conv,
            layer_name: "Conv".into(),
            param_name: "pads".into(),
            value: format!("len {} < 4", params.pads.len()),
        }
        .into());
    }
    if params.strides.contains(&0) {
        return Err(LayerError::InvalidParameterValue {
            layer: LayerKind::Conv,
            layer_name: "Conv".into(),
            param_name: "strides".into(),
            value: format!("strides must be > 0, got {:?}", params.strides),
        }
        .into());
    }
    Ok(())
}

/// Performs the core 4D convolution computation.
///
/// Iterates over the batch, channels, height, and width dimensions, slices the input
/// and weight tensors, performs dot products, and accumulates the results.
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder api.
/// - `input_arr`: NCHW input tensor.
/// - `conv_params`: Convolution parameters including kernel, pads, strides, etc.
/// - `weights`: Convolution weights tensor.
/// - `bias`: Convolution bias tensor.
///
/// # Returns
/// A 4D tensor containing the convolution output.
///
/// # Errors
/// - [`LayerError::InvalidShape`] if any tensor slice dimensions are incompatible.
/// - [`LayerError::InvalidParameterValue`] if convolution parameters are invalid (e.g., stride = 0).
/// - [`CircuitError`] other errors that propograte internally.
#[allow(clippy::too_many_lines)]
pub fn conv_shape_4<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: &ArrayD<Variable>,
    conv_params: &Conv2DParams,
    weights: &ArrayD<Variable>,
    bias: &ArrayD<Variable>,
) -> Result<ArrayD<Variable>, CircuitError> {
    validate_conv_params(conv_params)?;
    let s_n = get_u(&conv_params.input_shape, 0, "input_shape[0]")?;
    let s_c = get_u(&conv_params.input_shape, 1, "input_shape[1]")?;
    let s_h = get_u(&conv_params.input_shape, 2, "input_shape[2]")?;
    let s_w = get_u(&conv_params.input_shape, 3, "input_shape[3]")?;

    // # M, C_group, kH, kW = W.shape
    let kh = get_u(&conv_params.kernel_shape, 0, "kernel_shape[0]")?;
    let kw = get_u(&conv_params.kernel_shape, 1, "kernel_shape[1]")?;

    let stride_h = get_u(&conv_params.strides, 0, "strides[0]")?;
    let stride_w = get_u(&conv_params.strides, 1, "strides[1]")?;

    let pad_0 = get_u(&conv_params.pads, 0, "pads[0]")?;
    let pad_1 = get_u(&conv_params.pads, 1, "pads[1]")?;
    let pad_2 = get_u(&conv_params.pads, 2, "pads[2]")?;
    let pad_3 = get_u(&conv_params.pads, 3, "pads[3]")?;

    let h_out = (((s_h + pad_0 + pad_2).saturating_sub(kh)) / stride_h) + 1;
    let w_out = (((s_w + pad_1 + pad_3).saturating_sub(kw)) / stride_w) + 1;

    let kw = kw.as_i32()?;
    let kh = kh.as_i32()?;
    let s_w = s_w.as_i32()?;
    let s_h = s_h.as_i32()?;

    let h0 = &conv_params
        .pads
        .first()
        .ok_or_else(|| LayerError::InvalidParameterValue {
            layer: LayerKind::Conv,
            layer_name: "Conv".into(),
            param_name: "pads[0]".into(),
            value: format!("pads={:?}", &conv_params.pads),
        })?;
    let w0 = &conv_params
        .pads
        .get(1)
        .ok_or_else(|| LayerError::InvalidParameterValue {
            layer: LayerKind::Conv,
            layer_name: "Conv".into(),
            param_name: "pads[1]".into(),
            value: format!("pads={:?}", &conv_params.pads),
        })?;

    let oh = -((kh % 2) as i32);
    let ow = -((kw % 2) as i32);

    let bh = -((*h0).as_i32()?);
    let bw = -((*w0).as_i32()?);

    let eh = (h_out * stride_h).as_i32()?;
    let ew = (w_out * stride_w).as_i32()?;

    let mut res = conv_shape_4_setup_res(
        api,
        bias,
        s_n,
        weights.shape()[0],
        h_out as usize,
        w_out as usize,
    );

    for n in 0..s_n {
        for nw in 0..weights.shape()[0] {
            for c in 0..s_c {
                let w = weights
                    .slice(s![nw..=nw, c..=c, .., ..])
                    .into_dyn()
                    .to_owned();

                for io in (bh..eh as i32).step_by(stride_h as usize) {
                    let hr = ((io - bh) / stride_h.as_i32()?).as_usize()?;
                    if hr >= h_out {
                        continue;
                    }
                    let i = io + kh % 2;
                    let i_height1: usize = i32_to_usize(max(0, i + oh))?;
                    let i_height2 = i32_to_usize(min(i + oh + kh, s_h))?;

                    for jo in (bw..ew).step_by(stride_w) {
                        let wr = ((jo - bw) / stride_w.as_i32()?).as_usize()?;
                        if wr >= w_out {
                            continue;
                        }
                        let j = jo + kw % 2;
                        let iw1 = i32_to_usize(max(0, j + ow))?;
                        let iw2 = i32_to_usize(min(j + ow + kw, s_w))?;

                        let n_usize = n;

                        let img = input_arr
                            .slice(s![n_usize..=n_usize, c..=c, i_height1..i_height2, iw1..iw2])
                            .into_dyn()
                            .to_owned();

                        if have_matching_shapes(&img, &w) {
                            // TODO check if bias is empty
                            let s = flatten_and_perform_dot(api, &img, &w)?;

                            res[[n, nw, hr, wr]] = api.add(s, res[[n, nw, hr, wr]]);
                        } else {
                            let j_height1 = i32_to_usize(max(-oh - i, 0))?;
                            let j_height2 = i32_to_usize(min(kh, kh + s_h - (i + oh + kh)))?;

                            let j_width1 = i32_to_usize(max(-ow - j, 0))?;
                            let j_width2 = i32_to_usize(min(kw, kw + s_w - (j + ow + kw as i32)))?;
                            let w_ = w
                                .slice(s![0..1, 0..1, j_height1..j_height2, j_width1..j_width2])
                                .into_dyn()
                                .to_owned();

                            if !have_matching_shapes(&w_.clone(), &img) {
                                return Err(LayerError::InvalidShape {
                                    layer: LayerKind::Conv,
                                    msg: format!(
                                        "Mismatched shapes in convolution: img {:?} vs w_ {:?}. \
                                        oh={oh}, ow={ow}, i={i}, j={j}, kh={kh}, kw={kw}, \
                                        sH={s_h}, sW={s_w}, sth={stride_h}, stw={stride_w}",
                                        img.shape(),
                                        w_.shape(),
                                    ),
                                }
                                .into());
                            }
                            // TODO check if bias is empty
                            let s = flatten_and_perform_dot(api, &img, &w_)?;
                            res[[n, nw, hr, wr]] = api.add(s, res[[n, nw, hr, wr]]);
                        }
                    }
                }
            }
        }
    }
    Ok(res)
}

/// Flatten vector and perform dot product
fn flatten_and_perform_dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    img: &ArrayD<Variable>,
    w_: &ArrayD<Variable>,
) -> Result<Variable, CircuitError> {
    let flattened_img: ArrayD<&Variable> =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[img.len()]), img.iter().collect()).map_err(
            |e| LayerError::InvalidShape {
                layer: LayerKind::Conv,
                msg: format!("Failed to flatten img: {e}"),
            },
        )?;
    let flattened_w_: ArrayD<&Variable> =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[w_.len()]), w_.iter().collect()).map_err(|e| {
            LayerError::InvalidShape {
                layer: LayerKind::Conv,
                msg: format!("Failed to flatten weights: {e}"),
            }
        })?;

    let mut sum = api.constant(0);
    for (a, b) in flattened_img.iter().zip(flattened_w_.iter()) {
        let prod = api.mul(*a, *b);
        sum = api.add(sum, prod);
    }
    Ok(sum)
}

/// Executes a 4D convolution operation with optional quantization and `ReLU`.
///
/// Applies convolution over the input array using the given weights and bias,
/// respecting the convolution parameters. If quantization is enabled, the output
/// is rescaled according to the quantization parameters.
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder api.
/// - `input_arr`: Input tensor.
/// - `weights`: Convolution weights tensor.
/// - `bias`: Convolution bias tensor.
/// - `conv_params`: Convolution parameters such as kernel size, strides, pads, etc.
/// - `quantization_params`: Quantization and activation parameters.
///
/// # Returns
/// A 4D tensor containing the result of the convolution (and rescaling if applied).
///
/// # Errors
/// - [`LayerError::InvalidShape`] if the input or weight shapes are incompatible.
/// - [`LayerError::UnsupportedConfig`] if group > 1 or unsupported dilation is used.
/// - [`LayerError::Other`] if scaling cannot be converted to `usize`.
/// - [`CircuitError`] other errors that propogate through.
pub fn conv_4d_run<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: &ArrayD<Variable>,
    weights: &ArrayD<Variable>,
    bias: &ArrayD<Variable>,
    conv_params: &Conv2DParams,
    quantization_params: &ConvQuantizationParams,
) -> Result<ArrayD<Variable>, CircuitError> {
    let conv_params = set_default_params(conv_params)?;
    not_yet_implemented_conv(
        &conv_params.input_shape,
        &conv_params.groups,
        &conv_params.dilations,
    )?;

    let out = conv_shape_4(api, input_arr, &conv_params, weights, bias)?;

    if quantization_params.quantized {
        let scaling_exponent =
            usize::try_from(quantization_params.scaling).map_err(|_| LayerError::Other {
                layer: LayerKind::Conv,
                msg: "Cannot convert scaling to usize".to_string(),
            })?;
        let shift_exponent = quantization_params
            .v_plus_one
            .checked_sub(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::Conv,
                layer_name: "Conv".into(),
                param_name: "v_plus_one".into(),
                value: quantization_params.v_plus_one.to_string(),
            })?;
        let rescaled = rescale_array(
            api,
            out,
            scaling_exponent,
            shift_exponent,
            quantization_params.is_relu,
        )?;
        Ok(rescaled)
    } else {
        Ok(out)
    }
}
