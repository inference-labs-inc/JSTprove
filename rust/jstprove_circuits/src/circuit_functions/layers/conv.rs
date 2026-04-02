/// Standard library imports
use std::{
    cmp::{max, min},
    collections::HashMap,
};

/// External crate imports
use ndarray::{ArrayD, s};

/// `ExpanderCompilerCollection` imports
use arith::{FFTField, Field};
use expander_compiler::frontend::{CircuitField, Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind},
    utils::{
        UtilsError,
        constants::{DILATION, GROUP, INPUT, KERNEL_SHAPE, PADS, STRIDES, WEIGHTS},
        graph_pattern_matching::PatternRegistry,
        onnx_model::{
            extract_params, get_input_name, get_optional_input_name, get_param,
            get_param_or_default, get_w_or_b,
        },
        typecasting::{AsI32, AsUsize, UsizeAsU32, i32_to_usize},
    },
};

use crate::circuit_functions::{
    gadgets::LogupRangeCheckContext,
    layers::layer_ops::LayerOp,
    utils::{
        quantization::{MaybeRescaleParams, maybe_rescale},
        tensor_ops::load_array_constants_or_get_inputs,
    },
};

// -------- Struct --------

#[derive(Debug)]
pub struct ConvLayer {
    weights: Option<ArrayD<i64>>,
    bias: Option<ArrayD<i64>>,
    strides: Vec<u32>,
    kernel_shape: Vec<u32>,
    group: Vec<u32>,
    dilation: Vec<u32>,
    pads: Vec<u32>,
    scaling: u64,
    optimization_pattern: PatternRegistry,
    v_plus_one: usize,
    is_rescale: bool,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvLayer
where
    CircuitField<C>: FFTField,
{
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
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
        let weights =
            load_array_constants_or_get_inputs(api, input, w_name, &self.weights, LayerKind::Conv)?;

        let bias = match get_optional_input_name(&self.inputs, 2) {
            Some(b_name) => {
                load_array_constants_or_get_inputs(api, input, b_name, &self.bias, LayerKind::Conv)?
            }
            None => ArrayD::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).expect("empty bias"),
        };

        let in_shape = layer_input
            .shape()
            .iter()
            .map(|&x| x.as_u32())
            .collect::<Result<Vec<u32>, UtilsError>>()?;

        let out = conv_4d_run(
            api,
            logup_ctx,
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
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let params = extract_params(layer).map_err(|e| LayerError::Other {
            layer: LayerKind::Conv,
            msg: format!("extract_params failed: {e}"),
        })?;
        let w_name = get_input_name(&layer.inputs, 1, LayerKind::Conv, WEIGHTS)?;
        let b_name = get_optional_input_name(&layer.inputs, 2);

        let (weights, bias) = if layer_context.weights_as_inputs {
            (None, None)
        } else {
            let w = get_w_or_b(layer_context.w_and_b_map, w_name).map_err(|e| {
                LayerError::MissingParameter {
                    layer: LayerKind::Conv,
                    param: format!("weights (W), requested={w_name}: {e}"),
                }
            })?;
            let b = match b_name {
                Some(name) => Some(get_w_or_b(layer_context.w_and_b_map, name).map_err(|e| {
                    LayerError::MissingParameter {
                        layer: LayerKind::Conv,
                        param: format!("bias (B), requested={name}: {e}"),
                    }
                })?),
                None => None,
            };
            (Some(w), b)
        };

        let kernel_shape: Vec<u32> = get_param(&layer.name, KERNEL_SHAPE, &params)?;
        let spatial_rank = kernel_shape.len();

        let default_pads: Vec<u32> = vec![0; 2 * spatial_rank];

        let conv = Self {
            weights,
            bias,
            strides: get_param(&layer.name, STRIDES, &params)?,
            kernel_shape,
            group: vec![get_param_or_default(&layer.name, GROUP, &params, Some(&1))?],
            dilation: get_param(&layer.name, DILATION, &params)?,
            pads: get_param_or_default(&layer.name, PADS, &params, Some(&default_pads))?,
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
    if input_shape.len() < 2 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Conv,
            msg: format!("input_shape length {} < 2", input_shape.len()),
        }
        .into());
    }
    let g = *group.first().ok_or_else(|| LayerError::InvalidShape {
        layer: LayerKind::Conv,
        msg: "group parameter is empty".into(),
    })?;
    if g == 0 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Conv,
            msg: "group parameter cannot be zero".into(),
        }
        .into());
    }
    if input_shape[1] % g != 0 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::Conv,
            msg: format!(
                "input channels {} not divisible by group {g}",
                input_shape[1]
            ),
        }
        .into());
    }
    if g > 1 {
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
) -> Result<ArrayD<Variable>, CircuitError> {
    let shape = vec![shape_0, shape_1, shape_2, shape_3];
    let zero = api.constant(0);

    if bias.is_empty() {
        Ok(ArrayD::from_elem(shape, zero))
    } else {
        if bias.shape()[0] != shape_1 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Conv,
                msg: format!(
                    "bias length {} != output channels {shape_1}",
                    bias.shape()[0]
                ),
            }
            .into());
        }
        let mut res = ArrayD::from_elem(shape, zero);
        for j in 0..shape_1 {
            res.slice_mut(s![.., j, .., ..]).fill(bias[[j]]);
        }
        Ok(res)
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
    )?;

    for n in 0..s_n {
        for nw in 0..weights.shape()[0] {
            for c in 0..s_c {
                let w = weights.slice(s![nw..=nw, c..=c, .., ..]).into_dyn();

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

                        let img = input_arr
                            .slice(s![n..=n, c..=c, i_height1..i_height2, iw1..iw2])
                            .into_dyn();

                        if img.shape() == w.shape() {
                            let s = flatten_and_perform_dot(api, &img, &w.view());
                            res[[n, nw, hr, wr]] = api.add(s, res[[n, nw, hr, wr]]);
                        } else {
                            let j_height1 = i32_to_usize(max(-oh - i, 0))?;
                            let j_height2 = i32_to_usize(min(kh, kh + s_h - (i + oh + kh)))?;

                            let j_width1 = i32_to_usize(max(-ow - j, 0))?;
                            let j_width2 = i32_to_usize(min(kw, kw + s_w - (j + ow + kw as i32)))?;
                            let w_ = w
                                .slice(s![0..1, 0..1, j_height1..j_height2, j_width1..j_width2])
                                .into_dyn();

                            if w_.shape() != img.shape() {
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
                            let s = flatten_and_perform_dot(api, &img, &w_);
                            res[[n, nw, hr, wr]] = api.add(s, res[[n, nw, hr, wr]]);
                        }
                    }
                }
            }
        }
    }
    Ok(res)
}

fn flatten_and_perform_dot<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    img: &ndarray::ArrayViewD<'_, Variable>,
    w_: &ndarray::ArrayViewD<'_, Variable>,
) -> Variable {
    let mut sum = api.constant(0);
    for (a, b) in img.iter().zip(w_.iter()) {
        let prod = api.mul(*a, *b);
        sum = api.add(sum, prod);
    }
    sum
}

fn bit_reverse(x: usize, log_n: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - log_n)
}

fn ntt_1d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    data: &mut [Variable],
    omega: CircuitField<C>,
) where
    CircuitField<C>: FFTField,
{
    let n = data.len();
    assert!(n.is_power_of_two());
    let log_n = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            data.swap(i, j);
        }
    }

    for s in 0..log_n {
        let half = 1 << s;
        let full = half << 1;
        let exp = n / full;
        let mut w_step = CircuitField::<C>::ONE;
        let mut base = omega;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                w_step *= base;
            }
            base = base * base;
            e >>= 1;
        }

        let mut twiddles = Vec::with_capacity(half);
        let mut tw = CircuitField::<C>::ONE;
        for _ in 0..half {
            twiddles.push(tw);
            tw *= w_step;
        }

        for k in (0..n).step_by(full) {
            for j in 0..half {
                let u = data[k + j];
                let t = api.mul(data[k + j + half], twiddles[j]);
                data[k + j] = api.add(u, t);
                data[k + j + half] = api.sub(u, t);
            }
        }
    }
}

fn ntt_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    data: &mut [Variable],
    rows: usize,
    cols: usize,
) where
    CircuitField<C>: FFTField,
{
    let omega_cols = CircuitField::<C>::two_adic_generator(cols.trailing_zeros() as usize);
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        ntt_1d(api, &mut data[start..end], omega_cols);
    }

    let omega_rows = CircuitField::<C>::two_adic_generator(rows.trailing_zeros() as usize);
    let mut col_buf = vec![Variable::default(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = data[r * cols + c];
        }
        ntt_1d(api, &mut col_buf, omega_rows);
        for r in 0..rows {
            data[r * cols + c] = col_buf[r];
        }
    }
}

fn intt_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    data: &mut [Variable],
    rows: usize,
    cols: usize,
) where
    CircuitField<C>: FFTField,
{
    let omega_cols = CircuitField::<C>::two_adic_generator(cols.trailing_zeros() as usize);
    let omega_cols_inv = omega_cols.inv().expect("omega must be invertible");
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        ntt_1d(api, &mut data[start..end], omega_cols_inv);
    }

    let omega_rows = CircuitField::<C>::two_adic_generator(rows.trailing_zeros() as usize);
    let omega_rows_inv = omega_rows.inv().expect("omega must be invertible");
    let mut col_buf = vec![Variable::default(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = data[r * cols + c];
        }
        ntt_1d(api, &mut col_buf, omega_rows_inv);
        for r in 0..rows {
            data[r * cols + c] = col_buf[r];
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    let n_inv = CircuitField::<C>::from((rows * cols) as u32)
        .inv()
        .expect("n must be invertible in the field");
    for x in data.iter_mut() {
        *x = api.mul(*x, n_inv);
    }
}

fn should_use_ntt(
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
) -> bool {
    if kh <= 1 && kw <= 1 {
        return false;
    }
    if stride_h > 1 || stride_w > 1 {
        return false;
    }
    if h < kh || w < kw {
        return false;
    }
    let nh = (h + kh - 1).next_power_of_two();
    let nw = (w + kw - 1).next_power_of_two();
    let n_padded = nh * nw;
    let h_out = h - kh + 1;
    let w_out = w - kw + 1;
    let direct_muls_per_pair = h_out * w_out * kh * kw;
    let ntt_add_overhead = nh * nw * ((nh.trailing_zeros() + nw.trailing_zeros()) as usize) * 2;
    let add_cost_ratio = 4;
    n_padded + ntt_add_overhead / add_cost_ratio < direct_muls_per_pair
}

#[allow(clippy::too_many_lines)]
fn conv_ntt_4d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_arr: &ArrayD<Variable>,
    conv_params: &Conv2DParams,
    weights: &ArrayD<Variable>,
    bias: &ArrayD<Variable>,
) -> Result<ArrayD<Variable>, CircuitError>
where
    CircuitField<C>: FFTField,
{
    let s_n = get_u(&conv_params.input_shape, 0, "input_shape[0]")?;
    let s_c = get_u(&conv_params.input_shape, 1, "input_shape[1]")?;
    let s_h = get_u(&conv_params.input_shape, 2, "input_shape[2]")?;
    let s_w = get_u(&conv_params.input_shape, 3, "input_shape[3]")?;

    let kh = get_u(&conv_params.kernel_shape, 0, "kernel_shape[0]")?;
    let kw = get_u(&conv_params.kernel_shape, 1, "kernel_shape[1]")?;

    let pad_0 = get_u(&conv_params.pads, 0, "pads[0]")?;
    let pad_1 = get_u(&conv_params.pads, 1, "pads[1]")?;
    let pad_2 = get_u(&conv_params.pads, 2, "pads[2]")?;
    let pad_3 = get_u(&conv_params.pads, 3, "pads[3]")?;

    let padded_h = s_h + pad_0 + pad_2;
    let padded_w = s_w + pad_1 + pad_3;

    let h_out = padded_h - kh + 1;
    let w_out = padded_w - kw + 1;
    let c_out = weights.shape()[0];

    let nh = (padded_h + kh - 1).next_power_of_two();
    let nw = (padded_w + kw - 1).next_power_of_two();

    let mut res = conv_shape_4_setup_res(api, bias, s_n, c_out, h_out, w_out)?;
    let zero = api.constant(0);

    for n in 0..s_n {
        let mut input_ntt = Vec::with_capacity(s_c);
        for c in 0..s_c {
            let mut buf = vec![zero; nh * nw];
            for r in 0..s_h {
                for col in 0..s_w {
                    let pr = r + pad_0;
                    let pc = col + pad_1;
                    if pr < padded_h && pc < padded_w {
                        buf[pr * nw + pc] = input_arr[[n, c, r, col]];
                    }
                }
            }
            ntt_2d(api, &mut buf, nh, nw);
            input_ntt.push(buf);
        }

        for co in 0..c_out {
            let mut accum = vec![zero; nh * nw];

            for ci in 0..s_c {
                let mut w_buf = vec![zero; nh * nw];
                for r in 0..kh {
                    for col in 0..kw {
                        w_buf[r * nw + col] = weights[[co, ci, r, col]];
                    }
                }
                ntt_2d(api, &mut w_buf, nh, nw);

                for i in 0..nh * nw {
                    let prod = api.mul(input_ntt[ci][i], w_buf[i]);
                    accum[i] = api.add(accum[i], prod);
                }
            }

            intt_2d(api, &mut accum, nh, nw);

            for r in 0..h_out {
                for col in 0..w_out {
                    let val = accum[r * nw + col];
                    res[[n, co, r, col]] = api.add(res[[n, co, r, col]], val);
                }
            }
        }
    }

    Ok(res)
}

#[allow(clippy::missing_errors_doc)]
pub fn conv_4d_run<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    logup_ctx: &mut LogupRangeCheckContext,
    input_arr: &ArrayD<Variable>,
    weights: &ArrayD<Variable>,
    bias: &ArrayD<Variable>,
    conv_params: &Conv2DParams,
    quantization_params: &ConvQuantizationParams,
) -> Result<ArrayD<Variable>, CircuitError>
where
    CircuitField<C>: FFTField,
{
    let conv_params = set_default_params(conv_params)?;
    not_yet_implemented_conv(
        &conv_params.input_shape,
        &conv_params.groups,
        &conv_params.dilations,
    )?;

    let s_h = get_u(&conv_params.input_shape, 2, "input_shape[2]").unwrap_or(0);
    let s_w = get_u(&conv_params.input_shape, 3, "input_shape[3]").unwrap_or(0);
    let kh = get_u(&conv_params.kernel_shape, 0, "kernel_shape[0]").unwrap_or(0);
    let kw = get_u(&conv_params.kernel_shape, 1, "kernel_shape[1]").unwrap_or(0);
    let stride_h = get_u(&conv_params.strides, 0, "strides[0]").unwrap_or(1);
    let stride_w = get_u(&conv_params.strides, 1, "strides[1]").unwrap_or(1);
    let pad_0 = get_u(&conv_params.pads, 0, "pads[0]").unwrap_or(0);
    let pad_1 = get_u(&conv_params.pads, 1, "pads[1]").unwrap_or(0);
    let pad_2 = get_u(&conv_params.pads, 2, "pads[2]").unwrap_or(0);
    let pad_3 = get_u(&conv_params.pads, 3, "pads[3]").unwrap_or(0);
    let padded_h = s_h + pad_0 + pad_2;
    let padded_w = s_w + pad_1 + pad_3;

    let out = if should_use_ntt(padded_h, padded_w, kh, kw, stride_h, stride_w) {
        conv_ntt_4d(api, input_arr, &conv_params, weights, bias)?
    } else {
        conv_shape_4(api, input_arr, &conv_params, weights, bias)?
    };

    maybe_rescale(
        api,
        logup_ctx,
        out,
        &MaybeRescaleParams {
            is_rescale: quantization_params.quantized,
            scaling_exponent: quantization_params.scaling,
            n_bits: quantization_params.v_plus_one,
            is_relu: quantization_params.is_relu,
            layer_kind: LayerKind::Conv,
            layer_name: "Conv".into(),
        },
    )
}
