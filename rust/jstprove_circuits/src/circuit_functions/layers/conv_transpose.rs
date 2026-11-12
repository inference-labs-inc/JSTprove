// ONNX `ConvTranspose` layer for ZK circuits.
//
// # ZK approach
// ConvTranspose (transposed convolution) is implemented as an inverse-mapping
// convolution: for each output element, we find the input positions and kernel
// positions that contribute to it, multiply them together, and accumulate.
//
// The weight layout (ONNX convention) is [C_in, C_out/group, kH, kW].
// The algorithm matches the ONNX ConvTranspose specification exactly.
//
// Only group=1 and dilation=1 are supported; 4-D inputs (NCHW) are required.

use std::collections::HashMap;

use ndarray::{ArrayD, s};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{DILATION, GROUP, INPUT, KERNEL_SHAPE, PADS, STRIDES, WEIGHTS},
        onnx_model::{
            extract_params, get_input_name, get_optional_input_name, get_param,
            get_param_or_default, get_w_or_b,
        },
        quantization::{MaybeRescaleParams, maybe_rescale},
        tensor_ops::load_array_constants_or_get_inputs,
    },
};

// -------- Struct --------

#[derive(Debug)]
pub struct ConvTransposeLayer {
    weights: Option<ArrayD<i64>>,
    bias: Option<ArrayD<i64>>,
    strides: Vec<u32>,
    kernel_shape: Vec<u32>,
    group: u32,
    dilation: Vec<u32>,
    /// ONNX format: [begin_H, begin_W, end_H, end_W]
    pads: Vec<u32>,
    output_padding: Vec<u32>,
    scaling: u64,
    v_plus_one: usize,
    is_rescale: bool,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

struct Axis {
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad_begin: usize,
    pad_end: usize,
    output_padding: usize,
}

impl Axis {
    fn output_extent(&self) -> Option<usize> {
        let effective_kernel = (self.kernel - 1) * self.dilation + 1;
        (self.stride * self.input + effective_kernel + self.output_padding)
            .checked_sub(self.stride + self.pad_begin + self.pad_end)
            .filter(|&extent| extent > 0)
    }

    fn source_index(&self, out: usize, kernel_pos: usize) -> Option<usize> {
        let offset = (out + self.pad_begin).checked_sub(kernel_pos * self.dilation)?;
        (offset % self.stride == 0)
            .then_some(offset / self.stride)
            .filter(|&index| index < self.input)
    }
}

impl ConvTransposeLayer {
    fn axis(&self, dim: usize, kernel: usize, input: usize) -> Axis {
        Axis {
            input,
            kernel,
            stride: self.strides[dim] as usize,
            dilation: self.dilation[dim] as usize,
            pad_begin: self.pads[dim] as usize,
            pad_end: self.pads[dim + 2] as usize,
            output_padding: self.output_padding[dim] as usize,
        }
    }

    fn check_weights(
        &self,
        w_shape: &[usize],
        c_in: usize,
        kh: usize,
        kw: usize,
    ) -> Result<usize, CircuitError> {
        if w_shape.len() != 4 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ConvTranspose,
                msg: format!(
                    "weights must be 4-D [C_in, C_out/group, kH, kW], got {}D",
                    w_shape.len()
                ),
            }
            .into());
        }
        if w_shape[0] != c_in {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ConvTranspose,
                msg: format!("weights C_in dim {} != input C_in {c_in}", w_shape[0]),
            }
            .into());
        }
        if w_shape[2] != kh || w_shape[3] != kw {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ConvTranspose,
                msg: format!(
                    "weights spatial shape [{}, {}] != kernel_shape [{kh}, {kw}]",
                    w_shape[2], w_shape[3]
                ),
            }
            .into());
        }
        Ok(w_shape[1] * self.group as usize)
    }
}

fn check_spatial_params(
    layer_name: &str,
    strides: &[u32],
    dilation: &[u32],
    output_padding: &[u32],
    pads: &[u32],
) -> Result<(), CircuitError> {
    for (param_name, values, expected_len) in [
        (STRIDES, strides, 2usize),
        (DILATION, dilation, 2),
        ("output_padding", output_padding, 2),
        (PADS, pads, 4),
    ] {
        if values.len() != expected_len {
            return Err(LayerError::UnsupportedConfig {
                layer: LayerKind::ConvTranspose,
                msg: format!(
                    "layer '{layer_name}': parameter '{param_name}' has length {} but expected {expected_len}",
                    values.len()
                ),
            }
            .into());
        }
    }
    if strides.contains(&0) {
        return Err(LayerError::UnsupportedConfig {
            layer: LayerKind::ConvTranspose,
            msg: format!("layer '{layer_name}': strides must all be > 0, got {strides:?}"),
        }
        .into());
    }
    Ok(())
}

fn check_bias(bias_shape: &[usize], c_out: usize) -> Result<(), CircuitError> {
    if bias_shape.len() != 1 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::ConvTranspose,
            msg: format!("bias must be 1-D, got {}D", bias_shape.len()),
        }
        .into());
    }
    if bias_shape[0] != c_out {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::ConvTranspose,
            msg: format!("bias length {} != output channels {c_out}", bias_shape[0]),
        }
        .into());
    }
    Ok(())
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConvTransposeLayer {
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::ConvTranspose, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ConvTranspose,
                name: input_name.clone(),
            })?
            .clone();

        let w_name = get_input_name(&self.inputs, 1, LayerKind::ConvTranspose, WEIGHTS)?;
        let weights = load_array_constants_or_get_inputs(
            api,
            input,
            w_name,
            &self.weights,
            LayerKind::ConvTranspose,
        )?;

        let bias = match get_optional_input_name(&self.inputs, 2) {
            Some(b_name) => load_array_constants_or_get_inputs(
                api,
                input,
                b_name,
                &self.bias,
                LayerKind::ConvTranspose,
            )?,
            None => ArrayD::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).expect("empty bias"),
        };

        // Validate input rank
        if layer_input.ndim() != 4 {
            return Err(LayerError::UnsupportedConfig {
                layer: LayerKind::ConvTranspose,
                msg: format!("input must be 4-D (NCHW), got {}D", layer_input.ndim()),
            }
            .into());
        }

        // Input shape: [N, C_in, H_in, W_in]
        let n_batch = layer_input.shape()[0];
        let c_in = layer_input.shape()[1];
        let h_in = layer_input.shape()[2];
        let w_in = layer_input.shape()[3];

        // Weight shape: [C_in, C_out/group, kH, kW]
        let kh = self.kernel_shape[0] as usize;
        let kw = self.kernel_shape[1] as usize;

        let c_out = self.check_weights(weights.shape(), c_in, kh, kw)?;

        let rows = self.axis(0, kh, h_in);
        let cols = self.axis(1, kw, w_in);
        let (Some(out_h), Some(out_w)) = (rows.output_extent(), cols.output_extent()) else {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::ConvTranspose,
                msg: "computed output spatial size is non-positive; check kernel/stride/pad/output_padding".into(),
            }
            .into());
        };

        // Initialise result array; populate bias along the output-channel dimension.
        let zero = api.constant(0);
        let mut res = ArrayD::from_elem(vec![n_batch, c_out, out_h, out_w], zero);
        if !bias.is_empty() {
            check_bias(bias.shape(), c_out)?;
            for oc in 0..c_out {
                res.slice_mut(s![.., oc, .., ..]).fill(bias[[oc]]);
            }
        }

        // ConvTranspose inverse-mapping loop.
        //
        // For each output position (oh, ow) we find which input positions (ih, iw)
        // and kernel positions (ki_h, ki_w) contribute:
        //
        //   oh = ih * stride_h + ki_h * dil_h - pad_h_begin
        //   => ih_num = oh + pad_h_begin - ki_h * dil_h
        //   => ih = ih_num / stride_h  (requires ih_num >= 0, ih_num % stride_h == 0, ih < h_in)
        for n in 0..n_batch {
            for oc in 0..c_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        for ci in 0..c_in {
                            for ki_h in 0..kh {
                                let Some(ih) = rows.source_index(oh, ki_h) else {
                                    continue;
                                };
                                for ki_w in 0..kw {
                                    let Some(iw) = cols.source_index(ow, ki_w) else {
                                        continue;
                                    };
                                    // Weight layout: [C_in, C_out/group, kH, kW]
                                    let prod = api.mul(
                                        layer_input[[n, ci, ih, iw]],
                                        weights[[ci, oc, ki_h, ki_w]],
                                    );
                                    res[[n, oc, oh, ow]] = api.add(res[[n, oc, oh, ow]], prod);
                                }
                            }
                        }
                    }
                }
            }
        }

        let out = maybe_rescale(
            api,
            logup_ctx,
            res,
            &MaybeRescaleParams {
                is_rescale: self.is_rescale,
                scaling_exponent: self.scaling,
                n_bits: self.v_plus_one,
                is_relu: false,
                layer_kind: LayerKind::ConvTranspose,
                layer_name: "ConvTranspose".into(),
            },
        )?;

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let params = extract_params(layer).map_err(|e| LayerError::Other {
            layer: LayerKind::ConvTranspose,
            msg: format!("extract_params failed: {e}"),
        })?;

        let w_name = get_input_name(&layer.inputs, 1, LayerKind::ConvTranspose, WEIGHTS)?;
        let b_name = get_optional_input_name(&layer.inputs, 2);

        let (weights, bias) = if layer_context.weights_as_inputs {
            (None, None)
        } else {
            let w = get_w_or_b(layer_context.w_and_b_map, w_name).map_err(|e| {
                LayerError::MissingParameter {
                    layer: LayerKind::ConvTranspose,
                    param: format!("weights (W), requested={w_name}: {e}"),
                }
            })?;
            let b = match b_name {
                Some(name) => Some(get_w_or_b(layer_context.w_and_b_map, name).map_err(|e| {
                    LayerError::MissingParameter {
                        layer: LayerKind::ConvTranspose,
                        param: format!("bias (B), requested={name}: {e}"),
                    }
                })?),
                None => None,
            };
            (Some(w), b)
        };

        let kernel_shape: Vec<u32> = get_param(&layer.name, KERNEL_SHAPE, &params)?;
        let spatial_rank = kernel_shape.len();

        if spatial_rank != 2 {
            return Err(LayerError::UnsupportedConfig {
                layer: LayerKind::ConvTranspose,
                msg: format!(
                    "layer '{}': only 2-D spatial ConvTranspose is supported, got spatial_rank={}",
                    layer.name, spatial_rank
                ),
            }
            .into());
        }

        if kernel_shape.contains(&0) {
            return Err(LayerError::UnsupportedConfig {
                layer: LayerKind::ConvTranspose,
                msg: format!(
                    "layer '{}': kernel_shape contains a zero dimension: {:?}",
                    layer.name, kernel_shape
                ),
            }
            .into());
        }

        let default_zeros: Vec<u32> = vec![0; 2 * spatial_rank];
        let default_op: Vec<u32> = vec![0; spatial_rank];
        let default_ones: Vec<u32> = vec![1; spatial_rank];

        let group: u32 = get_param_or_default(&layer.name, GROUP, &params, Some(&1u32))?;

        if group != 1 {
            return Err(LayerError::UnsupportedConfig {
                layer: LayerKind::ConvTranspose,
                msg: "group > 1 not yet implemented".into(),
            }
            .into());
        }

        let dilation: Vec<u32> =
            get_param_or_default(&layer.name, DILATION, &params, Some(&default_ones))?;

        if dilation.iter().any(|&d| d != 1) {
            return Err(LayerError::UnsupportedConfig {
                layer: LayerKind::ConvTranspose,
                msg: format!("dilation != 1 not yet implemented: {dilation:?}"),
            }
            .into());
        }

        let strides: Vec<u32> =
            get_param_or_default(&layer.name, STRIDES, &params, Some(&default_ones))?;
        let pads: Vec<u32> =
            get_param_or_default(&layer.name, PADS, &params, Some(&default_zeros))?;
        let output_padding: Vec<u32> =
            get_param_or_default(&layer.name, "output_padding", &params, Some(&default_op))?;

        check_spatial_params(&layer.name, &strides, &dilation, &output_padding, &pads)?;

        let conv_transpose = Self {
            weights,
            bias,
            strides,
            kernel_shape,
            group,
            dilation,
            pads,
            output_padding,
            scaling: circuit_params.scale_exponent.into(),
            v_plus_one: layer_context.n_bits_for(&layer.name),
            is_rescale,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };

        Ok(Box::new(conv_transpose))
    }
}
