//! 2D MaxPool layer over int64 fixed-point tensors.
//!
//! This layer:
//! - applies sliding-window max pooling over the input tensor, and
//! - uses the `constrained_max` gadget with LogUp-based range checks
//!   to enforce that each pooled output is the maximum over its window.

use std::collections::HashMap;

/// External crate imports
use ndarray::{Array4, ArrayD, Ix4};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{DILATION, INPUT, KERNEL_SHAPE, PADS, STRIDES},
        onnx_model::{
            extract_params_and_expected_shape, get_input_name, get_param, get_param_or_default,
        },
        typecasting::AsIsize,
    },
};

use crate::circuit_functions::gadgets::{
    LogupRangeCheckContext, ShiftRangeContext, constrained_max,
};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct MaxPoolLayer {
    name: String,
    kernel_shape: Vec<usize>,
    strides: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    input_shape: Vec<usize>,
    shift_exponent: usize,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for MaxPoolLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::MaxPool, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::MaxPool,
                name: input_name.clone(),
            })?
            .clone();

        let shape = layer_input.shape();
        if shape.len() != 4 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::MaxPool,
                msg: format!("Expected 4D input for max pooling, got shape: {shape:?}"),
            }
            .into());
        }

        let ceil_mode = false;
        let (out_shape, pool_params) = setup_maxpooling_2d(
            &self.padding,
            &self.kernel_shape,
            &self.strides,
            &self.dilation,
            ceil_mode,
            &self.input_shape,
        )?;

        let output = maxpooling_2d::<C, Builder>(
            api,
            &layer_input,
            &self.input_shape,
            self.shift_exponent,
            &out_shape,
            &pool_params,
        )?;
        Ok((self.outputs.clone(), output))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)?;

        // Get kernel_shape first, because its length defines spatial dims
        let kernel_shape: Vec<usize> = get_param(&layer.name, KERNEL_SHAPE, &params)?;
        let spatial_rank = kernel_shape.len();

        // Default values according to ONNX spec
        let default_dilation: Vec<usize> = vec![1; spatial_rank];
        let default_pads: Vec<usize> = vec![0; 2 * spatial_rank]; // begin + end for each dim
        let default_strides: Vec<usize> = vec![1; spatial_rank];

        let maxpool = Self {
            name: layer.name.clone(),
            kernel_shape,
            strides: get_param_or_default(&layer.name, STRIDES, &params, Some(&default_strides))?,
            dilation: get_param_or_default(
                &layer.name,
                DILATION,
                &params,
                Some(&default_dilation),
            )?,
            padding: get_param_or_default(&layer.name, PADS, &params, Some(&default_pads))?,
            input_shape: expected_shape.clone(),
            shift_exponent: layer_context
                .n_bits_for(&layer.name)
                .checked_sub(1)
                .ok_or_else(|| LayerError::Other {
                    layer: LayerKind::MaxPool,
                    msg: "n_bits too small to derive shift_exponent".to_string(),
                })?,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        };
        Ok(Box::new(maxpool))
    }
}

pub struct PoolingParams {
    pub kernel_shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dilation: Vec<usize>,
    pub new_pads: Vec<[usize; 2]>,
}

/// Normalizes and validates pooling parameters for a 2D max pooling layer.
///
/// Expands scalar or incomplete parameter lists to full 2D vectors and constructs
/// the `PoolingParams` struct for downstream use in layer setup.
///
/// # Arguments
/// - `padding`: Padding per spatial dimension (0, 1, 2, or 4 elements allowed).
/// - `kernel_shape`: Size of the pooling kernel (1 or 2 elements allowed).
/// - `strides`: Stride per spatial dimension (0, 1, or 2 elements allowed).
/// - `dilation`: Dilation per spatial dimension (0, 1, or 2 elements allowed).
///
/// # Returns
/// A `PoolingParams` struct containing normalized kernel shape, strides, dilation, and padding.
///
/// # Errors
/// - [`LayerError::InvalidParameterValue`] if any input array has an invalid length.
pub fn setup_maxpooling_2d_params(
    padding: &[usize],
    kernel_shape: &[usize],
    strides: &[usize],
    dilation: &[usize],
) -> Result<PoolingParams, CircuitError> {
    let padding = match padding.len() {
        0 => vec![0; 4],
        1 => vec![padding[0]; 4],
        2 => vec![padding[0], padding[1], padding[0], padding[1]],
        4 => padding.to_vec(),
        n => {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxPoolLayer".to_string(),
                param_name: PADS.to_string(),
                value: format!("{n} elements"),
            }
            .into());
        }
    };

    let kernel_shape = match kernel_shape.len() {
        1 => vec![kernel_shape[0]; 2],
        2 => vec![kernel_shape[0], kernel_shape[1]],
        n => {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxPoolLayer".to_string(),
                param_name: KERNEL_SHAPE.to_string(),
                value: format!("{n} elements"),
            }
            .into());
        }
    };

    let dilation = match dilation.len() {
        0 => vec![1; 2],
        1 => vec![dilation[0]; 2],
        2 => vec![dilation[0], dilation[1]],
        n => {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxPoolLayer".to_string(),
                param_name: DILATION.to_string(),
                value: format!("{n} elements"),
            }
            .into());
        }
    };

    let strides = match strides.len() {
        0 => vec![1; 2],
        1 => vec![strides[0]; 2],
        2 => vec![strides[0], strides[1]],
        n => {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxPoolLayer".to_string(),
                param_name: STRIDES.to_string(),
                value: format!("{n} elements"),
            }
            .into());
        }
    };

    let new_pads = vec![[padding[0], padding[2]], [padding[1], padding[3]]];

    Ok(PoolingParams {
        kernel_shape,
        strides,
        dilation,
        new_pads,
    })
}

/// Computes the output shape and pooling parameters for a 2D max pooling layer.
///
/// Calculates the spatial dimensions of the output based on input shape, kernel,
/// padding, stride, dilation, and ceil mode. Returns both the output spatial shape
/// and a `PoolingParams` struct containing all relevant pooling parameters.
///
/// # Arguments
/// - `padding`: Padding applied to each spatial dimension.
/// - `kernel_shape`: Size of the pooling kernel for each spatial dimension.
/// - `strides`: Step size for sliding the pooling window.
/// - `dilation`: Dilation applied to the pooling kernel.
/// - `ceil_mode`: Whether to use ceiling mode when computing output dimensions.
/// - `x_shape`: Shape of the input tensor, including batch and channel dimensions.
///
/// # Returns
/// A tuple of:
/// 1. `Vec<usize>`: Computed output spatial dimensions.
/// 2. `PoolingParams`: Struct containing kernel shape, strides, dilation, and padding.
///
/// # Errors
/// - [`LayerError::InvalidParameterValue`] if any parameter (kernel, padding, stride, dilation) is invalid.
/// - Propagates any error from `setup_maxpooling_2d_params`.
pub fn setup_maxpooling_2d(
    padding: &[usize],
    kernel_shape: &[usize],
    strides: &[usize],
    dilation: &[usize],
    ceil_mode: bool,
    x_shape: &[usize],
) -> Result<(Vec<usize>, PoolingParams), CircuitError> {
    let params = setup_maxpooling_2d_params(padding, kernel_shape, strides, dilation)?;
    let pads = params.new_pads;
    let kernel_shape = params.kernel_shape;
    let strides = params.strides;
    let dilation = params.dilation;

    if kernel_shape.is_empty() {
        return Err(LayerError::InvalidParameterValue {
            layer: LayerKind::MaxPool,
            layer_name: "MaxPoolLayer".to_string(),
            param_name: KERNEL_SHAPE.to_string(),
            value: "empty".to_string(),
        }
        .into());
    }

    let new_pads: Vec<[usize; 2]> = pads.clone();

    let input_spatial_shape = &x_shape[2..];
    let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

    for i in 0..input_spatial_shape.len() {
        let total_padding = new_pads[i][0] + new_pads[i][1];
        let kernel_extent = (kernel_shape[i] - 1) * dilation[i] + 1;
        let numerator = if input_spatial_shape[i] + total_padding >= kernel_extent {
            input_spatial_shape[i] + total_padding - kernel_extent
        } else {
            return Err(LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxPoolLayer".to_string(),
                param_name: KERNEL_SHAPE.to_string(),
                value: format!(
                    "kernel extent ({kernel_extent}) larger than input+padding ({} + {})",
                    input_spatial_shape[i], total_padding
                ),
            }
            .into());
        };

        if ceil_mode {
            // Use integer ceiling division
            let mut out_dim = numerator.div_ceil(strides[i]) + 1;
            let need_to_reduce =
                (out_dim - 1) * strides[i] >= input_spatial_shape[i] + new_pads[i][0];
            if need_to_reduce {
                out_dim -= 1;
            }
            output_spatial_shape[i] = out_dim;
        } else {
            output_spatial_shape[i] = (numerator / strides[i]) + 1;
        }
    }
    Ok((
        output_spatial_shape,
        PoolingParams {
            kernel_shape,
            strides,
            dilation,
            new_pads,
        },
    ))
}

/// Reshapes a flat 1D array of variables into a 4D tensor.
///
/// # Arguments
/// - `flat`: Slice of variables representing the flattened tensor.
/// - `dims`: Target 4D shape `[batch, channels, height, width]`.
///
/// # Returns
/// A 4D array as `ArrayD<Variable>` reshaped from the input slice.
///
/// # Errors
/// - [`LayerError::InvalidShape`] if the input slice cannot be reshaped into the specified 4D dimensions.
///
pub fn reshape_4d(flat: &[Variable], dims: [usize; 4]) -> Result<ArrayD<Variable>, LayerError> {
    Array4::from_shape_vec(dims, flat.to_vec())
        .map(ndarray::ArrayBase::into_dyn)
        .map_err(|_| LayerError::InvalidShape {
            layer: LayerKind::MaxPool,
            msg: format!("reshape_4d: cannot reshape into {dims:?}"),
        })
}

/// Performs 2D max pooling on a 4D input tensor.
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `x`: Input tensor as a dynamic-dimensional array (`ArrayD<Variable>`).
/// - `x_shape`: Shape of the input tensor.
/// - `shift_exponent`: Exponent for fixed-point shifting in the circuit.
/// - `params`: Pooling parameters including kernel size, strides, dilation, padding, and output shape.
///
/// # Returns
/// A new `ArrayD<Variable>` with the max-pooled values.
///
/// # Errors
/// - [`LayerError::InvalidShape`] if `x` has fewer than 3 dimensions or is incompatible with the kernel shape.
/// - [`UtilsError::ValueConversionError`] if any indices cannot be converted from `usize` to `isize`.
/// - [`CircuitError`] if `unconstrained_max` or `constrained_max` fails within the circuit.
pub fn maxpooling_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &ArrayD<Variable>,
    x_shape: &[usize],
    shift_exponent: usize,
    output_spatial_shape: &[usize],
    params: &PoolingParams,
) -> Result<ArrayD<Variable>, CircuitError> {
    let global_pooling = false;
    let batch = x_shape[0];
    let channels = x_shape[1];
    let height = x_shape[2];

    let kernel_shape = &params.kernel_shape;
    let strides = &params.strides;
    let dilation = &params.dilation;
    let new_pads = &params.new_pads;

    let width = if kernel_shape.len() > 1 {
        x_shape[3]
    } else {
        1
    };

    if x_shape.len() < 3 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::MaxPool,
            msg: format!("Expected at least 3D input, got {x_shape:?}"),
        }
        .into());
    }
    if kernel_shape.len() > 1 && x_shape.len() < 4 {
        return Err(LayerError::InvalidShape {
            layer: LayerKind::MaxPool,
            msg: format!("Expected 4D input for 2D kernel, got {x_shape:?}"),
        }
        .into());
    }

    let pooled_height = output_spatial_shape[0];
    let pooled_width = if kernel_shape.len() > 1 {
        output_spatial_shape[1]
    } else {
        1
    };

    let y_dims = [batch, channels, pooled_height, pooled_width];
    let mut y = Array4::from_elem(y_dims, api.constant(0));

    let stride_h = if global_pooling { 1 } else { strides[0] };
    let stride_w = if global_pooling {
        1
    } else if strides.len() > 1 {
        strides[1]
    } else {
        1
    };

    let dilation_h = dilation[0];
    let dilation_w = if dilation.len() > 1 { dilation[1] } else { 1 };

    let array4 = x
        .clone()
        .into_dimensionality::<Ix4>()
        .map_err(|_| LayerError::InvalidShape {
            layer: LayerKind::MaxPool,
            msg: "Expected 4D input for maxpooling".to_string(),
        })?;

    let context = ShiftRangeContext::new(api, shift_exponent)?;

    let mut logup_ctx = LogupRangeCheckContext::new_default();
    logup_ctx.init::<C, Builder>(api);

    for n in 0..batch {
        for c in 0..channels {
            for ph in 0..pooled_height {
                let hstart = ph.as_isize()? * stride_h.as_isize()? - new_pads[0][0].as_isize()?;
                let hend = hstart + (kernel_shape[0] * dilation_h).as_isize()?;

                for pw in 0..pooled_width {
                    let wstart =
                        pw.as_isize()? * stride_w.as_isize()? - new_pads[1][0].as_isize()?;
                    let wend = wstart + (kernel_shape[1] * dilation_w).as_isize()?;

                    let mut values: Vec<Variable> = Vec::new();

                    for h in (hstart..hend).step_by(dilation_h) {
                        let h_usize = match usize::try_from(h) {
                            Ok(val) if h < 0 || val < height => val,
                            _ => continue,
                        };

                        for w in (wstart..wend).step_by(dilation_w) {
                            let w_usize = match usize::try_from(w) {
                                Ok(val) if w < 0 || val < width => val,
                                _ => continue,
                            };

                            let val = array4[[n, c, h_usize, w_usize]];
                            values.push(val);
                        }
                    }

                    if !values.is_empty() {
                        let max = constrained_max(api, &context, &mut logup_ctx, &values)?;
                        y[[n, c, ph, pw]] = max;
                    }
                }
            }
        }
    }
    // Commit LogUp constraints once
    logup_ctx.finalize::<C, Builder>(api);
    Ok(y.into_dyn())
}
