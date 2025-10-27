use std::collections::HashMap;

/// External crate imports
use ndarray::{Array4, ArrayD, Ix4};

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use super::super::utils::core_math::{assert_is_bitstring_and_reconstruct, unconstrained_to_bits};
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

        let maxpool = Self {
            name: layer.name.clone(),
            kernel_shape,
            strides: get_param(&layer.name, STRIDES, &params)?,
            dilation: get_param_or_default(
                &layer.name,
                DILATION,
                &params,
                Some(&default_dilation),
            )?,
            padding: get_param_or_default(&layer.name, PADS, &params, Some(&default_pads))?,
            input_shape: expected_shape.clone(),
            shift_exponent: layer_context.n_bits - 1,
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

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: unconstrained_max
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the maximum value in a nonempty slice of field elements (interpreted as integers in `[0, p−1]`),
/// using only unconstrained witness operations and explicit selection logic.
///
/// Internally, this function performs pairwise comparisons using `unconstrained_greater` and `unconstrained_lesser_eq`,
/// and selects the maximum via weighted sums:
/// `current_max ← v·(v > current_max) + current_max·(v ≤ current_max)`
///
/// # Errors
/// - If `values` is empty.
///
/// # Arguments
/// - `api`: A mutable reference to the circuit builder implementing `RootAPI<C>`.
/// - `values`: A slice of `Variable`s, each assumed to lie in the range `[0, p−1]`.
///
/// # Returns
/// A `Variable` encoding `max_i values[i]`, the maximum value in the slice.
///
/// # Example
/// ```ignore
/// // For values = [7, 2, 9, 5], returns 9.
/// ```
pub fn unconstrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::MaxPool,
            msg: "unconstrained_max: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // Initialize with the first element
    let mut current_max = values[0];
    for &v in &values[1..] {
        // Compute indicators: is_greater = 1 if v > current_max, else 0
        let is_greater = api.unconstrained_greater(v, current_max);
        let is_not_greater = api.unconstrained_lesser_eq(v, current_max);

        // Select either v or current_max based on indicator bits
        let take_v = api.unconstrained_mul(v, is_greater);
        let keep_old = api.unconstrained_mul(current_max, is_not_greater);

        // Update current_max
        current_max = api.unconstrained_add(take_v, keep_old);
    }

    Ok(current_max)
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: MaxAssertionContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for applying `constrained_max` with a fixed shift exponent `s`,
/// to avoid recomputing constants in repeated calls (e.g., in max pooling).
pub struct MaxAssertionContext {
    /// The exponent `s` such that `S = 2^s`.
    pub shift_exponent: usize,

    /// The offset `S = 2^s`, lifted as a constant into the circuit.
    pub offset: Variable,
}

impl MaxAssertionContext {
    /// Creates a new context for asserting maximums, given a `shift_exponent = s`.
    ///
    /// # Type Parameters
    /// - `C`: Configuration type.
    /// - `Builder`: Builder implementing the `RootAPI<C>` trait.
    ///
    /// # Arguments
    /// - `api`: Mutable reference to a builder for creating constants.
    /// - `shift_exponent`: Exponent `s` used to compute `2^s`.
    ///
    /// # Returns
    /// An instance of the assertion context containing `shift_exponent` and `offset`.
    ///
    /// # Errors
    /// - [`LayerError::Other`] if `shift_exponent` is too large to fit in a `u32`.
    /// - [`LayerError::InvalidParameterValue`] if the computed offset overflows `u32`.
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        shift_exponent: usize,
    ) -> Result<Self, LayerError> {
        let offset_: u32 = 1u32
            .checked_shl(
                u32::try_from(shift_exponent).map_err(|_| LayerError::Other {
                    layer: LayerKind::MaxPool,
                    msg: format!("Shift exponent {shift_exponent} is too large for type: u32"),
                })?,
            )
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxAssertionContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: shift_exponent.to_string(),
            })?;
        let offset = api.constant(offset_);
        Ok(Self {
            shift_exponent,
            offset,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: constrained_max
// ─────────────────────────────────────────────────────────────────────────────

/// Asserts that a given slice of `Variable`s contains a maximum value `M`,
/// by verifying that some `x_i` satisfies `M = max_i x_i`, using a combination of
/// unconstrained helper functions and explicit constraint assertions,
/// along with an offset-shifting technique to reduce comparisons to the
/// nonnegative range `[0, 2^(s + 1) − 1]`.
///
/// # Idea
/// Each `x_i` is a field element (i.e., a `Variable` representing the least nonnegative residue mod `p`)
/// that is **assumed** to encode a signed integer in the interval `[-S, T − S] = [−2^s, 2^s − 1]`,
/// where `S = 2^s` and `T = 2·S - 1 = 2^(s + 1) − 1]`.
///
/// Since all circuit operations take place in `𝔽_p` and each `x_i` is already reduced modulo `p`,
/// we shift each value by `S` on-circuit to ensure that the quantity `x_i + S` lands in `[0, T]`.
/// Under the assumption that `x_i ∈ [−S, T − S]`, this shift does **not** wrap around modulo `p`,
/// so `x_i + S` in `𝔽_p` reflects the true integer sum.
///
/// We then compute:
/// ```text
///     M^♯ = max_i (x_i + S)
///     M   = M^♯ − S mod p
/// ```
/// to recover the **least nonnegative residue** of the maximum value, `M`.
///
/// To verify that `M` is indeed the maximum:
/// - For each `x_i`, we compute `Δ_i = M − x_i`, and use bit decomposition to enforce
///   `Δ_i ∈ [0, T]`, using `s + 1` bits.
/// - Then we constrain the product `∏_i Δ_i` to be zero. This ensures that at least one
///   `Δ_i = 0`, i.e., that some `x_i = M`.
///
/// # Example
/// Suppose the input slice encodes the signed integers `[-2, 0, 3]`, and `s = 2`, so `S = 4`, `T = 7`.
///
/// - Shift:
///   `x_0 = -2` ⇒ `x_0 + S = 2`
///   `x_1 =  0` ⇒ `x_1 + S = 4`
///   `x_2 =  3` ⇒ `x_2 + S = 7`
///
/// - Compute:
///   `M^♯ = max{x_i + S} = 7`
///   `M   = M^♯ − S = 3`
///
/// - Verify:
///   For each `x_i`, compute `Δ_i = M − x_i ∈ [0, 7]`
///   The values are: `Δ = [5, 3, 0]`
///   Since one `Δ_i = 0`, we conclude that some `x_i = M`.
///
/// # Assumptions
/// - All values `x_i` are `Variable`s in `𝔽_p` that **encode signed integers** in `[-S, T − S]`.
/// - The prime `p` satisfies `p > T = 2^(s + 1) − 1`, so no wraparound occurs in `x_i + S`.
///
/// # Errors
/// - If `values` is empty.
/// - If computing `2^s` or `s + 1` overflows a `u32`.
///
/// # Type Parameters
/// - `C`: The circuit field configuration.
/// - `Builder`: A builder implementing `RootAPI<C>`.
///
/// # Arguments
/// - `api`: Your circuit builder.
/// - `context`: A `MaxAssertionContext` holding shift-related parameters.
/// - `values`: A nonempty slice of `Variable`s, each encoding an integer in `[-S, T − S]`.
pub fn constrained_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &MaxAssertionContext, // S = 2^s = context.offset
    values: &[Variable],
) -> Result<Variable, CircuitError> {
    // 0) Require nonempty input
    if values.is_empty() {
        return Err(LayerError::Other {
            layer: LayerKind::MaxPool,
            msg: "constrained_max: input slice must be nonempty".to_string(),
        }
        .into());
    }

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute max_i (x_i^♯), which equals M^♯ = M + S
    let max_offset = unconstrained_max(api, &values_offset)?;

    // 3) Recover M = M^♯ − S
    let max_raw = api.sub(max_offset, context.offset);

    // 4) For each x_i, range-check Δ_i = M − x_i ∈ [0, T] using s + 1 bits
    let n_bits =
        context
            .shift_exponent
            .checked_add(1)
            .ok_or_else(|| LayerError::InvalidParameterValue {
                layer: LayerKind::MaxPool,
                layer_name: "MaxAssertionContext".to_string(),
                param_name: "shift_exponent".to_string(),
                value: context.shift_exponent.to_string(),
            })?;
    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        // Δ ∈ [0, T] ⇔ ∃ bitstring of length s + 1 summing to Δ
        let bits = unconstrained_to_bits(api, delta, n_bits).map_err(|e| LayerError::Other {
            layer: LayerKind::MaxPool,
            msg: format!("unconstrained_to_bits failed: {e}"),
        })?;
        // TO DO: elaborate/make more explicit, e.g. "Range check enforcing Δ >= 0"
        let recon =
            assert_is_bitstring_and_reconstruct(api, &bits).map_err(|e| LayerError::Other {
                layer: LayerKind::MaxPool,
                msg: format!("assert_is_bitstring_and_reconstruct failed: {e}"),
            })?;
        api.assert_is_equal(delta, recon);

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
    Ok(max_raw)
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

    let context = MaxAssertionContext::new(api, shift_exponent)?;

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
                        let max = constrained_max(api, &context, &values)?;
                        y[[n, c, ph, pw]] = max;
                    }
                }
            }
        }
    }

    Ok(y.into_dyn())
}
