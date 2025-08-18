use std::collections::HashMap;

/// External crate imports
use ndarray::{Array4, ArrayD, Ix4};

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::{constants::{DILATION, KERNEL_SHAPE, PADS, STRIDES}, onnx_model::{extract_params_and_expected_shape, get_param}}};
/// Internal crate imports
use super::super::utils::core_math::{assert_is_bitstring_and_reconstruct, unconstrained_to_bits};

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
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let layer_input = input.get(&self.inputs[0])
        .ok_or_else(|| panic!("Missing input {}", self.inputs[0].clone())).unwrap()
    .clone();
        let shape = layer_input.shape();
        assert_eq!(shape.len(), 4, "Expected 4D input for max pooling, got shape: {:?}", shape);

        let ceil_mode = false;
        let (kernel, strides, dilation, out_shape, pads) = setup_maxpooling_2d(
            &self.padding,
            &self.kernel_shape,
            &self.strides,
            &self.dilation,
            ceil_mode,
            &self.input_shape,
        );

        let output = maxpooling_2d::<C, Builder>(
            api,
            &layer_input,
            &kernel,
            &strides,
            &dilation,
            &out_shape,
            &self.input_shape,
            &pads,
            self.shift_exponent,
        );

        Ok((self.outputs.clone(), output))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {

        let (params, expected_shape) = extract_params_and_expected_shape(layer_context, layer);

        let maxpool = Self{
            name: layer.name.clone(),
            kernel_shape: get_param(&layer.name, KERNEL_SHAPE, &params),
            strides: get_param(&layer.name, STRIDES, &params),
            dilation: get_param(&layer.name, DILATION, &params),
            padding: get_param(&layer.name, PADS, &params),
            input_shape: expected_shape.clone(),
            shift_exponent: layer_context.n_bits - 1,
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
        };
        Ok(Box::new(maxpool))
    }
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
/// # Panics
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
) -> Variable {
    assert!(
        !values.is_empty(),
        "unconstrained_max: input slice must be nonempty"
    );

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

    current_max
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: MaxAssertionContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for applying `assert_is_max` with a fixed shift exponent `s`,
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
    /// Computes `S = 2^s` and lifts it to a constant for reuse.
    pub fn new<C: Config, Builder: RootAPI<C>>(api: &mut Builder, shift_exponent: usize) -> Self {
        let offset_: u32 = 1u32
            .checked_shl(shift_exponent as u32)
            .expect("shift_exponent must be less than 32");
        let offset = api.constant(offset_);
        Self {
            shift_exponent,
            offset,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: assert_is_max
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
/// # Panics
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
pub fn assert_is_max<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    context: &MaxAssertionContext, // S = 2^s = context.offset
    values: &[Variable],
) {
    // 0) Require nonempty input
    assert!(
        !values.is_empty(),
        "assert_is_max: input slice must be nonempty"
    );

    // 1) Form offset-shifted values: x_i^♯ = x_i + S
    let mut values_offset = Vec::with_capacity(values.len());
    for &x in values {
        values_offset.push(api.add(x, context.offset));
    }

    // 2) Compute max_i (x_i^♯), which equals M^♯ = M + S
    let max_offset = unconstrained_max(api, &values_offset);

    // 3) Recover M = M^♯ − S
    let max_raw = api.sub(max_offset, context.offset);

    // 4) For each x_i, range-check Δ_i = M − x_i ∈ [0, T] using s + 1 bits
    let n_bits = context
        .shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 must fit in usize");
    let mut prod = api.constant(1);

    for &x in values {
        let delta = api.sub(max_raw, x);

        // Δ ∈ [0, T] ⇔ ∃ bitstring of length s + 1 summing to Δ
        let bits = unconstrained_to_bits(api, delta, n_bits);
        // TO DO: elaborate/make more explicit, e.g. "Range check enforcing Δ >= 0"
        let recon = assert_is_bitstring_and_reconstruct(api, &bits);
        api.assert_is_equal(delta, recon);

        // Multiply all Δ_i together
        prod = api.mul(prod, delta);
    }

    // 5) Final check: ∏ Δ_i = 0 ⇔ ∃ x_i such that Δ_i = 0 ⇔ x_i = M
    api.assert_is_zero(prod);
}

pub fn setup_maxpooling_2d_params(
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let padding = match padding.len() {
        0 => vec![0; 4],
        1 => vec![padding[0]; 4],
        2 => vec![padding[0], padding[1], padding[0], padding[1]],
        3 => panic!("Padding cannot have 3 elements"),
        4 => padding.clone(),
        _ => panic!("Padding must have between 1 and 4 elements"),
    };

    let kernel_shape = match kernel_shape.len() {
        1 => vec![kernel_shape[0]; 2],
        2 => vec![kernel_shape[0], kernel_shape[1]],
        _ => panic!("Kernel shape must have between 1 and 2 elements"),
    };

    let dilation = match dilation.len() {
        0 => vec![1; 2],
        1 => vec![dilation[0]; 2],
        2 => vec![dilation[0], dilation[1]],
        _ => panic!("Dilation must have between 1 and 2 elements"),
    };

    let strides = match strides.len() {
        0 => vec![1; 2],
        1 => vec![strides[0]; 2],
        2 => vec![strides[0], strides[1]],
        _ => panic!("Strides must have between 1 and 2 elements"),
    };
    return (padding, kernel_shape, strides, dilation);
}

pub fn setup_maxpooling_2d(
    padding: &Vec<usize>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
    ceil_mode: bool,
    x_shape: &Vec<usize>,
) -> (
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<[usize; 2]>,
) {
    let (pads, kernel_shape, strides, dilation) =
        setup_maxpooling_2d_params(&padding, &kernel_shape, &strides, &dilation);

    let n_dims = kernel_shape.len();

    // Create new_pads as Vec<[usize; 2]>
    let mut new_pads = vec![[0; 2]; n_dims];
    for i in 0..n_dims {
        new_pads[i] = [pads[i], pads[i + n_dims]];
    }

    let input_spatial_shape = &x_shape[2..];
    let mut output_spatial_shape = vec![0; input_spatial_shape.len()];

    for i in 0..input_spatial_shape.len() {
        let total_padding = new_pads[i][0] + new_pads[i][1];
        let kernel_extent = (kernel_shape[i] - 1) * dilation[i] + 1;
        let numerator = input_spatial_shape[i] + total_padding - kernel_extent;

        if ceil_mode {
            let mut out_dim = (numerator as f64 / strides[i] as f64).ceil() as usize + 1;
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
    return (
        kernel_shape,
        strides,
        dilation,
        output_spatial_shape,
        new_pads,
    );
}

/// Reshape a flat array into a 4D `ArrayD`.
pub fn reshape_4d(flat: &[Variable], dims: [usize; 4]) -> ArrayD<Variable> {
    let array4 = Array4::from_shape_vec(dims, flat.to_vec()).expect("reshape_4d: shape mismatch");
    array4.into_dyn()
}

/// Perform 2D max pooling using `ArrayD` instead of nested vectors.
pub fn maxpooling_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &ArrayD<Variable>,
    kernel_shape: &Vec<usize>,
    strides: &Vec<usize>,
    dilation: &Vec<usize>,
    output_spatial_shape: &Vec<usize>,
    x_shape: &Vec<usize>,
    new_pads: &Vec<[usize; 2]>,
    shift_exponent: usize,
) -> ArrayD<Variable> {
    let global_pooling = false;
    let batch = x_shape[0];
    let channels = x_shape[1];
    let height = x_shape[2];
    let width = if kernel_shape.len() > 1 {
        x_shape[3]
    } else {
        1
    };

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
        .expect("Expected 4D input for maxpooling");

    let context = MaxAssertionContext::new(api, shift_exponent);

    for n in 0..batch {
        for c in 0..channels {
            for ph in 0..pooled_height {
                let hstart = ph as isize * stride_h as isize - new_pads[0][0] as isize;
                let hend = hstart + (kernel_shape[0] * dilation_h) as isize;

                for pw in 0..pooled_width {
                    let wstart = pw as isize * stride_w as isize - new_pads[1][0] as isize;
                    let wend = wstart + (kernel_shape[1] * dilation_w) as isize;

                    let mut values: Vec<Variable> = Vec::new();

                    for h in (hstart..hend).step_by(dilation_h) {
                        if h < 0 || h >= height as isize {
                            continue;
                        }
                        for w in (wstart..wend).step_by(dilation_w) {
                            if w < 0 || w >= width as isize {
                                continue;
                            }

                            let val = array4[[n, c, h as usize, w as usize]];
                            values.push(val);
                        }
                    }

                    if !values.is_empty() {
                        let max = unconstrained_max(api, &values);
                        assert_is_max(api, &context, &values);
                        y[[n, c, ph, pw]] = max;
                    }
                }
            }
        }
    }

    y.into_dyn()
}
