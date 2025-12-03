use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal crate imports
use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        ArrayConversionError, RescaleError,
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

use crate::circuit_functions::gadgets::constrained_reconstruct_from_bits;
use crate::circuit_functions::hints::{
    unconstrained_clip, unconstrained_max, unconstrained_min, unconstrained_to_bits,
};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReluLayer {
    name: String,
    index: usize,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
}
// -------- Implementations --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReluLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::ReLU, INPUT)?;
        let layer_input = input
            .get(&input_name.clone())
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ReLU,
                name: input_name.clone(),
            })?
            .clone();
        let out = layer_input;
        let out = relu_array(api, &out, self.n_bits - 1)?;

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (_params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::ReLU,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;
        let relu = Self {
            name: layer.name.clone(),
            index,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits: layer_context.n_bits,
        };
        Ok(Box::new(relu))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: ReluContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for asserting a `ReLU` operation on signed integers represented in least residue form.
///
/// # Details
/// - We assume each input `x` satisfies:
///   `x = c mod p`, where `c ∈ [-2^s, 2^s - 1]`
///
/// - We shift `x` by `2^s` (a nonnegative constant) to obtain `x' = x + 2^s ∈ [0, 2^{s+1})`
///
/// - We bit-decompose `x'` to `s+1` bits and recover the sign bit `d_s`.
///   If `c < 0`, `d_s = 0`; if `c ≥ 0`, `d_s = 1`.
///
/// - The output is then `d_s * x`, which zeroes out `x` when negative.
pub struct ReluContext {
    pub shift_exponent: usize, // s
    pub shift: Variable,       // 2^s, lifted into the circuit
}

// ─────────────────────────────────────────────────────────────────────────────
// IMPL: ReluContext
// ─────────────────────────────────────────────────────────────────────────────

impl ReluContext {
    /// Constructs a `ReluContext` with the given `shift_exponent` (s).
    ///
    /// Precomputes `shift = 2^s` as a constant lifted into the circuit.
    ///
    /// # Arguments
    /// - `api`: Mutable reference to the circuit builder.
    /// - `shift_exponent`: The bit exponent `s` defining the signed range.
    ///
    /// # Returns
    /// A `ReluContext` holding the shift constant and exponent.
    ///
    /// # Errors
    /// - [`RescaleError::ShiftExponentTooLargeError`] if `shift_exponent`
    ///   cannot fit in a `u32` or causes a shift overflow.
    ///
    pub fn new<C: Config, Builder: RootAPI<C>>(
        api: &mut Builder,
        shift_exponent: usize,
    ) -> Result<Self, RescaleError> {
        let shift_const = 1u32
            .checked_shl(u32::try_from(shift_exponent).map_err(|_| {
                RescaleError::ShiftExponentTooLargeError {
                    exp: shift_exponent,
                    type_name: "u32",
                }
            })?)
            .ok_or(RescaleError::ShiftExponentTooLargeError {
                exp: shift_exponent,
                type_name: "u32",
            })?;
        let shift = api.constant(shift_const);
        Ok(Self {
            shift_exponent,
            shift,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: relu
// ─────────────────────────────────────────────────────────────────────────────

/// Applies the `ReLU` function to a signed input represented as a field element.
///
/// # Overview
/// The input `x` is assumed to be the least nonnegative residue of a signed integer `c`,
/// where `c` is considered *valid* if it lies in the symmetric range:
///
/// ```text
///     c ∈ [-S, T - S] = [-2^s, 2^s - 1]
/// ```
///
/// To extract the sign of `c` in a nonnegative domain, we shift by `S = 2^s`:
///
/// ```text
///     x' = c + 2^s
/// ```
///
/// If `c` is valid, then `x' ∈ [0, T] = [0, 2^{s + 1} - 1]`.
/// The function then proceeds as follows:
///
/// 1. Extract the `s + 1` least significant bits of `x'`
/// 2. Assert that each bit is boolean (0 or 1)
/// 3. Reconstruct `x'` from its bit decomposition and assert equality
/// 4. Extract the `(s + 1)`-st bit `d_s`, which indicates whether `c ≥ 0`
/// 5. Return `d_s * x`, which equals `ReLU(c)` when `c` is valid
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder.
/// - `x`: A field element representing the least nonnegative residue of a signed integer `c`.
/// - `context`: Precomputed shift constant `S = 2^s` and the exponent `s`.
///
/// # Returns
/// - A `Variable` representing the `ReLU` output: `ReLU(c) = max(0, c)` if `c` is valid.
///
/// # Errors
/// - [`CircuitError`] if bit decomposition or equality constraints fail.
/// - Propagates any error from `unconstrained_to_bits` or
///   `constrained_reconstruct_from_bits`.
///
// TODO can make use of memorized calls instead, by flattening the array and expanding?
pub fn relu<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Variable,
    context: &ReluContext,
) -> Result<Variable, CircuitError> {
    let shifted_x = api.add(context.shift, *x);
    let n_bits = context.shift_exponent + 1;

    let bits = unconstrained_to_bits(api, shifted_x, n_bits)?;
    let reconstructed = constrained_reconstruct_from_bits(api, &bits)?;
    api.assert_is_equal(shifted_x, reconstructed);

    let sign_bit = bits[context.shift_exponent];
    Ok(api.mul(*x, sign_bit))
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: relu_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies `ReLU` elementwise to an `ArrayD<Variable>` of signed values.
///
/// Assumes values are least nonnegative residues of signed integers.
///
/// # Arguments
/// - `api`: Mutable reference to the circuit builder api.
/// - `array`: Multi-dimensional array of input variables.
/// - `shift_exponent`: Bit exponent `s`, defining the signed range `[-2^s, 2^s - 1]`.
///
/// # Returns
/// A new `ArrayD<Variable>` where each element is replaced by its `ReLU` output.
///
/// # Errors
/// - [`CircuitError`] if `ReLU` construction fails for any element.
/// - [`ArrayConversionError::ShapeError`] if the result cannot be reshaped
///   back into the input array’s dimensions.
pub fn relu_array<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    array: &ArrayD<Variable>,
    shift_exponent: usize,
) -> Result<ArrayD<Variable>, CircuitError> {
    let context = ReluContext::new(api, shift_exponent)?;
    let flat_results: Result<Vec<_>, _> = array.iter().map(|x| relu(api, x, &context)).collect();
    let out = ArrayD::from_shape_vec(array.raw_dim(), flat_results?)
        .map_err(ArrayConversionError::ShapeError)?;
    Ok(out)
}
