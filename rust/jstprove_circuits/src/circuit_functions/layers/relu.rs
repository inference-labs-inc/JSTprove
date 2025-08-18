use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// ExpanderCompilerCollection imports
use expander_compiler::frontend::*;

/// Internal crate imports
use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::{core_math::{
    assert_is_bitstring_and_reconstruct,
    unconstrained_to_bits,
}, onnx_model::extract_params_and_expected_shape}};

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
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        eprintln!("{:?}", self);
        let layer_input = input.get(&self.inputs[0])
        .ok_or_else(|| panic!("Missing input {}", self.inputs[0].clone())).unwrap()
    .clone();
        let out = layer_input;
        let out = relu_array(api, out, self.n_bits - 1);

        Ok((self.outputs.clone(), out))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::GraphPattern,
        _is_rescale: bool,
        index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext
    ) -> Result<Box<dyn LayerOp<C, Builder>>, Error> {

        let (_params, expected_shape) = extract_params_and_expected_shape(layer_context, layer);
        let relu = Self{
            name: layer.name.clone(),
            index: index,
            input_shape: expected_shape.to_vec(),
            inputs: layer.inputs.to_vec(),
            outputs: layer.outputs.to_vec(),
            n_bits: layer_context.n_bits,
        };
        Ok(Box::new(relu))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT: ReluContext
// ─────────────────────────────────────────────────────────────────────────────

/// Context for asserting a ReLU operation on signed integers represented in least residue form.
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
    /// # Panics
    /// Panics if `shift_exponent ≥ 32` due to `u32` shift overflow.
    pub fn new<C: Config, Builder: RootAPI<C>>(api: &mut Builder, shift_exponent: usize) -> Self {
        let shift_const = 1u32
            .checked_shl(shift_exponent as u32)
            .expect("shift_exponent must be < 32");
        let shift = api.constant(shift_const);
        Self {
            shift_exponent,
            shift,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: relu
// ─────────────────────────────────────────────────────────────────────────────

/// Applies the ReLU function to a signed input represented as a field element.
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
/// - A `Variable` representing the ReLU output: `ReLU(c) = max(0, c)` if `c` is valid.
// TODO can make use of memorized calls instead, by flattening the array and expanding?
pub fn relu<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Variable,
    context: &ReluContext,
) -> Variable {
    let shifted_x = api.add(context.shift, *x);
    let n_bits = context.shift_exponent + 1;

    let bits = unconstrained_to_bits(api, shifted_x, n_bits);
    let reconstructed = assert_is_bitstring_and_reconstruct(api, &bits);
    api.assert_is_equal(shifted_x, reconstructed);

    let sign_bit = bits[context.shift_exponent];
    api.mul(*x, sign_bit)
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION: relu_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies ReLU elementwise to an `ArrayD<Variable>` of signed values.
///
/// Assumes values are least nonnegative residues of signed integers.
pub fn relu_array<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    array: ArrayD<Variable>,
    shift_exponent: usize,
) -> ArrayD<Variable> {
    let context = ReluContext::new(api, shift_exponent);
    array.map(|x| relu(api, x, &context))
}
