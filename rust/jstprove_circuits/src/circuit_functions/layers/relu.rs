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
        ArrayConversionError,
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

use crate::circuit_functions::gadgets::LogupRangeCheckContext;
use crate::circuit_functions::utils::core_math::{ShiftRangeContext, constrained_max};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReluLayer {
    name: String,
    index: usize,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// Total bit-width of the signed fixed-point representation.
    /// We assume signed range [-2^(n_bits-1), 2^(n_bits-1) - 1].
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
            .get(&input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::ReLU,
                name: input_name.clone(),
            })?
            .clone();

        // Values live in a symmetric signed range with s = n_bits - 1.
        let shift_exponent = self
            .n_bits
            .checked_sub(1)
            .ok_or_else(|| CircuitError::Other("ReLU: n_bits must be at least 1".to_string()))?;

        let out = relu_array::<C, Builder>(api, &layer_input, shift_exponent)?;

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
// FUNCTION: relu_array
// ─────────────────────────────────────────────────────────────────────────────

/// Applies `ReLU` elementwise to an `ArrayD<Variable>` of signed values.
///
/// Semantics are the same as in `rescale`: we implement
///
/// ```text
/// ReLU(x) = max(x, 0)
/// ```
///
/// using the `constrained_max` gadget plus a shared `LogupRangeCheckContext`.
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
/// - [`CircuitError`] if the internal gadgets fail.
/// - [`ArrayConversionError::ShapeError`] if the result cannot be reshaped
///   back into the input array’s dimensions.
pub fn relu_array<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    array: &ArrayD<Variable>,
    shift_exponent: usize,
) -> Result<ArrayD<Variable>, CircuitError> {
    // Shared shift context S = 2^s.
    let shift_ctx = ShiftRangeContext::new::<C, Builder>(api, shift_exponent)?;

    // Shared LogUp range-check context for all elements in this layer.
    let mut logup_ctx = LogupRangeCheckContext::new_default();
    logup_ctx.init::<C, Builder>(api);

    let zero = api.constant(0);

    let flat_results: Result<Vec<_>, CircuitError> = array
        .iter()
        .map(|x| {
            // ReLU(x) = max(x, 0)
            constrained_max::<C, Builder>(api, &shift_ctx, &mut logup_ctx, &[*x, zero])
        })
        .collect();

    // Single final LogUp consistency check for the whole layer.
    logup_ctx.finalize::<C, Builder>(api);

    let out = ArrayD::from_shape_vec(array.raw_dim(), flat_results?)
        .map_err(ArrayConversionError::ShapeError)?;
    Ok(out)
}
