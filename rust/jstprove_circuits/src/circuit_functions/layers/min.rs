//! Elementwise Min layer over int64 fixed-point tensors, using the
//! max/min-selection gadget to assert that outputs are true minima.

use std::collections::HashMap;

/// External crate imports
use ndarray::ArrayD;

/// `ExpanderCompilerCollection` imports
use expander_compiler::frontend::{Config, RootAPI, Variable};

/// Internal helpers shared with other layers
use crate::circuit_functions::utils::core_math::{
    LogupRangeCheckContext, ShiftRangeContext, constrained_min,
};
use crate::circuit_functions::utils::onnx_model::get_optional_w_or_b;
use crate::circuit_functions::utils::tensor_ops::{
    broadcast_two_arrays, load_array_constants_or_get_inputs,
};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        graph_pattern_matching::PatternRegistry,
        onnx_model::{extract_params_and_expected_shape, get_input_name},
    },
};

// -------- Struct --------

#[allow(dead_code)]
#[derive(Debug)]
pub struct MinLayer {
    name: String,
    optimization_pattern: PatternRegistry,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    initializer_a: Option<ArrayD<i64>>,
    initializer_b: Option<ArrayD<i64>>,
    /// s such that the signed fixed-point range is roughly [-2^s, 2^s - 1]
    shift_exponent: usize,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for MinLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // 1. Resolve input names (mirrors AddLayer)
        let a_name = get_input_name(&self.inputs, 0, LayerKind::Min, INPUT)?;
        let b_name = get_input_name(&self.inputs, 1, LayerKind::Min, INPUT)?;

        // 2. Load either constants (initializers) or runtime inputs
        let a_input = load_array_constants_or_get_inputs(
            api,
            &input,
            a_name,
            &self.initializer_a,
            LayerKind::Min,
        )?;

        let b_input = load_array_constants_or_get_inputs(
            api,
            &input,
            b_name,
            &self.initializer_b,
            LayerKind::Min,
        )?;

        // 3. Broadcast inputs to a common shape (same helper as AddLayer)
        let (a_bc, b_bc) = broadcast_two_arrays(&a_input, &b_input)?;

         // 4. Prepare shift context
        let shift_ctx =
            ShiftRangeContext::new(api, self.shift_exponent).map_err(|e| LayerError::Other {
                layer: LayerKind::Min,
                msg: format!("MinLayer: ShiftRangeContext::new failed: {e}"),
            })?;

        // Shared LogUp range-check context for this Min layer
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        // 5. Elementwise min: for each position, z = min(a, b)
        //
        // We use `constrained_min` with a 2-element slice [a, b], which:
        //   - shifts both by 2^s,
        //   - uses `unconstrained_min` to pick the min of shifted values,
        //   - shifts back and asserts correctness via LogUp-based range checks and a product = 0 constraint.
        let shape = a_bc.shape().to_vec();
        if a_bc.len() != b_bc.len() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::Min,
                msg: format!(
                    "broadcast_two_arrays returned arrays of different sizes: {:?} vs {:?}",
                    a_bc.shape(),
                    b_bc.shape()
                ),
            }
            .into());
        }

        let mut out_storage = Vec::with_capacity(a_bc.len());

        for (a_val, b_val) in a_bc.iter().zip(b_bc.iter()) {
            // Constrained pairwise min using existing gadget + shared LogUp context
            let min_var = constrained_min(api, &shift_ctx, &mut logup_ctx, &[*a_val, *b_val])?;
            out_storage.push(min_var);
        }

        // Finalize LogUp constraints for this layer
        logup_ctx.finalize::<C, Builder>(api);

        let result = ArrayD::from_shape_vec(shape.clone(), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Min,
                msg: format!("MinLayer: cannot reshape result into shape {shape:?}"),
            }
        })?;


    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // The same helper used by AddLayer to infer expected shapes
        let (_params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Min,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;

        // Optional initializers for A and B (same pattern as AddLayer)
        let initializer_a = get_optional_w_or_b(layer_context, &layer.inputs[0])?;
        let initializer_b = get_optional_w_or_b(layer_context, &layer.inputs[1])?;

        // Match MaxPool’s choice: use n_bits - 1 as the shift exponent
        let shift_exponent =
            layer_context
                .n_bits
                .checked_sub(1)
                .ok_or_else(|| LayerError::Other {
                    layer: LayerKind::Min,
                    msg: "layer_context.n_bits too small to derive shift_exponent".to_string(),
                })?;

        let min_layer = Self {
            name: layer.name.clone(),
            optimization_pattern,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_a,
            initializer_b,
            shift_exponent,
        };

        Ok(Box::new(min_layer))
    }
}
