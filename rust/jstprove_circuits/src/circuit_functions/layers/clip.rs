//! Elementwise Clip layer over int64 fixed-point tensors, implemented as
//! a composition of Max/Min gadgets:
//!
//!   clip(x; a, b) = min(max(x, a), b)
//!
//! with support for optional lower (`a`) and upper (`b`) bounds, matching
//! ONNX Clip semantics.
//!
//! Inputs:
//!   - X (required): data tensor
//!   - min (optional): lower bound (scalar or tensor)
//!   - max (optional): upper bound (scalar or tensor)
//! All inputs are broadcast to a common shape using the same helper as AddLayer.

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
        constants::INPUT,
        graph_pattern_matching::PatternRegistry,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_optional_w_or_b},
        tensor_ops::{broadcast_two_arrays, load_array_constants_or_get_inputs},
    },
};

use crate::circuit_functions::gadgets::{
    LogupRangeCheckContext, ShiftRangeContext, constrained_clip,
};

// -------- Struct --------

#[allow(dead_code)]
#[derive(Debug)]
pub struct ClipLayer {
    name: String,
    optimization_pattern: PatternRegistry,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,

    /// Optional initializers for X, min, max (may be scalars or tensors).
    initializer_x: Option<ArrayD<i64>>,
    initializer_min: Option<ArrayD<i64>>,
    initializer_max: Option<ArrayD<i64>>,

    /// s such that signed range is approximately [-2^s, 2^s - 1]
    shift_exponent: usize,
}

// -------- Implementation --------

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ClipLayer {
    #[allow(clippy::too_many_lines)]
    fn apply(
        &self,
        api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // 1. Resolve required input X
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Clip, INPUT)?;
        let x_input = load_array_constants_or_get_inputs(
            api,
            &input,
            x_name,
            &self.initializer_x,
            LayerKind::Clip,
        )?;

        // 2. Optional min / max inputs.
        //    We only attempt to read them if the ONNX node actually has those inputs.
        let min_input = if self.inputs.len() > 1 {
            let min_name = get_input_name(&self.inputs, 1, LayerKind::Clip, INPUT)?;
            Some(load_array_constants_or_get_inputs(
                api,
                &input,
                min_name,
                &self.initializer_min,
                LayerKind::Clip,
            )?)
        } else {
            None
        };

        let max_input = if self.inputs.len() > 2 {
            let max_name = get_input_name(&self.inputs, 2, LayerKind::Clip, INPUT)?;
            Some(load_array_constants_or_get_inputs(
                api,
                &input,
                max_name,
                &self.initializer_max,
                LayerKind::Clip,
            )?)
        } else {
            None
        };

        // Fast path: if both bounds are missing, Clip is the identity.
        if min_input.is_none() && max_input.is_none() {
            return Ok((self.outputs.clone(), x_input));
        }

        // 3. Broadcast X, min, and max to a common shape using pairwise helpers.
        //
        // We keep X as the "reference" tensor and progressively broadcast min
        // and max to match its shape. If both are present and require further
        // broadcasting, we reconcile them as needed.
        let mut x_bc = x_input;
        let mut min_bc_opt = min_input;
        let mut max_bc_opt = max_input;

        // Broadcast X and min first, if min exists
        if let Some(min_bc) = min_bc_opt.take() {
            let (x_new, min_new) = broadcast_two_arrays(&x_bc, &min_bc)?;
            x_bc = x_new;
            min_bc_opt = Some(min_new);
        }

        // Broadcast X (now possibly expanded) and max, if max exists
        if let Some(max_bc) = max_bc_opt.take() {
            let (x_new, max_new) = broadcast_two_arrays(&x_bc, &max_bc)?;
            x_bc = x_new;
            max_bc_opt = Some(max_new);

            // If min also exists, ensure it matches the final shape by broadcasting.
            if let Some(min_bc) = min_bc_opt.take() {
                if min_bc.shape() == x_bc.shape() {
                    min_bc_opt = Some(min_bc);
                } else {
                    let (_dummy, min_new2) = broadcast_two_arrays(&x_bc, &min_bc)?;
                    min_bc_opt = Some(min_new2);
                }
            }
        }

        // At this point, x_bc has the final shape. Any existing min/max must
        // have shapes compatible with x_bc.
        if let Some(ref min_bc) = min_bc_opt {
            if min_bc.shape() != x_bc.shape() {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Clip,
                    msg: format!(
                        "ClipLayer: min shape {:?} not broadcastable to X shape {:?}",
                        min_bc.shape(),
                        x_bc.shape()
                    ),
                }
                .into());
            }
        }
        if let Some(ref max_bc) = max_bc_opt {
            if max_bc.shape() != x_bc.shape() {
                return Err(LayerError::InvalidShape {
                    layer: LayerKind::Clip,
                    msg: format!(
                        "ClipLayer: max shape {:?} not broadcastable to X shape {:?}",
                        max_bc.shape(),
                        x_bc.shape()
                    ),
                }
                .into());
            }
        }

        let shape = x_bc.shape().to_vec();
        let total_len = x_bc.len();

        // 4. Prepare Max/Min assertion context (same signed-range assumptions as Max/Min/MaxPool).
        let range_ctx =
            ShiftRangeContext::new(api, self.shift_exponent).map_err(|e| LayerError::Other {
                layer: LayerKind::Clip,
                msg: format!("ShiftRangeContext::new failed: {e}"),
            })?;

        // Shared LogUp context for all range checks in this Clip layer
        let mut logup_ctx = LogupRangeCheckContext::new_default();
        logup_ctx.init::<C, Builder>(api);

        // 5. Elementwise clip using constrained_clip
        let mut out_storage = Vec::with_capacity(total_len);

        match (min_bc_opt.as_ref(), max_bc_opt.as_ref()) {
            (None, None) => {
                // This case should have early-returned, but keep for completeness.
                for &x_val in &x_bc {
                    let clipped =
                        constrained_clip(api, &range_ctx, &mut logup_ctx, x_val, None, None)?;
                    out_storage.push(clipped);
                }
            }
            (Some(min_bc), None) => {
                for (x_val, min_val) in x_bc.iter().zip(min_bc.iter()) {
                    let clipped = constrained_clip(
                        api,
                        &range_ctx,
                        &mut logup_ctx,
                        *x_val,
                        Some(*min_val),
                        None,
                    )?;
                    out_storage.push(clipped);
                }
            }
            (None, Some(max_bc)) => {
                for (x_val, max_val) in x_bc.iter().zip(max_bc.iter()) {
                    let clipped = constrained_clip(
                        api,
                        &range_ctx,
                        &mut logup_ctx,
                        *x_val,
                        None,
                        Some(*max_val),
                    )?;
                    out_storage.push(clipped);
                }
            }
            (Some(min_bc), Some(max_bc)) => {
                for ((x_val, min_val), max_val) in x_bc.iter().zip(min_bc.iter()).zip(max_bc.iter())
                {
                    let clipped = constrained_clip(
                        api,
                        &range_ctx,
                        &mut logup_ctx,
                        *x_val,
                        Some(*min_val),
                        Some(*max_val),
                    )?;
                    out_storage.push(clipped);
                }
            }
        }

        // Finalize LogUp lookup constraints for this layer
        logup_ctx.finalize::<C, Builder>(api);

        let result = ArrayD::from_shape_vec(shape.clone(), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Clip,
                msg: format!("ClipLayer: cannot reshape result into shape {shape:?}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        // Same helper used by Add/Max/Min to infer expected shape.
        let (_params, expected_shape) = extract_params_and_expected_shape(layer_context, layer)
            .map_err(|e| LayerError::Other {
                layer: LayerKind::Clip,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            })?;

        // Optional initializers for X, min, max.
        let initializer_x = get_optional_w_or_b(layer_context, &layer.inputs[0])?;
        let initializer_min = if layer.inputs.len() > 1 {
            get_optional_w_or_b(layer_context, &layer.inputs[1])?
        } else {
            None
        };
        let initializer_max = if layer.inputs.len() > 2 {
            get_optional_w_or_b(layer_context, &layer.inputs[2])?
        } else {
            None
        };

        // Match Max/Min/MaxPool: use n_bits - 1 as shift exponent.
        let shift_exponent = layer_context
            .n_bits_for(&layer.name)
            .checked_sub(1)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Clip,
                msg: "n_bits too small to derive shift_exponent".to_string(),
            })?;

        let clip_layer = Self {
            name: layer.name.clone(),
            optimization_pattern,
            input_shape: expected_shape.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            initializer_x,
            initializer_min,
            initializer_max,
            shift_exponent,
        };

        Ok(Box::new(clip_layer))
    }
}
