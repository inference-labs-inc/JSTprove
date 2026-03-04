use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param},
    },
};

/// ONNX TensorProto.DataType codes that are safe no-ops in the ZK circuit.
///
/// Every value in this pipeline is quantised to INT64 field elements before the
/// circuit is built; field elements carry no type information, so a Cast to any
/// numeric type is an identity in the circuit domain and requires no constraint.
///
/// FLOAT and DOUBLE are included because models commonly insert a Cast to
/// promote float32 inputs to float64 precision before transcendental ops
/// (Exp, Softmax, Sigmoid); after quantisation that cast is a no-op.
///
/// BOOL (9) and STRING (8) are excluded: a Cast that produces a boolean or
/// string almost certainly means the model was not quantised correctly.
const NOOP_CAST_TYPES: &[i64] = &[
    1,  // FLOAT
    2,  // UINT8
    3,  // INT8
    5,  // INT16
    6,  // INT32
    7,  // INT64
    11, // DOUBLE
    12, // UINT32
    13, // UINT64
];

fn is_noop_cast(to_type: i64) -> bool {
    NOOP_CAST_TYPES.contains(&to_type)
}

#[derive(Debug)]
pub struct CastLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    to_type: i64, // validated ONNX TensorProto.DataType code
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for CastLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        // Guard: reject any to_type that is not a proven no-op integer target.
        // build() already validates this, but apply() enforces it at the point
        // where the passthrough result is actually produced so unexpected types
        // can never silently become identity casts.
        if !is_noop_cast(self.to_type) {
            return Err(LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!(
                    "{}: Cast to ONNX DataType {} is not supported in the ZK circuit; \
                    only integer targets {:?} are proven no-ops",
                    self.name, self.to_type, NOOP_CAST_TYPES
                ),
            }
            .into());
        }

        let input_name = get_input_name(&self.inputs, 0, LayerKind::Cast, INPUT)?;
        let layer_input = input
            .get(input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Cast,
                name: input_name.clone(),
            })?
            .clone();
        // Cast is a no-op in the ZK circuit: the quantizer has already resolved
        // types; field elements are typeless.
        Ok((self.outputs.clone(), layer_input))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let to_type: i64 =
            get_param(&layer.name, "to", &params).map_err(|e| LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!("failed to read 'to' attribute: {e}"),
            })?;

        // Reject unsupported Cast targets at build time so they never reach apply().
        if !is_noop_cast(to_type) {
            return Err(LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!(
                    "{}: Cast to ONNX DataType {to_type} is not supported in the ZK circuit; \
                    only integer targets {:?} are proven no-ops",
                    layer.name, NOOP_CAST_TYPES
                ),
            }
            .into());
        }

        Ok(Box::new(Self {
            name: layer.name.clone(),
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            to_type,
        }))
    }
}
