use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use ndarray::ArrayD;

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params_and_expected_shape, get_input_name, get_param},
    },
};

const ONNX_BOOL: i64 = 9;

const SUPPORTED_CAST_TYPES: &[i64] = &[
    1,  // FLOAT
    2,  // UINT8
    3,  // INT8
    4,  // UINT16
    5,  // INT16
    6,  // INT32
    7,  // INT64
    9,  // BOOL
    10, // FLOAT16
    11, // DOUBLE
    12, // UINT32
    13, // UINT64
    16, // BFLOAT16
];

fn is_supported_cast(to_type: i64) -> bool {
    SUPPORTED_CAST_TYPES.contains(&to_type)
}

#[derive(Debug)]
pub struct CastLayer {
    name: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    to_type: i64,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for CastLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        if !is_supported_cast(self.to_type) {
            return Err(LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!(
                    "{}: Cast to ONNX DataType {} is not supported in the ZK circuit; \
                    supported targets: {:?}",
                    self.name, self.to_type, SUPPORTED_CAST_TYPES
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

        if self.to_type == ONNX_BOOL {
            let out_flat: Vec<Variable> = layer_input.iter().map(|&val| api.is_zero(val)).collect();
            let one = api.constant(1u32);
            let boolified: Vec<Variable> = out_flat.iter().map(|&iz| api.sub(one, iz)).collect();
            let out_array =
                ArrayD::from_shape_vec(layer_input.raw_dim(), boolified).map_err(|e| {
                    LayerError::InvalidShape {
                        layer: LayerKind::Cast,
                        msg: format!("Cast-to-bool output reshape failed: {e}"),
                    }
                })?;
            return Ok((self.outputs.clone(), out_array));
        }

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

        if !is_supported_cast(to_type) {
            return Err(LayerError::Other {
                layer: LayerKind::Cast,
                msg: format!(
                    "{}: Cast to ONNX DataType {to_type} is not supported in the ZK circuit; \
                    supported targets: {:?}",
                    layer.name, SUPPORTED_CAST_TYPES
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
