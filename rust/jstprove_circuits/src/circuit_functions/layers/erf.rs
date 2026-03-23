// Elementwise ONNX `Erf` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: computes erf(x_q / scale) * scale in native f64 via A&S 7.1.26.
// 2. **No range check**: erf can be negative.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::erf::ERF_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct ErfLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ErfLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Erf, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Erf,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input.iter() {
            let hint_out = api.new_hint(ERF_HINT_KEY, &[x, scale_var], 1);
            out_storage.push(hint_out[0]);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Erf,
                msg: format!("ErfLayer: cannot reshape result into shape {shape:?}"),
            }
        })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        _layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Erf,
                name: "input X".to_string(),
            })?;

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Erf,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            scaling,
        }))
    }
}
