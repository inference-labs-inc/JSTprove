use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{FunctionLookupTable, function_lookup_bits, range_check::LogupRangeCheckContext},
    hints::gelu::{GELU_HINT_KEY, compute_gelu_quantized},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{extract_params, get_input_name, get_param_or_default},
    },
};

#[derive(Debug)]
pub struct GeluLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GeluLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Gelu, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Gelu,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let lookup_bits = function_lookup_bits(self.scale_exponent);
        let mut lookup = FunctionLookupTable::build_signed::<C, Builder>(
            api,
            compute_gelu_quantized,
            lookup_bits,
            self.scaling,
        );

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let hint_out = api.new_hint(GELU_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];
            lookup.query(x, y);
            out_storage.push(y);
        }

        lookup.finalize::<C, Builder>(api);

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Gelu,
                msg: format!("GeluLayer: cannot reshape result into shape {shape:?}"),
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
                layer: LayerKind::Gelu,
                name: "input X".to_string(),
            })?;

        let params = extract_params(layer).ok();
        let default_approximate = "none".to_string();
        let approximate: String = params
            .as_ref()
            .and_then(|p| {
                get_param_or_default(&layer.name, "approximate", p, Some(&default_approximate)).ok()
            })
            .unwrap_or(default_approximate);

        if approximate != "tanh" {
            return Err(LayerError::Other {
                layer: LayerKind::Gelu,
                msg: format!(
                    "Gelu approximate='{approximate}' is not supported in the Expander backend: \
                     only approximate='tanh' is implemented."
                ),
            }
            .into());
        }

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Gelu,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            scaling,
            scale_exponent: circuit_params.scale_exponent,
        }))
    }
}
