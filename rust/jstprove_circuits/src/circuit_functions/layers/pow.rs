// Elementwise ONNX `Pow` layer for int64 fixed-point tensors.
//
// # ZK approach
// 1. **Hint**: computes pow(x_q/scale, exp_q/scale) * scale in native f64.
// 2. **No range check**: pow can be negative.

use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::pow::POW_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::INPUT,
        onnx_model::{get_input_name, get_w_or_b},
    },
};

#[derive(Debug)]
pub struct PowLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    scaling: u64,
    /// Pre-loaded constant exponent (if available at build time).
    const_exponent: Option<ArrayD<i64>>,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for PowLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Pow, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Pow,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));
        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        // Get exponent as variables.
        let exp_vars: Vec<Variable> = if let Some(ref const_exp) = self.const_exponent {
            // Broadcast scalar or match shape.
            let exp_flat: Vec<i64> = const_exp.as_slice().unwrap_or(&[]).to_vec();
            if exp_flat.len() == 1 {
                let exp_val = exp_flat[0];
                let exp_var = if exp_val >= 0 {
                    api.constant(CircuitField::<C>::from_u256(U256::from(exp_val as u64)))
                } else {
                    let mag = U256::from(exp_val.unsigned_abs());
                    api.constant(CircuitField::<C>::from_u256(
                        CircuitField::<C>::MODULUS - mag,
                    ))
                };
                vec![exp_var; x_input.len()]
            } else {
                exp_flat
                    .iter()
                    .map(|&v| {
                        if v >= 0 {
                            api.constant(CircuitField::<C>::from_u256(U256::from(v as u64)))
                        } else {
                            let mag = U256::from(v.unsigned_abs());
                            api.constant(CircuitField::<C>::from_u256(
                                CircuitField::<C>::MODULUS - mag,
                            ))
                        }
                    })
                    .collect()
            }
        } else {
            let exp_name = get_input_name(&self.inputs, 1, LayerKind::Pow, "exponent")?;
            let exp_input = input
                .get(exp_name)
                .ok_or_else(|| LayerError::MissingInput {
                    layer: LayerKind::Pow,
                    name: exp_name.to_string(),
                })?;
            if exp_input.len() == 1 {
                vec![*exp_input.first().unwrap(); x_input.len()]
            } else {
                exp_input.iter().copied().collect()
            }
        };

        for (i, &x) in x_input.iter().enumerate() {
            let exp_var = exp_vars[i.min(exp_vars.len() - 1)];
            let hint_out = api.new_hint(POW_HINT_KEY, &[x, exp_var, scale_var], 1);
            out_storage.push(hint_out[0]);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Pow,
                msg: format!("PowLayer: cannot reshape result into shape {shape:?}"),
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
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Pow,
                name: "input X".to_string(),
            })?;

        // Try to load exponent as a compile-time constant.
        let const_exponent = if let Some(exp_name) = layer.inputs.get(1) {
            get_w_or_b(layer_context.w_and_b_map, exp_name).ok()
        } else {
            None
        };

        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Pow,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            scaling,
            const_exponent,
        }))
    }
}
