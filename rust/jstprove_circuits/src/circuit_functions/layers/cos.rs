use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{FunctionLookupTable, i64_to_field, range_check::LogupRangeCheckContext},
    hints::cos::{COS_HINT_KEY, compute_cos_quantized},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

#[derive(Debug)]
pub struct CosLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    n_bits: usize,
    scaling: u64,
    scale_exponent: u32,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for CosLayer {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::Cos, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::Cos,
            name: x_name.to_string(),
        })?;

        let shape = x_input.shape().to_vec();
        let scale_var = api.constant(CircuitField::<C>::from_u256(U256::from(self.scaling)));

        let table_bits = ((self.scale_exponent as usize) + 3).min(20);

        let mut table = FunctionLookupTable::build_signed::<C, Builder>(
            api,
            compute_cos_quantized,
            table_bits,
            self.scaling,
        );

        let half = 1i64 << (table_bits - 1);
        let lower_var = api.constant(i64_to_field::<C>(-half));
        let upper_var = api.constant(i64_to_field::<C>(half - 1));

        let shift_exponent = self.n_bits.saturating_sub(1);
        let range_ctx = crate::circuit_functions::gadgets::ShiftRangeContext::new::<C, Builder>(
            api,
            LayerKind::Cos,
            shift_exponent,
        )?;

        let mut out_storage: Vec<Variable> = Vec::with_capacity(x_input.len());

        for &x in x_input {
            let clamped = crate::circuit_functions::gadgets::constrained_clip::<C, Builder>(
                api,
                &range_ctx,
                logup_ctx,
                x,
                Some(lower_var),
                Some(upper_var),
            )?;

            let hint_out = api.new_hint(COS_HINT_KEY, &[x, scale_var], 1);
            let y = hint_out[0];

            table.query(clamped, y);

            let y_plus_scale = api.add(y, scale_var);
            logup_ctx.range_check::<C, Builder>(api, y_plus_scale, self.n_bits + 1)?;
            let scale_minus_y = api.sub(scale_var, y);
            logup_ctx.range_check::<C, Builder>(api, scale_minus_y, self.n_bits + 1)?;

            out_storage.push(y);
        }

        if !x_input.is_empty() {
            table.finalize::<C, Builder>(api);
        }

        let result = ArrayD::from_shape_vec(IxDyn(&shape), out_storage).map_err(|_| {
            LayerError::InvalidShape {
                layer: LayerKind::Cos,
                msg: format!("CosLayer: cannot reshape result into shape {shape:?}"),
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
                layer: LayerKind::Cos,
                name: "input X".to_string(),
            })?;

        if circuit_params.scale_exponent >= 32 {
            return Err(LayerError::Other {
                layer: LayerKind::Cos,
                msg: format!(
                    "scale_exponent {} >= 32: overflows u64 in Cos lookup table",
                    circuit_params.scale_exponent
                ),
            }
            .into());
        }

        let n_bits = layer_context.n_bits_for(&layer.name);
        let scaling: u64 = 1u64
            .checked_shl(circuit_params.scale_exponent)
            .ok_or_else(|| LayerError::Other {
                layer: LayerKind::Cos,
                msg: format!(
                    "scale_exponent {} is too large to shift u64",
                    circuit_params.scale_exponent
                ),
            })?;

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            n_bits,
            scaling,
            scale_exponent: circuit_params.scale_exponent,
        }))
    }
}
