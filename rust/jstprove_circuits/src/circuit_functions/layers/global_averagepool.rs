use std::collections::HashMap;

use ethnum::U256;
use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{CircuitField, Config, FieldArith, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::LogupRangeCheckContext,
    hints::global_averagepool::GLOBAL_AVERAGEPOOL_HINT_KEY,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{constants::INPUT, onnx_model::get_input_name},
};

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize
    }
}

#[derive(Debug)]
pub struct GlobalAveragePoolLayer {
    inputs: Vec<String>,
    outputs: Vec<String>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    n_bits: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for GlobalAveragePoolLayer {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::similar_names,
        clippy::too_many_lines,
        clippy::many_single_char_names
    )]
    fn apply(
        &self,
        api: &mut Builder,
        logup_ctx: &mut LogupRangeCheckContext,
        input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let x_name = get_input_name(&self.inputs, 0, LayerKind::GlobalAveragePool, INPUT)?;
        let x_input = input.get(x_name).ok_or_else(|| LayerError::MissingInput {
            layer: LayerKind::GlobalAveragePool,
            name: x_name.to_string(),
        })?;

        let data_flat = x_input.as_slice().ok_or_else(|| LayerError::InvalidShape {
            layer: LayerKind::GlobalAveragePool,
            msg: "input tensor is not contiguous".to_string(),
        })?;

        if x_input.shape() != self.input_shape.as_slice() {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::GlobalAveragePool,
                msg: format!(
                    "GlobalAveragePool runtime input shape {:?} does not match expected {:?}",
                    x_input.shape(),
                    self.input_shape
                ),
            }
            .into());
        }

        let rank = self.input_shape.len();
        if rank < 3 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::GlobalAveragePool,
                msg: format!(
                    "GlobalAveragePool expects at least 3-D input [N,C,spatial...], got rank {rank}"
                ),
            }
            .into());
        }

        let big_n = self.input_shape[0];
        let c = self.input_shape[1];
        let n_valid: usize = self.input_shape[2..].iter().product();

        let n_bits = self.n_bits;
        let mut out_storage: Vec<Variable> = Vec::with_capacity(big_n * c);

        for n_idx in 0..big_n {
            for c_idx in 0..c {
                let spatial_start = n_idx * c * n_valid + c_idx * n_valid;
                let valid_vars = &data_flat[spatial_start..spatial_start + n_valid];

                if n_valid == 0 {
                    out_storage.push(api.constant(CircuitField::<C>::from_u256(U256::from(0u64))));
                    continue;
                }

                if n_valid == 1 {
                    let y = valid_vars[0];
                    logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;
                    out_storage.push(y);
                    continue;
                }

                for &var in valid_vars {
                    logup_ctx.range_check::<C, Builder>(api, var, n_bits)?;
                }

                let hint_out = api.new_hint(GLOBAL_AVERAGEPOOL_HINT_KEY, valid_vars, 1);
                let y = hint_out[0];

                let sum_var = valid_vars[1..]
                    .iter()
                    .fold(valid_vars[0], |acc, &x| api.add(acc, x));

                let n_const =
                    api.constant(CircuitField::<C>::from_u256(U256::from(n_valid as u64)));
                let n_y = api.mul(n_const, y);
                let r = api.sub(sum_var, n_y);

                let half_n = (n_valid / 2) as u64;
                let half_const = api.constant(CircuitField::<C>::from_u256(U256::from(half_n)));
                let s = api.add(r, half_const);

                let k = ceil_log2(n_valid);
                logup_ctx.range_check::<C, Builder>(api, s, k)?;

                let nm1_const = api.constant(CircuitField::<C>::from_u256(U256::from(
                    (n_valid - 1) as u64,
                )));
                let t = api.sub(nm1_const, s);
                logup_ctx.range_check::<C, Builder>(api, t, k)?;

                logup_ctx.range_check::<C, Builder>(api, y, n_bits)?;

                out_storage.push(y);
            }
        }

        let result =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_storage).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::GlobalAveragePool,
                    msg: format!("GlobalAveragePool output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), result))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::GlobalAveragePool,
                param: "input tensor".to_string(),
            })?;
        let output_name = layer
            .outputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::GlobalAveragePool,
                param: "output tensor".to_string(),
            })?;

        let input_shape = layer_context
            .shapes_map
            .get(input_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GlobalAveragePool,
                msg: format!("missing input shape for '{input_name}'"),
            })?
            .clone();

        if input_shape.len() < 3 {
            return Err(LayerError::InvalidShape {
                layer: LayerKind::GlobalAveragePool,
                msg: format!(
                    "GlobalAveragePool requires at least 3-D input [N,C,spatial...], got rank {}",
                    input_shape.len()
                ),
            }
            .into());
        }

        let output_shape = layer_context
            .shapes_map
            .get(output_name.as_str())
            .ok_or_else(|| LayerError::InvalidShape {
                layer: LayerKind::GlobalAveragePool,
                msg: format!("missing output shape for '{output_name}'"),
            })?
            .clone();

        let n_bits = layer_context.n_bits_for(&layer.name);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            input_shape,
            output_shape,
            n_bits,
        }))
    }
}
