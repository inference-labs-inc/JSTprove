use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    gadgets::{LogupRangeCheckContext, i64_to_field},
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::onnx_model::get_w_or_b,
};

pub struct ConstantOfShapeLayer {
    #[allow(dead_code)]
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_shape: Vec<usize>,
    fill_value: i64,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantOfShapeLayer {
    fn apply(
        &self,
        api: &mut Builder,
        _logup_ctx: &mut LogupRangeCheckContext,
        _input: &HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let total: usize = self.output_shape.iter().product();
        let val = api.constant(i64_to_field::<C>(self.fill_value));
        let out_flat: Vec<Variable> = vec![val; total];

        let out_array =
            ArrayD::from_shape_vec(IxDyn(&self.output_shape), out_flat).map_err(|e| {
                LayerError::InvalidShape {
                    layer: LayerKind::ConstantOfShape,
                    msg: format!("ConstantOfShape output reshape failed: {e}"),
                }
            })?;

        Ok((self.outputs.clone(), out_array))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        _circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        _is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let shape_name = layer
            .inputs
            .first()
            .ok_or_else(|| LayerError::MissingParameter {
                layer: LayerKind::ConstantOfShape,
                param: "shape input".to_string(),
            })?;

        let shape_data: ArrayD<i64> =
            get_w_or_b(layer_context.w_and_b_map, shape_name).map_err(|e| LayerError::Other {
                layer: LayerKind::ConstantOfShape,
                msg: format!("ConstantOfShape shape input '{shape_name}' must be a compile-time constant: {e}"),
            })?;

        let output_shape: Vec<usize> = shape_data.iter().map(|&v| v as usize).collect();

        let fill_value: i64 = layer
            .params
            .as_ref()
            .and_then(|p| {
                if let rmpv::Value::Map(m) = p {
                    m.iter().find_map(|(k, v)| {
                        if k == &rmpv::Value::String("value".into()) {
                            match v {
                                rmpv::Value::Integer(i) => i.as_i64(),
                                rmpv::Value::F64(f) => Some(*f as i64),
                                rmpv::Value::F32(f) => Some(*f as i64),
                                rmpv::Value::Array(arr) => arr.first().and_then(|el| match el {
                                    rmpv::Value::Integer(i) => i.as_i64(),
                                    rmpv::Value::F64(f) => Some(*f as i64),
                                    rmpv::Value::F32(f) => Some(*f as i64),
                                    _ => None,
                                }),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
            .unwrap_or(0);

        Ok(Box::new(Self {
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
            output_shape,
            fill_value,
        }))
    }
}
