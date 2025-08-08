use std::collections::HashMap;

use ndarray::ArrayD;
use expander_compiler::frontend::*;

use crate::circuit_functions::{layers::layer_ops::LayerOp, utils::shaping::onnx_flatten};

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct FlattenLayer {
    name: String,
    axis: usize,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

// -------- Implementations --------
impl FlattenLayer{
    pub fn new(
        name: String,
        axis: usize,
        input_shape: Vec<usize>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name,
            axis: axis,
            input_shape: input_shape,
            inputs: inputs,
            outputs: outputs,
        }
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for FlattenLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let reshape_axis = self.axis.clone();
        let layer_input = input.get(&self.inputs[0]).unwrap();

        let out = onnx_flatten(layer_input.clone(), reshape_axis);

        Ok((self.outputs.clone(), out.clone()))
    }
}