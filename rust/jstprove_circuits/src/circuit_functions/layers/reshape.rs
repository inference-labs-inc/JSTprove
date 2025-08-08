use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use expander_compiler::frontend::*;

use crate::circuit_functions::layers::layer_ops::LayerOp;


// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ReshapeLayer {
    name: String,
    shape: Vec<usize>,
    input_shape: Vec<usize>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}
// -------- Implementations --------
impl ReshapeLayer{
    pub fn new(
        name: String,
        input_shape: Vec<usize>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        shape: Vec<usize> 
    ) -> Self {
        Self {
            name: name,
            input_shape: input_shape,
            inputs: inputs,
            outputs: outputs,
            shape: shape
        }
    }
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ReshapeLayer {
    fn apply(
        &self,
        _api: &mut Builder,
        input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {
        let reshape_shape = self.shape.clone();
        let layer_input = input.get(&self.inputs[0]).unwrap();
        let out = &layer_input.clone()
            .into_shape_with_order(IxDyn(&reshape_shape))
            .expect("Shape mismatch: Cannot reshape into the given dimensions.");

        Ok((self.outputs.clone(), out.clone()))
    }
}