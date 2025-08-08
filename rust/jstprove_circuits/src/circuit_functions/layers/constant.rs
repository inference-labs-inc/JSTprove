use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};
use serde_json::Value;
use expander_compiler::frontend::*;

use crate::circuit_functions::layers::layer_ops::LayerOp;

// -------- Struct --------
#[allow(dead_code)]
#[derive(Debug)]
pub struct ConstantLayer {
    name: String,
    value: Value,
    outputs: Vec<String>,
}

// -------- Implementations --------
impl ConstantLayer{
    pub fn new(
        name: String,
        value: Value,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name,
            value: value,
            outputs: outputs,
        }
    }
}

// TODO remove constants from python side. Incorporate into the layer that uses it instead
impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for ConstantLayer {
    // Passthrough
    fn apply(
        &self,
        api: &mut Builder,
        _input: HashMap<String,ArrayD<Variable>>,
    ) -> Result<(Vec<String>,ArrayD<Variable>), String> {

        Ok((self.outputs.clone(), ArrayD::from_shape_vec(IxDyn(&[1]), vec![api.constant(0)]).unwrap()))
    }
}
