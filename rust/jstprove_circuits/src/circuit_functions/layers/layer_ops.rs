use std::collections::HashMap;

use expander_compiler::frontend::*;
use ndarray::ArrayD;
pub trait LayerOp<C: Config, Builder: RootAPI<C>> {
    fn apply(&self, api: &mut Builder, input: HashMap<String,ArrayD<Variable>>)
        -> Result<(Vec<String>,ArrayD<Variable>), String>;
}