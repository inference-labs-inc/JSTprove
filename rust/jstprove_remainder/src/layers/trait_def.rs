use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::tensor::ShapedMLE;

pub struct WitnessData {
    pub outputs: HashMap<String, Vec<i64>>,
    pub hints: HashMap<String, Vec<i64>>,
}

impl WitnessData {
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
            hints: HashMap::new(),
        }
    }

    pub fn with_output(mut self, name: &str, data: Vec<i64>) -> Self {
        self.outputs.insert(name.to_string(), data);
        self
    }

    pub fn with_hint(mut self, name: &str, data: Vec<i64>) -> Self {
        self.hints.insert(name.to_string(), data);
        self
    }
}

pub trait LayerOp<F: Field> {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>>;

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData>;
}
