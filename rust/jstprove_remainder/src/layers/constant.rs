use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct ConstantLayer {
    pub values: Vec<i64>,
    pub shape: Vec<usize>,
}

impl<F: Field> LayerOp<F> for ConstantLayer {
    fn build_graph(
        &self,
        _builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        inputs
            .get("constant")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing constant input shred"))
    }

    fn compute_witness(
        &self,
        _inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        Ok(WitnessData::new().with_output("output", self.values.clone()))
    }
}
