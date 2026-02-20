use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct FlattenLayer {
    pub axis: usize,
}

impl<F: Field> LayerOp<F> for FlattenLayer {
    fn build_graph(
        &self,
        _builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let shape = &input.shape;
        let front: usize = shape[..self.axis].iter().product();
        let back: usize = shape[self.axis..].iter().product();
        Ok(input.reshape(vec![front, back]))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        Ok(WitnessData::new().with_output("output", input.clone()))
    }
}
