use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct ReshapeLayer {
    pub target_shape: Vec<usize>,
}

impl<F: Field> LayerOp<F> for ReshapeLayer {
    fn build_graph(
        &self,
        _builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        Ok(input.reshape(self.target_shape.clone()))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        Ok(WitnessData::new().with_output("output", input.clone()))
    }
}
