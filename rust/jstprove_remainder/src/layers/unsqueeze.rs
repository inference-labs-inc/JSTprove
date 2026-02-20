use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct UnsqueezeLayer {
    pub axes: Vec<usize>,
}

impl<F: Field> LayerOp<F> for UnsqueezeLayer {
    fn build_graph(
        &self,
        _builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let mut new_shape = input.shape.clone();
        let mut sorted_axes = self.axes.clone();
        sorted_axes.sort();
        for &ax in &sorted_axes {
            new_shape.insert(ax, 1);
        }
        Ok(input.reshape(new_shape))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        Ok(WitnessData::new().with_output("output", input.clone()))
    }
}
