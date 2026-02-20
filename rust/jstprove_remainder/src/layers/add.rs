use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::linear_algebra;
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct AddLayer;

impl<F: Field> LayerOp<F> for AddLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing input A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing input B"))?;
        Ok(linear_algebra::elementwise_add(builder, a, b))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing input A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing input B"))?;
        let output: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        Ok(WitnessData::new().with_output("output", output))
    }
}
