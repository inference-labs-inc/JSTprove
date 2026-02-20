use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::max_min_clip;
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct MinLayer;

impl<F: Field> LayerOp<F> for MinLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?;
        let min_hint = inputs.get("min_hint").ok_or_else(|| anyhow::anyhow!("missing min_hint"))?;
        let delta_a = inputs.get("delta_a").ok_or_else(|| anyhow::anyhow!("missing delta_a"))?;
        let delta_b = inputs.get("delta_b").ok_or_else(|| anyhow::anyhow!("missing delta_b"))?;

        Ok(max_min_clip::constrained_min(
            builder,
            &[a.clone(), b.clone()],
            min_hint,
            &[delta_a.clone(), delta_b.clone()],
        ))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?;
        let output: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x.min(y)).collect();
        let delta_a: Vec<i64> = a.iter().zip(output.iter()).map(|(x, o)| x - o).collect();
        let delta_b: Vec<i64> = b.iter().zip(output.iter()).map(|(y, o)| y - o).collect();

        Ok(WitnessData::new()
            .with_output("output", output.clone())
            .with_hint("min_hint", output)
            .with_hint("delta_a", delta_a)
            .with_hint("delta_b", delta_b))
    }
}
