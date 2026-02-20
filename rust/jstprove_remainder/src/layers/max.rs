use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::max_min_clip;
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct MaxLayer;

impl<F: Field> LayerOp<F> for MaxLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?;
        let max_hint = inputs.get("max_hint").ok_or_else(|| anyhow::anyhow!("missing max_hint"))?;
        let delta_a = inputs.get("delta_a").ok_or_else(|| anyhow::anyhow!("missing delta_a"))?;
        let delta_b = inputs.get("delta_b").ok_or_else(|| anyhow::anyhow!("missing delta_b"))?;

        Ok(max_min_clip::constrained_max(
            builder,
            &[a.clone(), b.clone()],
            max_hint,
            &[delta_a.clone(), delta_b.clone()],
        ))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?;
        let output: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect();
        let delta_a: Vec<i64> = output.iter().zip(a.iter()).map(|(o, x)| o - x).collect();
        let delta_b: Vec<i64> = output.iter().zip(b.iter()).map(|(o, y)| o - y).collect();

        Ok(WitnessData::new()
            .with_output("output", output.clone())
            .with_hint("max_hint", output)
            .with_hint("delta_a", delta_a)
            .with_hint("delta_b", delta_b))
    }
}
