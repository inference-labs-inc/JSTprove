use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::max_min_clip;
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct ReluLayer;

impl<F: Field> LayerOp<F> for ReluLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let zero = inputs.get("zero").ok_or_else(|| anyhow::anyhow!("missing zero"))?;
        let max_hint = inputs.get("max_hint").ok_or_else(|| anyhow::anyhow!("missing max_hint"))?;
        let delta_input = inputs.get("delta_input").ok_or_else(|| anyhow::anyhow!("missing delta_input"))?;
        let delta_zero = inputs.get("delta_zero").ok_or_else(|| anyhow::anyhow!("missing delta_zero"))?;

        Ok(max_min_clip::constrained_max(
            builder,
            &[input.clone(), zero.clone()],
            max_hint,
            &[delta_input.clone(), delta_zero.clone()],
        ))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let output: Vec<i64> = input.iter().map(|&x| x.max(0)).collect();
        let delta_input: Vec<i64> = input.iter().zip(output.iter()).map(|(x, o)| o - x).collect();
        let delta_zero: Vec<i64> = output.iter().map(|&o| o).collect();

        Ok(WitnessData::new()
            .with_output("output", output.clone())
            .with_hint("max_hint", output)
            .with_hint("delta_input", delta_input)
            .with_hint("delta_zero", delta_zero))
    }
}
