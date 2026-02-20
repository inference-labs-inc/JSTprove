use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::max_min_clip;
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct ClipLayer {
    pub lo: i64,
    pub hi: i64,
}

impl<F: Field> LayerOp<F> for ClipLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let lo = inputs.get("lo").ok_or_else(|| anyhow::anyhow!("missing lo"))?;
        let hi = inputs.get("hi").ok_or_else(|| anyhow::anyhow!("missing hi"))?;
        let max_hint = inputs.get("max_hint").ok_or_else(|| anyhow::anyhow!("missing max_hint"))?;
        let min_hint = inputs.get("min_hint").ok_or_else(|| anyhow::anyhow!("missing min_hint"))?;
        let max_deltas_input = inputs.get("max_delta_input").ok_or_else(|| anyhow::anyhow!("missing max_delta_input"))?;
        let max_deltas_lo = inputs.get("max_delta_lo").ok_or_else(|| anyhow::anyhow!("missing max_delta_lo"))?;
        let min_deltas_result = inputs.get("min_delta_result").ok_or_else(|| anyhow::anyhow!("missing min_delta_result"))?;
        let min_deltas_hi = inputs.get("min_delta_hi").ok_or_else(|| anyhow::anyhow!("missing min_delta_hi"))?;

        Ok(max_min_clip::constrained_clip(
            builder,
            input,
            lo,
            hi,
            max_hint,
            min_hint,
            &[max_deltas_input.clone(), max_deltas_lo.clone()],
            &[min_deltas_result.clone(), min_deltas_hi.clone()],
        ))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let output: Vec<i64> = input
            .iter()
            .map(|&x| max_min_clip::compute_clip(x, self.lo, self.hi))
            .collect();

        let after_max: Vec<i64> = input.iter().map(|&x| x.max(self.lo)).collect();
        let max_delta_input: Vec<i64> = after_max.iter().zip(input.iter()).map(|(m, x)| m - x).collect();
        let max_delta_lo: Vec<i64> = after_max.iter().map(|m| m - self.lo).collect();
        let min_delta_result: Vec<i64> = after_max.iter().zip(output.iter()).map(|(m, o)| m - o).collect();
        let min_delta_hi: Vec<i64> = output.iter().map(|o| self.hi - o).collect();

        Ok(WitnessData::new()
            .with_output("output", output.clone())
            .with_hint("max_hint", after_max)
            .with_hint("min_hint", output)
            .with_hint("max_delta_input", max_delta_input)
            .with_hint("max_delta_lo", max_delta_lo)
            .with_hint("min_delta_result", min_delta_result)
            .with_hint("min_delta_hi", min_delta_hi))
    }
}
