use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::{linear_algebra, rescale};
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct BatchNormLayer {
    pub channels: usize,
    pub spatial_size: usize,
    pub alpha: i64,
    pub shift_exponent: usize,
}

impl<F: Field> LayerOp<F> for BatchNormLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let scale = inputs.get("scale").ok_or_else(|| anyhow::anyhow!("missing scale"))?;
        let bias = inputs.get("bias").ok_or_else(|| anyhow::anyhow!("missing bias"))?;

        let scaled = linear_algebra::hadamard_product(builder, input, scale);
        let after_bias = linear_algebra::elementwise_add(builder, &scaled, bias);

        let q = inputs.get("quotient").ok_or_else(|| anyhow::anyhow!("missing quotient hint"))?;
        let r = inputs.get("remainder").ok_or_else(|| anyhow::anyhow!("missing remainder hint"))?;
        Ok(rescale::constrained_rescale(builder, &after_bias, q, r))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let scale = inputs.get("scale").ok_or_else(|| anyhow::anyhow!("missing scale"))?;
        let bias = inputs.get("bias").ok_or_else(|| anyhow::anyhow!("missing bias"))?;

        let mut result = vec![0i64; input.len()];
        for i in 0..input.len() {
            let c = i / self.spatial_size % self.channels;
            result[i] = input[i] * scale.get(c).copied().unwrap_or(0)
                + bias.get(c).copied().unwrap_or(0);
        }

        let offset = 1i64 << self.shift_exponent;
        let (quotients, remainders) = rescale::compute_rescale_array(&result, self.alpha, offset);

        Ok(WitnessData::new()
            .with_output("output", quotients.clone())
            .with_hint("quotient", quotients)
            .with_hint("remainder", remainders))
    }
}
