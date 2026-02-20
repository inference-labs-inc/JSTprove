use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::linear_algebra;
use crate::gadgets::rescale;
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct MulLayer {
    pub needs_rescale: bool,
    pub alpha: i64,
    pub shift_exponent: usize,
}

impl<F: Field> LayerOp<F> for MulLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing input A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing input B"))?;
        let product = linear_algebra::hadamard_product(builder, a, b);

        if self.needs_rescale {
            let q = inputs.get("quotient").ok_or_else(|| anyhow::anyhow!("missing quotient hint"))?;
            let r = inputs.get("remainder").ok_or_else(|| anyhow::anyhow!("missing remainder hint"))?;
            Ok(rescale::constrained_rescale(builder, &product, q, r))
        } else {
            Ok(product)
        }
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing input A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing input B"))?;
        let product: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

        if self.needs_rescale {
            let offset = 1i64 << self.shift_exponent;
            let (quotients, remainders) = rescale::compute_rescale_array(&product, self.alpha, offset);
            Ok(WitnessData::new()
                .with_output("output", quotients.clone())
                .with_hint("quotient", quotients)
                .with_hint("remainder", remainders))
        } else {
            Ok(WitnessData::new().with_output("output", product))
        }
    }
}
