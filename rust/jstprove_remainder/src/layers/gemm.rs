use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::{linear_algebra, rescale};
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct GemmLayer {
    pub trans_a: bool,
    pub trans_b: bool,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub has_bias: bool,
    pub alpha: i64,
    pub shift_exponent: usize,
}

impl<F: Field> LayerOp<F> for GemmLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?;

        let (a_rows, a_cols) = if self.trans_a {
            (self.k, self.m)
        } else {
            (self.m, self.k)
        };
        let (b_rows, b_cols) = if self.trans_b {
            (self.n, self.k)
        } else {
            (self.k, self.n)
        };

        let product = linear_algebra::matmul_with_dims(builder, a, a_rows, a_cols, b, b_rows, b_cols);

        let after_bias = if self.has_bias {
            let bias = inputs.get("bias").ok_or_else(|| anyhow::anyhow!("missing bias"))?;
            linear_algebra::elementwise_add(builder, &product, bias)
        } else {
            product
        };

        let q = inputs.get("quotient").ok_or_else(|| anyhow::anyhow!("missing quotient hint"))?;
        let r = inputs.get("remainder").ok_or_else(|| anyhow::anyhow!("missing remainder hint"))?;
        Ok(rescale::constrained_rescale(builder, &after_bias, q, r))
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let a = inputs.get("A").ok_or_else(|| anyhow::anyhow!("missing A"))?;
        let b = inputs.get("B").ok_or_else(|| anyhow::anyhow!("missing B"))?;

        let (a_rows, a_cols, b_cols) = (self.m, self.k, self.n);

        let mut result = vec![0i64; a_rows * b_cols];
        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0i64;
                for p in 0..a_cols {
                    let a_idx = if self.trans_a {
                        p * a_rows + i
                    } else {
                        i * a_cols + p
                    };
                    let b_idx = if self.trans_b {
                        j * a_cols + p
                    } else {
                        p * b_cols + j
                    };
                    sum += a.get(a_idx).copied().unwrap_or(0)
                        * b.get(b_idx).copied().unwrap_or(0);
                }
                result[i * b_cols + j] = sum;
            }
        }

        if self.has_bias {
            if let Some(bias) = inputs.get("bias") {
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        result[i * b_cols + j] += bias.get(j).copied().unwrap_or(0);
                    }
                }
            }
        }

        let offset = 1i64 << self.shift_exponent;
        let (quotients, remainders) = rescale::compute_rescale_array(&result, self.alpha, offset);

        Ok(WitnessData::new()
            .with_output("output", quotients.clone())
            .with_hint("quotient", quotients)
            .with_hint("remainder", remainders))
    }
}
