use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct MaxPoolLayer {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
}

impl MaxPoolLayer {
    pub fn output_h(&self) -> usize {
        (self.input_h + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1
    }

    pub fn output_w(&self) -> usize {
        (self.input_w + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1
    }
}

impl<F: Field> LayerOp<F> for MaxPoolLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let max_hint = inputs.get("max_hint").ok_or_else(|| anyhow::anyhow!("missing max_hint"))?;

        let oh = self.output_h();
        let ow = self.output_w();
        Ok(max_hint.clone())
    }

    fn compute_witness(
        &self,
        inputs: &HashMap<String, Vec<i64>>,
    ) -> Result<WitnessData> {
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let oh = self.output_h();
        let ow = self.output_w();

        let mut output = vec![i64::MIN; self.channels * oh * ow];
        let mut deltas = Vec::new();

        for c in 0..self.channels {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut window_vals = Vec::new();
                    for ky in 0..self.kernel_h {
                        for kx in 0..self.kernel_w {
                            let iy = (oy * self.stride_h + ky) as isize - self.pad_h as isize;
                            let ix = (ox * self.stride_w + kx) as isize - self.pad_w as isize;
                            if iy >= 0
                                && iy < self.input_h as isize
                                && ix >= 0
                                && ix < self.input_w as isize
                            {
                                let idx =
                                    c * self.input_h * self.input_w + (iy as usize) * self.input_w + ix as usize;
                                window_vals.push(input[idx]);
                            }
                        }
                    }
                    let max_val = *window_vals.iter().max().unwrap_or(&0);
                    let out_idx = c * oh * ow + oy * ow + ox;
                    output[out_idx] = max_val;

                    for &v in &window_vals {
                        deltas.push(max_val - v);
                    }
                }
            }
        }

        Ok(WitnessData::new()
            .with_output("output", output.clone())
            .with_hint("max_hint", output)
            .with_hint("deltas", deltas))
    }
}
