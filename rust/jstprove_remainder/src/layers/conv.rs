use std::collections::HashMap;

use anyhow::Result;
use frontend::layouter::builder::CircuitBuilder;
use shared_types::Field;

use crate::gadgets::{linear_algebra, rescale};
use crate::layers::trait_def::{LayerOp, WitnessData};
use crate::tensor::ShapedMLE;

pub struct ConvLayer {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub has_bias: bool,
    pub alpha: i64,
    pub shift_exponent: usize,
}

impl ConvLayer {
    pub fn output_h(&self) -> usize {
        (self.input_h + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1
    }

    pub fn output_w(&self) -> usize {
        (self.input_w + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1
    }
}

impl<F: Field> LayerOp<F> for ConvLayer {
    fn build_graph(
        &self,
        builder: &mut CircuitBuilder<F>,
        inputs: &HashMap<String, ShapedMLE<F>>,
    ) -> Result<ShapedMLE<F>> {
        let im2col = inputs.get("im2col").ok_or_else(|| anyhow::anyhow!("missing im2col"))?;
        let weight = inputs.get("weight").ok_or_else(|| anyhow::anyhow!("missing weight"))?;

        let oh = self.output_h();
        let ow = self.output_w();
        let col_rows = oh * ow;
        let col_cols = self.in_channels * self.kernel_h * self.kernel_w;

        let product = linear_algebra::matmul_with_dims(
            builder,
            im2col,
            col_rows,
            col_cols,
            weight,
            col_cols,
            self.out_channels,
        );

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
        let input = inputs.get("input").ok_or_else(|| anyhow::anyhow!("missing input"))?;
        let weight = inputs.get("weight").ok_or_else(|| anyhow::anyhow!("missing weight"))?;

        let oh = self.output_h();
        let ow = self.output_w();
        let col_cols = self.in_channels * self.kernel_h * self.kernel_w;

        let im2col_data = im2col(
            input,
            self.in_channels,
            self.input_h,
            self.input_w,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
        );

        let mut result = vec![0i64; oh * ow * self.out_channels];
        for i in 0..(oh * ow) {
            for j in 0..self.out_channels {
                let mut sum = 0i64;
                for p in 0..col_cols {
                    sum += im2col_data[i * col_cols + p] * weight[j * col_cols + p];
                }
                result[i * self.out_channels + j] = sum;
            }
        }

        if self.has_bias {
            if let Some(bias) = inputs.get("bias") {
                for i in 0..(oh * ow) {
                    for j in 0..self.out_channels {
                        result[i * self.out_channels + j] += bias.get(j).copied().unwrap_or(0);
                    }
                }
            }
        }

        let offset = 1i64 << self.shift_exponent;
        let (quotients, remainders) = rescale::compute_rescale_array(&result, self.alpha, offset);

        Ok(WitnessData::new()
            .with_output("output", quotients.clone())
            .with_hint("im2col", im2col_data)
            .with_hint("quotient", quotients)
            .with_hint("remainder", remainders))
    }
}

pub fn im2col(
    input: &[i64],
    channels: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<i64> {
    let oh = (h + 2 * pad_h - kh) / stride_h + 1;
    let ow = (w + 2 * pad_w - kw) / stride_w + 1;
    let col_cols = channels * kh * kw;
    let mut col = vec![0i64; oh * ow * col_cols];

    for out_y in 0..oh {
        for out_x in 0..ow {
            let row = out_y * ow + out_x;
            for c in 0..channels {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let in_y = out_y * stride_h + ky;
                        let in_x = out_x * stride_w + kx;
                        let in_y = in_y as isize - pad_h as isize;
                        let in_x = in_x as isize - pad_w as isize;

                        let val = if in_y >= 0
                            && in_y < h as isize
                            && in_x >= 0
                            && in_x < w as isize
                        {
                            let idx = c * h * w + (in_y as usize) * w + (in_x as usize);
                            input[idx]
                        } else {
                            0
                        };

                        let col_idx = c * kh * kw + ky * kw + kx;
                        col[row * col_cols + col_idx] = val;
                    }
                }
            }
        }
    }

    col
}
