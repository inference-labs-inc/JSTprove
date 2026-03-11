use anyhow::{ensure, Result};

use super::graph::LayerNode;

pub struct GemmParams {
    pub input_name: String,
    pub weight_tensor_name: String,
    pub bias_tensor_name: Option<String>,
    pub trans_b: bool,
    pub k_dim: usize,
    pub n_dim: usize,
    pub w_rows: usize,
    pub w_cols: usize,
}

impl GemmParams {
    pub fn from_layer(layer: &LayerNode) -> Result<Self> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Gemm {} has no input", layer.name))?
            .clone();
        let weight_tensor_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("Gemm {} has no weight", layer.name))?
            .clone();
        let bias_tensor_name = layer.inputs.get(2).cloned();

        let trans_a = layer
            .get_int_attr("transA")
            .map(|v| v != 0)
            .unwrap_or(false);
        ensure!(
            !trans_a,
            "Gemm {} has transA=1 which is not supported",
            layer.name
        );
        let trans_b = layer
            .get_int_attr("transB")
            .map(|v| v != 0)
            .unwrap_or(false);

        let weight_data = layer.weights.get(&weight_tensor_name).ok_or_else(|| {
            anyhow::anyhow!("Gemm {} missing weight {}", layer.name, weight_tensor_name)
        })?;
        let w_shape = weight_data.shape();
        ensure!(
            w_shape.len() >= 2,
            "Gemm {} weight has {} dims, need >= 2",
            layer.name,
            w_shape.len()
        );
        let (w_rows, w_cols) = (w_shape[0], w_shape[1]);
        let (k_dim, n_dim) = if trans_b {
            (w_cols, w_rows)
        } else {
            (w_rows, w_cols)
        };

        Ok(Self {
            input_name,
            weight_tensor_name,
            bias_tensor_name,
            trans_b,
            k_dim,
            n_dim,
            w_rows,
            w_cols,
        })
    }
}

pub struct ConvParams {
    pub input_name: String,
    pub weight_tensor_name: String,
    pub bias_tensor_name: Option<String>,
    pub c_out: usize,
    pub c_in: usize,
    pub kh: usize,
    pub kw: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub pad_bottom: usize,
    pub pad_right: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub patch_size: usize,
    pub num_patches: usize,
}

impl ConvParams {
    pub fn from_layer(
        layer: &LayerNode,
        input_ch: usize,
        in_h: usize,
        in_w: usize,
    ) -> Result<Self> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("Conv {} has no input", layer.name))?
            .clone();
        let weight_tensor_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("Conv {} has no weight", layer.name))?
            .clone();
        let bias_tensor_name = layer.inputs.get(2).cloned();

        let weight_data = layer.weights.get(&weight_tensor_name).ok_or_else(|| {
            anyhow::anyhow!("Conv {} missing weight {}", layer.name, weight_tensor_name)
        })?;

        let w_shape = weight_data.shape();
        ensure!(
            w_shape.len() >= 4,
            "Conv {} weight has {} dims, need >= 4",
            layer.name,
            w_shape.len()
        );
        let c_out = w_shape[0];
        let c_in = w_shape[1];
        let kh = w_shape[2];
        let kw = w_shape[3];

        ensure!(
            input_ch == c_in,
            "Conv {}: weight c_in {} does not match input channels {}",
            layer.name,
            c_in,
            input_ch
        );

        let pads = layer.get_ints_attr("pads");
        let pad_top = pads.and_then(|p| p.first().copied()).unwrap_or(0) as usize;
        let pad_left = pads.and_then(|p| p.get(1).copied()).unwrap_or(0) as usize;
        let pad_bottom = pads.and_then(|p| p.get(2).copied()).unwrap_or(0) as usize;
        let pad_right = pads.and_then(|p| p.get(3).copied()).unwrap_or(0) as usize;

        let strides = layer.get_ints_attr("strides");
        let raw_stride_h = strides.and_then(|s| s.first().copied()).unwrap_or(1);
        let raw_stride_w = strides.and_then(|s| s.get(1).copied()).unwrap_or(1);
        ensure!(
            raw_stride_h > 0 && raw_stride_w > 0,
            "Conv {} stride_h={} stride_w={} must be positive",
            layer.name,
            raw_stride_h,
            raw_stride_w
        );
        let stride_h = raw_stride_h as usize;
        let stride_w = raw_stride_w as usize;

        let padded_h = in_h + pad_top + pad_bottom;
        let padded_w = in_w + pad_left + pad_right;
        ensure!(
            padded_h >= kh && padded_w >= kw,
            "Conv {}: padded input {}x{} smaller than kernel {}x{}",
            layer.name,
            padded_h,
            padded_w,
            kh,
            kw
        );
        let out_h = (padded_h - kh) / stride_h + 1;
        let out_w = (padded_w - kw) / stride_w + 1;
        let patch_size = c_in
            .checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Conv {} patch_size overflow: {} * {} * {}",
                    layer.name,
                    c_in,
                    kh,
                    kw
                )
            })?;
        let num_patches = out_h.checked_mul(out_w).ok_or_else(|| {
            anyhow::anyhow!(
                "Conv {} num_patches overflow: {} * {}",
                layer.name,
                out_h,
                out_w
            )
        })?;

        Ok(Self {
            input_name,
            weight_tensor_name,
            bias_tensor_name,
            c_out,
            c_in,
            kh,
            kw,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            stride_h,
            stride_w,
            out_h,
            out_w,
            patch_size,
            num_patches,
        })
    }
}

pub struct MaxPoolParams {
    pub input_name: String,
    pub pool_h: usize,
    pub pool_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pool_oh: usize,
    pub pool_ow: usize,
    pub window_size: usize,
}

impl MaxPoolParams {
    pub fn from_layer(layer: &LayerNode, in_h: usize, in_w: usize) -> Result<Self> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("MaxPool {} has no input", layer.name))?
            .clone();

        let kernel_shape = layer
            .get_ints_attr("kernel_shape")
            .ok_or_else(|| anyhow::anyhow!("MaxPool {} missing kernel_shape", layer.name))?;
        ensure!(
            kernel_shape.len() == 2,
            "MaxPool {} requires exactly 2D kernel_shape, got {} dims",
            layer.name,
            kernel_shape.len()
        );
        ensure!(
            kernel_shape[0] > 0 && kernel_shape[1] > 0,
            "MaxPool {} kernel_shape values must be positive, got [{}, {}]",
            layer.name,
            kernel_shape[0],
            kernel_shape[1]
        );
        let pool_h = kernel_shape[0] as usize;
        let pool_w = kernel_shape[1] as usize;

        let strides = layer.get_ints_attr("strides");
        let raw_stride_h = strides.and_then(|s| s.first().copied()).unwrap_or(1);
        let raw_stride_w = strides.and_then(|s| s.get(1).copied()).unwrap_or(1);
        ensure!(
            raw_stride_h > 0 && raw_stride_w > 0,
            "MaxPool {} stride_h={} stride_w={} must be positive",
            layer.name,
            raw_stride_h,
            raw_stride_w
        );
        let stride_h = raw_stride_h as usize;
        let stride_w = raw_stride_w as usize;

        ensure!(
            in_h >= pool_h && in_w >= pool_w,
            "MaxPool {}: input {}x{} smaller than kernel {}x{}",
            layer.name,
            in_h,
            in_w,
            pool_h,
            pool_w
        );
        let pool_oh = (in_h - pool_h) / stride_h + 1;
        let pool_ow = (in_w - pool_w) / stride_w + 1;
        let window_size = pool_h.checked_mul(pool_w).ok_or_else(|| {
            anyhow::anyhow!(
                "MaxPool {} window_size overflow: {} * {}",
                layer.name,
                pool_h,
                pool_w
            )
        })?;

        Ok(Self {
            input_name,
            pool_h,
            pool_w,
            stride_h,
            stride_w,
            pool_oh,
            pool_ow,
            window_size,
        })
    }
}

pub struct BatchNormParams {
    pub input_name: String,
    pub mul_tensor_name: String,
    pub add_tensor_name: String,
}

impl BatchNormParams {
    pub fn from_layer(layer: &LayerNode) -> Result<Self> {
        let input_name = layer
            .inputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("BatchNorm {} has no input", layer.name))?
            .clone();
        let mul_tensor_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing mul input", layer.name))?
            .clone();
        let add_tensor_name = layer
            .inputs
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("BatchNorm {} missing add input", layer.name))?
            .clone();
        Ok(Self {
            input_name,
            mul_tensor_name,
            add_tensor_name,
        })
    }
}

pub struct AddSubParams {
    pub input_a_name: String,
    pub input_b_name: String,
}

impl AddSubParams {
    pub fn from_layer(layer: &LayerNode) -> Result<Self> {
        let input_a_name = layer
            .inputs
            .first()
            .ok_or_else(|| {
                anyhow::anyhow!("{:?} {} has no first input", layer.op_type, layer.name)
            })?
            .clone();
        let input_b_name = layer
            .inputs
            .get(1)
            .ok_or_else(|| {
                anyhow::anyhow!("{:?} {} has no second input", layer.op_type, layer.name)
            })?
            .clone();
        Ok(Self {
            input_a_name,
            input_b_name,
        })
    }
}
