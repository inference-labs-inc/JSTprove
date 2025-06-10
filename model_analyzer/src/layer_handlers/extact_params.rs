use crate::layer_handlers::layer_ir::LayerParams;
// use crate::extract_params::ExtractParams;
use tract_core::ops::{cast::Cast, cnn::{Conv, PaddingSpec}, einsum::EinSum};


pub trait ExtractParams {
    fn extract_params(&self) -> Option<LayerParams>;
}

impl ExtractParams for Conv {
    fn extract_params(&self) -> Option<LayerParams> {
        Some(LayerParams::Conv {
            group: self.group,
            strides: self.pool_spec.strides.clone().unwrap().to_vec(),
            dilations: self.pool_spec.dilations.clone().unwrap().to_vec(),
            padding: match &self.pool_spec.padding {
                PaddingSpec::Explicit(before, after)
                | PaddingSpec::ExplicitOnnxPool(before, after, _) => (
                    before.clone().to_vec().into(),
                    after.clone().to_vec().into(),
                ),
                _ => {
                    let dim = self.pool_spec.kernel_shape.len(); // or hardcode e.g. 2
                    (vec![0; dim], vec![0; dim])
                }
            },
            kernel_shape: self.pool_spec.kernel_shape.to_vec(),
            output_channels: self.pool_spec.output_channels,
            input_channels: self.pool_spec.input_channels,
        })
    }
}

impl ExtractParams for EinSum {
    fn extract_params(&self) -> Option<LayerParams> {
        Some(LayerParams::EinSum  {
            axes: self.axes.to_string(),
            operating_dt: format!("{:?}",self.operating_dt),
            q_params: format!("{:?}",self.q_params),
            
        })
    }
}

impl ExtractParams for Cast {
    fn extract_params(&self) -> Option<LayerParams> {
        Some(LayerParams::Cast  {
            to: format!("{:?}", self.to),
        })
    }
}
