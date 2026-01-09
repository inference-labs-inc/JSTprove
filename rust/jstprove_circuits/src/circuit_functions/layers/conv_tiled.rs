use std::collections::HashMap;

use ndarray::{s, ArrayD};
use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::circuit_functions::{
    CircuitError,
    layers::{LayerError, LayerKind, layer_ops::LayerOp},
    utils::{
        constants::{BIAS, DILATION, GROUP, INPUT, KERNEL_SHAPE, PADS, STRIDES, WEIGHTS},
        onnx_model::{
            extract_params_and_expected_shape, get_input_name, get_param, get_param_or_default,
            get_w_or_b,
        },
        quantization::rescale_array,
        tensor_ops::{load_array_constants, load_circuit_constant},
        typecasting::{AsU32, AsUsize},
    },
};

pub const DEFAULT_TILE_SIZE: usize = 64;

#[derive(Debug)]
pub struct TiledConvLayer {
    name: String,
    weights: ArrayD<i64>,
    bias: ArrayD<i64>,
    strides: Vec<usize>,
    kernel_shape: Vec<usize>,
    pads: Vec<usize>,
    tile_size: usize,
    scaling: u64,
    v_plus_one: usize,
    is_rescale: bool,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

pub struct TileCoord {
    pub tile_y: usize,
    pub tile_x: usize,
    pub input_h: usize,
    pub input_w: usize,
}

impl<C: Config, Builder: RootAPI<C>> LayerOp<C, Builder> for TiledConvLayer {
    fn apply(
        &self,
        api: &mut Builder,
        input: HashMap<String, ArrayD<Variable>>,
    ) -> Result<(Vec<String>, ArrayD<Variable>), CircuitError> {
        let input_name = get_input_name(&self.inputs, 0, LayerKind::Conv, INPUT)?;
        let layer_input = input
            .get(&input_name)
            .ok_or_else(|| LayerError::MissingInput {
                layer: LayerKind::Conv,
                name: input_name.clone(),
            })?;

        let shape = layer_input.shape();
        let (n, c_in, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let weights_var = load_array_constants(api, &self.weights);
        let bias_var: ArrayD<Variable> = self.bias.mapv(|x| load_circuit_constant(api, x));
        let c_out = self.weights.shape()[0];

        let kh = self.kernel_shape[0];
        let kw = self.kernel_shape[1];
        let stride_h = self.strides[0];
        let stride_w = self.strides[1];
        let pad_h = self.pads[0];
        let pad_w = self.pads[1];

        let out_h = (h + 2 * pad_h - kh) / stride_h + 1;
        let out_w = (w + 2 * pad_w - kw) / stride_w + 1;

        let tile_size = self.tile_size;
        let tiles_y = (h + tile_size - 1) / tile_size;
        let tiles_x = (w + tile_size - 1) / tile_size;

        let out_tile_h = tile_size / stride_h;
        let out_tile_w = tile_size / stride_w;

        let zero = api.constant(0u32);
        let mut output = ArrayD::from_elem(vec![n, c_out, out_h, out_w], zero);

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let tile_out = self.process_tile(
                    api,
                    layer_input,
                    &weights_var,
                    &bias_var,
                    &TileCoord {
                        tile_y: ty,
                        tile_x: tx,
                        input_h: h,
                        input_w: w,
                    },
                )?;

                let out_y_start = ty * out_tile_h;
                let out_x_start = tx * out_tile_w;
                let out_y_end = (out_y_start + out_tile_h).min(out_h);
                let out_x_end = (out_x_start + out_tile_w).min(out_w);

                let tile_h_actual = out_y_end - out_y_start;
                let tile_w_actual = out_x_end - out_x_start;

                for nn in 0..n {
                    for co in 0..c_out {
                        for oy in 0..tile_h_actual {
                            for ox in 0..tile_w_actual {
                                output[[nn, co, out_y_start + oy, out_x_start + ox]] =
                                    tile_out[[nn, co, oy, ox]];
                            }
                        }
                    }
                }
            }
        }

        let output = if self.is_rescale {
            let shift_exp = self.v_plus_one.saturating_sub(1);
            rescale_array(api, output, self.scaling as usize, shift_exp, false)?
        } else {
            output
        };

        Ok((self.outputs.clone(), output))
    }

    fn build(
        layer: &crate::circuit_functions::utils::onnx_types::ONNXLayer,
        circuit_params: &crate::circuit_functions::utils::onnx_model::CircuitParams,
        _optimization_pattern: crate::circuit_functions::utils::graph_pattern_matching::PatternRegistry,
        is_rescale: bool,
        _index: usize,
        layer_context: &crate::circuit_functions::utils::build_layers::BuildLayerContext,
    ) -> Result<Box<dyn LayerOp<C, Builder>>, CircuitError> {
        let (params, _) = extract_params_and_expected_shape(layer_context, layer).map_err(|e| {
            LayerError::Other {
                layer: LayerKind::Conv,
                msg: format!("extract_params_and_expected_shape failed: {e}"),
            }
        })?;

        let w_name = get_input_name(&layer.inputs, 1, LayerKind::Conv, WEIGHTS)?;
        let weights = get_w_or_b(&layer_context.w_and_b_map, w_name)?;

        let bias = if layer.inputs.len() > 2 {
            let b_name = get_input_name(&layer.inputs, 2, LayerKind::Conv, BIAS)?;
            get_w_or_b(&layer_context.w_and_b_map, b_name).unwrap_or_else(|_| ArrayD::default(vec![]))
        } else {
            ArrayD::default(vec![])
        };

        let kernel_shape: Vec<u32> = get_param(&layer.name, KERNEL_SHAPE, &params)?;
        let strides: Vec<u32> = get_param(&layer.name, STRIDES, &params)?;
        let pads: Vec<u32> = get_param_or_default(&layer.name, PADS, &params, Some(&vec![0u32; 4]))?;

        Ok(Box::new(Self {
            name: layer.name.clone(),
            weights,
            bias,
            strides: strides.iter().map(|&x| x as usize).collect(),
            kernel_shape: kernel_shape.iter().map(|&x| x as usize).collect(),
            pads: pads.iter().map(|&x| x as usize).collect(),
            tile_size: DEFAULT_TILE_SIZE,
            scaling: circuit_params.scale_exponent.into(),
            v_plus_one: layer_context.n_bits,
            is_rescale,
            inputs: layer.inputs.clone(),
            outputs: layer.outputs.clone(),
        }))
    }
}

impl TiledConvLayer {
    fn process_tile<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        input: &ArrayD<Variable>,
        weights: &ArrayD<Variable>,
        bias: &ArrayD<Variable>,
        coord: &TileCoord,
    ) -> Result<ArrayD<Variable>, CircuitError> {
        let shape = input.shape();
        let (n, c_in, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let c_out = weights.shape()[0];

        let kh = self.kernel_shape[0];
        let kw = self.kernel_shape[1];
        let stride_h = self.strides[0];
        let stride_w = self.strides[1];
        let pad_h = self.pads[0];
        let pad_w = self.pads[1];

        let halo = (kh - 1) / 2;

        let in_y_start = (coord.tile_y * self.tile_size).saturating_sub(halo);
        let in_x_start = (coord.tile_x * self.tile_size).saturating_sub(halo);
        let in_y_end = ((coord.tile_y + 1) * self.tile_size + halo).min(h);
        let in_x_end = ((coord.tile_x + 1) * self.tile_size + halo).min(w);

        let out_tile_h = self.tile_size / stride_h;
        let out_tile_w = self.tile_size / stride_w;

        let zero = api.constant(0u32);
        let mut tile_output = ArrayD::from_elem(vec![n, c_out, out_tile_h, out_tile_w], zero);

        for nn in 0..n {
            for co in 0..c_out {
                let b_val = if bias.is_empty() {
                    zero
                } else {
                    bias[[co]]
                };

                for oy in 0..out_tile_h {
                    for ox in 0..out_tile_w {
                        let global_oy = coord.tile_y * out_tile_h + oy;
                        let global_ox = coord.tile_x * out_tile_w + ox;

                        let iy_base = (global_oy * stride_h) as isize - pad_h as isize;
                        let ix_base = (global_ox * stride_w) as isize - pad_w as isize;

                        let mut acc = b_val;

                        for ci in 0..c_in {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = iy_base + ky as isize;
                                    let ix = ix_base + kx as isize;

                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                        let iy = iy as usize;
                                        let ix = ix as usize;

                                        let in_val = input[[nn, ci, iy, ix]];
                                        let w_val = weights[[co, ci, ky, kx]];
                                        let prod = api.mul(in_val, w_val);
                                        acc = api.add(acc, prod);
                                    }
                                }
                            }
                        }

                        tile_output[[nn, co, oy, ox]] = acc;
                    }
                }
            }
        }

        Ok(tile_output)
    }
}
