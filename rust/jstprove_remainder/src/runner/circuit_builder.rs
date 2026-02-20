use std::collections::HashMap;

use anyhow::{Result, bail};
use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::{CircuitBuilder, Circuit, InputLayerNodeRef, LayerVisibility, NodeRef};
use shared_types::Fr;

use crate::onnx::graph::OpType;
use crate::onnx::quantizer::QuantizedModel;
use crate::padding::{next_power_of_two, num_vars_for};
use crate::util::i64_to_fr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Committed,
}

#[derive(Debug, Clone)]
pub struct ShredEntry {
    pub num_vars: usize,
    pub visibility: Visibility,
}

pub type ShredManifest = HashMap<String, ShredEntry>;

pub struct BuildResult {
    pub circuit: Circuit<Fr>,
    pub manifest: ShredManifest,
}

#[derive(Debug, Clone)]
pub enum SpatialInfo {
    CHW { c: usize, h: usize, w: usize },
    HWC { h: usize, w: usize, c: usize, stride_c: usize },
}

impl SpatialInfo {
    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        match self {
            SpatialInfo::CHW { h: hd, w: wd, .. } => c * hd * wd + h * wd + w,
            SpatialInfo::HWC { w: wd, stride_c, .. } => (h * wd + w) * stride_c + c,
        }
    }

    pub fn spatial_dims(&self) -> (usize, usize, usize) {
        match self {
            SpatialInfo::CHW { c, h, w } => (*c, *h, *w),
            SpatialInfo::HWC { h, w, c, .. } => (*c, *h, *w),
        }
    }

    pub fn hw(&self) -> (usize, usize) {
        match self {
            SpatialInfo::CHW { h, w, .. } => (*h, *w),
            SpatialInfo::HWC { h, w, .. } => (*h, *w),
        }
    }
}

pub fn build_circuit(model: &QuantizedModel, input_size: usize) -> Result<BuildResult> {
    let alpha = model.scale_config.alpha;
    let alpha_fr = i64_to_fr(alpha);

    let mut manifest = ShredManifest::new();
    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_vars = num_vars_for(input_size);

    let input_shred_name = model.graph.input_names.first()
        .cloned()
        .unwrap_or_else(|| "input".to_string());

    let input_node = builder.add_input_shred(&input_shred_name, input_vars, &public);
    manifest.insert(input_shred_name.clone(), ShredEntry { num_vars: input_vars, visibility: Visibility::Public });

    let mut tensor_nodes: HashMap<String, NodeRef<Fr>> = HashMap::new();
    let mut tensor_num_vars: HashMap<String, usize> = HashMap::new();
    let mut tensor_layouts: HashMap<String, SpatialInfo> = HashMap::new();

    for name in &model.graph.input_names {
        tensor_nodes.insert(name.clone(), input_node.clone());
        tensor_num_vars.insert(name.clone(), input_vars);
    }

    for (name, shape) in &model.graph.input_shapes {
        if shape.len() == 3 {
            tensor_layouts.insert(
                name.clone(),
                SpatialInfo::CHW { c: shape[0], h: shape[1], w: shape[2] },
            );
        }
    }

    let mut last_output_name = String::new();

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm => {
                build_gemm_layer(
                    &mut builder,
                    &public,
                    &committed,
                    layer,
                    &mut tensor_nodes,
                    &mut tensor_num_vars,
                    &mut manifest,
                    alpha_fr,
                )?;
            }
            OpType::Conv => {
                build_conv_layer(
                    &mut builder,
                    &public,
                    &committed,
                    layer,
                    &mut tensor_nodes,
                    &mut tensor_num_vars,
                    &mut tensor_layouts,
                    &mut manifest,
                    alpha_fr,
                )?;
            }
            OpType::Relu => {
                build_relu_layer(
                    &mut builder,
                    &public,
                    &committed,
                    layer,
                    &mut tensor_nodes,
                    &mut tensor_num_vars,
                    &mut tensor_layouts,
                    &mut manifest,
                )?;
            }
            OpType::MaxPool => {
                build_maxpool_layer(
                    &mut builder,
                    &public,
                    &committed,
                    layer,
                    &mut tensor_nodes,
                    &mut tensor_num_vars,
                    &mut tensor_layouts,
                    &mut manifest,
                )?;
            }
            OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
                let input_name = layer.inputs.first()
                    .ok_or_else(|| anyhow::anyhow!("shape op {} has no inputs", layer.name))?;
                let node = tensor_nodes.get(input_name).cloned();
                let nv = tensor_num_vars.get(input_name).copied();
                if let Some(node) = node {
                    for out in &layer.outputs {
                        tensor_nodes.insert(out.clone(), node.clone());
                        if let Some(nv) = nv {
                            tensor_num_vars.insert(out.clone(), nv);
                        }
                    }
                }
            }
            other => {
                bail!("circuit builder: unsupported op type {:?} in layer {}", other, layer.name);
            }
        }

        for out in &layer.outputs {
            last_output_name = out.clone();
        }
    }

    let expected_name = "expected_output".to_string();
    let output_vars = tensor_num_vars.get(&last_output_name)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("no output tensor found"))?;
    let expected_node = builder.add_input_shred(&expected_name, output_vars, &public);
    manifest.insert(expected_name, ShredEntry { num_vars: output_vars, visibility: Visibility::Public });

    let last_output_node = tensor_nodes.get(&last_output_name)
        .ok_or_else(|| anyhow::anyhow!("no output node for {}", last_output_name))?;
    let out_check = builder.add_sector(last_output_node.expr() - expected_node.expr());
    builder.set_output(&out_check);

    let circuit = builder.build_with_layer_combination()?;

    Ok(BuildResult { circuit, manifest })
}

fn build_gemm_layer(
    builder: &mut CircuitBuilder<Fr>,
    public: &InputLayerNodeRef<Fr>,
    committed: &InputLayerNodeRef<Fr>,
    layer: &crate::onnx::graph::LayerNode,
    tensor_nodes: &mut HashMap<String, NodeRef<Fr>>,
    tensor_num_vars: &mut HashMap<String, usize>,
    manifest: &mut ShredManifest,
    alpha_fr: Fr,
) -> Result<()> {
    let input_name = layer.inputs.first()
        .ok_or_else(|| anyhow::anyhow!("Gemm {} has no input", layer.name))?;
    let weight_tensor_name = layer.inputs.get(1)
        .ok_or_else(|| anyhow::anyhow!("Gemm {} has no weight input", layer.name))?;

    let weight_data = layer.weights.get(weight_tensor_name)
        .ok_or_else(|| anyhow::anyhow!("Gemm {} missing weight tensor {}", layer.name, weight_tensor_name))?;

    let trans_b = layer.get_int_attr("transB").map(|v| v != 0).unwrap_or(false);

    let w_shape = weight_data.shape();
    let (w_rows, w_cols) = if w_shape.len() >= 2 {
        (w_shape[0], w_shape[1])
    } else {
        bail!("Gemm {} weight has < 2 dims", layer.name);
    };

    let (k_dim, n_dim) = if trans_b {
        (w_cols, w_rows)
    } else {
        (w_rows, w_cols)
    };

    let k_vars = num_vars_for(k_dim);
    let n_vars = num_vars_for(n_dim);

    let weight_shred_name = format!("{}_weight", layer.name);
    let weight_node = builder.add_input_shred(&weight_shred_name, k_vars + n_vars, public);
    manifest.insert(weight_shred_name, ShredEntry { num_vars: k_vars + n_vars, visibility: Visibility::Public });

    let bias_shred_name = format!("{}_bias", layer.name);
    let bias_node = builder.add_input_shred(&bias_shred_name, n_vars, public);
    manifest.insert(bias_shred_name, ShredEntry { num_vars: n_vars, visibility: Visibility::Public });

    let q_name = format!("{}_q", layer.name);
    let r_name = format!("{}_r", layer.name);
    let q_node = builder.add_input_shred(&q_name, n_vars, committed);
    let r_node = builder.add_input_shred(&r_name, n_vars, committed);
    manifest.insert(q_name, ShredEntry { num_vars: n_vars, visibility: Visibility::Committed });
    manifest.insert(r_name, ShredEntry { num_vars: n_vars, visibility: Visibility::Committed });

    let input_node = tensor_nodes.get(input_name)
        .ok_or_else(|| anyhow::anyhow!("Gemm {} input {} not in tensor_nodes", layer.name, input_name))?;

    let mm_node = builder.add_matmult_node(
        input_node,
        (0, k_vars),
        &weight_node,
        (k_vars, n_vars),
    );

    let rescale_check = builder.add_sector(
        mm_node.expr() + bias_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_check);

    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), q_node.clone());
        tensor_num_vars.insert(out.clone(), n_vars);
    }

    Ok(())
}

fn build_conv_layer(
    builder: &mut CircuitBuilder<Fr>,
    public: &InputLayerNodeRef<Fr>,
    committed: &InputLayerNodeRef<Fr>,
    layer: &crate::onnx::graph::LayerNode,
    tensor_nodes: &mut HashMap<String, NodeRef<Fr>>,
    tensor_num_vars: &mut HashMap<String, usize>,
    tensor_layouts: &mut HashMap<String, SpatialInfo>,
    manifest: &mut ShredManifest,
    alpha_fr: Fr,
) -> Result<()> {
    let input_name = layer.inputs.first()
        .ok_or_else(|| anyhow::anyhow!("Conv {} has no input", layer.name))?;
    let weight_tensor_name = layer.inputs.get(1)
        .ok_or_else(|| anyhow::anyhow!("Conv {} has no weight", layer.name))?;

    let weight_data = layer.weights.get(weight_tensor_name)
        .ok_or_else(|| anyhow::anyhow!("Conv {} missing weight {}", layer.name, weight_tensor_name))?;

    let w_shape = weight_data.shape();
    if w_shape.len() < 4 {
        bail!("Conv {} weight has < 4 dims", layer.name);
    }
    let c_out = w_shape[0];
    let c_in = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];

    let strides = layer.get_ints_attr("strides");
    let stride_h = strides.and_then(|s| s.first()).map(|&v| v as usize).unwrap_or(1);
    let stride_w = strides.and_then(|s| s.get(1)).map(|&v| v as usize).unwrap_or(1);

    let input_layout = tensor_layouts.get(input_name)
        .ok_or_else(|| anyhow::anyhow!("Conv {} input {} has no spatial layout", layer.name, input_name))?
        .clone();

    let (in_h, in_w) = input_layout.hw();
    anyhow::ensure!(in_h >= kh && in_w >= kw, "Conv {}: input {}x{} smaller than kernel {}x{}", layer.name, in_h, in_w, kh, kw);
    let out_h = (in_h - kh) / stride_h + 1;
    let out_w = (in_w - kw) / stride_w + 1;
    let patch_size = c_in * kh * kw;
    let num_patches = out_h * out_w;

    let pad_psize = next_power_of_two(patch_size);
    let pad_cout = next_power_of_two(c_out);

    let im2col_row_vars = num_vars_for(num_patches);
    let im2col_col_vars = num_vars_for(patch_size);
    let im2col_vars = im2col_row_vars + im2col_col_vars;
    let out_col_vars = num_vars_for(c_out);
    let weight_vars = im2col_col_vars + out_col_vars;
    let result_vars = im2col_row_vars + out_col_vars;

    let mut wiring: Vec<(u32, u32)> = Vec::new();
    for oh in 0..out_h {
        for ow in 0..out_w {
            let patch = oh * out_w + ow;
            for c in 0..c_in {
                for kr in 0..kh {
                    for kc in 0..kw {
                        let col = c * kh * kw + kr * kw + kc;
                        let ih = oh * stride_h + kr;
                        let iw = ow * stride_w + kc;
                        let src = input_layout.index(c, ih, iw);
                        let dest = patch * pad_psize + col;
                        wiring.push((dest as u32, src as u32));
                    }
                }
            }
        }
    }

    let input_node = tensor_nodes.get(input_name)
        .ok_or_else(|| anyhow::anyhow!("Conv {} input {} not in tensor_nodes", layer.name, input_name))?;

    let weight_shred_name = format!("{}_weight", layer.name);
    let weight_node = builder.add_input_shred(&weight_shred_name, weight_vars, public);
    manifest.insert(weight_shred_name, ShredEntry { num_vars: weight_vars, visibility: Visibility::Public });

    let bias_shred_name = format!("{}_bias", layer.name);
    let bias_node = builder.add_input_shred(&bias_shred_name, result_vars, public);
    manifest.insert(bias_shred_name, ShredEntry { num_vars: result_vars, visibility: Visibility::Public });

    let q_name = format!("{}_q", layer.name);
    let r_name = format!("{}_r", layer.name);
    let q_node = builder.add_input_shred(&q_name, result_vars, committed);
    let r_node = builder.add_input_shred(&r_name, result_vars, committed);
    manifest.insert(q_name, ShredEntry { num_vars: result_vars, visibility: Visibility::Committed });
    manifest.insert(r_name, ShredEntry { num_vars: result_vars, visibility: Visibility::Committed });

    let im2col_node = builder.add_identity_gate_node(input_node, wiring, im2col_vars, None);
    let mm_node = builder.add_matmult_node(
        &im2col_node,
        (im2col_row_vars, im2col_col_vars),
        &weight_node,
        (im2col_col_vars, out_col_vars),
    );

    let rescale_check = builder.add_sector(
        mm_node.expr() + bias_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_check);

    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), q_node.clone());
        tensor_num_vars.insert(out.clone(), result_vars);
        tensor_layouts.insert(out.clone(), SpatialInfo::HWC {
            h: out_h, w: out_w, c: c_out, stride_c: pad_cout,
        });
    }

    Ok(())
}

fn build_relu_layer(
    builder: &mut CircuitBuilder<Fr>,
    public: &InputLayerNodeRef<Fr>,
    committed: &InputLayerNodeRef<Fr>,
    layer: &crate::onnx::graph::LayerNode,
    tensor_nodes: &mut HashMap<String, NodeRef<Fr>>,
    tensor_num_vars: &mut HashMap<String, usize>,
    tensor_layouts: &mut HashMap<String, SpatialInfo>,
    manifest: &mut ShredManifest,
) -> Result<()> {
    let input_name = layer.inputs.first()
        .ok_or_else(|| anyhow::anyhow!("Relu {} has no input", layer.name))?;

    let nv = tensor_num_vars.get(input_name)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("Relu {} input {} has no num_vars", layer.name, input_name))?;

    let zero_name = format!("{}_zero", layer.name);
    let max_name = format!("{}_max", layer.name);
    let di_name = format!("{}_di", layer.name);
    let dz_name = format!("{}_dz", layer.name);

    let zero_node = builder.add_input_shred(&zero_name, nv, public);
    let max_node = builder.add_input_shred(&max_name, nv, committed);
    let di_node = builder.add_input_shred(&di_name, nv, committed);
    let dz_node = builder.add_input_shred(&dz_name, nv, committed);

    manifest.insert(zero_name, ShredEntry { num_vars: nv, visibility: Visibility::Public });
    manifest.insert(max_name.clone(), ShredEntry { num_vars: nv, visibility: Visibility::Committed });
    manifest.insert(di_name.clone(), ShredEntry { num_vars: nv, visibility: Visibility::Committed });
    manifest.insert(dz_name.clone(), ShredEntry { num_vars: nv, visibility: Visibility::Committed });

    let input_node = tensor_nodes.get(input_name)
        .ok_or_else(|| anyhow::anyhow!("Relu {} input {} not in tensor_nodes", layer.name, input_name))?;

    let c1 = builder.add_sector(max_node.expr() - input_node.expr() - di_node.expr());
    builder.set_output(&c1);

    let c2 = builder.add_sector(max_node.expr() - zero_node.expr() - dz_node.expr());
    builder.set_output(&c2);

    let prod = builder.add_sector(
        AbstractExpression::products(vec![di_node.id(), dz_node.id()]),
    );
    builder.set_output(&prod);

    let layout = tensor_layouts.get(input_name).cloned();

    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), max_node.clone());
        tensor_num_vars.insert(out.clone(), nv);
        if let Some(ref layout) = layout {
            tensor_layouts.insert(out.clone(), layout.clone());
        }
    }

    Ok(())
}

fn build_maxpool_layer(
    builder: &mut CircuitBuilder<Fr>,
    _public: &InputLayerNodeRef<Fr>,
    committed: &InputLayerNodeRef<Fr>,
    layer: &crate::onnx::graph::LayerNode,
    tensor_nodes: &mut HashMap<String, NodeRef<Fr>>,
    tensor_num_vars: &mut HashMap<String, usize>,
    tensor_layouts: &mut HashMap<String, SpatialInfo>,
    manifest: &mut ShredManifest,
) -> Result<()> {
    let input_name = layer.inputs.first()
        .ok_or_else(|| anyhow::anyhow!("MaxPool {} has no input", layer.name))?;

    let input_layout = tensor_layouts.get(input_name)
        .ok_or_else(|| anyhow::anyhow!("MaxPool {} input {} has no spatial layout", layer.name, input_name))?
        .clone();

    let (c, in_h, in_w) = input_layout.spatial_dims();

    let kernel_shape = layer.get_ints_attr("kernel_shape")
        .ok_or_else(|| anyhow::anyhow!("MaxPool {} missing kernel_shape", layer.name))?;
    anyhow::ensure!(kernel_shape.len() >= 2, "MaxPool {} kernel_shape has fewer than 2 dimensions", layer.name);
    let pool_h = kernel_shape[0] as usize;
    let pool_w = kernel_shape[1] as usize;

    let strides = layer.get_ints_attr("strides");
    let stride_h = strides.and_then(|s| s.first()).map(|&v| v as usize).unwrap_or(pool_h);
    let stride_w = strides.and_then(|s| s.get(1)).map(|&v| v as usize).unwrap_or(pool_w);

    anyhow::ensure!(in_h >= pool_h && in_w >= pool_w, "MaxPool {}: input {}x{} smaller than kernel {}x{}", layer.name, in_h, in_w, pool_h, pool_w);
    let pool_oh = (in_h - pool_h) / stride_h + 1;
    let pool_ow = (in_w - pool_w) / stride_w + 1;
    let window_size = pool_h * pool_w;
    let num_pool_out = pool_oh * pool_ow * c;
    let pool_out_vars = num_vars_for(num_pool_out);

    let mut gate_wiring: Vec<Vec<(u32, u32)>> = vec![Vec::new(); window_size];
    for ch in 0..c {
        for poh in 0..pool_oh {
            for pow in 0..pool_ow {
                let dest_idx = (poh * pool_ow + pow) * c + ch;
                for ph in 0..pool_h {
                    for pw in 0..pool_w {
                        let elem_pos = ph * pool_w + pw;
                        let soh = poh * stride_h + ph;
                        let sow = pow * stride_w + pw;
                        let src_idx = input_layout.index(ch, soh, sow);
                        gate_wiring[elem_pos].push((dest_idx as u32, src_idx as u32));
                    }
                }
            }
        }
    }

    let input_node = tensor_nodes.get(input_name)
        .ok_or_else(|| anyhow::anyhow!("MaxPool {} input {} not in tensor_nodes", layer.name, input_name))?;

    let max_name = format!("{}_max", layer.name);
    let max_node = builder.add_input_shred(&max_name, pool_out_vars, committed);
    manifest.insert(max_name, ShredEntry { num_vars: pool_out_vars, visibility: Visibility::Committed });

    let delta_nodes: Vec<_> = (0..window_size).map(|i| {
        let name = format!("{}_d{}", layer.name, i);
        let node = builder.add_input_shred(&name, pool_out_vars, committed);
        manifest.insert(name, ShredEntry { num_vars: pool_out_vars, visibility: Visibility::Committed });
        node
    }).collect();

    let gate_nodes: Vec<_> = (0..window_size).map(|i| {
        builder.add_identity_gate_node(input_node, gate_wiring[i].clone(), pool_out_vars, None)
    }).collect();

    for i in 0..window_size {
        let chk = builder.add_sector(
            max_node.expr() - gate_nodes[i].expr() - delta_nodes[i].expr(),
        );
        builder.set_output(&chk);
    }

    let prod = builder.add_sector(AbstractExpression::products(
        delta_nodes.iter().map(|d| d.id()).collect(),
    ));
    builder.set_output(&prod);

    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), max_node.clone());
        tensor_num_vars.insert(out.clone(), pool_out_vars);
        tensor_layouts.insert(out.clone(), SpatialInfo::HWC {
            h: pool_oh, w: pool_ow, c, stride_c: c,
        });
    }

    Ok(())
}

pub fn transpose_matrix(data: &[i64], rows: usize, cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

pub fn pad_matrix(data: &[i64], orig_rows: usize, orig_cols: usize, pad_rows: usize, pad_cols: usize) -> Vec<i64> {
    debug_assert!(orig_rows <= pad_rows && orig_cols <= pad_cols,
        "pad_matrix: orig {}x{} exceeds target {}x{}", orig_rows, orig_cols, pad_rows, pad_cols);
    let mut out = vec![0i64; pad_rows * pad_cols];
    for r in 0..orig_rows {
        for c in 0..orig_cols {
            out[r * pad_cols + c] = data[r * orig_cols + c];
        }
    }
    out
}

pub fn pad_to_size(data: &[i64], target: usize) -> Vec<i64> {
    let mut padded = data.to_vec();
    padded.resize(target, 0);
    padded
}
