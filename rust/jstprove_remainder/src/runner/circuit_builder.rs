use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Result};
use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::{
    Circuit, CircuitBuilder, InputLayerNodeRef, LayerVisibility, NodeRef,
};
use shared_types::Fr;

use crate::onnx::graph::OpType;
use crate::onnx::quantizer::{QuantizedModel, ScaleConfig};
use crate::padding::{next_power_of_two, num_vars_for};
use crate::util::i64_to_fr;

struct RangeCheckRequest {
    shred_name: String,
    node: NodeRef<Fr>,
    table_nv: usize,
    node_nv: usize,
}

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
    CHW {
        c: usize,
        h: usize,
        w: usize,
    },
    HWC {
        h: usize,
        w: usize,
        c: usize,
        stride_c: usize,
    },
}

impl SpatialInfo {
    pub fn index(&self, c: usize, h: usize, w: usize) -> usize {
        match self {
            SpatialInfo::CHW { h: hd, w: wd, .. } => c * hd * wd + h * wd + w,
            SpatialInfo::HWC {
                w: wd, stride_c, ..
            } => (h * wd + w) * stride_c + c,
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

pub fn delta_table_nv(layer_n_bits: usize, exponent: usize) -> usize {
    layer_n_bits.saturating_sub(exponent).max(1)
}

pub fn compute_range_check_plan(model: &QuantizedModel) -> Result<BTreeMap<usize, Vec<String>>> {
    let exponent = model.scale_config.exponent as usize;
    let mut plan: BTreeMap<usize, Vec<String>> = BTreeMap::new();

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm | OpType::Conv | OpType::BatchNormalization => {
                plan.entry(exponent)
                    .or_default()
                    .push(format!("{}_r", layer.name));
            }
            OpType::Relu => {
                if let Some(n_bits) = layer.n_bits {
                    let dnv = delta_table_nv(n_bits, exponent);
                    let entry = plan.entry(dnv).or_default();
                    entry.push(format!("{}_di", layer.name));
                    entry.push(format!("{}_dz", layer.name));
                }
            }
            OpType::MaxPool => {
                if let Some(n_bits) = layer.n_bits {
                    let dnv = delta_table_nv(n_bits, exponent);
                    let kernel_shape = layer.get_ints_attr("kernel_shape").ok_or_else(|| {
                        anyhow::anyhow!("MaxPool {} missing kernel_shape attribute", layer.name)
                    })?;
                    let window_size: usize = kernel_shape.iter().map(|&v| v as usize).product();
                    let entry = plan.entry(dnv).or_default();
                    for i in 0..window_size {
                        entry.push(format!("{}_d{}", layer.name, i));
                    }
                }
            }
            _ => {}
        }
    }

    Ok(plan)
}

pub fn build_circuit(model: &QuantizedModel, input_size: usize) -> Result<BuildResult> {
    let mut manifest = ShredManifest::new();
    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    anyhow::ensure!(
        model.graph.input_names.len() <= 1,
        "multi-input models not supported (found {} inputs: {:?})",
        model.graph.input_names.len(),
        model.graph.input_names
    );

    let input_vars = num_vars_for(input_size);

    let input_shred_name = model
        .graph
        .input_names
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no input names defined"))?
        .clone();

    let input_node = builder.add_input_shred(&input_shred_name, input_vars, &public);
    manifest.insert(
        input_shred_name.clone(),
        ShredEntry {
            num_vars: input_vars,
            visibility: Visibility::Public,
        },
    );

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
                SpatialInfo::CHW {
                    c: shape[0],
                    h: shape[1],
                    w: shape[2],
                },
            );
        }
    }

    anyhow::ensure!(
        model.graph.output_names.len() == 1,
        "expected exactly 1 output, found {} ({:?})",
        model.graph.output_names.len(),
        model.graph.output_names
    );
    let declared_output = &model.graph.output_names[0];

    anyhow::ensure!(
        model.scale_config.base == 2,
        "LogUp range checks require base == 2, got base = {}",
        model.scale_config.base
    );
    anyhow::ensure!(
        model.scale_config.exponent < 63,
        "scale_config.exponent {} is too large for range table construction",
        model.scale_config.exponent
    );

    let mut range_checks: Vec<RangeCheckRequest> = Vec::new();

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
                    &model.scale_config,
                    &mut range_checks,
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
                    &model.scale_config,
                    &mut range_checks,
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
                    &model.scale_config,
                    &mut range_checks,
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
                    &model.scale_config,
                    &mut range_checks,
                )?;
            }
            OpType::Add | OpType::Sub => {
                build_addsub_layer(
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
            OpType::BatchNormalization => {
                build_batchnorm_layer(
                    &mut builder,
                    &public,
                    &committed,
                    layer,
                    &mut tensor_nodes,
                    &mut tensor_num_vars,
                    &mut tensor_layouts,
                    &mut manifest,
                    &model.scale_config,
                    &mut range_checks,
                )?;
            }
            OpType::Reshape | OpType::Flatten | OpType::Squeeze | OpType::Unsqueeze => {
                let input_name = layer
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("shape op {} has no inputs", layer.name))?;
                let node = tensor_nodes
                    .get(input_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "shape op {} input {} not in tensor_nodes",
                            layer.name,
                            input_name
                        )
                    })?
                    .clone();
                let nv = tensor_num_vars.get(input_name).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "shape op {} input {} has no num_vars",
                        layer.name,
                        input_name
                    )
                })?;
                let layout = tensor_layouts.get(input_name).cloned();
                for out in &layer.outputs {
                    tensor_nodes.insert(out.clone(), node.clone());
                    tensor_num_vars.insert(out.clone(), nv);
                    if let Some(ref layout) = layout {
                        tensor_layouts.insert(out.clone(), layout.clone());
                    }
                }
            }
            other => {
                bail!(
                    "circuit builder: unsupported op type {:?} in layer {}",
                    other,
                    layer.name
                );
            }
        }
    }

    let output_vars = tensor_num_vars
        .get(declared_output)
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "declared output '{}' not found in tensor_num_vars",
                declared_output
            )
        })?;
    let expected_name = "expected_output".to_string();
    let expected_node = builder.add_input_shred(&expected_name, output_vars, &public);
    manifest.insert(
        expected_name,
        ShredEntry {
            num_vars: output_vars,
            visibility: Visibility::Public,
        },
    );

    let output_node = tensor_nodes.get(declared_output).ok_or_else(|| {
        anyhow::anyhow!(
            "declared output '{}' not found in tensor_nodes",
            declared_output
        )
    })?;
    let out_check = builder.add_sector(output_node.expr() - expected_node.expr());
    builder.set_output(&out_check);

    if !range_checks.is_empty() {
        let fs_node = builder.add_fiat_shamir_challenge_node(1);

        let mut checks_by_key: BTreeMap<(usize, usize), Vec<RangeCheckRequest>> = BTreeMap::new();
        for rc in range_checks {
            checks_by_key
                .entry((rc.table_nv, rc.node_nv))
                .or_default()
                .push(rc);
        }

        let mut table_nodes: HashMap<usize, NodeRef<Fr>> = HashMap::new();
        for &(table_nv, _) in checks_by_key.keys() {
            if !table_nodes.contains_key(&table_nv) {
                let table_shred_name = format!("range_table_{}", table_nv);
                let table_node = builder.add_input_shred(&table_shred_name, table_nv, &public);
                manifest.insert(
                    table_shred_name,
                    ShredEntry {
                        num_vars: table_nv,
                        visibility: Visibility::Public,
                    },
                );
                table_nodes.insert(table_nv, table_node);
            }
        }

        for ((table_nv, node_nv), requests) in &checks_by_key {
            let table_node = table_nodes
                .get(table_nv)
                .expect("table_node must exist for table_nv inserted above");
            let lookup_table = builder.add_lookup_table(table_node, &fs_node);

            let real_count = requests.len();
            let target_count = real_count.next_power_of_two();
            let dummy_count = target_count - real_count;

            for i in 0..dummy_count {
                let dummy_name = format!("range_dummy_t{}_n{}_{}", table_nv, node_nv, i);
                let dummy_node = builder.add_input_shred(&dummy_name, *node_nv, &committed);
                manifest.insert(
                    dummy_name.clone(),
                    ShredEntry {
                        num_vars: *node_nv,
                        visibility: Visibility::Committed,
                    },
                );

                let dummy_mults_name = format!("{}_mults", dummy_name);
                let dummy_mults_node =
                    builder.add_input_shred(&dummy_mults_name, *table_nv, &committed);
                manifest.insert(
                    dummy_mults_name,
                    ShredEntry {
                        num_vars: *table_nv,
                        visibility: Visibility::Committed,
                    },
                );

                builder.add_lookup_constraint(&lookup_table, &dummy_node, &dummy_mults_node);
            }

            for rc in requests {
                let mults_name = format!("{}_mults", rc.shred_name);
                let mults_node = builder.add_input_shred(&mults_name, *table_nv, &committed);
                manifest.insert(
                    mults_name,
                    ShredEntry {
                        num_vars: *table_nv,
                        visibility: Visibility::Committed,
                    },
                );

                builder.add_lookup_constraint(&lookup_table, &rc.node, &mults_node);
            }
        }
    }

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
    scale_config: &ScaleConfig,
    range_checks: &mut Vec<RangeCheckRequest>,
) -> Result<()> {
    let alpha_fr = i64_to_fr(scale_config.alpha);
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("Gemm {} has no input", layer.name))?;
    let weight_tensor_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("Gemm {} has no weight input", layer.name))?;

    let weight_data = layer.weights.get(weight_tensor_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Gemm {} missing weight tensor {}",
            layer.name,
            weight_tensor_name
        )
    })?;

    let trans_a = layer
        .get_int_attr("transA")
        .map(|v| v != 0)
        .unwrap_or(false);
    anyhow::ensure!(
        !trans_a,
        "Gemm {} has transA=1 which is not supported",
        layer.name
    );
    let trans_b = layer
        .get_int_attr("transB")
        .map(|v| v != 0)
        .unwrap_or(false);

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
    manifest.insert(
        weight_shred_name,
        ShredEntry {
            num_vars: k_vars + n_vars,
            visibility: Visibility::Public,
        },
    );

    let bias_shred_name = format!("{}_bias", layer.name);
    let bias_node = builder.add_input_shred(&bias_shred_name, n_vars, public);
    manifest.insert(
        bias_shred_name,
        ShredEntry {
            num_vars: n_vars,
            visibility: Visibility::Public,
        },
    );

    let q_name = format!("{}_q", layer.name);
    let r_name = format!("{}_r", layer.name);
    let q_node = builder.add_input_shred(&q_name, n_vars, committed);
    let r_node = builder.add_input_shred(&r_name, n_vars, committed);
    manifest.insert(
        q_name,
        ShredEntry {
            num_vars: n_vars,
            visibility: Visibility::Committed,
        },
    );
    manifest.insert(
        r_name.clone(),
        ShredEntry {
            num_vars: n_vars,
            visibility: Visibility::Committed,
        },
    );

    let input_node = tensor_nodes.get(input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Gemm {} input {} not in tensor_nodes",
            layer.name,
            input_name
        )
    })?;

    let mm_node = builder.add_matmult_node(input_node, (0, k_vars), &weight_node, (k_vars, n_vars));

    let rescale_check = builder.add_sector(
        mm_node.expr() + bias_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_check);

    range_checks.push(RangeCheckRequest {
        shred_name: r_name,
        node: r_node,
        table_nv: scale_config.exponent as usize,
        node_nv: n_vars,
    });

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
    scale_config: &ScaleConfig,
    range_checks: &mut Vec<RangeCheckRequest>,
) -> Result<()> {
    let alpha_fr = i64_to_fr(scale_config.alpha);
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("Conv {} has no input", layer.name))?;
    let weight_tensor_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("Conv {} has no weight", layer.name))?;

    let weight_data = layer.weights.get(weight_tensor_name).ok_or_else(|| {
        anyhow::anyhow!("Conv {} missing weight {}", layer.name, weight_tensor_name)
    })?;

    let w_shape = weight_data.shape();
    if w_shape.len() < 4 {
        bail!("Conv {} weight has < 4 dims", layer.name);
    }
    let c_out = w_shape[0];
    let c_in = w_shape[1];
    let kh = w_shape[2];
    let kw = w_shape[3];

    let pads = layer.get_ints_attr("pads");
    let pad_top = pads.and_then(|p| p.first().copied()).unwrap_or(0) as usize;
    let pad_left = pads.and_then(|p| p.get(1).copied()).unwrap_or(0) as usize;
    let pad_bottom = pads.and_then(|p| p.get(2).copied()).unwrap_or(0) as usize;
    let pad_right = pads.and_then(|p| p.get(3).copied()).unwrap_or(0) as usize;

    let strides = layer.get_ints_attr("strides");
    let stride_h = strides
        .and_then(|s| s.first())
        .map(|&v| v as usize)
        .unwrap_or(1);
    let stride_w = strides
        .and_then(|s| s.get(1))
        .map(|&v| v as usize)
        .unwrap_or(1);

    let input_layout = tensor_layouts
        .get(input_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Conv {} input {} has no spatial layout",
                layer.name,
                input_name
            )
        })?
        .clone();

    let (input_ch, in_h, in_w) = input_layout.spatial_dims();
    anyhow::ensure!(
        input_ch == c_in,
        "Conv {}: weight c_in {} does not match input channels {}",
        layer.name,
        c_in,
        input_ch
    );
    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    anyhow::ensure!(
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
                        let abs_h = oh * stride_h + kr;
                        let abs_w = ow * stride_w + kc;
                        if abs_h < pad_top || abs_w < pad_left {
                            continue;
                        }
                        let ih = abs_h - pad_top;
                        let iw = abs_w - pad_left;
                        if ih >= in_h || iw >= in_w {
                            continue;
                        }
                        let col = c * kh * kw + kr * kw + kc;
                        let src = input_layout.index(c, ih, iw);
                        let dest = patch * pad_psize + col;
                        wiring.push((dest as u32, src as u32));
                    }
                }
            }
        }
    }

    let input_node = tensor_nodes.get(input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Conv {} input {} not in tensor_nodes",
            layer.name,
            input_name
        )
    })?;

    let weight_shred_name = format!("{}_weight", layer.name);
    let weight_node = builder.add_input_shred(&weight_shred_name, weight_vars, public);
    manifest.insert(
        weight_shred_name,
        ShredEntry {
            num_vars: weight_vars,
            visibility: Visibility::Public,
        },
    );

    let bias_shred_name = format!("{}_bias", layer.name);
    let bias_node = builder.add_input_shred(&bias_shred_name, result_vars, public);
    manifest.insert(
        bias_shred_name,
        ShredEntry {
            num_vars: result_vars,
            visibility: Visibility::Public,
        },
    );

    let q_name = format!("{}_q", layer.name);
    let r_name = format!("{}_r", layer.name);
    let q_node = builder.add_input_shred(&q_name, result_vars, committed);
    let r_node = builder.add_input_shred(&r_name, result_vars, committed);
    manifest.insert(
        q_name,
        ShredEntry {
            num_vars: result_vars,
            visibility: Visibility::Committed,
        },
    );
    manifest.insert(
        r_name.clone(),
        ShredEntry {
            num_vars: result_vars,
            visibility: Visibility::Committed,
        },
    );

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

    range_checks.push(RangeCheckRequest {
        shred_name: r_name,
        node: r_node,
        table_nv: scale_config.exponent as usize,
        node_nv: result_vars,
    });

    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), q_node.clone());
        tensor_num_vars.insert(out.clone(), result_vars);
        tensor_layouts.insert(
            out.clone(),
            SpatialInfo::HWC {
                h: out_h,
                w: out_w,
                c: c_out,
                stride_c: pad_cout,
            },
        );
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
    scale_config: &ScaleConfig,
    range_checks: &mut Vec<RangeCheckRequest>,
) -> Result<()> {
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("Relu {} has no input", layer.name))?;

    let nv = tensor_num_vars.get(input_name).copied().ok_or_else(|| {
        anyhow::anyhow!("Relu {} input {} has no num_vars", layer.name, input_name)
    })?;

    let zero_name = format!("{}_zero", layer.name);
    let max_name = format!("{}_max", layer.name);
    let di_name = format!("{}_di", layer.name);
    let dz_name = format!("{}_dz", layer.name);

    let zero_node = builder.add_input_shred(&zero_name, nv, public);
    let max_node = builder.add_input_shred(&max_name, nv, committed);
    let di_node = builder.add_input_shred(&di_name, nv, committed);
    let dz_node = builder.add_input_shred(&dz_name, nv, committed);

    manifest.insert(
        zero_name,
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Public,
        },
    );
    manifest.insert(
        max_name.clone(),
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Committed,
        },
    );
    manifest.insert(
        di_name.clone(),
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Committed,
        },
    );
    manifest.insert(
        dz_name.clone(),
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Committed,
        },
    );

    let input_node = tensor_nodes.get(input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "Relu {} input {} not in tensor_nodes",
            layer.name,
            input_name
        )
    })?;

    let c1 = builder.add_sector(max_node.expr() - input_node.expr() - di_node.expr());
    builder.set_output(&c1);

    let c2 = builder.add_sector(max_node.expr() - zero_node.expr() - dz_node.expr());
    builder.set_output(&c2);

    let prod = builder.add_sector(AbstractExpression::products(vec![
        di_node.id(),
        dz_node.id(),
    ]));
    builder.set_output(&prod);

    if let Some(n_bits) = layer.n_bits {
        let dnv = delta_table_nv(n_bits, scale_config.exponent as usize);
        range_checks.push(RangeCheckRequest {
            shred_name: di_name,
            node: di_node,
            table_nv: dnv,
            node_nv: nv,
        });
        range_checks.push(RangeCheckRequest {
            shred_name: dz_name,
            node: dz_node,
            table_nv: dnv,
            node_nv: nv,
        });
    }

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
    scale_config: &ScaleConfig,
    range_checks: &mut Vec<RangeCheckRequest>,
) -> Result<()> {
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("MaxPool {} has no input", layer.name))?;

    let input_layout = tensor_layouts
        .get(input_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "MaxPool {} input {} has no spatial layout",
                layer.name,
                input_name
            )
        })?
        .clone();

    let (c, in_h, in_w) = input_layout.spatial_dims();

    let kernel_shape = layer
        .get_ints_attr("kernel_shape")
        .ok_or_else(|| anyhow::anyhow!("MaxPool {} missing kernel_shape", layer.name))?;
    anyhow::ensure!(
        kernel_shape.len() >= 2,
        "MaxPool {} kernel_shape has fewer than 2 dimensions",
        layer.name
    );
    let pool_h = kernel_shape[0] as usize;
    let pool_w = kernel_shape[1] as usize;

    let strides = layer.get_ints_attr("strides");
    let stride_h = strides
        .and_then(|s| s.first())
        .map(|&v| v as usize)
        .unwrap_or(1);
    let stride_w = strides
        .and_then(|s| s.get(1))
        .map(|&v| v as usize)
        .unwrap_or(1);

    anyhow::ensure!(
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

    let input_node = tensor_nodes.get(input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "MaxPool {} input {} not in tensor_nodes",
            layer.name,
            input_name
        )
    })?;

    let max_name = format!("{}_max", layer.name);
    let max_node = builder.add_input_shred(&max_name, pool_out_vars, committed);
    manifest.insert(
        max_name,
        ShredEntry {
            num_vars: pool_out_vars,
            visibility: Visibility::Committed,
        },
    );

    let delta_nodes: Vec<_> = (0..window_size)
        .map(|i| {
            let name = format!("{}_d{}", layer.name, i);
            let node = builder.add_input_shred(&name, pool_out_vars, committed);
            manifest.insert(
                name,
                ShredEntry {
                    num_vars: pool_out_vars,
                    visibility: Visibility::Committed,
                },
            );
            node
        })
        .collect();

    let gate_nodes: Vec<_> = (0..window_size)
        .map(|i| {
            builder.add_identity_gate_node(input_node, gate_wiring[i].clone(), pool_out_vars, None)
        })
        .collect();

    for i in 0..window_size {
        let chk =
            builder.add_sector(max_node.expr() - gate_nodes[i].expr() - delta_nodes[i].expr());
        builder.set_output(&chk);
    }

    let prod = builder.add_sector(AbstractExpression::products(
        delta_nodes.iter().map(|d| d.id()).collect(),
    ));
    builder.set_output(&prod);

    if let Some(n_bits) = layer.n_bits {
        let dnv = delta_table_nv(n_bits, scale_config.exponent as usize);
        for (i, dnode) in delta_nodes.into_iter().enumerate() {
            range_checks.push(RangeCheckRequest {
                shred_name: format!("{}_d{}", layer.name, i),
                node: dnode,
                table_nv: dnv,
                node_nv: pool_out_vars,
            });
        }
    }

    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), max_node.clone());
        tensor_num_vars.insert(out.clone(), pool_out_vars);
        tensor_layouts.insert(
            out.clone(),
            SpatialInfo::HWC {
                h: pool_oh,
                w: pool_ow,
                c,
                stride_c: c,
            },
        );
    }

    Ok(())
}

fn build_batchnorm_layer(
    builder: &mut CircuitBuilder<Fr>,
    public: &InputLayerNodeRef<Fr>,
    committed: &InputLayerNodeRef<Fr>,
    layer: &crate::onnx::graph::LayerNode,
    tensor_nodes: &mut HashMap<String, NodeRef<Fr>>,
    tensor_num_vars: &mut HashMap<String, usize>,
    tensor_layouts: &mut HashMap<String, SpatialInfo>,
    manifest: &mut ShredManifest,
    scale_config: &ScaleConfig,
    range_checks: &mut Vec<RangeCheckRequest>,
) -> Result<()> {
    let alpha_fr = i64_to_fr(scale_config.alpha);
    let input_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("BatchNorm {} has no input", layer.name))?;

    let nv = tensor_num_vars.get(input_name).copied().ok_or_else(|| {
        anyhow::anyhow!(
            "BatchNorm {} input {} has no num_vars",
            layer.name,
            input_name
        )
    })?;

    let mul_name = format!("{}_mul", layer.name);
    let add_name = format!("{}_add", layer.name);
    let q_name = format!("{}_q", layer.name);
    let r_name = format!("{}_r", layer.name);

    let mul_node = builder.add_input_shred(&mul_name, nv, public);
    let add_node = builder.add_input_shred(&add_name, nv, public);
    let q_node = builder.add_input_shred(&q_name, nv, committed);
    let r_node = builder.add_input_shred(&r_name, nv, committed);

    manifest.insert(
        mul_name,
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Public,
        },
    );
    manifest.insert(
        add_name,
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Public,
        },
    );
    manifest.insert(
        q_name,
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Committed,
        },
    );
    manifest.insert(
        r_name.clone(),
        ShredEntry {
            num_vars: nv,
            visibility: Visibility::Committed,
        },
    );

    let input_node = tensor_nodes.get(input_name).ok_or_else(|| {
        anyhow::anyhow!(
            "BatchNorm {} input {} not in tensor_nodes",
            layer.name,
            input_name
        )
    })?;

    let rescale_check = builder.add_sector(
        AbstractExpression::products(vec![input_node.id(), mul_node.id()]) + add_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_check);

    range_checks.push(RangeCheckRequest {
        shred_name: r_name,
        node: r_node,
        table_nv: scale_config.exponent as usize,
        node_nv: nv,
    });

    let layout = tensor_layouts.get(input_name).cloned();
    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), q_node.clone());
        tensor_num_vars.insert(out.clone(), nv);
        if let Some(ref layout) = layout {
            tensor_layouts.insert(out.clone(), layout.clone());
        }
    }

    Ok(())
}

fn build_addsub_layer(
    builder: &mut CircuitBuilder<Fr>,
    public: &InputLayerNodeRef<Fr>,
    committed: &InputLayerNodeRef<Fr>,
    layer: &crate::onnx::graph::LayerNode,
    tensor_nodes: &mut HashMap<String, NodeRef<Fr>>,
    tensor_num_vars: &mut HashMap<String, usize>,
    tensor_layouts: &mut HashMap<String, SpatialInfo>,
    manifest: &mut ShredManifest,
) -> Result<()> {
    let input_a_name = layer
        .inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("{:?} {} has no first input", layer.op_type, layer.name))?;
    let input_b_name = layer
        .inputs
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("{:?} {} has no second input", layer.op_type, layer.name))?;

    let a_is_tensor = tensor_nodes.contains_key(input_a_name);
    let b_is_tensor = tensor_nodes.contains_key(input_b_name);

    anyhow::ensure!(
        a_is_tensor || b_is_tensor,
        "{:?} {} has no computed tensor inputs",
        layer.op_type,
        layer.name
    );

    let mut resolve_input = |name: &str, is_tensor: bool| -> Result<(NodeRef<Fr>, usize)> {
        if is_tensor {
            let node = tensor_nodes
                .get(name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{:?} {} input {} not in tensor_nodes",
                        layer.op_type,
                        layer.name,
                        name
                    )
                })?
                .clone();
            let nv = tensor_num_vars.get(name).copied().ok_or_else(|| {
                anyhow::anyhow!(
                    "{:?} {} input {} has no num_vars",
                    layer.op_type,
                    layer.name,
                    name
                )
            })?;
            Ok((node, nv))
        } else {
            let weight = layer.weights.get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "{:?} {} input {} not in tensor_nodes or weights",
                    layer.op_type,
                    layer.name,
                    name
                )
            })?;
            let data = weight.as_i64_vec();
            let nv = num_vars_for(data.len());
            let shred_name = format!("{}_{}", layer.name, name);
            let node = builder.add_input_shred(&shred_name, nv, public);
            manifest.insert(
                shred_name,
                ShredEntry {
                    num_vars: nv,
                    visibility: Visibility::Public,
                },
            );
            Ok((node, nv))
        }
    };

    let (a_node, a_nv) = resolve_input(input_a_name, a_is_tensor)?;
    let (b_node, b_nv) = resolve_input(input_b_name, b_is_tensor)?;

    let out_nv = a_nv.max(b_nv);

    let result_name = format!("{}_result", layer.name);
    let result_node = builder.add_input_shred(&result_name, out_nv, committed);
    manifest.insert(
        result_name,
        ShredEntry {
            num_vars: out_nv,
            visibility: Visibility::Committed,
        },
    );

    let constraint = if layer.op_type == OpType::Add {
        builder.add_sector(a_node.expr() + b_node.expr() - result_node.expr())
    } else {
        builder.add_sector(a_node.expr() - b_node.expr() - result_node.expr())
    };
    builder.set_output(&constraint);

    let layout = tensor_layouts
        .get(input_a_name)
        .or_else(|| tensor_layouts.get(input_b_name))
        .cloned();
    for out in &layer.outputs {
        tensor_nodes.insert(out.clone(), result_node.clone());
        tensor_num_vars.insert(out.clone(), out_nv);
        if let Some(ref layout) = layout {
            tensor_layouts.insert(out.clone(), layout.clone());
        }
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

pub fn pad_matrix(
    data: &[i64],
    orig_rows: usize,
    orig_cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> anyhow::Result<Vec<i64>> {
    anyhow::ensure!(
        orig_rows <= pad_rows && orig_cols <= pad_cols,
        "pad_matrix: orig {}x{} exceeds target {}x{}",
        orig_rows,
        orig_cols,
        pad_rows,
        pad_cols
    );
    let mut out = vec![0i64; pad_rows * pad_cols];
    for r in 0..orig_rows {
        for c in 0..orig_cols {
            out[r * pad_cols + c] = data[r * orig_cols + c];
        }
    }
    Ok(out)
}

pub fn pad_to_size(data: &[i64], target: usize) -> Vec<i64> {
    let mut padded = data.to_vec();
    padded.resize(target, 0);
    padded
}
