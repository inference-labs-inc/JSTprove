use std::collections::{BTreeMap, HashMap};

use anyhow::Result;

use crate::onnx::graph::OpType;
use crate::onnx::quantizer::QuantizedModel;

pub const RANGE_CHECK_CHUNK_BITS: usize = 9;

pub fn delta_table_nv(layer_n_bits: usize, exponent: usize) -> usize {
    layer_n_bits.saturating_sub(exponent).max(1)
}

pub fn compute_multiplicities(values: &[i64], table_size: usize) -> Result<Vec<i64>> {
    let mut mults = vec![0i64; table_size];
    for (i, &v) in values.iter().enumerate() {
        anyhow::ensure!(
            v >= 0 && (v as usize) < table_size,
            "range check: value at index {i} is {v} which is outside table range [0, {table_size})"
        );
        mults[v as usize] += 1;
    }
    Ok(mults)
}

pub fn split_and_insert_range_shreds(
    shreds: &mut HashMap<String, Vec<i64>>,
    prefix: &str,
    values: &[i64],
    exponent: usize,
) -> Result<()> {
    if exponent > RANGE_CHECK_CHUNK_BITS {
        let chunk_scale = 1i64 << RANGE_CHECK_CHUNK_BITS;
        let chunk_table_size = 1usize << RANGE_CHECK_CHUNK_BITS;
        let hi_table_size = 1usize << (exponent - RANGE_CHECK_CHUNK_BITS);
        let c0: Vec<i64> = values.iter().map(|&v| v % chunk_scale).collect();
        let c1: Vec<i64> = values.iter().map(|&v| v / chunk_scale).collect();
        shreds.insert(format!("{prefix}_c0"), c0.clone());
        shreds.insert(format!("{prefix}_c1"), c1.clone());
        shreds.insert(
            format!("{prefix}_c0_mults"),
            compute_multiplicities(&c0, chunk_table_size)?,
        );
        shreds.insert(
            format!("{prefix}_c1_mults"),
            compute_multiplicities(&c1, hi_table_size)?,
        );
    } else {
        let table_size = 1usize << exponent;
        shreds.insert(
            format!("{prefix}_mults"),
            compute_multiplicities(values, table_size)?,
        );
    }
    Ok(())
}

pub fn compute_range_check_plan(
    model: &QuantizedModel,
    observed_n_bits: Option<&HashMap<String, usize>>,
) -> Result<BTreeMap<usize, Vec<String>>> {
    let exponent = model.scale_config.exponent as usize;
    let mut plan: BTreeMap<usize, Vec<String>> = BTreeMap::new();

    for layer in model.graph.iter_topo() {
        match layer.op_type {
            OpType::Gemm | OpType::Conv | OpType::BatchNormalization => {
                if exponent <= RANGE_CHECK_CHUNK_BITS {
                    plan.entry(exponent)
                        .or_default()
                        .push(format!("{}_r", layer.name));
                } else {
                    anyhow::ensure!(
                        exponent <= 2 * RANGE_CHECK_CHUNK_BITS,
                        "layer {} exponent {} exceeds two-chunk range-check capacity (max {} bits)",
                        layer.name,
                        exponent,
                        2 * RANGE_CHECK_CHUNK_BITS
                    );
                    plan.entry(RANGE_CHECK_CHUNK_BITS)
                        .or_default()
                        .push(format!("{}_r_c0", layer.name));
                    plan.entry(exponent - RANGE_CHECK_CHUNK_BITS)
                        .or_default()
                        .push(format!("{}_r_c1", layer.name));
                }
            }
            OpType::Relu => {
                let n_bits = observed_n_bits
                    .and_then(|m| m.get(&layer.name).copied())
                    .or(layer.n_bits);
                if let Some(n_bits) = n_bits {
                    let dnv = delta_table_nv(n_bits, exponent);
                    if dnv > RANGE_CHECK_CHUNK_BITS {
                        anyhow::ensure!(
                            dnv <= 2 * RANGE_CHECK_CHUNK_BITS,
                            "Relu layer {} delta nv {} exceeds two-chunk range-check capacity (max {} bits)",
                            layer.name,
                            dnv,
                            2 * RANGE_CHECK_CHUNK_BITS
                        );
                        plan.entry(RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_di_c0", layer.name));
                        plan.entry(RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_dz_c0", layer.name));
                        plan.entry(dnv - RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_di_c1", layer.name));
                        plan.entry(dnv - RANGE_CHECK_CHUNK_BITS)
                            .or_default()
                            .push(format!("{}_dz_c1", layer.name));
                    } else {
                        let entry = plan.entry(dnv).or_default();
                        entry.push(format!("{}_di", layer.name));
                        entry.push(format!("{}_dz", layer.name));
                    }
                }
            }
            OpType::MaxPool => {
                let n_bits = observed_n_bits
                    .and_then(|m| m.get(&layer.name).copied())
                    .or(layer.n_bits);
                if let Some(n_bits) = n_bits {
                    let dnv = delta_table_nv(n_bits, exponent);
                    let kernel_shape = layer.get_ints_attr("kernel_shape").ok_or_else(|| {
                        anyhow::anyhow!("MaxPool {} missing kernel_shape attribute", layer.name)
                    })?;
                    anyhow::ensure!(
                        kernel_shape.len() == 2,
                        "MaxPool {} requires exactly 2D kernel_shape, got {} dims",
                        layer.name,
                        kernel_shape.len()
                    );
                    anyhow::ensure!(
                        kernel_shape[0] > 0 && kernel_shape[1] > 0,
                        "MaxPool {} kernel_shape values must be positive, got [{}, {}]",
                        layer.name,
                        kernel_shape[0],
                        kernel_shape[1]
                    );
                    let window_size = (kernel_shape[0] as usize)
                        .checked_mul(kernel_shape[1] as usize)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "MaxPool {} kernel_shape overflow: {} * {}",
                                layer.name,
                                kernel_shape[0],
                                kernel_shape[1]
                            )
                        })?;
                    if dnv > RANGE_CHECK_CHUNK_BITS {
                        anyhow::ensure!(
                            dnv <= 2 * RANGE_CHECK_CHUNK_BITS,
                            "MaxPool layer {} delta nv {} exceeds two-chunk range-check capacity (max {} bits)",
                            layer.name,
                            dnv,
                            2 * RANGE_CHECK_CHUNK_BITS
                        );
                        for i in 0..window_size {
                            plan.entry(RANGE_CHECK_CHUNK_BITS)
                                .or_default()
                                .push(format!("{}_d{}_c0", layer.name, i));
                            plan.entry(dnv - RANGE_CHECK_CHUNK_BITS)
                                .or_default()
                                .push(format!("{}_d{}_c1", layer.name, i));
                        }
                    } else {
                        let entry = plan.entry(dnv).or_default();
                        for i in 0..window_size {
                            entry.push(format!("{}_d{}", layer.name, i));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(plan)
}
