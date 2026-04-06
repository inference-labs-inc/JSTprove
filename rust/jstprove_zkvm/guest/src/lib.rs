#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;
use alloc::vec::Vec;

pub const OP_ADD: u8 = 0;
pub const OP_DIV: u8 = 1;
pub const OP_SUB: u8 = 2;
pub const OP_MUL: u8 = 3;
pub const OP_GEMM: u8 = 4;
pub const OP_CONV: u8 = 5;
pub const OP_RELU: u8 = 6;
pub const OP_MAXPOOL: u8 = 7;
pub const OP_BATCHNORM: u8 = 8;
pub const OP_MAX: u8 = 9;
pub const OP_MIN: u8 = 10;
pub const OP_CAST: u8 = 11;
pub const OP_CLIP: u8 = 12;
pub const OP_EXP: u8 = 13;
pub const OP_RESHAPE: u8 = 14;
pub const OP_FLATTEN: u8 = 15;
pub const OP_SQUEEZE: u8 = 16;
pub const OP_UNSQUEEZE: u8 = 17;
pub const OP_CONSTANT: u8 = 18;
pub const OP_SOFTMAX: u8 = 19;
pub const OP_SIGMOID: u8 = 20;
pub const OP_GELU: u8 = 21;
pub const OP_TILE: u8 = 22;
pub const OP_GATHER: u8 = 23;
pub const OP_LAYERNORM: u8 = 24;
pub const OP_RESIZE: u8 = 25;
pub const OP_GRIDSAMPLE: u8 = 26;
pub const OP_TRANSPOSE: u8 = 27;
pub const OP_CONCAT: u8 = 28;
pub const OP_SLICE: u8 = 29;
pub const OP_TOPK: u8 = 30;
pub const OP_SHAPE: u8 = 31;
pub const OP_LOG: u8 = 32;
pub const OP_EXPAND: u8 = 33;
pub const OP_REDUCEMEAN: u8 = 34;

pub const MAGIC: &[u8; 8] = b"JSTCAN01";

#[derive(Clone)]
pub struct LayerDesc {
    pub op_type: u8,
    pub needs_rescale: bool,
    pub n_inputs: u32,
    pub weight_dims: Vec<u32>,
    pub weight_data: Vec<i64>,
    pub bias_dims: Vec<u32>,
    pub bias_data: Vec<i64>,
    pub int_attrs: Vec<i64>,
    pub extra_constant_data: Vec<i64>,
}

fn abs_u64(x: i64) -> u64 {
    x.unsigned_abs()
}

fn max_channel_l1(weights: &[i64], out_channels: usize) -> u64 {
    if out_channels == 0 || weights.is_empty() {
        return 0;
    }
    let per_channel = weights.len() / out_channels;
    let mut max_l1: u64 = 0;
    for oc in 0..out_channels {
        let start = oc * per_channel;
        let end = start + per_channel;
        let l1: u64 = weights[start..end].iter().map(|v| abs_u64(*v)).sum();
        if l1 > max_l1 {
            max_l1 = l1;
        }
    }
    max_l1
}

fn bias_max_abs(bias: &[i64]) -> u64 {
    bias.iter().map(|v| abs_u64(*v)).max().unwrap_or(0)
}

fn integer_log2_ceil(val: u64) -> u32 {
    if val <= 1 {
        return 0;
    }
    64 - (val - 1).leading_zeros()
}

fn compute_n_bits_int(_alpha: u64, bound_int: u64) -> u32 {
    if bound_int == 0 {
        return 2;
    }
    let val = bound_int;
    if val <= 1 {
        return 2;
    }
    integer_log2_ceil(val) + 1
}

fn compute_bounds_and_nbits(layers: &[LayerDesc], alpha: u64) -> Vec<u32> {
    let mut nbits_out = Vec::with_capacity(layers.len());
    let mut prev_bound_scaled: u64 = alpha;

    for layer in layers {
        let op = layer.op_type;
        match op {
            OP_CONV | OP_GEMM => {
                let out_ch = if layer.weight_dims.len() >= 2 {
                    layer.weight_dims[0] as usize
                } else {
                    1
                };
                let l1 = max_channel_l1(&layer.weight_data, out_ch);
                let bmax = bias_max_abs(&layer.bias_data);

                let bound_scaled = l1
                    .saturating_mul(prev_bound_scaled / alpha)
                    .saturating_add(bmax / alpha);

                let nb = compute_n_bits_int(alpha, bound_scaled);
                nbits_out.push(nb);
                prev_bound_scaled = alpha;
            }
            OP_BATCHNORM => {
                let mul_max: u64 = if !layer.weight_data.is_empty() {
                    layer
                        .weight_data
                        .iter()
                        .map(|v| abs_u64(*v))
                        .max()
                        .unwrap_or(alpha)
                } else {
                    alpha
                };
                let bmax = bias_max_abs(&layer.bias_data);
                let bound_scaled = (mul_max / alpha)
                    .saturating_mul(prev_bound_scaled)
                    .saturating_add(bmax / alpha);
                let nb = compute_n_bits_int(alpha, bound_scaled);
                nbits_out.push(nb);
                prev_bound_scaled = alpha;
            }
            OP_MUL => {
                nbits_out.push(0);
                prev_bound_scaled = alpha;
            }
            OP_ADD | OP_SUB => {
                nbits_out.push(0);
            }
            OP_DIV => {
                nbits_out.push(0);
            }
            OP_RELU | OP_MAXPOOL | OP_MAX | OP_MIN | OP_CLIP => {
                let nb = compute_n_bits_int(alpha, prev_bound_scaled);
                nbits_out.push(nb);
            }
            OP_EXP | OP_SOFTMAX | OP_SIGMOID | OP_GELU => {
                let nb = compute_n_bits_int(alpha, prev_bound_scaled);
                nbits_out.push(nb);
            }
            OP_RESIZE | OP_GRIDSAMPLE | OP_TOPK => {
                let nb = compute_n_bits_int(alpha, prev_bound_scaled);
                nbits_out.push(nb);
            }
            _ => {
                nbits_out.push(0);
            }
        }
    }
    nbits_out
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_i64(buf: &mut Vec<u8>, v: i64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn canonical_output(
    layers: &[LayerDesc],
    alpha: u64,
    base: u64,
    exponent: u32,
    input_dims: &[u32],
    nbits: &[u32],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    write_u64(&mut buf, base);
    write_u32(&mut buf, exponent);
    write_u64(&mut buf, alpha);
    write_u32(&mut buf, input_dims.len() as u32);
    for &d in input_dims {
        write_u32(&mut buf, d);
    }
    write_u32(&mut buf, layers.len() as u32);
    for (i, layer) in layers.iter().enumerate() {
        buf.push(layer.op_type);
        buf.push(if layer.needs_rescale { 1 } else { 0 });
        write_u32(&mut buf, layer.n_inputs);

        write_u32(&mut buf, layer.weight_dims.len() as u32);
        for &d in &layer.weight_dims {
            write_u32(&mut buf, d);
        }
        write_u32(&mut buf, layer.weight_data.len() as u32);
        for &v in &layer.weight_data {
            write_i64(&mut buf, v);
        }
        write_u32(&mut buf, layer.bias_dims.len() as u32);
        for &d in &layer.bias_dims {
            write_u32(&mut buf, d);
        }
        write_u32(&mut buf, layer.bias_data.len() as u32);
        for &v in &layer.bias_data {
            write_i64(&mut buf, v);
        }
        write_u32(&mut buf, layer.int_attrs.len() as u32);
        for &v in &layer.int_attrs {
            write_i64(&mut buf, v);
        }
        write_u32(&mut buf, layer.extra_constant_data.len() as u32);
        for &v in &layer.extra_constant_data {
            write_i64(&mut buf, v);
        }
        write_u32(&mut buf, nbits[i]);
    }
    buf
}

fn read_u32(data: &[u8], pos: &mut usize) -> u32 {
    assert!(
        data.len().saturating_sub(*pos) >= 4,
        "read_u32: need 4 bytes at offset {}, have {}",
        *pos,
        data.len()
    );
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    v
}

fn read_u64(data: &[u8], pos: &mut usize) -> u64 {
    assert!(
        data.len().saturating_sub(*pos) >= 8,
        "read_u64: need 8 bytes at offset {}, have {}",
        *pos,
        data.len()
    );
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    v
}

fn read_i64(data: &[u8], pos: &mut usize) -> i64 {
    assert!(
        data.len().saturating_sub(*pos) >= 8,
        "read_i64: need 8 bytes at offset {}, have {}",
        *pos,
        data.len()
    );
    let v = i64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    v
}

fn decode_layers(data: &[u8], pos: &mut usize) -> Vec<LayerDesc> {
    let n = read_u32(data, pos) as usize;
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        assert!(
            *pos < data.len(),
            "read_u8: need 1 byte at offset {}, have {}",
            *pos,
            data.len()
        );
        let op_type = data[*pos];
        *pos += 1;
        assert!(
            *pos < data.len(),
            "read_u8: need 1 byte at offset {}, have {}",
            *pos,
            data.len()
        );
        let needs_rescale = data[*pos] != 0;
        *pos += 1;
        let n_inputs = read_u32(data, pos);

        let nwd = read_u32(data, pos) as usize;
        let mut weight_dims = Vec::with_capacity(nwd);
        for _ in 0..nwd {
            weight_dims.push(read_u32(data, pos));
        }
        let nw = read_u32(data, pos) as usize;
        let mut weight_data = Vec::with_capacity(nw);
        for _ in 0..nw {
            weight_data.push(read_i64(data, pos));
        }
        let nbd = read_u32(data, pos) as usize;
        let mut bias_dims = Vec::with_capacity(nbd);
        for _ in 0..nbd {
            bias_dims.push(read_u32(data, pos));
        }
        let nb = read_u32(data, pos) as usize;
        let mut bias_data = Vec::with_capacity(nb);
        for _ in 0..nb {
            bias_data.push(read_i64(data, pos));
        }
        let na = read_u32(data, pos) as usize;
        let mut int_attrs = Vec::with_capacity(na);
        for _ in 0..na {
            int_attrs.push(read_i64(data, pos));
        }
        let ne = read_u32(data, pos) as usize;
        let mut extra_constant_data = Vec::with_capacity(ne);
        for _ in 0..ne {
            extra_constant_data.push(read_i64(data, pos));
        }
        layers.push(LayerDesc {
            op_type,
            needs_rescale,
            n_inputs,
            weight_dims,
            weight_data,
            bias_dims,
            bias_data,
            int_attrs,
            extra_constant_data,
        });
    }
    layers
}

#[jolt::provable(
    heap_size = 8388608,
    max_input_size = 524288,
    max_output_size = 256,
    max_trace_length = 67108864
)]
fn verify_compilation(model_bytes: &[u8]) -> [u8; 32] {
    let mut pos = 0;
    let base = read_u64(model_bytes, &mut pos);
    let exponent = read_u32(model_bytes, &mut pos);
    let alpha = read_u64(model_bytes, &mut pos);

    let n_dims = read_u32(model_bytes, &mut pos) as usize;
    let mut input_dims = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        input_dims.push(read_u32(model_bytes, &mut pos));
    }

    let layers = decode_layers(model_bytes, &mut pos);
    let nbits = compute_bounds_and_nbits(&layers, alpha);
    let canonical = canonical_output(&layers, alpha, base, exponent, &input_dims, &nbits);
    jolt_inlines_sha2::Sha256::digest(&canonical)
}
