use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use tiny_keccak::{Hasher, Sha3};

use crate::circuit_functions::utils::onnx_model::{Architecture, CircuitParams};

use super::DEFAULT_LOGUP_CHUNK_BITS;

const CANDIDATES: &[usize] = &[10, 11, 12, 13, 14];

const CACHE_SUBDIR: &str = "jstprove/chunk_width";

fn cache_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Caches").join(CACHE_SUBDIR))
    } else {
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .map(|h| PathBuf::from(h).join(".cache"))
            })
            .map(|d| d.join(CACHE_SUBDIR))
    }
}

fn sha3_hex(data: &[u8]) -> String {
    let mut hasher = Sha3::v256();
    hasher.update(data);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    out[..16]
        .iter()
        .fold(String::with_capacity(32), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn version_tag() -> String {
    format!(
        "{}-{}",
        env!("CARGO_PKG_VERSION"),
        option_env!("JSTPROVE_GIT_REV").unwrap_or("dev")
    )
}

#[must_use]
pub fn circuit_cache_key(params: &CircuitParams, architecture: &Architecture) -> String {
    let mut buf = Vec::new();
    buf.extend_from_slice(version_tag().as_bytes());
    buf.push(0);
    buf.extend_from_slice(&params.scale_exponent.to_le_bytes());
    buf.extend_from_slice(&params.scale_base.to_le_bytes());
    let rescale_json = serde_json::to_string(&params.rescale_config).unwrap_or_default();
    buf.extend_from_slice(rescale_json.as_bytes());
    buf.push(0);
    let nbits_json = serde_json::to_string(&params.n_bits_config).unwrap_or_default();
    buf.extend_from_slice(nbits_json.as_bytes());
    buf.push(0);
    buf.push(u8::from(params.weights_as_inputs));
    for layer in &architecture.architecture {
        buf.extend_from_slice(layer.name.as_bytes());
        buf.push(0);
        buf.extend_from_slice(layer.op_type.as_bytes());
        buf.push(0);
        for output_name in &layer.outputs {
            buf.extend_from_slice(output_name.as_bytes());
            if let Some(shape) = layer.shape.get(output_name) {
                for &dim in shape {
                    buf.extend_from_slice(&dim.to_le_bytes());
                }
            }
            buf.push(0);
        }
        buf.push(0);
    }
    sha3_hex(&buf)
}

#[must_use]
pub fn operator_cache_key(
    params: &CircuitParams,
    architecture: &Architecture,
    n_bits: usize,
) -> String {
    let mut profile: BTreeMap<&str, usize> = BTreeMap::new();
    let shapes = crate::circuit_functions::utils::onnx_model::collect_all_shapes(
        &architecture.architecture,
        &params.inputs,
    );
    for layer in &architecture.architecture {
        let is_rescale = params
            .rescale_config
            .get(&layer.name)
            .copied()
            .unwrap_or(true);
        if !is_rescale {
            continue;
        }
        let mut elems = 0usize;
        for output_name in &layer.outputs {
            if let Some(shape) = shapes.get(output_name) {
                elems += shape.iter().product::<usize>();
            }
        }
        *profile.entry(layer.op_type.as_str()).or_default() += elems;
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(version_tag().as_bytes());
    buf.push(0);
    buf.extend_from_slice(&params.scale_exponent.to_le_bytes());
    #[allow(clippy::cast_possible_truncation)]
    buf.extend_from_slice(&(n_bits as u32).to_le_bytes());
    for (op, count) in &profile {
        buf.extend_from_slice(op.as_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
    }
    sha3_hex(&buf)
}

fn read_cache(key: &str) -> Option<usize> {
    let dir = cache_dir()?;
    let path = dir.join(key);
    let contents = fs::read_to_string(path).ok()?;
    contents.trim().parse().ok()
}

fn write_cache(key: &str, chunk_bits: usize) {
    let Some(dir) = cache_dir() else { return };
    let _ = fs::create_dir_all(&dir);
    let _ = fs::write(dir.join(key), chunk_bits.to_string());
}

pub fn sweep_and_select<F>(candidates: &[usize], mut compile_cost: F) -> usize
where
    F: FnMut(usize) -> Option<usize>,
{
    let mut best_bits = DEFAULT_LOGUP_CHUNK_BITS;
    let mut best_cost = usize::MAX;
    for &c in candidates {
        if let Some(cost) = compile_cost(c) {
            if cost < best_cost {
                best_cost = cost;
                best_bits = c;
            }
        }
    }
    best_bits
}

#[must_use]
pub fn lookup_circuit(params: &CircuitParams, architecture: &Architecture) -> Option<usize> {
    let key = circuit_cache_key(params, architecture);
    read_cache(&key)
}

pub fn store_circuit(params: &CircuitParams, architecture: &Architecture, chunk_bits: usize) {
    let key = circuit_cache_key(params, architecture);
    write_cache(&key, chunk_bits);
}

#[must_use]
pub fn lookup_operator(
    params: &CircuitParams,
    architecture: &Architecture,
    n_bits: usize,
) -> Option<usize> {
    let key = operator_cache_key(params, architecture, n_bits);
    read_cache(&key)
}

pub fn store_operator(
    params: &CircuitParams,
    architecture: &Architecture,
    n_bits: usize,
    chunk_bits: usize,
) {
    let key = operator_cache_key(params, architecture, n_bits);
    write_cache(&key, chunk_bits);
}

#[must_use]
pub fn candidates() -> &'static [usize] {
    CANDIDATES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params() -> CircuitParams {
        serde_json::from_str(
            r#"{
                "scale_base": 2,
                "scale_exponent": 18,
                "rescale_config": {},
                "inputs": [{"name": "x", "elem_type": 1, "shape": [1, 3, 32, 32]}],
                "outputs": [{"name": "y", "elem_type": 1, "shape": [1, 10]}]
            }"#,
        )
        .unwrap()
    }

    fn make_architecture() -> Architecture {
        Architecture {
            architecture: vec![],
        }
    }

    #[test]
    fn circuit_key_is_deterministic() {
        let params = make_params();
        let arch = make_architecture();
        let k1 = circuit_cache_key(&params, &arch);
        let k2 = circuit_cache_key(&params, &arch);
        assert_eq!(k1, k2);
    }

    #[test]
    fn operator_key_is_deterministic() {
        let params = make_params();
        let arch = make_architecture();
        let k1 = operator_cache_key(&params, &arch, 64);
        let k2 = operator_cache_key(&params, &arch, 64);
        assert_eq!(k1, k2);
    }

    #[test]
    fn circuit_and_operator_keys_differ() {
        let params = make_params();
        let arch = make_architecture();
        let ck = circuit_cache_key(&params, &arch);
        let ok = operator_cache_key(&params, &arch, 64);
        assert_ne!(ck, ok);
    }

    fn make_conv_relu_slice(name_prefix: &str, output_elems: Vec<usize>) -> Architecture {
        use crate::circuit_functions::utils::onnx_types::ONNXLayer;
        let mut layers = Vec::new();
        let conv_name = format!("{name_prefix}_conv");
        let relu_name = format!("{name_prefix}_relu");
        let conv_out = format!("{name_prefix}_conv_out");
        let relu_out = format!("{name_prefix}_relu_out");
        let mut conv_shape = std::collections::HashMap::new();
        conv_shape.insert(conv_out.clone(), output_elems.clone());
        layers.push(ONNXLayer {
            id: 0,
            name: conv_name,
            op_type: "Conv".to_string(),
            inputs: vec![],
            outputs: vec![conv_out],
            shape: conv_shape,
            tensor: None,
            params: None,
            opset_version_number: 13,
        });
        let mut relu_shape = std::collections::HashMap::new();
        relu_shape.insert(relu_out.clone(), output_elems);
        layers.push(ONNXLayer {
            id: 1,
            name: relu_name,
            op_type: "Relu".to_string(),
            inputs: vec![],
            outputs: vec![relu_out],
            shape: relu_shape,
            tensor: None,
            params: None,
            opset_version_number: 13,
        });
        Architecture {
            architecture: layers,
        }
    }

    #[test]
    fn operator_key_transfers_across_slices_with_same_profile() {
        let params = make_params();
        let slice_a = make_conv_relu_slice("resnet_layer3", vec![1, 64, 16, 16]);
        let slice_b = make_conv_relu_slice("efficientnet_block5", vec![1, 64, 16, 16]);

        let op_key_a = operator_cache_key(&params, &slice_a, 64);
        let op_key_b = operator_cache_key(&params, &slice_b, 64);
        assert_eq!(
            op_key_a, op_key_b,
            "slices with same op profile must share operator key"
        );

        let circuit_key_a = circuit_cache_key(&params, &slice_a);
        let circuit_key_b = circuit_cache_key(&params, &slice_b);
        assert_ne!(
            circuit_key_a, circuit_key_b,
            "slices from different models must have distinct circuit keys"
        );
    }

    #[test]
    fn operator_key_differs_for_different_shapes() {
        let params = make_params();
        let slice_small = make_conv_relu_slice("layer", vec![1, 32, 8, 8]);
        let slice_large = make_conv_relu_slice("layer", vec![1, 128, 32, 32]);

        let key_small = operator_cache_key(&params, &slice_small, 64);
        let key_large = operator_cache_key(&params, &slice_large, 64);
        assert_ne!(key_small, key_large);
    }

    #[test]
    fn sweep_picks_lowest_cost() {
        let costs = [100, 80, 90, 85, 95];
        let result = sweep_and_select(&[10, 11, 12, 13, 14], |c| {
            let idx = c - 10;
            Some(costs[idx])
        });
        assert_eq!(result, 11);
    }

    #[test]
    fn sweep_skips_failed_candidates() {
        let result = sweep_and_select(
            &[10, 11, 12],
            |c| {
                if c == 11 { None } else { Some(100 - c) }
            },
        );
        assert_eq!(result, 12);
    }
}
