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
        buf.extend_from_slice(layer.op_type.as_bytes());
        for output_name in &layer.outputs {
            if let Some(shape) = layer.shape.get(output_name) {
                for &dim in shape {
                    buf.extend_from_slice(&dim.to_le_bytes());
                }
            }
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
