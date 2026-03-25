use std::path::Path;

use jstprove_circuits::circuit_functions::utils::build_layers::default_n_bits_for_config;
use jstprove_circuits::circuit_functions::utils::onnx_model::estimate_rescale_elements;
use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::compile_bn254;

use expander_compiler::frontend::BN254Config;

const CHUNK_CANDIDATES: &[usize] = &[10, 11, 12, 13, 14, 15, 16];

fn sweep_model(model_path: &Path) {
    let metadata = expander_metadata::generate_from_onnx(model_path).unwrap();
    let base_params = metadata.circuit_params.clone();

    let kappa = base_params.scale_exponent;
    let n_bits = default_n_bits_for_config::<BN254Config>();
    let est_elems = estimate_rescale_elements(&base_params, &metadata.architecture);

    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    eprintln!("sweeping {model_name}: kappa={kappa}, n_bits={n_bits}, est_elems={est_elems}");

    for &chunk_bits in CHUNK_CANDIDATES {
        let mut params = base_params.clone();
        params.logup_chunk_bits = Some(chunk_bits);

        OnnxContext::set_all(
            metadata.architecture.clone(),
            params.clone(),
            Some(metadata.wandb.clone()),
        );

        let tmp = tempfile::TempDir::new().unwrap();
        let circuit_path = tmp.path().join("circuit.bundle");
        let circuit_path_str = circuit_path.to_str().unwrap();

        let t = std::time::Instant::now();
        compile_bn254(circuit_path_str, false, Some(params)).unwrap();
        let compile_secs = t.elapsed().as_secs_f64();

        let bundle =
            jstprove_circuits::runner::main_runner::read_circuit_msgpack(circuit_path_str).unwrap();
        let circuit_bytes = bundle.circuit.len();

        println!(
            "{{\
             \"model\":\"{model_name}\",\
             \"kappa\":{kappa},\
             \"n_bits\":{n_bits},\
             \"est_elems\":{est_elems},\
             \"chunk_bits\":{chunk_bits},\
             \"compile_secs\":{compile_secs:.3},\
             \"circuit_bytes\":{circuit_bytes}\
             }}"
        );
    }
}

fn main() {
    let models: Vec<String> = if let Ok(m) = std::env::var("MODEL") {
        m.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        vec!["lenet".to_string(), "mini_resnet".to_string()]
    };

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jstprove_remainder/models");

    for model_name in &models {
        let model_path = models_dir.join(format!("{model_name}.onnx"));
        if !model_path.exists() {
            eprintln!("SKIP {model_name}: not found at {}", model_path.display());
            continue;
        }
        sweep_model(&model_path);
    }
}
