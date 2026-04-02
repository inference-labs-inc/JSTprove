use jstprove_circuits::circuit_functions::utils::onnx_model::{Architecture, CircuitParams, WANDB};

use jstprove_circuits::ProofSystem;
use jstprove_circuits::cli::OutputMode;
use jstprove_circuits::expander_metadata;
use jstprove_circuits::io::io_reader::FileReader;
use jstprove_circuits::runner::main_runner::{
    get_arg, get_args, get_curve, handle_args, try_load_metadata_from_circuit,
};

use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::Circuit;

use expander_compiler::frontend::{
    BN254Config, GoldilocksBasefoldConfig, GoldilocksConfig, GoldilocksExt2BasefoldConfig, Variable,
};
use jstprove_circuits::Curve;
use jstprove_onnx::quantizer::{N_BITS_GOLDILOCKS, N_BITS_GOLDILOCKS_EXT2};

fn load_wandb(matches: &clap::ArgMatches) -> Result<Option<WANDB>, String> {
    let Ok(wandb_file_path) = get_arg(matches, "wandb") else {
        return Ok(None);
    };
    let wandb_bytes = std::fs::read(&wandb_file_path)
        .map_err(|e| format!("Failed to read W&B file '{wandb_file_path}': {e}"))?;
    let wandb: WANDB = rmp_serde::from_slice(&wandb_bytes)
        .map_err(|e| format!("Invalid W&B msgpack in '{wandb_file_path}': {e}"))?;
    Ok(Some(wandb))
}

fn load_metadata(path: &str) -> CircuitParams {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Error: failed to read metadata file '{path}': {e}");
        std::process::exit(1);
    });
    rmp_serde::from_slice::<CircuitParams>(&bytes).unwrap_or_else(|e| {
        eprintln!("Error: failed to parse metadata '{path}' as msgpack: {e}");
        std::process::exit(1);
    })
}

fn set_onnx_context(matches: &clap::ArgMatches, needs_full: bool) {
    let meta_file_path = get_arg(matches, "meta").unwrap();
    let params = load_metadata(&meta_file_path);
    let wandb = match load_wandb(matches) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    if wandb.is_none() && (params.weights_as_inputs || needs_full) {
        let cmd_type = get_arg(matches, "type").unwrap_or_default();
        let reason = if params.weights_as_inputs {
            "weights_as_inputs is enabled in metadata"
        } else {
            "compile commands require weight data"
        };
        eprintln!("Error: command '{cmd_type}' requires --wandb ({reason}).");
        std::process::exit(1);
    }

    if needs_full {
        let arch_file_path = get_arg(matches, "arch").unwrap();
        let arch_bytes = std::fs::read(&arch_file_path).unwrap_or_else(|e| {
            eprintln!("Error: failed to read architecture file '{arch_file_path}': {e}");
            std::process::exit(1);
        });
        let arch: Architecture = rmp_serde::from_slice(&arch_bytes).unwrap_or_else(|e| {
            eprintln!("Error: invalid architecture msgpack in '{arch_file_path}': {e}");
            std::process::exit(1);
        });
        OnnxContext::set_all(arch, params, wandb);
    } else {
        OnnxContext::set_params(params);
        if let Some(w) = wandb {
            OnnxContext::set_wandb(w);
        }
    }
}

const ONNX_META_COMMANDS: &[&str] = &[
    "run_gen_witness",
    "run_gen_verify",
    "run_batch_witness",
    "run_batch_verify",
    "run_pipe_witness",
    "run_pipe_verify",
    "msgpack_witness_stdin",
];

const ONNX_FULL_COMMANDS: &[&str] = &[
    "run_compile_circuit",
    "run_debug_witness",
    "msgpack_compile",
];

#[allow(clippy::too_many_lines)]
fn main() {
    let mut file_reader = FileReader {
        path: "jstprove".to_owned(),
    };

    let matches = get_args();

    let cmd_type = get_arg(&matches, "type").unwrap_or_default();
    let cli_backend_explicit =
        matches.value_source("backend") == Some(clap::parser::ValueSource::CommandLine);
    let cli_is_remainder = cli_backend_explicit
        && matches
            .get_one::<String>("backend")
            .is_some_and(|b| b == "remainder");
    let mut is_remainder = cli_is_remainder;

    let needs_meta = ONNX_META_COMMANDS.contains(&cmd_type.as_str())
        || ONNX_FULL_COMMANDS.contains(&cmd_type.as_str());
    let needs_full = ONNX_FULL_COMMANDS.contains(&cmd_type.as_str());

    let has_meta = matches.get_one::<String>("meta").is_some();
    let has_arch = matches.get_one::<String>("arch").is_some();
    let has_onnx = matches.get_one::<String>("onnx").is_some();

    if !is_remainder && !cli_backend_explicit {
        if let Some(circuit_path) = matches.get_one::<String>("circuit_path") {
            if let Some(params) = try_load_metadata_from_circuit(circuit_path) {
                if params.proof_system.is_remainder() {
                    is_remainder = true;
                }
            }
        }
    }

    if has_onnx && is_remainder {
        eprintln!("Error: --onnx cannot be used with the remainder backend.");
        std::process::exit(1);
    }

    if has_onnx && (has_meta || has_arch) {
        eprintln!("Error: --onnx cannot be combined with --meta or --arch.");
        std::process::exit(1);
    }

    if has_onnx && !is_remainder {
        let onnx_path_str = get_arg(&matches, "onnx").unwrap();
        let onnx_path = std::path::Path::new(&onnx_path_str);
        let early_curve = matches
            .get_one::<String>("curve")
            .and_then(|s| s.parse::<Curve>().ok())
            .unwrap_or_default();
        let result = match early_curve {
            Curve::Bn254 => expander_metadata::generate_from_onnx(onnx_path),
            Curve::Goldilocks | Curve::GoldilocksBasefold => {
                expander_metadata::generate_from_onnx_for_field(onnx_path, N_BITS_GOLDILOCKS, None)
            }
            Curve::GoldilocksExt2 => expander_metadata::generate_from_onnx_for_field(
                onnx_path,
                N_BITS_GOLDILOCKS_EXT2,
                None,
            ),
        };
        match result {
            Ok(meta) => {
                OnnxContext::set_all(meta.architecture, meta.circuit_params, Some(meta.wandb));
            }
            Err(e) => {
                eprintln!("Error: ONNX metadata generation failed: {e:#}");
                std::process::exit(1);
            }
        }
    }

    if !is_remainder && needs_full && !has_onnx && (!has_meta || !has_arch) {
        eprintln!("Error: command '{cmd_type}' requires --onnx or --meta and --arch arguments.");
        std::process::exit(1);
    }

    if !is_remainder && !has_onnx {
        if has_meta {
            set_onnx_context(&matches, needs_full);
            if let Ok(params) = OnnxContext::get_params() {
                if params.proof_system.is_remainder() {
                    if cli_backend_explicit {
                        eprintln!(
                            "Error: command '{cmd_type}' --backend conflicts with metadata (backend: remainder)."
                        );
                        std::process::exit(1);
                    }
                    is_remainder = true;
                }
            }
        } else if needs_meta {
            let circuit_path = matches
                .get_one::<String>("circuit_path")
                .expect("command requires --meta or -c with bundled metadata");
            if let Some(params) = try_load_metadata_from_circuit(circuit_path) {
                if params.proof_system.is_remainder() {
                    if cli_backend_explicit {
                        eprintln!(
                            "Error: command '{cmd_type}' --backend conflicts with metadata (backend: remainder)."
                        );
                        std::process::exit(1);
                    }
                    is_remainder = true;
                }
                let wandb = match load_wandb(&matches) {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("Error: {e}");
                        std::process::exit(1);
                    }
                };
                if params.weights_as_inputs && wandb.is_none() {
                    eprintln!(
                        "Error: command '{cmd_type}' requires --wandb (weights_as_inputs is enabled in bundled metadata)."
                    );
                    std::process::exit(1);
                }
                OnnxContext::set_params(params);
                if let Some(w) = wandb {
                    OnnxContext::set_wandb(w);
                }
            } else {
                eprintln!(
                    "Error: command '{cmd_type}' requires --meta or circuit .msgpack with bundled metadata."
                );
                std::process::exit(1);
            }
        }
    }

    let mut metadata = if is_remainder {
        Some(CircuitParams {
            scale_base: 2,
            scale_exponent: 18,
            rescale_config: std::collections::HashMap::new(),
            inputs: vec![],
            outputs: vec![],
            freivalds_reps: 1,
            n_bits_config: std::collections::HashMap::new(),
            weights_as_inputs: false,
            proof_system: ProofSystem::Remainder,
            curve: Some(Curve::default()),
            logup_chunk_bits: None,
        })
    } else {
        OnnxContext::get_params().ok()
    };

    let curve = get_curve(&matches, metadata.as_ref());

    if let Some(ref meta) = metadata {
        let cli_explicit =
            matches.value_source("curve") == Some(clap::parser::ValueSource::CommandLine);
        if let Some(bundle_curve) = meta.curve {
            if cli_explicit && bundle_curve != curve {
                eprintln!(
                    "Error: curve mismatch — circuit was compiled with '{bundle_curve}' but '{curve}' was requested",
                );
                std::process::exit(1);
            }
        }
    }

    if let Some(ref mut meta) = metadata {
        meta.curve = Some(curve);
    }

    let mode = if matches.get_flag("json") {
        OutputMode::Json
    } else if matches.get_flag("quiet") {
        OutputMode::Quiet
    } else {
        OutputMode::Human
    };

    let result = match curve {
        Curve::Bn254 => handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(
            &matches,
            &mut file_reader,
            metadata,
            mode,
        ),
        Curve::Goldilocks => handle_args::<GoldilocksConfig, Circuit<Variable>, Circuit<_>, _>(
            &matches,
            &mut file_reader,
            metadata,
            mode,
        ),
        Curve::GoldilocksBasefold => handle_args::<
            GoldilocksBasefoldConfig,
            Circuit<Variable>,
            Circuit<_>,
            _,
        >(&matches, &mut file_reader, metadata, mode),
        Curve::GoldilocksExt2 => handle_args::<
            GoldilocksExt2BasefoldConfig,
            Circuit<Variable>,
            Circuit<_>,
            _,
        >(&matches, &mut file_reader, metadata, mode),
    };

    if let Err(err) = result {
        if !matches!(
            err,
            jstprove_circuits::runner::errors::CliError::AlreadyReported(_)
        ) {
            eprintln!("Error: {err}");
        }
        std::process::exit(1);
    }
}
