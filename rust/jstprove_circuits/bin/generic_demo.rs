use jstprove_circuits::circuit_functions::utils::onnx_model::{
    Architecture, CircuitParams, WANDB,
};

use jstprove_circuits::io::io_reader::FileReader;
use jstprove_circuits::runner::main_runner::{
    get_arg, get_args, handle_args, try_load_metadata_from_circuit,
};

use jstprove_circuits::io::io_reader::onnx_context::OnnxContext;
use jstprove_circuits::onnx::Circuit;

use expander_compiler::frontend::{BN254Config, Variable};

fn set_onnx_context(matches: &clap::ArgMatches, needs_full: bool, has_wandb: bool) {
    let meta_file_path = get_arg(matches, "meta").unwrap();
    let meta_file = std::fs::read_to_string(&meta_file_path).expect("Failed to read metadata file");
    let params: CircuitParams = serde_json::from_str(&meta_file).expect("Invalid metadata JSON");
    if params.weights_as_inputs && !has_wandb {
        let cmd_type = get_arg(matches, "type").unwrap_or_default();
        eprintln!(
            "Error: command '{cmd_type}' requires --wandb (weights_as_inputs is enabled in metadata)."
        );
        std::process::exit(1);
    }

    OnnxContext::set_params(params);

    if needs_full {
        let arch_file_path = get_arg(matches, "arch").unwrap();
        let arch_file =
            std::fs::read_to_string(&arch_file_path).expect("Failed to read architecture file");
        let arch: Architecture =
            serde_json::from_str(&arch_file).expect("Invalid architecture JSON");
        OnnxContext::set_architecture(arch);
    }

    if has_wandb {
        let wandb_file_path = get_arg(matches, "wandb").unwrap();
        let wandb_file =
            std::fs::read_to_string(&wandb_file_path).expect("Failed to read W&B file");
        let wandb: WANDB = serde_json::from_str(&wandb_file).expect("Invalid W&B JSON");
        OnnxContext::set_wandb(wandb);
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

fn main() {
    let mut file_reader = FileReader {
        path: "demo_cnn".to_owned(),
    };

    let matches = get_args();

    let cmd_type = get_arg(&matches, "type").unwrap_or_default();
    let needs_meta = ONNX_META_COMMANDS.contains(&cmd_type.as_str())
        || ONNX_FULL_COMMANDS.contains(&cmd_type.as_str());
    let needs_full = ONNX_FULL_COMMANDS.contains(&cmd_type.as_str());

    let has_meta = matches.get_one::<String>("meta").is_some();
    let has_arch = matches.get_one::<String>("arch").is_some();
    let has_wandb = matches.get_one::<String>("wandb").is_some();

    if needs_full && (!has_meta || !has_arch) {
        eprintln!("Error: command '{cmd_type}' requires --meta and --arch arguments.");
        std::process::exit(1);
    }

    if has_meta {
        set_onnx_context(&matches, needs_full, has_wandb);
    } else if needs_meta {
        let circuit_path = matches
            .get_one::<String>("circuit_path")
            .expect("command requires --meta or -c with bundled metadata");
        if let Some(params) = try_load_metadata_from_circuit(circuit_path) {
            if params.weights_as_inputs && !has_wandb {
                eprintln!(
                    "Error: command '{cmd_type}' requires --wandb (weights_as_inputs is enabled in bundled metadata)."
                );
                std::process::exit(1);
            }
            OnnxContext::set_params(params);
        } else {
            eprintln!(
                "Error: command '{cmd_type}' requires --meta or circuit .msgpack with bundled metadata."
            );
            std::process::exit(1);
        }
    }

    let metadata = OnnxContext::get_params().ok();

    if let Err(err) = handle_args::<BN254Config, Circuit<Variable>, Circuit<_>, _>(
        &matches,
        &mut file_reader,
        metadata,
    ) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
