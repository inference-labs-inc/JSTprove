use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use jstprove_remainder::cli::{self, OutputMode};

#[derive(Parser)]
#[command(name = "jstprove-remainder")]
#[command(about = "zkML proving for ONNX models using Remainder_CE")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(
        long,
        global = true,
        help = "Suppress all output",
        conflicts_with = "json"
    )]
    quiet: bool,

    #[arg(
        long,
        global = true,
        help = "Emit JSON lines to stderr",
        conflicts_with = "quiet"
    )]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    Compile {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    Witness {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    Prove {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        witness: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    Verify {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
    },
    BatchWitness {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    BatchProve {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        manifest: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    BatchVerify {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        manifest: PathBuf,
    },
    PipeWitness {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    PipeProve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = false)]
        no_compress: bool,
    },
    PipeVerify {
        #[arg(long)]
        model: PathBuf,
    },
}

#[allow(clippy::too_many_lines)]
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli_args = Cli::parse();

    let mode = if cli_args.json {
        OutputMode::Json
    } else if cli_args.quiet {
        OutputMode::Quiet
    } else {
        OutputMode::Human
    };

    let mut error_already_reported = false;
    let result = match cli_args.command {
        Commands::Compile {
            model,
            output,
            no_compress,
        } => {
            cli::header("compile", mode);
            jstprove_remainder::runner::compile::run(&model, &output, !no_compress, mode)
        }
        Commands::Witness {
            model,
            input,
            output,
            no_compress,
        } => {
            cli::header("witness", mode);
            jstprove_remainder::runner::witness::run(&model, &input, &output, !no_compress, mode)
        }
        Commands::Prove {
            model,
            witness,
            output,
            no_compress,
        } => {
            cli::header("prove", mode);
            jstprove_remainder::runner::prove::run(&model, &witness, &output, !no_compress, mode)
        }
        Commands::Verify {
            model,
            proof,
            input,
        } => {
            cli::header("verify", mode);
            let r = jstprove_remainder::runner::verify::run(&model, &proof, &input, mode);
            if r.is_err() {
                error_already_reported = true;
            }
            r
        }
        Commands::BatchWitness {
            model,
            manifest,
            no_compress,
        } => {
            cli::header("batch-witness", mode);
            jstprove_remainder::runner::batch::run_batch_witness(
                &model,
                &manifest,
                !no_compress,
                mode,
            )
            .map(|_| ())
        }
        Commands::BatchProve {
            model,
            manifest,
            no_compress,
        } => {
            cli::header("batch-prove", mode);
            jstprove_remainder::runner::batch::run_batch_prove(
                &model,
                &manifest,
                !no_compress,
                mode,
            )
            .map(|_| ())
        }
        Commands::BatchVerify { model, manifest } => {
            cli::header("batch-verify", mode);
            jstprove_remainder::runner::batch::run_batch_verify(&model, &manifest, mode).map(|_| ())
        }
        Commands::PipeWitness { model, no_compress } => {
            jstprove_remainder::runner::pipe::run_pipe_witness(&model, !no_compress)
        }
        Commands::PipeProve { model, no_compress } => {
            jstprove_remainder::runner::pipe::run_pipe_prove(&model, !no_compress)
        }
        Commands::PipeVerify { model } => jstprove_remainder::runner::pipe::run_pipe_verify(&model),
    };

    if let Err(ref e) = result {
        if error_already_reported {
            return result;
        }
        match mode {
            OutputMode::Human => {
                let x = console::style("error:").red().bold();
                eprintln!("\n{x} {e:#}");
            }
            OutputMode::Json => {
                let obj = serde_json::json!({"event": "fatal", "message": format!("{e:#}")});
                eprintln!("{obj}");
            }
            OutputMode::Quiet => {}
        }
    }

    result
}
