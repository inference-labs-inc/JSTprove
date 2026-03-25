use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use jstprove_remainder::cli;

#[derive(Parser)]
#[command(name = "jstprove-remainder")]
#[command(about = "zkML proving for ONNX models using Remainder_CE")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
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

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli_args = Cli::parse();

    let result = match cli_args.command {
        Commands::Compile {
            model,
            output,
            no_compress,
        } => {
            cli::header("compile");
            jstprove_remainder::runner::compile::run(&model, &output, !no_compress)
        }
        Commands::Witness {
            model,
            input,
            output,
            no_compress,
        } => {
            cli::header("witness");
            jstprove_remainder::runner::witness::run(&model, &input, &output, !no_compress)
        }
        Commands::Prove {
            model,
            witness,
            output,
            no_compress,
        } => {
            cli::header("prove");
            jstprove_remainder::runner::prove::run(&model, &witness, &output, !no_compress)
        }
        Commands::Verify {
            model,
            proof,
            input,
        } => {
            cli::header("verify");
            jstprove_remainder::runner::verify::run(&model, &proof, &input)
        }
        Commands::BatchWitness {
            model,
            manifest,
            no_compress,
        } => {
            cli::header("batch-witness");
            jstprove_remainder::runner::batch::run_batch_witness(&model, &manifest, !no_compress)
                .map(|_| ())
        }
        Commands::BatchProve {
            model,
            manifest,
            no_compress,
        } => {
            cli::header("batch-prove");
            jstprove_remainder::runner::batch::run_batch_prove(&model, &manifest, !no_compress)
                .map(|_| ())
        }
        Commands::BatchVerify { model, manifest } => {
            cli::header("batch-verify");
            jstprove_remainder::runner::batch::run_batch_verify(&model, &manifest).map(|_| ())
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
        let x = console::style("error:").red().bold();
        eprintln!("\n{x} {e:#}");
    }

    result
}
