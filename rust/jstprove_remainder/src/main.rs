use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

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
    },
    Witness {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Prove {
        #[arg(long)]
        model: PathBuf,
        #[arg(short, long)]
        witness: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    Verify {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { model, output } => {
            jstprove_remainder::runner::compile::run(&model, &output)
        }
        Commands::Witness {
            model,
            input,
            output,
        } => jstprove_remainder::runner::witness::run(&model, &input, &output),
        Commands::Prove {
            model,
            witness,
            output,
        } => jstprove_remainder::runner::prove::run(&model, &witness, &output),
        Commands::Verify {
            model,
            proof,
            input,
        } => jstprove_remainder::runner::verify::run(&model, &proof, &input),
    }
}
