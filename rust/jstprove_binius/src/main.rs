use std::time::Instant;

use binius_core::word::Word;
use clap::Parser;

#[derive(Parser)]
#[command(name = "jstprove-binius", about = "Binary tower field proving backend")]
struct Cli {
    #[arg(long, default_value_t = 1024)]
    n_ops: usize,

    #[arg(long, default_value_t = 1)]
    log_inv_rate: usize,

    #[arg(long, default_value = "iadd")]
    circuit: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let t = Instant::now();
    let bc = match cli.circuit.as_str() {
        "iadd" => jstprove_binius::circuit::build_iadd_chain(cli.n_ops),
        "bitwise" => jstprove_binius::circuit::build_bitwise_chain(cli.n_ops),
        other => anyhow::bail!("unknown circuit type: {other}"),
    };
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut filler = bc.circuit.new_witness_filler();
    for &w in &bc.input_wires {
        filler[w] = Word(1);
    }
    bc.circuit.populate_wire_witness(&mut filler)?;
    let witness = filler.into_value_vec();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    jstprove_binius::circuit::verify_circuit_constraints(&bc, &witness)?;
    let constraint_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (verifier, prover) = jstprove_binius::prove::setup(&bc, cli.log_inv_rate)?;
    let setup_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let artifact = jstprove_binius::prove::prove(&prover, witness)?;
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    jstprove_binius::verify::verify(&verifier, &artifact)?;
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    let proof_kib = artifact.proof_bytes.len() / 1024;
    println!("circuit: {} ({} ops)", cli.circuit, cli.n_ops);
    println!("{}", "-".repeat(50));
    println!("build:      {build_ms:>10.1}ms");
    println!("witness:    {witness_ms:>10.1}ms");
    println!("constraint: {constraint_ms:>10.1}ms");
    println!("setup:      {setup_ms:>10.1}ms");
    println!("prove:      {prove_ms:>10.1}ms  proof: {proof_kib} KiB");
    println!("verify:     {verify_ms:>10.1}ms");
    println!("{}", "-".repeat(50));
    let total_ms = build_ms + witness_ms + constraint_ms + setup_ms + prove_ms + verify_ms;
    println!("total:      {total_ms:>10.1}ms");

    Ok(())
}
