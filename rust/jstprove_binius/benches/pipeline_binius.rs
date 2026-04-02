use std::time::Instant;

use binius_core::word::Word;

fn fmt_duration(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn bench_circuit(name: &str, n_ops: usize, log_inv_rate: usize) {
    let t = Instant::now();
    let bc = match name {
        "iadd" => jstprove_binius::circuit::build_iadd_chain(n_ops),
        "bitwise" => jstprove_binius::circuit::build_bitwise_chain(n_ops),
        _ => panic!("unknown circuit: {name}"),
    };
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let mut filler = bc.circuit.new_witness_filler();
    for &w in &bc.input_wires {
        filler[w] = Word(1);
    }
    bc.circuit
        .populate_wire_witness(&mut filler)
        .expect("witness population failed");
    let witness = filler.into_value_vec();
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (verifier, prover) =
        jstprove_binius::prove::setup(&bc, log_inv_rate).expect("setup failed");
    let setup_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let artifact = jstprove_binius::prove::prove(&prover, witness).expect("prove failed");
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    jstprove_binius::verify::verify(&verifier, &artifact).expect("verify failed");
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{name:>8} n={n_ops:<6} build={:<10} witness={:<10} setup={:<10} prove={:<10} verify={:<10} proof={}KiB",
        fmt_duration(build_ms),
        fmt_duration(witness_ms),
        fmt_duration(setup_ms),
        fmt_duration(prove_ms),
        fmt_duration(verify_ms),
        artifact.proof_bytes.len() / 1024,
    );
}

fn main() {
    println!("jstprove-binius pipeline benchmark");
    println!("{}", "=".repeat(120));

    for &n in &[64, 256, 1024, 4096] {
        bench_circuit("iadd", n, 1);
    }
    println!();
    for &n in &[64, 256, 1024, 4096] {
        bench_circuit("bitwise", n, 1);
    }
}
