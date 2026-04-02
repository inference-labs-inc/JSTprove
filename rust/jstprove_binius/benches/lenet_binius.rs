use std::time::Instant;

fn fmt_duration(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{ms:.1}ms")
    }
}

fn rss_bytes() -> u64 {
    let mut u = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, u.as_mut_ptr()) };
    if ret != 0 {
        return 0;
    }
    let usage = unsafe { u.assume_init() };
    usage.ru_maxrss as u64
}

fn main() {
    println!("jstprove-binius LeNet benchmark");
    println!("{}", "=".repeat(60));

    let t = Instant::now();
    let lenet = jstprove_binius::models::lenet::build();
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("build:      {}", fmt_duration(build_ms));
    println!("gates:      {}", lenet.circuit.n_gates());

    let t = Instant::now();
    let witness = jstprove_binius::models::lenet::fill_dummy_witness(&lenet);
    let witness_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("witness:    {}", fmt_duration(witness_ms));

    let t = Instant::now();
    let (verifier, prover) = jstprove_binius::prove::setup(
        &jstprove_binius::circuit::BiniusCircuit {
            circuit: lenet.circuit,
            input_wires: lenet.input_wires,
            output_wires: lenet.output_wires,
        },
        1,
    )
    .expect("setup failed");
    let setup_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("setup:      {}", fmt_duration(setup_ms));

    let t = Instant::now();
    let artifact = jstprove_binius::prove::prove(&prover, witness).expect("prove failed");
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    let proof_kib = artifact.proof_bytes.len() / 1024;
    println!(
        "prove:      {}  proof: {} KiB",
        fmt_duration(prove_ms),
        proof_kib
    );

    let t = Instant::now();
    jstprove_binius::verify::verify(&verifier, &artifact).expect("verify failed");
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("verify:     {}", fmt_duration(verify_ms));

    println!("{}", "-".repeat(60));
    let total_ms = build_ms + witness_ms + setup_ms + prove_ms + verify_ms;
    println!("total:      {}", fmt_duration(total_ms));
    println!("peak RSS:   {:.1} MiB", rss_bytes() as f64 / 1048576.0);
}
