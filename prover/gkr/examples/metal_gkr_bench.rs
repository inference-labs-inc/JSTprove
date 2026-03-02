#![allow(clippy::all, clippy::pedantic)]

use std::time::Instant;

use arith::Field;
use ark_std::test_rng;
use circuit::{Circuit, CircuitLayer, CoefType, Gate, StructureInfo};
use config_macros::declare_gkr_config;
use gkr::{Prover, Verifier};
use gkr_engine::{BN254Config, FieldEngine, GKREngine, GKRScheme, MPIConfig};
use gkr_hashers::SHA256hasher;
use poly_commit::{expander_pcs_init_testing_only, raw::RawExpanderGKR};
use transcript::BytesHashTranscript;

fn build_synthetic_circuit<C: FieldEngine>(log_size: usize) -> Circuit<C> {
    let mut rng = test_rng();
    let n = 1usize << log_size;

    let mut mul_gates: Vec<Gate<C, 2>> = Vec::with_capacity(n);
    let mut add_gates: Vec<Gate<C, 1>> = Vec::with_capacity(n);

    for i in 0..n {
        mul_gates.push(Gate {
            i_ids: [i, (i + 1) % n],
            o_id: i,
            coef_type: CoefType::Constant,
            coef: C::CircuitField::one(),
            gate_type: 0,
        });
        add_gates.push(Gate {
            i_ids: [i],
            o_id: i,
            coef_type: CoefType::Constant,
            coef: C::CircuitField::one(),
            gate_type: 0,
        });
    }

    let mk_layer = |mul: Vec<Gate<C, 2>>, add: Vec<Gate<C, 1>>| CircuitLayer::<C> {
        input_var_num: log_size,
        output_var_num: log_size,
        input_vals: vec![],
        output_vals: vec![],
        mul,
        add,
        const_: vec![],
        uni: vec![],
        structure_info: StructureInfo {
            skip_sumcheck_phase_two: false,
        },
    };

    let mut circuit = Circuit::<C> {
        layers: vec![
            mk_layer(mul_gates.clone(), add_gates.clone()),
            mk_layer(mul_gates.clone(), add_gates.clone()),
            mk_layer(mul_gates, add_gates),
        ],
        public_input: vec![],
        expected_num_output_zeros: 0,
        rnd_coefs_identified: false,
        rnd_coefs: vec![],
    };
    circuit.pre_process_gkr();

    circuit.layers[0].input_vals = (0..n)
        .map(|_| C::SimdCircuitField::random_unsafe(&mut rng))
        .collect();
    circuit
}

fn bench_prove<Cfg: GKREngine>(label: &str, log_size: usize, warmup: usize, iters: usize)
where
    Cfg::FieldConfig: FieldEngine,
{
    let mpi_config = MPIConfig::prover_new();
    let circuit = build_synthetic_circuit::<Cfg::FieldConfig>(log_size);

    let (pcs_params, pcs_proving_key, pcs_vk, mut pcs_scratch) =
        expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
            circuit.log_input_size(),
            &mpi_config,
        );

    let mut prover = Prover::<Cfg>::new(mpi_config.clone());
    prover.prepare_mem(&circuit);

    {
        let mut c = circuit.clone();
        c.evaluate();
        let (claimed_v, proof) =
            prover.prove(&mut c, &pcs_params, &pcs_proving_key, &mut pcs_scratch);
        let verifier = Verifier::<Cfg>::new(mpi_config.clone());
        let mut vc = circuit.clone();
        vc.evaluate();
        let ok = verifier.verify(&mut vc, &[], &claimed_v, &pcs_params, &pcs_vk, &proof);
        assert!(ok, "proof verification failed at log_size={log_size}");
        eprintln!("[{label}] log_size={log_size} verification: PASS");
    }

    for _ in 0..warmup {
        let mut c = circuit.clone();
        c.evaluate();
        prover.prove(&mut c, &pcs_params, &pcs_proving_key, &mut pcs_scratch);
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut c = circuit.clone();
        c.evaluate();
        let start = Instant::now();
        prover.prove(&mut c, &pcs_params, &pcs_proving_key, &mut pcs_scratch);
        times.push(start.elapsed());
    }

    let avg_ms = times.iter().map(|t| t.as_secs_f64() * 1000.0).sum::<f64>() / iters as f64;
    let min_ms = times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let max_ms = times
        .iter()
        .map(|t| t.as_secs_f64() * 1000.0)
        .fold(0.0f64, f64::max);

    eprintln!(
        "[{label}] log_size={log_size} layers=3 | avg={avg_ms:.2}ms min={min_ms:.2}ms max={max_ms:.2}ms ({iters} iters)"
    );
}

fn main() {
    env_logger::init();

    declare_gkr_config!(
        BN254Sha2,
        FieldType::BN254,
        FiatShamirHashType::SHA256,
        PCSCommitmentType::Raw,
        GKRScheme::Vanilla
    );

    let warmup = 3;
    let iters = 10;

    let args: Vec<String> = std::env::args().collect();
    let log_size: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(18);
    bench_prove::<BN254Sha2>("BN254-GKR", log_size, warmup, iters);
}
