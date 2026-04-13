#![allow(clippy::all, clippy::pedantic)]

use std::time::Instant;

use arith::Field;
use ark_std::test_rng;
use goldilocks::Goldilocks;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use circuit::{Circuit, CircuitLayer, CoefType, Gate, StructureInfo};
use config_macros::declare_gkr_config;
use gkr::{Prover, Verifier};
use gkr_engine::{FieldEngine, GKREngine, GKRScheme, Goldilocksx1Config, MPIConfig};
use gkr_hashers::SHA256hasher;
use poly_commit::{expander_pcs_init_testing_only, raw::RawExpanderGKR};
use transcript::BytesHashTranscript;

use jst_gkr::prover::prove_with_evaluations as jst_prove_eval;
use jst_gkr::transcript::Sha256Transcript;
use jst_gkr::verifier::verify as jst_verify;
use jst_gkr_engine::{
    AddGate as JstAddGate, CircuitLayer as JstCircuitLayer, LayeredCircuit, MulGate as JstMulGate,
};

declare_gkr_config!(
    GoldilocksSha2Raw,
    FieldType::Goldilocksx1,
    FiatShamirHashType::SHA256,
    PCSCommitmentType::Raw,
    GKRScheme::Vanilla
);

fn build_expander_circuit<C: FieldEngine>(depth: usize, log_size: usize) -> Circuit<C> {
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

    let layers: Vec<_> = (0..depth)
        .map(|_| mk_layer(mul_gates.clone(), add_gates.clone()))
        .collect();

    let mut circuit = Circuit::<C> {
        layers,
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

fn build_jst_circuit(
    depth: usize,
    log_size: usize,
) -> (LayeredCircuit<Goldilocks>, Vec<Goldilocks>) {
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let n = 1usize << log_size;

    let witness: Vec<Goldilocks> = (0..n)
        .map(|_| Goldilocks::random_unsafe(&mut rng))
        .collect();

    let layers: Vec<_> = (0..depth)
        .map(|_| {
            let mut mul_gates = Vec::with_capacity(n);
            let mut add_gates = Vec::with_capacity(n);

            for i in 0..n {
                mul_gates.push(JstMulGate {
                    o_id: i,
                    i_ids: [i, (i + 1) % n],
                    coef: Goldilocks::ONE,
                });
                add_gates.push(JstAddGate {
                    o_id: i,
                    i_id: i,
                    coef: Goldilocks::ONE,
                });
            }

            JstCircuitLayer {
                input_var_num: log_size,
                output_var_num: log_size,
                mul_gates,
                add_gates,
                const_gates: vec![],
            }
        })
        .collect();

    (LayeredCircuit { layers }, witness)
}

fn bench_one(label: &str, log_size: usize, depth: usize, warmup: usize, iters: usize) {
    eprintln!(
        "\n=== {label} | depth={depth} log_width={log_size} ({} gates/layer) ===",
        1 << log_size
    );

    {
        let mpi_config = MPIConfig::prover_new();
        let circuit = build_expander_circuit::<<GoldilocksSha2Raw as GKREngine>::FieldConfig>(
            depth, log_size,
        );

        let (pcs_params, pcs_pk, pcs_vk, mut pcs_scratch) =
            expander_pcs_init_testing_only::<
                <GoldilocksSha2Raw as GKREngine>::FieldConfig,
                <GoldilocksSha2Raw as GKREngine>::PCSConfig,
            >(circuit.log_input_size(), &mpi_config);

        let mut prover = Prover::<GoldilocksSha2Raw>::new(mpi_config.clone());
        prover.prepare_mem(&circuit);

        {
            let mut c = circuit.clone();
            c.evaluate();
            let (claimed_v, proof) = prover.prove(&mut c, &pcs_params, &pcs_pk, &mut pcs_scratch);
            let verifier = Verifier::<GoldilocksSha2Raw>::new(mpi_config.clone());
            let mut vc = circuit.clone();
            vc.evaluate();
            assert!(verifier.verify(&mut vc, &[], &claimed_v, &pcs_params, &pcs_vk, &proof));
        }

        for _ in 0..warmup {
            let mut c = circuit.clone();
            c.evaluate();
            prover.prove(&mut c, &pcs_params, &pcs_pk, &mut pcs_scratch);
        }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut c = circuit.clone();
            let start = Instant::now();
            c.evaluate();
            prover.prove(&mut c, &pcs_params, &pcs_pk, &mut pcs_scratch);
            times.push(start.elapsed());
        }

        let avg_us = times.iter().map(|t| t.as_micros() as f64).sum::<f64>() / iters as f64;
        let min_us = times
            .iter()
            .map(|t| t.as_micros() as f64)
            .fold(f64::INFINITY, f64::min);
        eprintln!("  Expander GKR:    avg={avg_us:.0}µs  min={min_us:.0}µs  ({iters} iters)");
    }

    {
        let (jst_circuit, witness) = build_jst_circuit(depth, log_size);
        let layer_vals = jst_circuit.evaluate(&witness);

        {
            let mut t = Sha256Transcript::default();
            let proof =
                jst_prove_eval::<Goldilocks, Sha256Transcript>(&jst_circuit, &layer_vals, &mut t);
            let mut vt = Sha256Transcript::default();
            assert!(
                jst_verify::<Goldilocks, Sha256Transcript>(&jst_circuit, &witness, &proof, &mut vt),
                "jst_gkr proof verification failed at log_size={log_size} depth={depth}"
            );
        }

        for _ in 0..warmup {
            let mut t = Sha256Transcript::default();
            jst_prove_eval::<Goldilocks, Sha256Transcript>(&jst_circuit, &layer_vals, &mut t);
        }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut t = Sha256Transcript::default();
            let start = Instant::now();
            let layer_vals = jst_circuit.evaluate(&witness);
            jst_prove_eval::<Goldilocks, Sha256Transcript>(&jst_circuit, &layer_vals, &mut t);
            times.push(start.elapsed());
        }

        let avg_us = times.iter().map(|t| t.as_micros() as f64).sum::<f64>() / iters as f64;
        let min_us = times
            .iter()
            .map(|t| t.as_micros() as f64)
            .fold(f64::INFINITY, f64::min);
        eprintln!("  Clean-room GKR:  avg={avg_us:.0}µs  min={min_us:.0}µs  ({iters} iters)");
    }
}

fn main() {
    let warmup = 5;
    let iters = 20;

    for &log_size in &[8, 10, 12, 14, 16] {
        bench_one("Width scaling (d=4)", log_size, 4, warmup, iters);
    }

    for &depth in &[2, 4, 8, 16] {
        bench_one("Depth scaling (w=12)", 12, depth, warmup, iters);
    }
}
