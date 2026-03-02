#![allow(clippy::pedantic, clippy::all)]
use std::{
    str::FromStr,
    sync::{Arc, Mutex},
    thread,
};

use circuit::Circuit;
use clap::Parser;
use gkr::{
    BN254ConfigMIMC5KZG, BN254ConfigSha2Hyrax, BN254ConfigSha2Raw, GF2ExtConfigSha2Orion,
    GF2ExtConfigSha2Raw, Goldilocksx8ConfigSha2Orion, Goldilocksx8ConfigSha2Raw,
    M31x1ConfigSha2RawVanilla, M31x16ConfigSha2OrionSquare, M31x16ConfigSha2OrionVanilla,
    M31x16ConfigSha2RawSquare, M31x16ConfigSha2RawVanilla, Prover,
    utils::{
        KECCAK_BABYBEAR_CIRCUIT, KECCAK_BABYBEAR_WITNESS, KECCAK_BN254_CIRCUIT,
        KECCAK_BN254_WITNESS, KECCAK_GF2_CIRCUIT, KECCAK_GF2_WITNESS, KECCAK_GOLDILOCKS_CIRCUIT,
        KECCAK_GOLDILOCKS_WITNESS, KECCAK_M31_CIRCUIT, KECCAK_M31_WITNESS, POSEIDON_M31_CIRCUIT,
        POSEIDON_M31_WITNESS,
    },
};
use gkr_engine::{
    ExpanderPCS, FieldEngine, FieldType, GKREngine, MPIConfig, MPIEngine, PolynomialCommitmentType,
};
use poly_commit::expander_pcs_init_testing_only;
use serdes::ExpSerde;

/// ...
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Field Identifier: fr, m31ext3, gf2ext128, goldilocks
    #[arg(short, long,default_value_t = String::from("m31ext3"))]
    field: String,

    // circuit: keccak, poseidon
    #[arg(short, long, default_value_t = String::from("keccak"))]
    circuit: String,

    /// Polynomial Commitment Scheme: Raw, Hyrax, Orion, KZG
    #[arg(short, long, default_value_t = String::from("Raw"))]
    pcs: String,

    /// number of repeat
    #[arg(short, long, default_value_t = 1)]
    repeats: usize,

    /// number of thread
    #[arg(short, long, default_value_t = 1)]
    threads: u64,
}

#[allow(static_mut_refs)]
fn main() {
    let args = Args::parse();
    print_info(&args);

    // This is a designated single-process benchmark
    let mpi_config = MPIConfig::prover_new();
    let pcs_type = PolynomialCommitmentType::from_str(&args.pcs).unwrap();

    match args.field.as_str() {
        "m31" => match pcs_type {
            PolynomialCommitmentType::Raw => match args.circuit.as_str() {
                "keccak" => run_benchmark::<M31x1ConfigSha2RawVanilla>(&args, mpi_config),
                _ => unreachable!(),
            },
            _ => unreachable!("Unsupported PCS type for M31"),
        },

        "m31ext3" => match pcs_type {
            PolynomialCommitmentType::Raw => match args.circuit.as_str() {
                "keccak" => run_benchmark::<M31x16ConfigSha2RawVanilla>(&args, mpi_config.clone()),
                "poseidon" => run_benchmark::<M31x16ConfigSha2RawSquare>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            PolynomialCommitmentType::Orion => match args.circuit.as_str() {
                "keccak" => {
                    run_benchmark::<M31x16ConfigSha2OrionVanilla>(&args, mpi_config.clone())
                }
                "poseidon" => {
                    run_benchmark::<M31x16ConfigSha2OrionSquare>(&args, mpi_config.clone())
                }
                _ => unreachable!(""),
            },
            _ => unreachable!("Unsupported PCS type for M31"),
        },
        "fr" => match pcs_type {
            PolynomialCommitmentType::Raw => match args.circuit.as_str() {
                "keccak" => run_benchmark::<BN254ConfigSha2Raw>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            PolynomialCommitmentType::Hyrax => match args.circuit.as_str() {
                "keccak" => run_benchmark::<BN254ConfigSha2Hyrax>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            PolynomialCommitmentType::KZG => match args.circuit.as_str() {
                "keccak" => run_benchmark::<BN254ConfigMIMC5KZG>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            _ => unreachable!("Unsupported PCS type for BN254"),
        },
        "gf2ext128" => match pcs_type {
            PolynomialCommitmentType::Raw => match args.circuit.as_str() {
                "keccak" => run_benchmark::<GF2ExtConfigSha2Raw>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            PolynomialCommitmentType::Orion => match args.circuit.as_str() {
                "keccak" => run_benchmark::<GF2ExtConfigSha2Orion>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            _ => unreachable!("Unsupported PCS type for GF2"),
        },
        "goldilocks" => match pcs_type {
            PolynomialCommitmentType::Raw => match args.circuit.as_str() {
                "keccak" => run_benchmark::<Goldilocksx8ConfigSha2Raw>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            PolynomialCommitmentType::Orion => match args.circuit.as_str() {
                "keccak" => run_benchmark::<Goldilocksx8ConfigSha2Orion>(&args, mpi_config.clone()),
                _ => unreachable!(),
            },
            _ => unreachable!("Unsupported PCS type for Goldilocks"),
        },
        _ => unreachable!(),
    };
}

fn run_benchmark<Cfg: GKREngine>(args: &Args, mpi_config: MPIConfig)
where
    <Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::ScratchPad: 'static,
    Cfg::FieldConfig: FieldEngine,
{
    let partial_proof_cnts = (0..args.threads)
        .map(|_| Arc::new(Mutex::new(0)))
        .collect::<Vec<_>>();

    let pack_size = <Cfg::FieldConfig as FieldEngine>::get_field_pack_size();

    // load circuit
    let mut circuit_template = match args.circuit.as_str() {
        "keccak" => match Cfg::FieldConfig::FIELD_TYPE {
            FieldType::GF2Ext128 => Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_GF2_CIRCUIT),
            FieldType::M31x1 => Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_M31_CIRCUIT),
            FieldType::M31x16 => Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_M31_CIRCUIT),
            FieldType::BN254 => Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_BN254_CIRCUIT),
            FieldType::BabyBearx16 => {
                Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_BABYBEAR_CIRCUIT)
            }
            FieldType::Goldilocksx1 => {
                Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_GOLDILOCKS_CIRCUIT)
            }
            FieldType::Goldilocksx8 => {
                Circuit::<Cfg::FieldConfig>::load_circuit(KECCAK_GOLDILOCKS_CIRCUIT)
            }
        },
        "poseidon" => match Cfg::FieldConfig::FIELD_TYPE {
            FieldType::M31x16 => Circuit::<Cfg::FieldConfig>::load_circuit(POSEIDON_M31_CIRCUIT),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };

    let witness_path = match args.circuit.as_str() {
        "keccak" => match Cfg::FieldConfig::FIELD_TYPE {
            FieldType::GF2Ext128 => KECCAK_GF2_WITNESS,
            FieldType::M31x1 => KECCAK_M31_WITNESS,
            FieldType::M31x16 => KECCAK_M31_WITNESS,
            FieldType::BN254 => KECCAK_BN254_WITNESS,
            FieldType::Goldilocksx1 => KECCAK_GOLDILOCKS_WITNESS,
            FieldType::Goldilocksx8 => KECCAK_GOLDILOCKS_WITNESS,
            FieldType::BabyBearx16 => KECCAK_BABYBEAR_WITNESS,
        },
        "poseidon" => match Cfg::FieldConfig::FIELD_TYPE {
            FieldType::M31x16 => POSEIDON_M31_WITNESS,
            _ => unreachable!("not supported"),
        },
        _ => unreachable!(),
    };

    circuit_template.load_witness_allow_padding_testing_only(witness_path, &mpi_config);

    let circuit_copy_size: usize = match (Cfg::FieldConfig::FIELD_TYPE, args.circuit.as_str()) {
        (FieldType::GF2Ext128, "keccak") => 1,
        (FieldType::M31x1, "keccak") => 2,
        (FieldType::M31x16, "keccak") => 2,
        (FieldType::M31x16, "poseidon") => 120,
        (FieldType::Goldilocksx8, "keccak") => 2,
        (FieldType::BabyBearx16, "keccak") => 2,
        (FieldType::BN254, "keccak") => 2,
        _ => unreachable!(),
    };

    let circuits = (0..args.threads)
        .map(|_| {
            let mut c = circuit_template.clone();
            c.evaluate();
            c
        })
        .collect::<Vec<_>>();

    println!("Circuit loaded!");

    let (pcs_params, pcs_proving_key, _pcs_verification_key, pcs_scratch) =
        expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
            circuit_template.log_input_size(),
            &mpi_config,
        );

    // calculate the proof size
    {
        let mut local_circuit = circuits[0].clone();
        let pcs_params = pcs_params.clone();
        let pcs_proving_key = pcs_proving_key.clone();
        let mut pcs_scratch = pcs_scratch.clone();
        let mut prover = Prover::<Cfg>::new(mpi_config.clone());
        prover.prepare_mem(&local_circuit);

        let (claim, proof) = prover.prove(
            &mut local_circuit,
            &pcs_params,
            &pcs_proving_key,
            &mut pcs_scratch,
        );
        let mut buf = Vec::new();
        claim.serialize_into(&mut buf).unwrap();
        proof.serialize_into(&mut buf).unwrap();
        println!("Proof size: {}", buf.len());

        // Write the serialized proof to a file
        std::fs::write("proof.txt", &buf).expect("Failed to write proof to file");
        println!("Proof written to proof.txt");
    }

    let start_time = std::time::Instant::now();
    let _ = circuits
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            let partial_proof_cnt = partial_proof_cnts[i].clone();
            let pcs_params = pcs_params.clone();
            let pcs_proving_key = pcs_proving_key.clone();
            let mut pcs_scratch = pcs_scratch.clone();
            thread::spawn(move || {
                // bench func
                let local_mpi_config = MPIConfig::prover_new();
                let mut prover = Prover::<Cfg>::new(local_mpi_config);
                prover.prepare_mem(&c);
                loop {
                    prover.prove(&mut c, &pcs_params, &pcs_proving_key, &mut pcs_scratch);
                    // update cnt
                    let mut cnt = partial_proof_cnt.lock().unwrap();
                    let proof_cnt_this_round = circuit_copy_size * pack_size;
                    *cnt += proof_cnt_this_round;
                }
            })
        })
        .collect::<Vec<_>>();

    println!("We are now calculating average throughput, please wait for 5 seconds");
    for i in 0..args.repeats {
        thread::sleep(std::time::Duration::from_secs(5));
        let stop_time = std::time::Instant::now();
        let duration = stop_time.duration_since(start_time);
        let mut total_proof_cnt = 0;
        for cnt in &partial_proof_cnts {
            total_proof_cnt += *cnt.lock().unwrap();
        }
        let throughput =
            total_proof_cnt as f64 / duration.as_secs_f64() * (mpi_config.world_size() as f64);
        println!("{}-bench: throughput: {} hashes/s", i, throughput.round());
    }
}

fn print_info(args: &Args) {
    let prover = match args.circuit.as_str() {
        "keccak" => "GKR",
        "poseidon" => "GKR^2",
        _ => unreachable!(),
    };

    println!("===============================");
    println!(
        "benchmarking {} with {} over {}",
        args.circuit, prover, args.field
    );
    println!("field:          {}", args.field);
    println!("#threads:       {}", args.threads);
    println!("#bench repeats: {}", args.repeats);
    println!("hash circuit:   {}", args.circuit);
    println!("PCS:            {}", args.pcs);
    println!("===============================")
}
