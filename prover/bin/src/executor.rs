use std::{
    fs,
    io::Cursor,
    process::exit,
    sync::{Arc, Mutex},
};

use arith::Field;
use circuit::Circuit;
use clap::{Parser, Subcommand};
use gkr::{Prover, Verifier};
use gkr_engine::{
    BN254Config, FieldEngine, FieldType, GF2ExtConfig, GKREngine, Goldilocksx8Config, M31x16Config,
    MPIConfig, MPIEngine, Proof,
};
use log::info;
use poly_commit::expander_pcs_init_testing_only;
use serdes::{ExpSerde, SerdeError};
use warp::{Filter, http::StatusCode, reply};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct ExpanderExecArgs {
    /// Fiat-Shamir Hash: SHA256, or Poseidon, or MiMC5
    #[arg(short, long, default_value = "SHA256")]
    pub fiat_shamir_hash: String,

    /// Polynomial Commitment Scheme: Raw, or Orion
    #[arg(short, long, default_value = "Raw")]
    pub poly_commitment_scheme: String,

    /// Prove, Verify, or Serve subcommands
    #[clap(subcommand)]
    pub subcommands: ExpanderExecSubCommand,
}

#[derive(Debug, Subcommand, Clone)]
pub enum ExpanderExecSubCommand {
    Prove {
        /// Circuit File Path
        #[arg(short, long)]
        circuit_file: String,

        /// Witness File Path
        #[arg(short, long)]
        witness_file: String,

        /// Output Proof Path
        #[arg(short, long)]
        output_proof_file: String,
    },
    Verify {
        /// Circuit File Path
        #[arg(short, long)]
        circuit_file: String,

        /// Witness File Path
        #[arg(short, long)]
        witness_file: String,

        /// Output Proof Path
        #[arg(short, long)]
        input_proof_file: String,

        /// MPI size
        #[arg(short, long, default_value_t = 1)]
        mpi_size: u32,
    },
    Serve {
        /// Circuit File Path
        #[arg(short, long)]
        circuit_file: String,

        /// IP host
        #[arg(short, long)]
        host_ip: String,

        /// IP Port
        #[arg(short, long)]
        port: u16,
    },
}

pub fn dump_proof_and_claimed_v<F: Field>(
    proof: &Proof,
    claimed_v: &F,
) -> Result<Vec<u8>, SerdeError> {
    let mut bytes = Vec::new();

    proof.serialize_into(&mut bytes)?;
    claimed_v.serialize_into(&mut bytes)?;

    Ok(bytes)
}

pub fn load_proof_and_claimed_v<F: Field>(bytes: &[u8]) -> Result<(Proof, F), SerdeError> {
    let mut cursor = Cursor::new(bytes);

    let proof = Proof::deserialize_from(&mut cursor)?;
    let claimed_v = F::deserialize_from(&mut cursor)?;

    Ok((proof, claimed_v))
}

pub fn detect_field_type_from_circuit_file(circuit_file: &str) -> FieldType {
    // read last 32 byte of sentinel field element to determine field type
    let bytes = fs::read(circuit_file).expect("Unable to read circuit file.");
    let field_bytes = &bytes[8..8 + 32];
    match field_bytes.try_into().unwrap() {
        M31x16Config::SENTINEL => FieldType::M31x16,
        BN254Config::SENTINEL => FieldType::BN254,
        GF2ExtConfig::SENTINEL => FieldType::GF2Ext128,
        Goldilocksx8Config::SENTINEL => FieldType::Goldilocksx8,
        _ => {
            println!("Unknown field type. Field byte value: {field_bytes:?}");
            exit(1);
        }
    }
}

pub fn prove<Cfg: GKREngine>(
    circuit: &mut Circuit<Cfg::FieldConfig>,
    mpi_config: MPIConfig,
) -> (
    <<Cfg as GKREngine>::FieldConfig as FieldEngine>::ChallengeField,
    Proof,
)
where
    Cfg::FieldConfig: FieldEngine,
{
    let mut prover = Prover::<Cfg>::new(mpi_config.clone());
    prover.prepare_mem(circuit);

    // TODO: Read PCS setup from files
    let (pcs_params, pcs_proving_key, _, mut pcs_scratch) =
        expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
            circuit.log_input_size(),
            &mpi_config,
        );

    eprintln!("proving");
    prover.prove(circuit, &pcs_params, &pcs_proving_key, &mut pcs_scratch)
}

pub fn verify<Cfg: GKREngine>(
    circuit: &mut Circuit<Cfg::FieldConfig>,
    mpi_config: MPIConfig,
    proof: &Proof,
    claimed_v: &<<Cfg as GKREngine>::FieldConfig as FieldEngine>::ChallengeField,
) -> bool {
    // TODO: Read PCS setup from files
    let (pcs_params, _, pcs_verification_key, _) = expander_pcs_init_testing_only::<
        Cfg::FieldConfig,
        Cfg::PCSConfig,
    >(circuit.log_input_size(), &mpi_config);
    let verifier = Verifier::<Cfg>::new(mpi_config);
    let public_input = circuit.public_input.clone();
    verifier.verify(
        circuit,
        &public_input,
        claimed_v,
        &pcs_params,
        &pcs_verification_key,
        proof,
    )
}

// The 'Prove' command can be run with mpi more than one process
// The 'Verify' command must be run with mpi size = 1
// The 'Serve' command must be run with mpi size = 1
pub async fn run_command<Cfg: GKREngine + 'static>(
    command: &ExpanderExecArgs,
    mpi_config: &MPIConfig,
) where
    Cfg::FieldConfig: FieldEngine,
{
    let subcommands = command.subcommands.clone();

    match subcommands {
        ExpanderExecSubCommand::Prove {
            circuit_file,
            witness_file,
            output_proof_file,
        } => {
            let mut circuit = Circuit::<Cfg::FieldConfig>::load_circuit(&circuit_file);
            let prover = Prover::<Cfg>::new(mpi_config.clone());

            circuit.prover_load_witness_file(&witness_file, mpi_config);
            let (claimed_v, proof) = prove::<Cfg>(&mut circuit, mpi_config.clone());

            if prover.mpi_config.is_root() {
                let bytes = dump_proof_and_claimed_v(&proof, &claimed_v)
                    .expect("Unable to serialize proof.");
                fs::write(output_proof_file, bytes).expect("Unable to write proof to file.");
            }
        }
        ExpanderExecSubCommand::Verify {
            circuit_file,
            witness_file,
            input_proof_file,
            mpi_size,
        } => {
            let mpi_config = MPIConfig::verifier_new(mpi_size as i32);
            let verifier = Verifier::<Cfg>::new(mpi_config);

            // this assertion is not right: the MPI size = 2 so that the verifier knows the prover
            // is running in 2 threads. The verifier itself is running in 1 thread.
            //
            // assert!(
            //     verifier.mpi_config.world_size() == 1,
            //     "Do not run verifier with mpiexec."
            // );

            println!("loading circuit file");

            let mut circuit = Circuit::<Cfg::FieldConfig>::load_circuit(&circuit_file);

            println!("loading witness file");

            circuit.verifier_load_witness_file(&witness_file, &verifier.mpi_config);

            println!("loading proof file");

            let bytes = fs::read(&input_proof_file).expect("Unable to read proof from file.");
            let (proof, claimed_v) = load_proof_and_claimed_v::<
                <Cfg::FieldConfig as FieldEngine>::ChallengeField,
            >(&bytes)
            .expect("Unable to deserialize proof.");

            println!("verifying proof");

            assert!(verify::<Cfg>(
                &mut circuit,
                verifier.mpi_config,
                &proof,
                &claimed_v
            ));

            println!("success");
        }
        ExpanderExecSubCommand::Serve {
            circuit_file,
            host_ip,
            port,
        } => {
            let mpi_config = mpi_config.clone();
            let prover = Prover::<Cfg>::new(mpi_config.clone());

            assert!(
                prover.mpi_config.world_size() == 1,
                "Serve mode is not compatible with mpi for now."
            );
            let host: [u8; 4] = host_ip
                .split('.')
                .map(|s| s.parse().unwrap())
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap();

            let circuit = Circuit::<Cfg::FieldConfig>::load_circuit(&circuit_file);

            // TODO: Read PCS setup from files
            let (pcs_params, pcs_proving_key, pcs_verification_key, pcs_scratch) =
                expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
                    circuit.log_input_size(),
                    &prover.mpi_config,
                );

            let circuit = Arc::new(Mutex::new(circuit));
            let circuit_clone_for_verifier = circuit.clone();
            let pcs_params = Arc::new(Mutex::new(pcs_params));
            let pcs_params_clone_for_verifier = pcs_params.clone();
            let pcs_proving_key = Arc::new(Mutex::new(pcs_proving_key));
            let pcs_verification_key = Arc::new(Mutex::new(pcs_verification_key));
            let pcs_scratch = Arc::new(Mutex::new(pcs_scratch));

            let ready_time = chrono::offset::Utc::now();
            let ready = warp::path("ready").map(move || {
                info!("Received ready request.");
                reply::with_status(format!("Ready since {ready_time:?}"), StatusCode::OK)
            });
            let prove =
                warp::path("prove")
                    .and(warp::body::bytes())
                    .map(move |bytes: bytes::Bytes| {
                        info!("Received prove request.");
                        let witness_bytes: Vec<u8> = bytes.to_vec();

                        let mut circuit = circuit.lock().unwrap();
                        let mut prover = Prover::<Cfg>::new(MPIConfig::prover_new());
                        prover.prepare_mem(&circuit);
                        let pcs_params = pcs_params.lock().unwrap();
                        let pcs_proving_key = pcs_proving_key.lock().unwrap();
                        let mut pcs_scratch = pcs_scratch.lock().unwrap();

                        circuit.load_witness_bytes(&witness_bytes, &prover.mpi_config, true, true);
                        let (claimed_v, proof) = prover.prove(
                            &mut circuit,
                            &pcs_params,
                            &pcs_proving_key,
                            &mut pcs_scratch,
                        );
                        reply::with_status(
                            dump_proof_and_claimed_v(&proof, &claimed_v).unwrap(),
                            StatusCode::OK,
                        )
                    });
            let verify =
                warp::path("verify")
                    .and(warp::body::bytes())
                    .map(move |bytes: bytes::Bytes| {
                        info!("Received verify request.");
                        let witness_and_proof_bytes: Vec<u8> = bytes.to_vec();
                        let length_of_witness_bytes =
                            u64::from_le_bytes(witness_and_proof_bytes[0..8].try_into().unwrap())
                                as usize;
                        let length_of_proof_bytes =
                            u64::from_le_bytes(witness_and_proof_bytes[8..16].try_into().unwrap())
                                as usize;
                        let witness_bytes =
                            &witness_and_proof_bytes[16..16 + length_of_witness_bytes];
                        let proof_bytes = &witness_and_proof_bytes[16 + length_of_witness_bytes
                            ..16 + length_of_witness_bytes + length_of_proof_bytes];

                        let mut circuit = circuit_clone_for_verifier.lock().unwrap();
                        let verifier = Verifier::<Cfg>::new(MPIConfig::verifier_new(1));
                        let pcs_verification_key = pcs_verification_key.lock().unwrap();
                        circuit.load_witness_bytes(
                            witness_bytes,
                            &verifier.mpi_config,
                            false,
                            true,
                        );
                        let public_input = circuit.public_input.clone();
                        let (proof, claimed_v) = load_proof_and_claimed_v(proof_bytes).unwrap();
                        if verifier.verify(
                            &mut circuit,
                            &public_input,
                            &claimed_v,
                            &pcs_params_clone_for_verifier.lock().unwrap(),
                            &pcs_verification_key,
                            &proof,
                        ) {
                            "success".to_string()
                        } else {
                            "failure".to_string()
                        }
                    });
            warp::serve(
                warp::post()
                    .and(prove.or(verify))
                    .or(warp::get().and(ready)),
            )
            .run((host, port))
            .await;
        }
    }
}
