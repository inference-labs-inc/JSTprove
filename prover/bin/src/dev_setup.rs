#![allow(clippy::pedantic, clippy::all)]
//! Development environment setup
//! Download precomputed circuit and witness files.
//! (Those files can be generated and double checked with `ExpanderCompilerCollection`)
//! Download the precomputed proofs for these circuits and witnesses.

use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use circuit::Circuit;
use clap::Parser;
use gkr::Verifier;
use gkr::{
    BN254ConfigSha2Raw, GF2ExtConfigSha2Raw, M31x16ConfigSha2RawVanilla, Prover,
    utils::{
        KECCAK_BN254_CIRCUIT, KECCAK_BN254_WITNESS, KECCAK_GF2_CIRCUIT, KECCAK_GF2_WITNESS,
        KECCAK_M31_CIRCUIT, KECCAK_M31_WITNESS, dev_env_data_setup,
    },
};
use gkr_engine::{FieldEngine, FieldType, GKREngine, MPIConfig};
use poly_commit::expander_pcs_init_testing_only;
use serdes::ExpSerde;

/// ...
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// compare generated and downloaded proof files
    /// default: false
    #[arg(short, long, default_value_t = false)]
    compare: bool,
}

fn main() {
    let args = Args::parse();

    dev_env_data_setup();
    if args.compare {
        // check if the downloaded proofs match the one been generated
        let mpi_config = MPIConfig::prover_new();

        proof_gen::<GF2ExtConfigSha2Raw>(mpi_config.clone());
        proof_gen::<M31x16ConfigSha2RawVanilla>(mpi_config.clone());
        proof_gen::<BN254ConfigSha2Raw>(mpi_config.clone());
        compare_proof_files();
    }
}

fn proof_gen<C: GKREngine>(mpi_config: MPIConfig) {
    // load circuit
    let mut circuit = match C::FieldConfig::FIELD_TYPE {
        FieldType::GF2Ext128 => Circuit::<C::FieldConfig>::load_circuit(KECCAK_GF2_CIRCUIT),
        FieldType::M31x16 => Circuit::<C::FieldConfig>::load_circuit(KECCAK_M31_CIRCUIT),
        FieldType::BN254 => Circuit::<C::FieldConfig>::load_circuit(KECCAK_BN254_CIRCUIT),
        _ => unreachable!(),
    };

    let witness_path = match C::FieldConfig::FIELD_TYPE {
        FieldType::GF2Ext128 => KECCAK_GF2_WITNESS,
        FieldType::M31x16 => KECCAK_M31_WITNESS,
        FieldType::BN254 => KECCAK_BN254_WITNESS,
        _ => unreachable!(),
    };

    let proof_file_name = match C::FieldConfig::FIELD_TYPE {
        FieldType::GF2Ext128 => "data/proof_gf2_regen.txt",
        FieldType::M31x16 => "data/proof_m31_regen.txt",
        FieldType::BN254 => "data/proof_bn254_regen.txt",
        _ => unreachable!(),
    };

    circuit.load_witness_allow_padding_testing_only(witness_path, &mpi_config);

    circuit.evaluate();

    let (pcs_params, pcs_proving_key, pcs_verification_key, pcs_scratch) =
        expander_pcs_init_testing_only::<C::FieldConfig, C::PCSConfig>(
            circuit.log_input_size(),
            &mpi_config,
        );

    let (claim, proof) = {
        // generate the proof
        let mut local_circuit = circuit.clone();
        let pcs_params = pcs_params.clone();
        let pcs_proving_key = pcs_proving_key.clone();
        let mut pcs_scratch = pcs_scratch.clone();
        let mut prover = Prover::<C>::new(mpi_config.clone());
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

        let mut file = std::fs::File::create(proof_file_name).unwrap();
        file.write_all(buf.as_ref()).expect("Unable to write data");
        println!("{proof_file_name} generated");

        (claim, proof)
    };

    {
        // verify the proof
        let verifier = Verifier::<C>::new(mpi_config.clone());

        let public_input = circuit.public_input.clone();

        assert!(verifier.verify(
            &mut circuit,
            &public_input,
            &claim,
            &pcs_params,
            &pcs_verification_key,
            &proof
        ));

        println!("{proof_file_name} verified");
    }
}

fn compare_proof_files() {
    let field_types = [
        ("GF2", "data/proof_gf2_regen.txt", "data/proof_gf2.txt"),
        ("M31", "data/proof_m31_regen.txt", "data/proof_m31.txt"),
        (
            "BN254",
            "data/proof_bn254_regen.txt",
            "data/proof_bn254.txt",
        ),
    ];

    for (field_type, generated_path, downloaded_path) in &field_types {
        println!("\nComparing {field_type} proof files:");

        // Check if both files exist
        if !Path::new(generated_path).exists() {
            println!("Error: Generated file '{generated_path}' does not exist");
            continue;
        }

        if !Path::new(downloaded_path).exists() {
            println!("Error: Downloaded file '{downloaded_path}' does not exist");
            continue;
        }

        // Read the files
        let mut generated_content = Vec::new();
        let mut downloaded_content = Vec::new();

        let mut generated_file = File::open(generated_path).expect("Failed to open generated file");
        let mut downloaded_file =
            File::open(downloaded_path).expect("Failed to open downloaded file");

        generated_file
            .read_to_end(&mut generated_content)
            .expect("Failed to read generated file");
        downloaded_file
            .read_to_end(&mut downloaded_content)
            .expect("Failed to read downloaded file");

        // Compare file sizes
        println!("Generated file size: {} bytes", generated_content.len());
        println!("Downloaded file size: {} bytes", downloaded_content.len());

        if generated_content.len() != downloaded_content.len() {
            println!("Files have different sizes!");
            continue;
        }

        // Compare content
        let mut differences = 0;
        for (i, (gen_byte, down_byte)) in generated_content
            .iter()
            .zip(downloaded_content.iter())
            .enumerate()
        {
            if gen_byte != down_byte {
                differences += 1;
                if differences <= 10 {
                    // Show only first 10 differences
                    println!(
                        "Difference at byte {i}: generated={gen_byte:02x}, downloaded={down_byte:02x}"
                    );
                }
            }
        }

        if differences == 0 {
            println!("Files are identical!");
        } else {
            println!("Files differ in {differences} bytes");
        }
    }
}
