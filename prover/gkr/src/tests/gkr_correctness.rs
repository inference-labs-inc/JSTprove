use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::time::Instant;
use std::{fs, panic};

use arith::Field;
use circuit::Circuit;
use config_macros::declare_gkr_config;
use gf2::GF2x128;
use gkr_engine::{
    root_println, BN254Config, BabyBearx16Config, FieldEngine, FieldType, GF2ExtConfig, GKREngine,
    GKRScheme, Goldilocksx1Config, Goldilocksx8Config, M31x16Config, M31x1Config, MPIConfig,
    MPIEngine,
};
use gkr_hashers::{Keccak256hasher, MiMC5FiatShamirHasher, PoseidonFiatShamirHasher, SHA256hasher};
use halo2curves::bn256::{Bn256, G1Affine};
use mersenne31::M31x16;
use poly_commit::{
    expander_pcs_init_testing_only, HyperBiKZGPCS, HyraxPCS, OrionPCSForGKR, RawExpanderGKR,
};
use rand::Rng;
use serdes::ExpSerde;
use sha2::Digest;
use transcript::BytesHashTranscript;

use crate::{utils::*, Prover, Verifier};

#[test]
fn test_gkr_correctness() {
    // Initialize logger
    env_logger::init();
    let mpi_config = MPIConfig::prover_new();

    declare_gkr_config!(
        C0,
        FieldType::GF2Ext128,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C1,
        FieldType::M31x16,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C2,
        FieldType::BN254,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C3,
        FieldType::GF2Ext128,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C4,
        FieldType::M31x16,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C5,
        FieldType::BN254,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C6,
        FieldType::BN254,
        FiatShamirHashType::MIMC5,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C7,
        FieldType::GF2Ext128,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Orion,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C8,
        FieldType::M31x16,
        FiatShamirHashType::Poseidon,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C9,
        FieldType::M31x16,
        FiatShamirHashType::Poseidon,
        PolynomialCommitmentType::Orion,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C10,
        FieldType::BN254,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Hyrax,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C11,
        FieldType::BN254,
        FiatShamirHashType::MIMC5,
        PolynomialCommitmentType::KZG,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C12,
        FieldType::Goldilocksx8,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C13,
        FieldType::M31x1,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C14,
        FieldType::BabyBearx16,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        C15,
        FieldType::Goldilocksx1,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    test_gkr_correctness_helper::<C0>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C1>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C2>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C3>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C4>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C5>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C6>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C7>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C8>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C9>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C10>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C11>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C12>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C13>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C14>(mpi_config.clone(), None);
    test_gkr_correctness_helper::<C15>(mpi_config.clone(), None);
}

#[allow(unreachable_patterns)]
fn test_gkr_correctness_helper<Cfg: GKREngine>(mpi_config: MPIConfig, write_proof_to: Option<&str>)
where
    Cfg::FieldConfig: FieldEngine,
{
    root_println!(mpi_config, "============== start ===============");
    root_println!(
        mpi_config,
        "Field Type: {:?}",
        <Cfg::FieldConfig as FieldEngine>::FIELD_TYPE
    );
    let circuit_copy_size: usize = match <Cfg::FieldConfig as FieldEngine>::FIELD_TYPE {
        FieldType::GF2Ext128 => 1,
        FieldType::M31x16 => 2,
        FieldType::BN254 => 2,
        FieldType::Goldilocksx1 => 2,
        FieldType::Goldilocksx8 => 2,
        FieldType::M31x1 => 2,
        FieldType::BabyBearx16 => 2,
        _ => unreachable!(),
    };
    root_println!(
        mpi_config,
        "Proving {} keccak instances at once.",
        circuit_copy_size * <Cfg::FieldConfig as FieldEngine>::get_field_pack_size()
    );
    root_println!(mpi_config, "Config created.");

    let circuit_path = match <Cfg::FieldConfig as FieldEngine>::FIELD_TYPE {
        FieldType::GF2Ext128 => "../".to_owned() + KECCAK_GF2_CIRCUIT,
        FieldType::M31x1 => "../".to_owned() + KECCAK_M31_CIRCUIT,
        FieldType::M31x16 => "../".to_owned() + KECCAK_M31_CIRCUIT,
        FieldType::BN254 => "../".to_owned() + KECCAK_BN254_CIRCUIT,
        FieldType::Goldilocksx1 => "../".to_owned() + KECCAK_GOLDILOCKS_CIRCUIT,
        FieldType::Goldilocksx8 => "../".to_owned() + KECCAK_GOLDILOCKS_CIRCUIT,
        FieldType::BabyBearx16 => "../".to_owned() + KECCAK_BABYBEAR_CIRCUIT,
        _ => unreachable!(),
    };
    let mut circuit = Circuit::<Cfg::FieldConfig>::load_circuit(&circuit_path);
    root_println!(mpi_config, "Circuit loaded.");

    let witness_path = match <Cfg::FieldConfig as FieldEngine>::FIELD_TYPE {
        FieldType::GF2Ext128 => "../".to_owned() + KECCAK_GF2_WITNESS,
        FieldType::M31x1 => "../".to_owned() + KECCAK_M31_WITNESS,
        FieldType::M31x16 => "../".to_owned() + KECCAK_M31_WITNESS,
        FieldType::BN254 => "../".to_owned() + KECCAK_BN254_WITNESS,
        FieldType::Goldilocksx1 => "../".to_owned() + KECCAK_GOLDILOCKS_WITNESS,
        FieldType::Goldilocksx8 => "../".to_owned() + KECCAK_GOLDILOCKS_WITNESS,
        FieldType::BabyBearx16 => "../".to_owned() + KECCAK_BABYBEAR_WITNESS,
        _ => unreachable!(),
    };
    circuit.load_witness_allow_padding_testing_only(&witness_path, &mpi_config);
    root_println!(mpi_config, "Witness loaded.");

    circuit.evaluate();
    let output = &circuit.layers.last().unwrap().output_vals;
    assert!(output[..circuit.expected_num_output_zeros]
        .iter()
        .all(|f| f.is_zero()));

    let mut prover = Prover::<Cfg>::new(mpi_config.clone());
    prover.prepare_mem(&circuit);

    let (pcs_params, pcs_proving_key, pcs_verification_key, mut pcs_scratch) =
        expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
            circuit.log_input_size(),
            &mpi_config,
        );

    let proving_start = Instant::now();
    let (claimed_v, proof) = prover.prove(
        &mut circuit,
        &pcs_params,
        &pcs_proving_key,
        &mut pcs_scratch,
    );
    root_println!(
        mpi_config,
        "Proving time: {} μs",
        proving_start.elapsed().as_micros()
    );

    root_println!(
        mpi_config,
        "Proof generated. Size: {} bytes",
        proof.bytes.len()
    );
    root_println!(mpi_config,);

    root_println!(mpi_config, "Proof hash: ");
    sha2::Sha256::digest(&proof.bytes)
        .iter()
        .for_each(|b| print!("{} ", b));
    root_println!(mpi_config,);

    let mut public_input_gathered = if mpi_config.is_root() {
        vec![
            <Cfg::FieldConfig as FieldEngine>::SimdCircuitField::ZERO;
            circuit.public_input.len() * mpi_config.world_size()
        ]
    } else {
        vec![]
    };
    mpi_config.gather_vec(&circuit.public_input, &mut public_input_gathered);

    // Verify
    if mpi_config.is_root() {
        if let Some(str) = write_proof_to {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(str)
                .unwrap();

            let mut buf = vec![];
            proof.serialize_into(&mut buf).unwrap();
            file.write_all(&buf).unwrap();
        }
        let verifier = Verifier::<Cfg>::new(mpi_config.clone());
        println!("Verifier created.");
        let verification_start = Instant::now();
        assert!(verifier.verify(
            &mut circuit,
            &public_input_gathered,
            &claimed_v,
            &pcs_params,
            &pcs_verification_key,
            &proof
        ));
        println!(
            "Verification time: {} μs",
            verification_start.elapsed().as_micros()
        );

        let par_verification_start = Instant::now();
        assert!(verifier.par_verify(
            &mut circuit,
            &public_input_gathered,
            &claimed_v,
            &pcs_params,
            &pcs_verification_key,
            &proof
        ));
        println!(
            "Multi-core Verification time: {} μs",
            par_verification_start.elapsed().as_micros()
        );
        println!("Correct proof verified.");

        let mut bad_proof = proof.clone();
        let rng = &mut rand::thread_rng();
        let random_idx = rng.gen_range(0..bad_proof.bytes.len());
        let random_change = rng.gen_range(1..256) as u8;
        bad_proof.bytes[random_idx] ^= random_change;

        // Catch the panic and treat it as returning `false`
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            verifier.verify(
                &mut circuit,
                &public_input_gathered,
                &claimed_v,
                &pcs_params,
                &pcs_verification_key,
                &bad_proof,
            )
        }));
        let final_result = result.unwrap_or_default();
        assert!(!final_result,);

        let par_result = panic::catch_unwind(AssertUnwindSafe(|| {
            verifier.par_verify(
                &mut circuit,
                &public_input_gathered,
                &claimed_v,
                &pcs_params,
                &pcs_verification_key,
                &bad_proof,
            )
        }));
        let final_par_result = par_result.unwrap_or_default();
        assert!(!final_par_result,);

        println!("Bad proof rejected.");
        println!("============== end ===============");
    }
}
