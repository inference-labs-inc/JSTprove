use circuit::Circuit;
use config_macros::declare_gkr_config;
use gkr_engine::{
    root_println, BN254Config, FieldEngine, FieldType, GF2ExtConfig, GKREngine, GKRScheme,
    M31x16Config, MPIConfig, MPIEngine,
};
use gkr_hashers::SHA256hasher;
use poly_commit::RawExpanderGKR;
use serdes::ExpSerde;
use transcript::BytesHashTranscript;

// circuit for repeating Keccak for 2 times
pub const KECCAK_M31_CIRCUIT: &str = "data/circuit_m31.txt";
pub const KECCAK_GF2_CIRCUIT: &str = "data/circuit_gf2.txt";
pub const KECCAK_BN254_CIRCUIT: &str = "data/circuit_bn254.txt";
pub const KECCAK_GOLDILOCKS_CIRCUIT: &str = "data/circuit_goldilocks.txt";

declare_gkr_config!(
    pub M31x16ConfigSha2Raw,
    FieldType::M31x16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

declare_gkr_config!(
    pub GF2ExtConfigSha2Raw,
    FieldType::GF2Ext128,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

declare_gkr_config!(
    pub BN254ConfigSha2Raw,
    FieldType::BN254,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

#[test]
#[ignore]
fn test_circuit_serde() {
    let mpi_config = MPIConfig::prover_new();
    test_circuit_serde_helper::<M31x16ConfigSha2Raw>(&mpi_config);
    test_circuit_serde_helper::<GF2ExtConfigSha2Raw>(&mpi_config);
    test_circuit_serde_helper::<BN254ConfigSha2Raw>(&mpi_config);
}

#[allow(unreachable_patterns)]
fn test_circuit_serde_helper<Cfg: GKREngine>(mpi_config: &MPIConfig) {
    root_println!(
        mpi_config,
        "Field Type: {:?}",
        <Cfg as GKREngine>::FieldConfig::FIELD_TYPE
    );

    let circuit_path = match Cfg::FieldConfig::FIELD_TYPE {
        FieldType::GF2Ext128 => "../".to_owned() + KECCAK_GF2_CIRCUIT,
        FieldType::M31x16 => "../".to_owned() + KECCAK_M31_CIRCUIT,
        FieldType::BN254 => "../".to_owned() + KECCAK_BN254_CIRCUIT,
        FieldType::Goldilocksx8 => "../".to_owned() + KECCAK_GOLDILOCKS_CIRCUIT,
        _ => unreachable!(),
    };
    let circuit = Circuit::<Cfg::FieldConfig>::load_circuit(&circuit_path);
    root_println!(mpi_config, "Circuit loaded.");

    let mut buffer = vec![];
    circuit.serialize_into(&mut buffer).unwrap();
    let circuit_deserialized = Circuit::<Cfg::FieldConfig>::deserialize_from(&buffer[..]).unwrap();

    let mut buffer2 = vec![];
    circuit_deserialized.serialize_into(&mut buffer2).unwrap();
    assert_eq!(buffer, buffer2);
}
