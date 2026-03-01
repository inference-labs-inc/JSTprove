use circuit::Circuit;
use config_macros::declare_gkr_config;
use gkr_engine::{
    BN254Config, FieldEngine, FieldType, GF2ExtConfig, GKREngine, GKRScheme, Goldilocksx8Config,
    M31x16Config, MPIConfig, MPIEngine,
};
use gkr_hashers::SHA256hasher;
use poly_commit::RawExpanderGKR;
use transcript::BytesHashTranscript;

pub const KECCAK_M31_CIRCUIT: &str = "data/circuit_m31.txt";
pub const KECCAK_GF2_CIRCUIT: &str = "data/circuit_gf2.txt";
pub const KECCAK_BN254_CIRCUIT: &str = "data/circuit_bn254.txt";
pub const KECCAK_GOLDILOCKS_CIRCUIT: &str = "data/circuit_goldilocks.txt";

declare_gkr_config!(
    M31x16ConfigSha2Raw,
    FieldType::M31x16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

declare_gkr_config!(
    GF2ExtConfigSha2Raw,
    FieldType::GF2Ext128,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

declare_gkr_config!(
    BN254ConfigSha2Raw,
    FieldType::BN254,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

declare_gkr_config!(
    Goldilocksx8ConfigSha2Raw,
    FieldType::Goldilocksx8,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

#[allow(unreachable_patterns)]
fn load_circuit<Cfg: GKREngine>(mpi_config: &MPIConfig) -> Option<Circuit<Cfg::FieldConfig>> {
    let circuit_path = match <Cfg as GKREngine>::FieldConfig::FIELD_TYPE {
        FieldType::GF2Ext128 => "../".to_owned() + KECCAK_GF2_CIRCUIT,
        FieldType::M31x16 => "../".to_owned() + KECCAK_M31_CIRCUIT,
        FieldType::BN254 => "../".to_owned() + KECCAK_BN254_CIRCUIT,
        FieldType::Goldilocksx8 => "../".to_owned() + KECCAK_GOLDILOCKS_CIRCUIT,
        _ => unreachable!(),
    };

    if mpi_config.is_root() {
        Some(Circuit::<Cfg::FieldConfig>::load_circuit(&circuit_path))
    } else {
        None
    }
}

#[test]
#[ignore]
fn test_shared_mem() {
    let mpi_config = MPIConfig::prover_new();
    let _circuit = load_circuit::<M31x16ConfigSha2Raw>(&mpi_config);
    let _circuit = load_circuit::<GF2ExtConfigSha2Raw>(&mpi_config);
    let _circuit = load_circuit::<BN254ConfigSha2Raw>(&mpi_config);
    let _circuit = load_circuit::<Goldilocksx8ConfigSha2Raw>(&mpi_config);
}
