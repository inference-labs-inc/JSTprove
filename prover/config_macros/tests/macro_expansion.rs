use std::any::type_name;

use config_macros::declare_gkr_config;
use gf2::GF2x128;
use gkr_engine::{
    BN254Config, BabyBearx16Config, FieldEngine, GF2ExtConfig, GKREngine, GKRScheme,
    Goldilocksx8Config, M31x16Config, MPIConfig,
};
use gkr_hashers::{Keccak256hasher, MiMC5FiatShamirHasher, PoseidonFiatShamirHasher, SHA256hasher};
use halo2curves::bn256::Bn256;
use mersenne31::M31x16;
use poly_commit::{HyperBiKZGPCS, OrionPCSForGKR, RawExpanderGKR};
use transcript::BytesHashTranscript;

fn print_type_name<Cfg: GKREngine>() {
    println!("{}: {:?}", type_name::<Cfg>(), Cfg::SCHEME);
}

#[test]
fn main() {
    declare_gkr_config!(
        M31ExtSha256Config,
        FieldType::M31x16,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        M31ExtPoseidonRawConfig,
        FieldType::M31x16,
        FiatShamirHashType::Poseidon,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        M31ExtPoseidonOrionConfig,
        FieldType::M31x16,
        FiatShamirHashType::Poseidon,
        PolynomialCommitmentType::Orion,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        BN254MIMCConfig,
        FieldType::BN254,
        FiatShamirHashType::MIMC5,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        BN254MIMCKZGConfig,
        FieldType::BN254,
        FiatShamirHashType::MIMC5,
        PolynomialCommitmentType::KZG,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        GF2ExtKeccak256Config,
        FieldType::GF2Ext128,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        GF2ExtKeccak256OrionConfig,
        FieldType::GF2Ext128,
        FiatShamirHashType::Keccak256,
        PolynomialCommitmentType::Orion,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        GoldilocksExtSHA256Config,
        FieldType::Goldilocksx8,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );
    declare_gkr_config!(
        BabyBearExtSHA256Config,
        FieldType::BabyBearx16,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::Vanilla,
    );

    print_type_name::<M31ExtSha256Config>();
    print_type_name::<M31ExtPoseidonRawConfig>();
    print_type_name::<M31ExtPoseidonOrionConfig>();
    print_type_name::<BN254MIMCConfig>();
    print_type_name::<BN254MIMCKZGConfig>();
    print_type_name::<GF2ExtKeccak256Config>();
    print_type_name::<GF2ExtKeccak256OrionConfig>();
    print_type_name::<GoldilocksExtSHA256Config>();
    print_type_name::<BabyBearExtSHA256Config>();
}
