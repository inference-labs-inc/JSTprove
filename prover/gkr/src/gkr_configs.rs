use config_macros::declare_gkr_config;
use gf2::GF2x128;
use gkr_engine::{
    BN254Config, BabyBearx16Config, FieldEngine, GF2ExtConfig, GKREngine, GKRScheme,
    Goldilocksx1Config, Goldilocksx8Config, M31x16Config, M31x1Config, MPIConfig,
};
use gkr_hashers::{MiMC5FiatShamirHasher, PoseidonFiatShamirHasher, SHA256hasher};
use goldilocks::Goldilocksx8;
use halo2curves::bn256::{Bn256, G1Affine};
use mersenne31::M31x16;
use poly_commit::{raw::RawExpanderGKR, HyperBiKZGPCS, HyraxPCS, OrionPCSForGKR};
use transcript::BytesHashTranscript;

// ============== M31 ==============
declare_gkr_config!(
    pub M31x1ConfigSha2RawVanilla,
    FieldType::M31x1,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);
// ============== M31Ext3 ==============
declare_gkr_config!(
    pub M31x16ConfigPoseidonRawVanilla,
    FieldType::M31x16,
    FiatShamirHashType::Poseidon,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);
declare_gkr_config!(
    pub M31x16ConfigPoseidonRawSquare,
    FieldType::M31x16,
    FiatShamirHashType::Poseidon,
    PolynomialCommitmentType::Raw,
    GKRScheme::GkrSquare,
);
declare_gkr_config!(
    pub M31x16ConfigSha2OrionVanilla,
    FieldType::M31x16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Orion,
    GKRScheme::Vanilla,
);
declare_gkr_config!(
    pub M31x16ConfigSha2OrionSquare,
    FieldType::M31x16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Orion,
    GKRScheme::GkrSquare,
);
declare_gkr_config!(
    pub M31x16ConfigSha2RawVanilla,
    FieldType::M31x16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);
declare_gkr_config!(
    pub M31x16ConfigSha2RawSquare,
    FieldType::M31x16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::GkrSquare,
);

// ============== BN254 ==============
declare_gkr_config!(
    pub BN254ConfigMIMC5Raw,
    FieldType::BN254,
    FiatShamirHashType::MIMC5,
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
declare_gkr_config!(
    pub BN254ConfigSha2Hyrax,
    FieldType::BN254,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Hyrax,
    GKRScheme::Vanilla,
);
declare_gkr_config!(
    pub BN254ConfigSha2KZG,
    FieldType::BN254,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::KZG,
    GKRScheme::Vanilla,
);
declare_gkr_config!(
    pub BN254ConfigMIMC5KZG,
    FieldType::BN254,
    FiatShamirHashType::MIMC5,
    PolynomialCommitmentType::KZG,
    GKRScheme::Vanilla,
);

// ============== GF2 ==============
declare_gkr_config!(
    pub GF2ExtConfigSha2Orion,
    FieldType::GF2Ext128,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Orion,
    GKRScheme::Vanilla,
);
declare_gkr_config!(
    pub GF2ExtConfigSha2Raw,
    FieldType::GF2Ext128,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

// ============== Goldilocks ==============
declare_gkr_config!(
    pub Goldilocksx1ConfigSha2Raw,
    FieldType::Goldilocksx1,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

// ============== GoldilocksExt2 ==============
declare_gkr_config!(
    pub Goldilocksx8ConfigSha2Raw,
    FieldType::Goldilocksx8,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);

declare_gkr_config!(
    pub Goldilocksx8ConfigSha2Orion,
    FieldType::Goldilocksx8,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Orion,
    GKRScheme::Vanilla,
);

// ============== Babybear ==============
declare_gkr_config!(
    pub BabyBearx16ConfigSha2Raw,
    FieldType::BabyBearx16,
    FiatShamirHashType::SHA256,
    PolynomialCommitmentType::Raw,
    GKRScheme::Vanilla,
);
