use gkr::{BN254ConfigSha2Hyrax, BN254ConfigSha2Raw, M31x16ConfigSha2RawVanilla};
use gkr_engine::GKREngine;

use crate::{
    frontend::{BN254Config, Config, M31Config},
    zkcuda::proving_system::expander_pcs_defered::BN254ConfigSha2UniKZG,
};

pub trait ZKCudaConfig {
    type ECCConfig: Config;
    type GKRConfig: GKREngine<FieldConfig = <Self::ECCConfig as GKREngine>::FieldConfig>;

    const BATCH_PCS: bool = false;
}

pub type GetPCS<ZKCConfig> = <<ZKCConfig as ZKCudaConfig>::GKRConfig as GKREngine>::PCSConfig;
pub type GetTranscript<ZKCConfig> =
    <<ZKCConfig as ZKCudaConfig>::GKRConfig as GKREngine>::TranscriptConfig;
pub type GetFieldConfig<ZKCConfig> =
    <<ZKCConfig as ZKCudaConfig>::GKRConfig as GKREngine>::FieldConfig;

pub struct ZKCudaConfigImpl<ECC, GKR, const BATCH_PCS: bool>
where
    ECC: Config,
    GKR: GKREngine<FieldConfig = <ECC as GKREngine>::FieldConfig>,
{
    _phantom: std::marker::PhantomData<(ECC, GKR, bool)>,
}

impl<ECC, GKR, const BATCH_PCS: bool> ZKCudaConfig for ZKCudaConfigImpl<ECC, GKR, BATCH_PCS>
where
    ECC: Config,
    GKR: GKREngine<FieldConfig = <ECC as GKREngine>::FieldConfig>,
{
    type ECCConfig = ECC;
    type GKRConfig = GKR;

    const BATCH_PCS: bool = BATCH_PCS;
}

pub type ZKCudaBN254Hyrax = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Hyrax, false>;
pub type ZKCudaBN254KZG = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2UniKZG, false>;

pub type ZKCudaM31 = ZKCudaConfigImpl<M31Config, M31x16ConfigSha2RawVanilla, false>;
pub type ZKCudaGF2 = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Raw, false>;
pub type ZKCudaGoldilocks = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Raw, false>;
pub type ZKCudaBabyBear = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Raw, false>;

pub type ZKCudaBN254HyraxBatchPCS = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Hyrax, true>;
pub type ZKCudaBN254KZGBatchPCS = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2UniKZG, true>;

pub type ZKCudaM31BatchPCS = ZKCudaConfigImpl<M31Config, M31x16ConfigSha2RawVanilla, true>;
pub type ZKCudaGF2BatchPCS = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Raw, true>;
pub type ZKCudaGoldilocksBatchPCS = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Raw, true>;
pub type ZKCudaBabyBearBatchPCS = ZKCudaConfigImpl<BN254Config, BN254ConfigSha2Raw, true>;
