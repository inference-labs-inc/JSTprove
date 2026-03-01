pub mod prove_impl;
pub mod server_fns;
pub mod setup_impl;
pub mod verify_impl;

pub mod api_pcs_defered;
pub use api_pcs_defered::*;

use crate::frontend::BN254Config;
use expander_transcript::BytesHashTranscript;
use gkr_engine::{GKREngine, GKRScheme, MPIConfig};
use gkr_hashers::SHA256hasher;
use halo2curves::bn256::Bn256;
use poly_commit::HyperUniKZGPCS;

pub struct BN254ConfigSha2UniKZG;

impl GKREngine for BN254ConfigSha2UniKZG {
    type FieldConfig = <BN254Config as GKREngine>::FieldConfig;
    type MPIConfig = MPIConfig;
    type TranscriptConfig = BytesHashTranscript<SHA256hasher>;
    type PCSConfig = HyperUniKZGPCS<Bn256>;
    const SCHEME: GKRScheme = GKRScheme::Vanilla;
}
