use expander_utils::timer::Timer;
use gkr_engine::{ExpanderPCS, GKREngine, MPIConfig, StructuredReferenceString};
use polynomials::RefMultiLinearPoly;

use crate::{
    frontend::{Config, SIMDField},
    zkcuda::proving_system::expander::structs::{ExpanderCommitment, ExpanderCommitmentState},
};

pub fn local_commit_impl<C, ECCConfig>(
    p_key: &<<C::PCSConfig as ExpanderPCS<C::FieldConfig>>::SRS as StructuredReferenceString>::PKey,
    vals: &[SIMDField<C>],
) -> (
    ExpanderCommitment<C::FieldConfig, C::PCSConfig>,
    ExpanderCommitmentState<C::FieldConfig, C::PCSConfig>,
)
where
    C: GKREngine,
    ECCConfig: Config<FieldConfig = C::FieldConfig>,
{
    let timer = Timer::new("commit", true);

    let n_vars = vals.len().ilog2() as usize;
    let params = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::gen_params(n_vars, 1);

    let mpi_config = MPIConfig::prover_new();

    let mut scratch =
        <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::init_scratch_pad(&params, &mpi_config);

    let commitment = <C::PCSConfig as ExpanderPCS<C::FieldConfig>>::commit(
        &params,
        &mpi_config,
        p_key,
        &RefMultiLinearPoly::from_ref(vals),
        &mut scratch,
    )
    .unwrap();

    timer.stop();
    (
        ExpanderCommitment {
            vals_len: vals.len(),
            commitment,
        },
        ExpanderCommitmentState { scratch },
    )
}
