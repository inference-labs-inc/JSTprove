use arith::Field;
use ark_std::test_rng;
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, StructuredReferenceString,
};
use polynomials::{MultiLinearPoly, MultilinearExtension, MutableMultilinearExtension};

#[allow(clippy::type_complexity)]
pub fn expander_pcs_init_testing_only<FieldConfig: FieldEngine, PCS: ExpanderPCS<FieldConfig>>(
    n_input_vars: usize,
    mpi_config: &impl MPIEngine,
) -> (
    PCS::Params,
    <PCS::SRS as StructuredReferenceString>::PKey,
    <PCS::SRS as StructuredReferenceString>::VKey,
    PCS::ScratchPad,
) {
    let mut rng = test_rng();

    let pcs_params =
        <PCS as ExpanderPCS<FieldConfig>>::gen_params(n_input_vars, mpi_config.world_size());
    let pcs_setup = <PCS as ExpanderPCS<FieldConfig>>::gen_or_load_srs_for_testing(
        &pcs_params,
        mpi_config,
        &mut rng,
        None,
    );

    let (pcs_proving_key, pcs_verification_key) = pcs_setup.into_keys();
    let pcs_scratch = <PCS as ExpanderPCS<FieldConfig>>::init_scratch_pad(&pcs_params, mpi_config);

    (
        pcs_params,
        pcs_proving_key,
        pcs_verification_key,
        pcs_scratch,
    )
}

#[inline(always)]
pub fn lift_poly_to_n_vars<F: Field>(
    poly: &impl MultilinearExtension<F>,
    n_vars: usize,
) -> MultiLinearPoly<F> {
    let mut poly = MultiLinearPoly::new(poly.hypercube_basis());
    poly.lift_to_n_vars(n_vars);
    poly
}

#[inline(always)]
pub fn lift_expander_challenge_to_n_vars<F: FieldEngine>(
    expander_challenge: &ExpanderSingleVarChallenge<F>,
    n_vars: usize,
) -> ExpanderSingleVarChallenge<F> {
    let mut expander_challenge = expander_challenge.clone();
    expander_challenge
        .rz
        .resize(n_vars, F::ChallengeField::zero());
    expander_challenge
}

#[inline(always)]
pub fn lift_poly_and_expander_challenge_to_n_vars<F: FieldEngine>(
    poly: &impl MultilinearExtension<F::SimdCircuitField>,
    expander_challenge: &ExpanderSingleVarChallenge<F>,
    n_vars: usize,
) -> (
    MultiLinearPoly<F::SimdCircuitField>,
    ExpanderSingleVarChallenge<F>,
) {
    (
        lift_poly_to_n_vars(poly, n_vars),
        lift_expander_challenge_to_n_vars(expander_challenge, n_vars),
    )
}
