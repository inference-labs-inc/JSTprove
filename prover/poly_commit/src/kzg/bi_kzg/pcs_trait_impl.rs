use std::marker::PhantomData;

use arith::ExtensionField;
use gkr_engine::{StructuredReferenceString, Transcript};
use halo2curves::{
    ff::PrimeField,
    pairing::{Engine, MultiMillerLoop},
    CurveAffine,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};
use serdes::ExpSerde;

use crate::{
    coeff_form_uni_hyperkzg_open, coeff_form_uni_hyperkzg_verify, coeff_form_uni_kzg_commit,
    HyperUniKZGOpening, PolynomialCommitmentScheme,
};

use super::{
    generate_coef_form_bi_kzg_local_srs_for_testing, BiKZGCommitment, CoefFormBiKZGLocalSRS,
    HyperBiKZGOpening,
};

pub struct HyperBiKZGPCS<E>
where
    E: Engine,
    E::Fr: ExtensionField,
{
    _marker_e: PhantomData<E>,
}

impl<E> HyperBiKZGPCS<E>
where
    E: Engine,
    E::Fr: ExtensionField,
{
    pub const MINIMUM_SUPPORTED_NUM_VARS: usize = 2;
}

impl<E> PolynomialCommitmentScheme<E::Fr> for HyperBiKZGPCS<E>
where
    E: Engine + MultiMillerLoop,
    E::Fr: ExtensionField + PrimeField,
    E::G1Affine: ExpSerde + Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    E::G2Affine: ExpSerde + Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2>,
{
    const NAME: &'static str = "HyperBiKZGPCS";

    type Params = usize;
    type Poly = MultiLinearPoly<E::Fr>;
    type EvalPoint = Vec<E::Fr>;
    type ScratchPad = ();

    type SRS = CoefFormBiKZGLocalSRS<E>;
    type Commitment = BiKZGCommitment<E>;
    type Opening = HyperBiKZGOpening<E>;

    fn init_scratch_pad(_params: &Self::Params) -> Self::ScratchPad {}

    fn gen_srs_for_testing(params: &Self::Params, rng: impl rand::RngCore) -> (Self::SRS, usize) {
        let local_num_vars = if *params == 0 { 1 } else { *params };

        let length = 1 << local_num_vars;
        let srs = generate_coef_form_bi_kzg_local_srs_for_testing(length, 1, 0, rng);
        (srs, local_num_vars)
    }

    fn commit(
        _params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        _scratch_pad: &mut Self::ScratchPad,
    ) -> Self::Commitment {
        let local_commitment =
            coeff_form_uni_kzg_commit(&proving_key.tau_x_srs, poly.hypercube_basis_ref());

        BiKZGCommitment(local_commitment)
    }

    fn open(
        _params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        x: &Self::EvalPoint,
        _scratch_pad: &Self::ScratchPad,
        transcript: &mut impl Transcript,
    ) -> (E::Fr, Self::Opening) {
        let (eval, hyperkzg_opening) = coeff_form_uni_hyperkzg_open(
            &proving_key.tau_x_srs,
            poly.hypercube_basis_ref(),
            x,
            transcript,
        );

        let hyper_bikzg_opening: HyperBiKZGOpening<E> = hyperkzg_opening.into();
        (eval, hyper_bikzg_opening)
    }

    fn verify(
        _params: &Self::Params,
        verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        x: &Self::EvalPoint,
        v: E::Fr,
        opening: &Self::Opening,
        transcript: &mut impl Transcript,
    ) -> bool {
        let hyper_bikzg_opening = opening.clone();
        let hyper_kzg_opening: HyperUniKZGOpening<E> = hyper_bikzg_opening.into();

        coeff_form_uni_hyperkzg_verify(
            &verifying_key.into(),
            commitment.0,
            x,
            v,
            &hyper_kzg_opening,
            transcript,
        )
    }
}
