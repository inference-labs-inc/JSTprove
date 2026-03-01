use std::marker::PhantomData;

use arith::ExtensionField;
use gkr_engine::{StructuredReferenceString, Transcript};
use halo2curves::{ff::PrimeField, group::UncompressedEncoding, CurveAffine};
use polynomials::MultiLinearPoly;
use serdes::ExpSerde;

use crate::traits::BatchOpening;
use crate::{
    hyrax::hyrax_impl::{hyrax_commit, hyrax_open, hyrax_setup, hyrax_verify},
    traits::BatchOpeningPCS,
    HyraxCommitment, HyraxOpening, PedersenParams, PolynomialCommitmentScheme,
};

use super::hyrax_impl::hyrax_multi_points_batch_open_internal;
use super::hyrax_impl::hyrax_multi_points_batch_verify_internal;
use super::hyrax_impl::{hyrax_batch_open, hyrax_batch_verify};

pub struct HyraxPCS<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField,
    C::ScalarExt: ExtensionField,
{
    _phantom_c: PhantomData<C>,
}

impl<C> PolynomialCommitmentScheme<C::Scalar> for HyraxPCS<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    const NAME: &'static str = "HyraxPCS";

    type Params = usize;
    type Poly = MultiLinearPoly<C::Scalar>;
    type EvalPoint = Vec<C::Scalar>;
    type ScratchPad = ();

    type SRS = PedersenParams<C>;
    type Commitment = HyraxCommitment<C>;
    type Opening = HyraxOpening<C>;

    fn init_scratch_pad(_params: &Self::Params) -> Self::ScratchPad {}

    fn gen_srs_for_testing(params: &Self::Params, rng: impl rand::RngCore) -> (Self::SRS, usize) {
        (hyrax_setup(*params, 0, rng), *params)
    }

    fn commit(
        _params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        _scratch_pad: &mut Self::ScratchPad,
    ) -> Self::Commitment {
        hyrax_commit(proving_key, poly)
    }

    fn open(
        _params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        x: &Self::EvalPoint,
        _scratch_pad: &Self::ScratchPad,
        _transcript: &mut impl Transcript,
    ) -> (C::Scalar, Self::Opening) {
        hyrax_open(proving_key, poly, x)
    }

    fn verify(
        _params: &Self::Params,
        verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        x: &Self::EvalPoint,
        v: C::Scalar,
        opening: &Self::Opening,
        _transcript: &mut impl Transcript,
    ) -> bool {
        hyrax_verify(verifying_key, commitment, x, v, opening)
    }
}

impl<C> BatchOpeningPCS<C::Scalar> for HyraxPCS<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    fn single_point_batch_open(
        _params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        mle_poly_list: &[Self::Poly],
        eval_point: &Self::EvalPoint,
        _scratch_pad: &Self::ScratchPad,
        transcript: &mut impl Transcript,
    ) -> (Vec<C::Scalar>, Self::Opening) {
        hyrax_batch_open(proving_key, mle_poly_list, eval_point, transcript)
    }

    fn single_point_batch_verify(
        _params: &Self::Params,
        verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        comm_list: &[Self::Commitment],
        eval_point: &Self::EvalPoint,
        eval_list: &[C::Scalar],
        batch_proof: &Self::Opening,
        transcript: &mut impl Transcript,
    ) -> bool {
        hyrax_batch_verify(
            verifying_key,
            comm_list,
            eval_point,
            eval_list,
            batch_proof,
            transcript,
        )
    }

    /// Open a set of polynomials at a multiple points.
    /// Requires the length of the polys to be the same as points.
    ///
    /// Returns:
    /// - the evaluations of the polynomials at their corresponding points
    /// - the batch opening proof containing the sumcheck proof and the opening of g'(X)
    fn multiple_points_batch_open(
        _params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        polys: &[Self::Poly],
        points: &[Self::EvalPoint],
        _scratch_pad: &Self::ScratchPad,
        transcript: &mut impl Transcript,
    ) -> (Vec<C::Scalar>, BatchOpening<C::Scalar, Self>) {
        hyrax_multi_points_batch_open_internal(proving_key, polys, points, transcript)
    }

    /// Verify the opening of a set of polynomials at a single point.
    fn multiple_points_batch_verify(
        _params: &Self::Params,
        verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitments: &[Self::Commitment],
        points: &[Self::EvalPoint],
        values: &[C::Scalar],
        batch_opening: &BatchOpening<C::Scalar, Self>,
        transcript: &mut impl Transcript,
    ) -> bool {
        hyrax_multi_points_batch_verify_internal(
            verifying_key,
            commitments,
            points,
            values,
            batch_opening,
            transcript,
        )
    }
}
