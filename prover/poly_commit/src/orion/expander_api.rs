use arith::{Field, SimdField};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::MultilinearExtension;

use crate::{
    orion::{
        simd_field_impl::{orion_commit_simd_field, orion_open_simd_field},
        simd_field_mpi_impl::{orion_mpi_commit_simd_field, orion_mpi_open_simd_field},
        verify::orion_verify,
        OrionCommitment, OrionProof, OrionSIMDFieldPCS, OrionSRS, OrionScratchPad,
        ORION_CODE_PARAMETER_INSTANCE,
    },
    utils::{
        lift_expander_challenge_to_n_vars, lift_poly_and_expander_challenge_to_n_vars,
        lift_poly_to_n_vars,
    },
};

use super::utils::orion_eval_shape;

impl<C, ComPackF> ExpanderPCS<C>
    for OrionSIMDFieldPCS<C::CircuitField, C::SimdCircuitField, C::ChallengeField, ComPackF>
where
    C: FieldEngine,
    ComPackF: SimdField<Scalar = C::CircuitField>,
{
    const NAME: &'static str = "OrionPCSForExpanderGKR";

    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::Orion;

    type Params = usize;
    type ScratchPad = OrionScratchPad;

    type Commitment = OrionCommitment;
    type Opening = OrionProof<C::ChallengeField>;
    type BatchOpening = ();
    type SRS = OrionSRS;

    /// NOTE(HS): this is the number of variables for local polynomial w.r.t. SIMD field elements.
    fn gen_params(n_input_vars: usize, world_size: usize) -> Self::Params {
        let num_vars_each_core = n_input_vars + C::SimdCircuitField::PACK_SIZE.ilog2() as usize;
        let (_num_leaves_per_mt_query, scaled_num_local_vars, _msg_size) = orion_eval_shape(
            world_size,
            num_vars_each_core,
            C::CircuitField::FIELD_SIZE,
            C::SimdCircuitField::PACK_SIZE,
        );

        scaled_num_local_vars - C::SimdCircuitField::PACK_SIZE.ilog2() as usize
    }

    fn gen_srs(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        rng: impl rand::RngCore,
    ) -> Self::SRS {
        let num_vars_each_core = *params + C::SimdCircuitField::PACK_SIZE.ilog2() as usize;
        let (srs, calibrated_num_vars_each_core) = OrionSRS::from_random(
            mpi_engine.world_size(),
            num_vars_each_core,
            C::CircuitField::FIELD_SIZE,
            ComPackF::PACK_SIZE,
            ORION_CODE_PARAMETER_INSTANCE,
            rng,
        );
        assert_eq!(num_vars_each_core, calibrated_num_vars_each_core);
        srs
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi_engine: &impl MPIEngine) -> Self::ScratchPad {
        Self::ScratchPad::default()
    }

    fn commit(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        if poly.num_vars() < *params {
            let poly = lift_poly_to_n_vars(poly, *params);
            return <Self as ExpanderPCS<C>>::commit(
                params,
                mpi_engine,
                proving_key,
                &poly,
                scratch_pad,
            );
        }

        let num_vars_each_core = *params + C::SimdCircuitField::PACK_SIZE.ilog2() as usize;
        assert_eq!(num_vars_each_core, proving_key.num_vars);

        if mpi_engine.is_single_process() {
            return orion_commit_simd_field::<_, C::SimdCircuitField, ComPackF>(
                proving_key,
                poly,
                scratch_pad,
            )
            .ok();
        }

        orion_mpi_commit_simd_field::<_, C::SimdCircuitField, ComPackF>(
            mpi_engine,
            proving_key,
            poly,
            scratch_pad,
        )
        .ok()
    }

    fn open(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        eval_point: &ExpanderSingleVarChallenge<C>,
        transcript: &mut impl Transcript,
        scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        if poly.num_vars() < *params {
            let (poly, eval_point) =
                lift_poly_and_expander_challenge_to_n_vars(poly, eval_point, *params);
            return <Self as ExpanderPCS<C>>::open(
                params,
                mpi_engine,
                proving_key,
                &poly,
                &eval_point,
                transcript,
                scratch_pad,
            );
        }

        let num_vars_each_core = *params + C::SimdCircuitField::PACK_SIZE.ilog2() as usize;
        assert_eq!(num_vars_each_core, proving_key.num_vars);

        if mpi_engine.is_single_process() {
            let (_, opening) = orion_open_simd_field::<_, C::SimdCircuitField, _, ComPackF>(
                proving_key,
                poly,
                &eval_point.local_xs(),
                transcript,
                scratch_pad,
            );
            return opening.into();
        }

        orion_mpi_open_simd_field::<_, C::SimdCircuitField, _, ComPackF>(
            mpi_engine,
            proving_key,
            poly,
            &eval_point.local_xs(),
            &eval_point.r_mpi,
            transcript,
            scratch_pad,
        )
    }

    fn verify(
        params: &Self::Params,
        verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        eval_point: &ExpanderSingleVarChallenge<C>,
        eval: C::ChallengeField,
        transcript: &mut impl Transcript, /* add transcript here to allow
                                           * interactive arguments */
        opening: &Self::Opening,
    ) -> bool {
        if eval_point.num_vars() < *params {
            let eval_point = lift_expander_challenge_to_n_vars(eval_point, *params);
            return <Self as ExpanderPCS<C>>::verify(
                params,
                verifying_key,
                commitment,
                &eval_point,
                eval,
                transcript,
                opening,
            );
        }

        orion_verify::<_, C::SimdCircuitField, _, ComPackF>(
            verifying_key,
            commitment,
            &eval_point.local_xs(),
            &eval_point.r_mpi,
            eval,
            transcript,
            opening,
        )
    }
}

pub type OrionPCSForGKR<C, ComPack> = OrionSIMDFieldPCS<
    <C as FieldEngine>::CircuitField,
    <C as FieldEngine>::SimdCircuitField,
    <C as FieldEngine>::ChallengeField,
    ComPack,
>;
