use std::marker::PhantomData;

use arith::{FFTField, SimdField};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};

use super::commit::basefold_commit;
use super::open::basefold_open;
use super::types::{BasefoldCommitment, BasefoldOpening, BasefoldSRS, BasefoldScratchPad};
use super::verify::basefold_verify;
use crate::utils::{lift_expander_challenge_to_n_vars, lift_poly_to_n_vars};

pub struct BasefoldPCSForGKR<C: FieldEngine> {
    _phantom: PhantomData<C>,
}

fn prepare_base_evals<C: FieldEngine>(
    poly: &impl MultilinearExtension<C::SimdCircuitField>,
    params: usize,
) -> Vec<C::CircuitField>
where
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
{
    let local_poly = if poly.num_vars() < params {
        lift_poly_to_n_vars(poly, params)
    } else {
        MultiLinearPoly::new(poly.hypercube_basis())
    };
    local_poly
        .hypercube_basis()
        .iter()
        .flat_map(|simd| simd.unpack())
        .collect()
}

impl<C> ExpanderPCS<C> for BasefoldPCSForGKR<C>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    C::ChallengeField: FFTField + SimdField<Scalar = C::ChallengeField>,
{
    const NAME: &'static str = "BasefoldPCS";
    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::FRI;

    type Params = usize;
    type ScratchPad = BasefoldScratchPad;
    type SRS = BasefoldSRS;
    type Commitment = BasefoldCommitment;
    type Opening = BasefoldOpening<C::ChallengeField>;
    type BatchOpening = ();

    fn gen_params(n_input_vars: usize, _world_size: usize) -> Self::Params {
        n_input_vars
    }

    fn gen_srs(
        _params: &Self::Params,
        _mpi_engine: &impl MPIEngine,
        _rng: impl rand::RngCore,
    ) -> Self::SRS {
        BasefoldSRS
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi_engine: &impl MPIEngine) -> Self::ScratchPad {
        BasefoldScratchPad::default()
    }

    fn commit(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        if !mpi_engine.is_single_process() {
            unimplemented!("Basefold MPI not yet supported");
        }

        if poly.num_vars() > *params {
            return None;
        }

        let base_evals = prepare_base_evals::<C>(poly, *params);
        let num_evals = base_evals.len();
        let (commitment, tree, codeword) = basefold_commit(&base_evals);
        scratch_pad.store_commit(tree, codeword, num_evals);
        Some(commitment)
    }

    fn open(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        eval_point: &ExpanderSingleVarChallenge<C>,
        transcript: &mut impl Transcript,
        scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        if !mpi_engine.is_single_process() {
            unimplemented!("Basefold MPI not yet supported");
        }

        if poly.num_vars() > *params || eval_point.num_vars() > *params {
            return None;
        }

        let effective_point = if eval_point.num_vars() < *params {
            lift_expander_challenge_to_n_vars(eval_point, *params)
        } else {
            eval_point.clone()
        };

        let base_evals = prepare_base_evals::<C>(poly, *params);
        let xs = effective_point.local_xs();

        let opening = if let Some((tree, codeword)) =
            scratch_pad.get_commit::<C::CircuitField>(base_evals.len())
        {
            basefold_open::<C::CircuitField, C::ChallengeField>(
                &base_evals,
                codeword,
                tree,
                *params,
                &xs,
                transcript,
            )
        } else {
            let (_commitment, tree, codeword) = basefold_commit(&base_evals);
            basefold_open::<C::CircuitField, C::ChallengeField>(
                &base_evals,
                &codeword,
                &tree,
                *params,
                &xs,
                transcript,
            )
        };

        Some(opening)
    }

    fn verify(
        params: &Self::Params,
        _verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        eval_point: &ExpanderSingleVarChallenge<C>,
        claimed_eval: C::ChallengeField,
        transcript: &mut impl Transcript,
        opening: &Self::Opening,
    ) -> bool {
        if eval_point.num_vars() > *params {
            return false;
        }

        let effective_point = if eval_point.num_vars() < *params {
            lift_expander_challenge_to_n_vars(eval_point, *params)
        } else {
            eval_point.clone()
        };

        let xs = effective_point.local_xs();

        basefold_verify::<C::CircuitField, C::ChallengeField>(
            commitment,
            &xs,
            claimed_eval,
            opening,
            transcript,
        )
    }
}
