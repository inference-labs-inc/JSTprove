use std::marker::PhantomData;

use arith::{FFTField, SimdField};
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};

use super::adapter;
use super::types::{WhirCommitment, WhirOpening, WhirSRS, WhirScratchPad};
use crate::utils::{lift_expander_challenge_to_n_vars, lift_poly_to_n_vars};

pub struct WhirPCSForGKR<C: FieldEngine> {
    _phantom: PhantomData<C>,
}

fn prepare_base_u64s<C: FieldEngine>(
    poly: &impl MultilinearExtension<C::SimdCircuitField>,
    params: usize,
) -> Vec<u64>
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
        .map(|f| {
            let mut buf = [0u8; 8];
            serdes::ExpSerde::serialize_into(&f, &mut buf[..]).unwrap();
            u64::from_le_bytes(buf)
        })
        .collect()
}

fn challenge_to_ext2_pairs<C: FieldEngine>(
    eval_point: &ExpanderSingleVarChallenge<C>,
) -> Vec<(u64, u64)> {
    let xs = eval_point.local_xs();
    xs.iter()
        .map(|x| {
            let mut buf = [0u8; 16];
            serdes::ExpSerde::serialize_into(x, &mut buf[..]).unwrap();
            let c0 = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let c1 = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            (c0, c1)
        })
        .collect()
}

fn claimed_eval_to_ext2<C: FieldEngine>(v: C::ChallengeField) -> (u64, u64) {
    let mut buf = [0u8; 16];
    serdes::ExpSerde::serialize_into(&v, &mut buf[..]).unwrap();
    let c0 = u64::from_le_bytes(buf[..8].try_into().unwrap());
    let c1 = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    (c0, c1)
}

impl<C> ExpanderPCS<C> for WhirPCSForGKR<C>
where
    C: FieldEngine,
    C::CircuitField: FFTField + SimdField<Scalar = C::CircuitField>,
    C::ChallengeField: FFTField + SimdField<Scalar = C::ChallengeField>,
{
    const NAME: &'static str = "WhirPCS";
    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::Whir;

    type Params = usize;
    type ScratchPad = WhirScratchPad;
    type SRS = WhirSRS;
    type Commitment = WhirCommitment;
    type Opening = WhirOpening;
    type BatchOpening = ();

    fn gen_params(n_input_vars: usize, _world_size: usize) -> Self::Params {
        n_input_vars
    }

    fn gen_srs(
        _params: &Self::Params,
        _mpi_engine: &impl MPIEngine,
        _rng: impl rand::RngCore,
    ) -> Self::SRS {
        WhirSRS
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi_engine: &impl MPIEngine) -> Self::ScratchPad {
        WhirScratchPad::default()
    }

    fn commit(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        if !mpi_engine.is_single_process() {
            unimplemented!("WHIR MPI not yet supported");
        }

        if poly.num_vars() > *params {
            return None;
        }

        let base_u64s = prepare_base_u64s::<C>(poly, *params);
        scratch_pad.cached_vector = Some(base_u64s);

        Some(WhirCommitment { num_vars: *params })
    }

    fn open(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        eval_point: &ExpanderSingleVarChallenge<C>,
        _transcript: &mut impl Transcript,
        scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        if !mpi_engine.is_single_process() {
            unimplemented!("WHIR MPI not yet supported");
        }

        if poly.num_vars() > *params || eval_point.num_vars() > *params {
            return None;
        }

        let effective_point = if eval_point.num_vars() < *params {
            lift_expander_challenge_to_n_vars(eval_point, *params)
        } else {
            eval_point.clone()
        };

        let base_u64s = match &scratch_pad.cached_vector {
            Some(v) => v.clone(),
            None => prepare_base_u64s::<C>(poly, *params),
        };

        let ext2_pairs = challenge_to_ext2_pairs::<C>(&effective_point);

        let proof = adapter::whir_commit_and_open(&base_u64s, &ext2_pairs, *params);

        Some(WhirOpening { proof })
    }

    fn verify(
        params: &Self::Params,
        _verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        _commitment: &Self::Commitment,
        eval_point: &ExpanderSingleVarChallenge<C>,
        claimed_eval: C::ChallengeField,
        _transcript: &mut impl Transcript,
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

        let ext2_pairs = challenge_to_ext2_pairs::<C>(&effective_point);
        let eval_pair = claimed_eval_to_ext2::<C>(claimed_eval);

        adapter::whir_verify(&ext2_pairs, eval_pair, *params, &opening.proof)
    }
}
