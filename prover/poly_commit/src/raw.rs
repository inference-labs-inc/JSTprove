/// Raw commitment for multi-linear polynomials
use arith::{ExtensionField, Field};
use ethnum::U256;
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::{MultiLinearPoly, MultilinearExtension};
use rand::RngCore;
use serdes::{ExpSerde, SerdeResult};

use crate::PolynomialCommitmentScheme;

#[derive(Clone, Debug, Default)]
pub struct RawCommitment<F: Field> {
    pub evals: Vec<F>,
}

impl<F: Field> ExpSerde for RawCommitment<F> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        // ideally we want to remove this and use Derive(ExpSerde) method to determine the length
        // of the vector.
        // however, it seems that the recursion circuit has hardcoded the length.
        // maybe we want to fix it?
        let u256_embedded = U256::from(self.evals.len() as u64);
        u256_embedded.serialize_into(&mut writer)?;

        self.evals
            .iter()
            .try_for_each(|v| v.serialize_into(&mut writer))?;

        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let mut v = Self::default();

        let len = U256::deserialize_from(&mut reader)?;

        for _ in 0..len.as_usize() {
            v.evals.push(F::deserialize_from(&mut reader)?);
        }
        Ok(v)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawMultiLinearScratchPad<F: Field> {
    pub eval_buffer: Vec<F>,
}

// Raw commitment for multi-linear polynomials
pub struct RawMultiLinearPCS {}

impl<F: ExtensionField> PolynomialCommitmentScheme<F> for RawMultiLinearPCS {
    const NAME: &'static str = "RawMultiLinear";

    type Params = usize;
    type ScratchPad = ();

    type Poly = MultiLinearPoly<F>;

    type EvalPoint = Vec<F>;

    type SRS = ();
    type Commitment = RawCommitment<F>;

    type Opening = ();

    fn gen_srs_for_testing(params: &Self::Params, _rng: impl RngCore) -> (Self::SRS, usize) {
        (Self::SRS::default(), *params)
    }

    fn init_scratch_pad(_params: &Self::Params) -> Self::ScratchPad {}

    fn commit(
        params: &Self::Params,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        _scratch_pad: &mut Self::ScratchPad,
    ) -> Self::Commitment {
        assert!(poly.coeffs.len() == 1 << params);
        Self::Commitment {
            evals: poly.coeffs.clone(),
        }
    }

    fn open(
        params: &Self::Params,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        x: &Self::EvalPoint,
        _scratch_pad: &Self::ScratchPad,
        _transcript: &mut impl Transcript,
    ) -> (F, Self::Opening) {
        assert!(x.len() == *params);
        (MultiLinearPoly::<F>::evaluate_jolt(poly, x), ())
    }

    fn verify(
        params: &Self::Params,
        _verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        x: &Self::EvalPoint,
        v: F,
        _opening: &Self::Opening,
        _transcript: &mut impl Transcript,
    ) -> bool {
        assert!(x.len() == *params);
        MultiLinearPoly::<F>::evaluate_with_buffer(
            &commitment.evals,
            x,
            &mut vec![F::ZERO; commitment.evals.len()],
        ) == v
    }
}

// =================================================================================================

pub struct RawExpanderGKR<C: FieldEngine> {
    _phantom: std::marker::PhantomData<C>,
}

impl<C: FieldEngine> ExpanderPCS<C> for RawExpanderGKR<C> {
    const NAME: &'static str = "RawExpanderGKR";

    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::Raw;

    type Params = usize;

    type ScratchPad = ();

    type SRS = ();

    type Commitment = RawCommitment<C::SimdCircuitField>;

    type Opening = ();

    type BatchOpening = ();

    fn gen_srs(
        _params: &Self::Params,
        _mpi_engine: &impl MPIEngine,
        _rng: impl RngCore,
    ) -> Self::SRS {
    }

    fn gen_params(n_input_vars: usize, _world_size: usize) -> Self::Params {
        n_input_vars
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi_engine: &impl MPIEngine) -> Self::ScratchPad {}

    fn commit(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        _scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        assert!(poly.num_vars() == *params);

        if mpi_engine.is_single_process() {
            return Self::Commitment {
                evals: poly.hypercube_basis(),
            }
            .into();
        }

        let mut buffer = if mpi_engine.is_root() {
            vec![C::SimdCircuitField::zero(); poly.hypercube_size() * mpi_engine.world_size()]
        } else {
            vec![]
        };

        mpi_engine.gather_vec(poly.hypercube_basis_ref(), &mut buffer);

        if !mpi_engine.is_root() {
            return None;
        }

        Self::Commitment { evals: buffer }.into()
    }

    fn open(
        _params: &Self::Params,
        _mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        _poly: &impl MultilinearExtension<C::SimdCircuitField>,
        _x: &ExpanderSingleVarChallenge<C>,
        _transcript: &mut impl Transcript,
        _scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        Some(())
    }

    fn verify(
        _params: &Self::Params,
        _verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        challenge: &ExpanderSingleVarChallenge<C>,
        v: C::ChallengeField,
        _transcript: &mut impl Transcript,
        _opening: &Self::Opening,
    ) -> bool {
        let v_target =
            C::single_core_eval_circuit_vals_at_expander_challenge(&commitment.evals, challenge);
        v == v_target
    }
}
