use arith::Field;
use gkr_engine::{
    ExpanderPCS, ExpanderSingleVarChallenge, FieldEngine, MPIEngine, PolynomialCommitmentType,
    StructuredReferenceString, Transcript,
};
use polynomials::MultilinearExtension;
use rand::RngCore;

use super::ajtai::{
    ajtai_commit, ajtai_verify, AjtaiCommitment, AjtaiOpening, AjtaiSRS, AjtaiScratchPad,
    LatticePCSParams, LATTICE_COMMITMENT_ROWS,
};

pub struct LatticePCSForGKR<C: FieldEngine> {
    _phantom: std::marker::PhantomData<C>,
}

impl<C: FieldEngine> ExpanderPCS<C> for LatticePCSForGKR<C> {
    const NAME: &'static str = "LatticePCS";

    const PCS_TYPE: PolynomialCommitmentType = PolynomialCommitmentType::Lattice;

    type Params = LatticePCSParams;
    type ScratchPad = AjtaiScratchPad;
    type SRS = AjtaiSRS;
    type Commitment = AjtaiCommitment<C::SimdCircuitField>;
    type Opening = AjtaiOpening<C::SimdCircuitField>;
    type BatchOpening = ();

    fn gen_srs(
        _params: &Self::Params,
        _mpi_engine: &impl MPIEngine,
        mut rng: impl RngCore,
    ) -> Self::SRS {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        AjtaiSRS { seed }
    }

    fn gen_params(n_input_vars: usize, _world_size: usize) -> Self::Params {
        LatticePCSParams {
            num_vars: n_input_vars,
            commitment_rows: LATTICE_COMMITMENT_ROWS,
        }
    }

    fn init_scratch_pad(_params: &Self::Params, _mpi_engine: &impl MPIEngine) -> Self::ScratchPad {
        AjtaiScratchPad
    }

    fn commit(
        params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        _scratch_pad: &mut Self::ScratchPad,
    ) -> Option<Self::Commitment> {
        assert!(poly.num_vars() == params.num_vars);

        if mpi_engine.is_single_process() {
            let coeffs = poly.hypercube_basis();
            let digest = ajtai_commit(&proving_key.seed, params.commitment_rows, &coeffs);
            return Some(AjtaiCommitment { digest });
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

        let digest = ajtai_commit(&proving_key.seed, params.commitment_rows, &buffer);
        Some(AjtaiCommitment { digest })
    }

    fn open(
        _params: &Self::Params,
        mpi_engine: &impl MPIEngine,
        _proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &impl MultilinearExtension<C::SimdCircuitField>,
        _x: &ExpanderSingleVarChallenge<C>,
        _transcript: &mut impl Transcript,
        _scratch_pad: &Self::ScratchPad,
    ) -> Option<Self::Opening> {
        if mpi_engine.is_single_process() {
            return Some(AjtaiOpening {
                coefficients: poly.hypercube_basis(),
            });
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

        Some(AjtaiOpening {
            coefficients: buffer,
        })
    }

    fn verify(
        params: &Self::Params,
        verifying_key: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        challenge: &ExpanderSingleVarChallenge<C>,
        v: C::ChallengeField,
        _transcript: &mut impl Transcript,
        opening: &Self::Opening,
    ) -> bool {
        let commitment_valid = ajtai_verify(
            &verifying_key.seed,
            params.commitment_rows,
            &commitment.digest,
            &opening.coefficients,
        );
        if !commitment_valid {
            return false;
        }

        let v_target = C::single_core_eval_circuit_vals_at_expander_challenge(
            &opening.coefficients,
            challenge,
        );
        v == v_target
    }
}
