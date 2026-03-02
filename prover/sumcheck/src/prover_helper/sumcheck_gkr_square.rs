use crate::{unpack_and_combine, ProverScratchPad};
use arith::{Field, SimdField};
use circuit::CircuitLayer;
use gkr_engine::{ExpanderSingleVarChallenge, FieldEngine, MPIConfig, MPIEngine};
use polynomials::EqPolynomial;

use super::{power_gate::SumcheckPowerGateHelper, simd_gate::SumcheckSimdProdGateHelper};

// todo: Move D to GKRFieldConfig
pub(crate) struct SumcheckGkrSquareHelper<'a, F: FieldEngine, const D: usize> {
    pub(crate) rx: Vec<F::ChallengeField>,
    pub(crate) r_simd_var: Vec<F::ChallengeField>,
    pub(crate) r_mpi_var: Vec<F::ChallengeField>,

    layer: &'a CircuitLayer<F>,
    sp: &'a mut ProverScratchPad<F>,

    challenge: &'a ExpanderSingleVarChallenge<F>,

    pub(crate) simd_var_num: usize,

    x_helper: SumcheckPowerGateHelper<D>,
    simd_helper: SumcheckSimdProdGateHelper<F>,
    mpi_helper: SumcheckSimdProdGateHelper<F>,

    // ZF: I think we do not need it to be a reference here, will remove it later
    mpi_config: &'a MPIConfig,
}

impl<'a, F: FieldEngine, const D: usize> SumcheckGkrSquareHelper<'a, F, D> {
    #[inline]
    pub(crate) fn new(
        layer: &'a CircuitLayer<F>,
        challenge: &'a ExpanderSingleVarChallenge<F>,
        sp: &'a mut ProverScratchPad<F>,
        mpi_config: &'a MPIConfig,
    ) -> Self {
        let simd_var_num = F::get_field_pack_size().trailing_zeros() as usize;

        SumcheckGkrSquareHelper {
            rx: vec![],
            r_simd_var: vec![],
            r_mpi_var: vec![],

            layer,
            sp,
            challenge,
            simd_var_num,

            x_helper: SumcheckPowerGateHelper::new(layer.input_var_num),
            simd_helper: SumcheckSimdProdGateHelper::new(simd_var_num),
            mpi_helper: SumcheckSimdProdGateHelper::new(
                mpi_config.world_size().trailing_zeros() as usize
            ),
            mpi_config,
        }
    }

    #[inline]
    pub(crate) fn poly_evals_at_x(&self, var_idx: usize) -> [F::ChallengeField; D] {
        let local_vals_simd = self.x_helper.poly_eval_at::<F>(
            var_idx,
            &self.sp.v_evals,
            &self.sp.hg_evals_5,
            &self.sp.hg_evals_1,
            &self.layer.input_vals,
            &self.sp.gate_exists_5,
            &self.sp.gate_exists_1,
        );

        // SIMD
        let local_vals = local_vals_simd
            .iter()
            .map(|p| unpack_and_combine(p, &self.sp.eq_evals_at_r_simd0))
            .collect::<Vec<F::ChallengeField>>();

        self.mpi_config
            .coef_combine_vec(&local_vals, &self.sp.eq_evals_at_r_mpi0)
            .try_into()
            .unwrap()
    }

    #[inline]
    pub(crate) fn poly_evals_at_simd(&self, var_idx: usize) -> [F::ChallengeField; D] {
        let local_vals = self
            .simd_helper
            .gkr2_poly_eval_at::<D>(
                var_idx,
                &self.sp.eq_evals_at_r_simd0,
                &self.sp.simd_var_v_evals,
                self.sp.hg_evals_1[0],
                self.sp.hg_evals_5[0],
            )
            .to_vec();

        self.mpi_config
            .coef_combine_vec(&local_vals, &self.sp.eq_evals_at_r_mpi0)
            .try_into()
            .unwrap()
    }

    pub(crate) fn poly_evals_at_mpi(&mut self, var_idx: usize) -> [F::ChallengeField; D] {
        assert!(var_idx < self.mpi_config.world_size().trailing_zeros() as usize);
        let mut evals = self.mpi_helper.gkr2_poly_eval_at::<D>(
            var_idx,
            &self.sp.eq_evals_at_r_mpi0,
            &self.sp.mpi_var_v_evals,
            self.sp.hg_evals_1[0],
            self.sp.hg_evals_5[0],
        );
        for eval in evals.iter_mut() {
            *eval *= self.sp.eq_evals_at_r_simd0[0];
        }
        evals
    }

    #[inline]
    pub(crate) fn receive_x_challenge(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.x_helper.receive_challenge::<F>(
            var_idx,
            r,
            &mut self.sp.v_evals,
            &mut self.sp.hg_evals_5,
            &mut self.sp.hg_evals_1,
            &self.layer.input_vals,
            &mut self.sp.gate_exists_5,
            &mut self.sp.gate_exists_1,
        );
        log::trace!("v_eval[0]:= {:?}", self.sp.v_evals[0]);
        self.rx.push(r);
    }

    #[inline]
    pub(crate) fn receive_simd_challenge(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.simd_helper.receive_challenge(
            var_idx,
            r,
            &mut self.sp.eq_evals_at_r_simd0,
            &mut self.sp.simd_var_v_evals,
            &mut self.sp.simd_var_hg_evals,
        );
        self.r_simd_var.push(r);
    }

    #[inline]
    pub(crate) fn receive_mpi_challenge(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.mpi_helper.receive_challenge(
            var_idx,
            r,
            &mut self.sp.eq_evals_at_r_mpi0,
            &mut self.sp.mpi_var_v_evals,
            &mut self.sp.mpi_var_hg_evals,
        );
        self.r_mpi_var.push(r);
    }

    #[inline(always)]
    pub(crate) fn vx_claim(&self) -> F::ChallengeField {
        self.sp.mpi_var_v_evals[0]
    }

    #[inline]
    pub(crate) fn prepare_simd(&mut self) {
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.challenge.r_simd,
            &F::ChallengeField::one(),
            &mut self.sp.eq_evals_at_r_simd0,
            &mut self.sp.eq_evals_first_half,
            &mut self.sp.eq_evals_second_half,
        );
    }

    #[inline]
    pub(crate) fn prepare_mpi(&mut self) {
        // TODO: No need to evaluate it at all world ranks, remove redundancy later.
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.challenge.r_mpi,
            &F::ChallengeField::one(),
            &mut self.sp.eq_evals_at_r_mpi0,
            &mut self.sp.eq_evals_first_half,
            &mut self.sp.eq_evals_second_half,
        );
    }

    #[inline]
    pub(crate) fn prepare_simd_var_vals(&mut self) {
        self.sp.simd_var_v_evals = self.sp.v_evals[0].unpack();
    }

    #[inline]
    pub(crate) fn prepare_mpi_var_vals(&mut self) {
        self.mpi_config
            .gather_vec(&[self.sp.simd_var_v_evals[0]], &mut self.sp.mpi_var_v_evals);
    }

    #[inline]
    pub(crate) fn prepare_g_x_vals(&mut self) {
        let uni = &self.layer.uni; // univariate things like square, pow5, etc.
        let vals = &self.layer.input_vals;
        let eq_evals_at_rz0 = &mut self.sp.eq_evals_at_rz0;
        let gate_exists_5 = &mut self.sp.gate_exists_5;
        let gate_exists_1 = &mut self.sp.gate_exists_1;
        let hg_evals_5 = &mut self.sp.hg_evals_5;
        let hg_evals_1 = &mut self.sp.hg_evals_1;
        // hg_vals[0..vals.len()].fill(F::zero()); // FIXED: consider memset unsafe?
        unsafe {
            std::ptr::write_bytes(hg_evals_5.as_mut_ptr(), 0, vals.len());
            std::ptr::write_bytes(hg_evals_1.as_mut_ptr(), 0, vals.len());
        }
        // gate_exists[0..vals.len()].fill(false); // FIXED: consider memset unsafe?
        unsafe {
            std::ptr::write_bytes(gate_exists_5.as_mut_ptr(), 0, vals.len());
            std::ptr::write_bytes(gate_exists_1.as_mut_ptr(), 0, vals.len());
        }
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.challenge.rz,
            &F::ChallengeField::one(),
            eq_evals_at_rz0,
            &mut self.sp.eq_evals_first_half,
            &mut self.sp.eq_evals_second_half,
        );

        for g in uni.iter() {
            match g.gate_type {
                12345 => {
                    hg_evals_5[g.i_ids[0]] += eq_evals_at_rz0[g.o_id] * g.coef;
                    gate_exists_5[g.i_ids[0]] = true;
                }
                12346 => {
                    hg_evals_1[g.i_ids[0]] += eq_evals_at_rz0[g.o_id] * g.coef;
                    gate_exists_1[g.i_ids[0]] = true;
                }
                _ => panic!("Unsupported gate type"),
            }
        }
    }
}
