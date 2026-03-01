use arith::{Field, SimdField};
use circuit::CircuitLayer;
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, MPIEngine};
use polynomials::EqPolynomial;

use crate::{unpack_and_combine, ProverScratchPad};

use super::{product_gate::SumcheckProductGateHelper, simd_gate::SumcheckSimdProdGateHelper};

pub(crate) struct SumcheckGkrVanillaHelper<'a, F: FieldEngine> {
    pub(crate) rx: Vec<F::ChallengeField>,
    pub(crate) ry: Vec<F::ChallengeField>,
    pub(crate) r_simd_var: Vec<F::ChallengeField>,
    pub(crate) r_mpi_var: Vec<F::ChallengeField>,

    layer: &'a CircuitLayer<F>,
    sp: &'a mut ProverScratchPad<F>,

    challenge: &'a ExpanderDualVarChallenge<F>,
    alpha: Option<F::ChallengeField>,

    pub(crate) input_var_num: usize,
    pub(crate) simd_var_num: usize,

    xy_helper: SumcheckProductGateHelper,
    simd_var_helper: SumcheckSimdProdGateHelper<F>,
    mpi_var_helper: SumcheckSimdProdGateHelper<F>,

    is_output_layer: bool,
}

/// internal helper functions
impl<'a, F: FieldEngine> SumcheckGkrVanillaHelper<'a, F> {
    #[inline(always)]
    fn xy_helper_receive_challenge(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.xy_helper.receive_challenge::<F>(
            var_idx,
            r,
            &mut self.sp.v_evals,
            &mut self.sp.hg_evals,
            &self.layer.input_vals,
            &mut self.sp.gate_exists_5,
        );
    }
}

/// Helper functions to be called
#[allow(clippy::too_many_arguments)]
impl<'a, F: FieldEngine> SumcheckGkrVanillaHelper<'a, F> {
    #[inline]
    pub(crate) fn new(
        layer: &'a CircuitLayer<F>,
        challenge: &'a ExpanderDualVarChallenge<F>,
        alpha: Option<F::ChallengeField>,
        sp: &'a mut ProverScratchPad<F>,
        mpi_config: &impl MPIEngine,
        is_output_layer: bool,
    ) -> Self {
        let simd_var_num = F::get_field_pack_size().trailing_zeros() as usize;
        SumcheckGkrVanillaHelper {
            rx: vec![],
            ry: vec![],
            r_simd_var: vec![],
            r_mpi_var: vec![],

            layer,
            sp,
            challenge,
            alpha,

            input_var_num: layer.input_var_num,
            simd_var_num,

            xy_helper: SumcheckProductGateHelper::new(layer.input_var_num),
            simd_var_helper: SumcheckSimdProdGateHelper::new(simd_var_num),
            mpi_var_helper: SumcheckSimdProdGateHelper::new(
                mpi_config.world_size().trailing_zeros() as usize,
            ),
            is_output_layer,
        }
    }

    pub(crate) fn poly_evals_at_rx(
        &mut self,
        var_idx: usize,
        degree: usize,
        mpi_config: &impl MPIEngine,
    ) -> [F::ChallengeField; 3] {
        assert!(var_idx < self.input_var_num);
        let local_vals_simd = self.xy_helper.poly_eval_at::<F>(
            var_idx,
            degree,
            &self.sp.v_evals,
            &self.sp.hg_evals,
            &self.layer.input_vals,
            &self.sp.gate_exists_5,
        );

        // SIMD
        let local_vals = local_vals_simd
            .iter()
            .map(|p| unpack_and_combine(p, &self.sp.eq_evals_at_r_simd0))
            .collect::<Vec<F::ChallengeField>>();

        // MPI
        mpi_config
            .coef_combine_vec(&local_vals, &self.sp.eq_evals_at_r_mpi0)
            .try_into()
            .unwrap()
    }

    pub(crate) fn poly_evals_at_r_simd_var(
        &mut self,
        var_idx: usize,
        degree: usize,
        mpi_config: &impl MPIEngine,
    ) -> [F::ChallengeField; 4] {
        assert!(var_idx < self.simd_var_num);
        let local_vals = self
            .simd_var_helper
            .poly_eval_at(
                var_idx,
                degree,
                &mut self.sp.eq_evals_at_r_simd0,
                &mut self.sp.simd_var_v_evals,
                &mut self.sp.simd_var_hg_evals,
            )
            .to_vec();

        mpi_config
            .coef_combine_vec(&local_vals, &self.sp.eq_evals_at_r_mpi0)
            .try_into()
            .unwrap()
    }

    pub(crate) fn poly_evals_at_r_mpi_var(
        &mut self,
        var_idx: usize,
        degree: usize,
    ) -> [F::ChallengeField; 4] {
        assert!(var_idx < self.mpi_var_helper.var_num);
        self.mpi_var_helper.poly_eval_at(
            var_idx,
            degree,
            &mut self.sp.eq_evals_at_r_mpi0,
            &mut self.sp.mpi_var_v_evals,
            &mut self.sp.mpi_var_hg_evals,
        )
    }

    #[inline(always)]
    pub(crate) fn poly_evals_at_ry(
        &mut self,
        var_idx: usize,
        degree: usize,
        mpi_config: &impl MPIEngine,
    ) -> [F::ChallengeField; 3] {
        let [p0, p1, p2] = self.poly_evals_at_rx(var_idx, degree, mpi_config);
        [
            p0 * self.sp.phase2_coef,
            p1 * self.sp.phase2_coef,
            p2 * self.sp.phase2_coef,
        ]
    }

    #[inline]
    pub(crate) fn receive_rx(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.xy_helper_receive_challenge(var_idx, r);
        self.rx.push(r);
    }

    #[inline]
    pub(crate) fn receive_r_simd_var(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.simd_var_helper.receive_challenge(
            var_idx,
            r,
            &mut self.sp.eq_evals_at_r_simd0,
            &mut self.sp.simd_var_v_evals,
            &mut self.sp.simd_var_hg_evals,
        );
        self.r_simd_var.push(r);
    }

    #[inline]
    pub(crate) fn receive_r_mpi_var(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.mpi_var_helper.receive_challenge(
            var_idx,
            r,
            &mut self.sp.eq_evals_at_r_mpi0,
            &mut self.sp.mpi_var_v_evals,
            &mut self.sp.mpi_var_hg_evals,
        );
        self.r_mpi_var.push(r);
    }

    #[inline]
    pub(crate) fn receive_ry(&mut self, var_idx: usize, r: F::ChallengeField) {
        self.xy_helper_receive_challenge(var_idx, r);
        self.ry.push(r);
    }

    /// Warning:
    /// The function must be called at a specific point of the protocol, otherwise it's incorrect
    /// Consider fix this.
    pub(crate) fn vx_claim(&self) -> F::ChallengeField {
        self.sp.mpi_var_v_evals[0]
    }

    #[inline(always)]
    pub(crate) fn vy_claim(&self, mpi_config: &impl MPIEngine) -> F::ChallengeField {
        let vy_local = unpack_and_combine(&self.sp.v_evals[0], &self.sp.eq_evals_at_r_simd0);
        mpi_config.coef_combine_vec(&[vy_local], &self.sp.eq_evals_at_r_mpi0)[0]
    }

    #[inline]
    pub(crate) fn prepare_simd(&mut self) {
        if self.is_output_layer || self.alpha.is_none() {
            EqPolynomial::<F::ChallengeField>::eq_eval_at(
                &self.challenge.r_simd,
                &F::ChallengeField::one(),
                &mut self.sp.eq_evals_at_r_simd0,
                &mut self.sp.eq_evals_first_half,
                &mut self.sp.eq_evals_second_half,
            );
        }
    }

    #[inline]
    pub(crate) fn prepare_mpi(&mut self) {
        if self.is_output_layer || self.alpha.is_none() {
            // TODO: No need to evaluate it at all world ranks, remove redundancy later.
            EqPolynomial::<F::ChallengeField>::eq_eval_at(
                &self.challenge.r_mpi,
                &F::ChallengeField::one(),
                &mut self.sp.eq_evals_at_r_mpi0,
                &mut self.sp.eq_evals_first_half,
                &mut self.sp.eq_evals_second_half,
            );
        }
    }

    #[inline]
    pub(crate) fn prepare_x_vals(&mut self) {
        let mul = &self.layer.mul;
        let add = &self.layer.add;
        let vals = &self.layer.input_vals;
        let eq_evals_at_rz0 = &mut self.sp.eq_evals_at_rz0;
        let gate_exists = &mut self.sp.gate_exists_5;
        let hg_vals = &mut self.sp.hg_evals;
        // hg_vals[0..vals.len()].fill(F::zero()); // FIXED: consider memset unsafe?
        unsafe {
            std::ptr::write_bytes(hg_vals.as_mut_ptr(), 0, vals.len());
        }
        // gate_exists[0..vals.len()].fill(false); // FIXED: consider memset unsafe?
        unsafe {
            std::ptr::write_bytes(gate_exists.as_mut_ptr(), 0, vals.len());
        }

        assert_eq!(self.challenge.rz_1.is_none(), self.alpha.is_none());

        if self.is_output_layer || self.challenge.rz_1.is_none() {
            // Case 1: Output layer. There is only 1 claim
            // Case 2: Internal layer, but there is only 1 claim to prove,
            //  eq_evals_at_rx was thus skipped in the previous round
            EqPolynomial::<F::ChallengeField>::eq_eval_at(
                &self.challenge.rz_0,
                &F::ChallengeField::ONE,
                eq_evals_at_rz0,
                &mut self.sp.eq_evals_first_half,
                &mut self.sp.eq_evals_second_half,
            );
        } else {
            let alpha = self.alpha.unwrap();
            let eq_evals_at_rx_previous = &self.sp.eq_evals_at_rx;
            EqPolynomial::<F::ChallengeField>::eq_eval_at(
                self.challenge.rz_1.as_ref().unwrap(),
                &alpha,
                eq_evals_at_rz0,
                &mut self.sp.eq_evals_first_half,
                &mut self.sp.eq_evals_second_half,
            );

            for i in 0..(1 << self.challenge.rz_0.len()) {
                eq_evals_at_rz0[i] += eq_evals_at_rx_previous[i];
            }
        }

        for g in mul.iter() {
            let r = eq_evals_at_rz0[g.o_id] * g.coef;
            hg_vals[g.i_ids[0]] += r * vals[g.i_ids[1]];

            gate_exists[g.i_ids[0]] = true;
        }

        for g in add.iter() {
            hg_vals[g.i_ids[0]] += F::Field::from(eq_evals_at_rz0[g.o_id] * g.coef);
            gate_exists[g.i_ids[0]] = true;
        }
    }

    #[inline]
    pub(crate) fn prepare_simd_var_vals(&mut self) {
        self.sp.simd_var_v_evals = self.sp.v_evals[0].unpack();
        self.sp.simd_var_hg_evals = self.sp.hg_evals[0].unpack();
    }

    #[inline]
    pub(crate) fn prepare_mpi_var_vals(&mut self, mpi_config: &impl MPIEngine) {
        mpi_config.gather_vec(&[self.sp.simd_var_v_evals[0]], &mut self.sp.mpi_var_v_evals);
        mpi_config.gather_vec(
            &[self.sp.simd_var_hg_evals[0] * self.sp.eq_evals_at_r_simd0[0]],
            &mut self.sp.mpi_var_hg_evals,
        );
    }

    #[inline]
    pub(crate) fn prepare_y_vals(&mut self, mpi_config: &impl MPIEngine) {
        let mut v_rx_rsimd_rw = self.sp.mpi_var_v_evals[0];
        mpi_config.root_broadcast_f(&mut v_rx_rsimd_rw);

        let mul = &self.layer.mul;
        let eq_evals_at_rz0 = &self.sp.eq_evals_at_rz0;
        let eq_evals_at_rx = &mut self.sp.eq_evals_at_rx;
        let gate_exists = &mut self.sp.gate_exists_5;
        let hg_vals = &mut self.sp.hg_evals;
        let fill_len = 1 << self.rx.len();
        // hg_vals[0..fill_len].fill(F::zero()); // FIXED: consider memset unsafe?
        unsafe {
            std::ptr::write_bytes(hg_vals.as_mut_ptr(), 0, fill_len);
        }
        // gate_exists[0..fill_len].fill(false); // FIXED: consider memset unsafe?
        unsafe {
            std::ptr::write_bytes(gate_exists.as_mut_ptr(), 0, fill_len);
        }

        // TODO-Optimization: For root process, _eq_vec does not have to be recomputed
        self.sp.phase2_coef =
            EqPolynomial::<F::ChallengeField>::eq_vec(&self.challenge.r_mpi, &self.r_mpi_var)
                * self.sp.eq_evals_at_r_simd0[0]
                * v_rx_rsimd_rw;

        // EQ Polys for next round
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.r_mpi_var,
            &F::ChallengeField::ONE,
            &mut self.sp.eq_evals_at_r_mpi0,
            &mut self.sp.eq_evals_first_half,
            &mut self.sp.eq_evals_second_half,
        );

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.rx,
            &F::ChallengeField::ONE,
            eq_evals_at_rx,
            &mut self.sp.eq_evals_first_half,
            &mut self.sp.eq_evals_second_half,
        );

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.r_simd_var,
            &F::ChallengeField::ONE,
            &mut self.sp.eq_evals_at_r_simd0,
            &mut self.sp.eq_evals_first_half,
            &mut self.sp.eq_evals_second_half,
        );

        // TODO-OPTIMIZATION: hg_vals does not have to be simd here
        for g in mul.iter() {
            hg_vals[g.i_ids[1]] +=
                F::Field::from(eq_evals_at_rz0[g.o_id] * eq_evals_at_rx[g.i_ids[0]] * g.coef);
            gate_exists[g.i_ids[1]] = true;
        }
    }
}
