use arith::{Field, SimdField};
use circuit::CircuitLayer;
#[cfg(all(target_os = "macos", feature = "metal"))]
use gkr_engine::Transcript;
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

#[cfg(all(target_os = "macos", feature = "metal"))]
enum MetalRoundTarget {
    Rx,
    Ry,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl<'a, F: FieldEngine> SumcheckGkrVanillaHelper<'a, F> {
    fn challenge_to_limbs(v: &[F::ChallengeField]) -> Vec<[u64; 4]> {
        debug_assert_eq!(
            std::mem::size_of::<F::ChallengeField>(),
            32,
            "ChallengeField must be 32 bytes for [u64; 4] transmute"
        );
        v.iter()
            .map(|x| unsafe { std::mem::transmute_copy(x) })
            .collect()
    }

    fn metal_fold_loop<T: Transcript>(
        &mut self,
        ctx: &crate::metal_sumcheck::MetalSumcheckCtx,
        input_var_num: usize,
        eq_simd: &[F::ChallengeField],
        eq_mpi: &[F::ChallengeField],
        phase2_coef: Option<F::ChallengeField>,
        transcript: &mut T,
        mpi_config: &impl MPIEngine,
        target: MetalRoundTarget,
    ) -> bool {
        use metal_accel::{metal_fold_all, metal_poly_eval, BN254_ELEM_SIZE};

        let mut f_ping = true;
        let mut hg_ping = true;
        let mut ge_ping = true;

        for var_idx in 0..input_var_num {
            let eval_size = 1usize << (input_var_num - var_idx - 1);
            let Ok(eval_size_u32) = u32::try_from(eval_size) else {
                return false;
            };

            let f_src = if f_ping {
                &ctx.pool.v_evals
            } else {
                &ctx.pool.fold_scratch
            };
            let hg_src = if hg_ping {
                &ctx.pool.hg_evals
            } else {
                &ctx.pool.hg_fold_scratch
            };
            let ge_src = if ge_ping {
                &ctx.pool.gate_exists
            } else {
                &ctx.pool.fold_ge_scratch
            };

            let raw_evals = metal_poly_eval(
                &ctx.accel,
                f_src,
                hg_src,
                ge_src,
                &ctx.pool.block_results,
                &ctx.pool.output,
                eval_size_u32,
            );

            let p0: F::Field = unsafe { std::mem::transmute_copy(&raw_evals[0]) };
            let p1: F::Field = unsafe { std::mem::transmute_copy(&raw_evals[1]) };
            let p2_raw: F::Field = unsafe { std::mem::transmute_copy(&raw_evals[2]) };
            let p2 = p1.mul_by_6() + p0.mul_by_3() - p2_raw.double();

            let evals_arr = [p0, p1, p2];
            let local_vals: Vec<F::ChallengeField> = evals_arr
                .iter()
                .map(|p| unpack_and_combine(p, eq_simd))
                .collect();
            let mut evals: [F::ChallengeField; 3] = mpi_config
                .coef_combine_vec(&local_vals, eq_mpi)
                .try_into()
                .unwrap();

            if let Some(coef) = phase2_coef {
                evals[0] = evals[0] * coef;
                evals[1] = evals[1] * coef;
                evals[2] = evals[2] * coef;
            }

            let r =
                crate::utils::transcript_io::<F::ChallengeField, T>(mpi_config, &evals, transcript);
            match target {
                MetalRoundTarget::Rx => self.rx.push(r),
                MetalRoundTarget::Ry => self.ry.push(r),
            }

            let r_limbs: [u64; 4] = unsafe { std::mem::transmute_copy(&r) };
            ctx.pool.write_challenge(&r_limbs);

            let f_dst = if f_ping {
                &ctx.pool.fold_scratch
            } else {
                &ctx.pool.v_evals
            };
            let hg_dst = if hg_ping {
                &ctx.pool.hg_fold_scratch
            } else {
                &ctx.pool.hg_evals
            };
            let ge_dst = if ge_ping {
                &ctx.pool.fold_ge_scratch
            } else {
                &ctx.pool.gate_exists
            };

            metal_fold_all(
                &ctx.accel,
                f_src,
                f_dst,
                hg_src,
                hg_dst,
                ge_src,
                ge_dst,
                &ctx.pool.challenge,
                eval_size_u32,
            );

            f_ping = !f_ping;
            hg_ping = !hg_ping;
            ge_ping = !ge_ping;
        }

        let f_final = if f_ping {
            &ctx.pool.v_evals
        } else {
            &ctx.pool.fold_scratch
        };
        let hg_final = if hg_ping {
            &ctx.pool.hg_evals
        } else {
            &ctx.pool.hg_fold_scratch
        };
        let ge_final = if ge_ping {
            &ctx.pool.gate_exists
        } else {
            &ctx.pool.fold_ge_scratch
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                f_final.contents() as *const u8,
                self.sp.v_evals.as_mut_ptr() as *mut u8,
                BN254_ELEM_SIZE,
            );
            std::ptr::copy_nonoverlapping(
                hg_final.contents() as *const u8,
                self.sp.hg_evals.as_mut_ptr() as *mut u8,
                BN254_ELEM_SIZE,
            );
            self.sp.gate_exists_5[0] = *(ge_final.contents() as *const u32) != 0;
        }
        true
    }

    pub(crate) fn try_metal_xy_rounds<T: Transcript>(
        &mut self,
        transcript: &mut T,
        mpi_config: &impl MPIEngine,
    ) -> bool {
        use metal_accel::{metal_eq_eval_at, BN254_ELEM_SIZE};

        if std::mem::size_of::<F::Field>() != BN254_ELEM_SIZE {
            return false;
        }
        if std::mem::size_of::<F::ChallengeField>() != BN254_ELEM_SIZE {
            return false;
        }
        if !crate::metal_sumcheck::metal_available() {
            return false;
        }
        if self.input_var_num < 12 {
            return false;
        }

        let input_var_num = self.input_var_num;
        let Some(total) = 1usize.checked_shl(input_var_num as u32) else {
            return false;
        };
        if total.checked_mul(BN254_ELEM_SIZE).is_none() {
            return false;
        }
        let Some(output_size) = 1usize.checked_shl(self.challenge.rz_0.len() as u32) else {
            return false;
        };
        if self.layer.input_vals.len() < total
            || self.sp.hg_evals.len() < total
            || self.sp.gate_exists_5.len() < total
            || self.sp.eq_evals_at_rz0.len() < output_size
        {
            return false;
        }

        let eq_simd = self.sp.eq_evals_at_r_simd0.clone();
        let eq_mpi = self.sp.eq_evals_at_r_mpi0.clone();

        crate::metal_sumcheck::METAL_CTX.with(|cell| {
            let mut ctx_ref = cell.borrow_mut();
            let Some(ctx) = ctx_ref.as_mut() else {
                return false;
            };
            if ctx.pool.max_input_size() < total {
                return false;
            }

            if self.challenge.rz_1.is_none() != self.alpha.is_none() {
                return false;
            }

            if self.is_output_layer || self.challenge.rz_1.is_none() {
                let rz0_limbs = Self::challenge_to_limbs(&self.challenge.rz_0);
                let one_limbs: [u64; 4] =
                    unsafe { std::mem::transmute_copy(&F::ChallengeField::ONE) };
                metal_eq_eval_at(
                    &ctx.accel,
                    &rz0_limbs,
                    &one_limbs,
                    &ctx.pool.eq_first_half,
                    &ctx.pool.eq_second_half,
                    &ctx.pool.eq_evals_rz0,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ctx.pool.eq_evals_rz0.contents() as *const u8,
                        self.sp.eq_evals_at_rz0.as_mut_ptr() as *mut u8,
                        output_size * BN254_ELEM_SIZE,
                    );
                }
            } else {
                if self.sp.eq_evals_at_rx.len() < output_size {
                    return false;
                }
                let alpha = self.alpha.unwrap();
                let alpha_limbs: [u64; 4] = unsafe { std::mem::transmute_copy(&alpha) };
                let rz1_limbs = Self::challenge_to_limbs(self.challenge.rz_1.as_ref().unwrap());
                metal_eq_eval_at(
                    &ctx.accel,
                    &rz1_limbs,
                    &alpha_limbs,
                    &ctx.pool.eq_first_half,
                    &ctx.pool.eq_second_half,
                    &ctx.pool.eq_evals_rz0,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ctx.pool.eq_evals_rz0.contents() as *const u8,
                        self.sp.eq_evals_at_rz0.as_mut_ptr() as *mut u8,
                        output_size * BN254_ELEM_SIZE,
                    );
                }
                for i in 0..output_size {
                    self.sp.eq_evals_at_rz0[i] += self.sp.eq_evals_at_rx[i];
                }
            }

            let mul = &self.layer.mul;
            let add = &self.layer.add;
            let vals = &self.layer.input_vals;
            let eq_evals_at_rz0 = &self.sp.eq_evals_at_rz0;
            let gate_exists = &mut self.sp.gate_exists_5;
            let hg_vals = &mut self.sp.hg_evals;
            unsafe {
                std::ptr::write_bytes(hg_vals.as_mut_ptr(), 0, total);
                std::ptr::write_bytes(gate_exists.as_mut_ptr(), 0, total);
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

            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.layer.input_vals.as_ptr() as *const u8,
                    ctx.pool.v_evals.contents() as *mut u8,
                    total * BN254_ELEM_SIZE,
                );
                std::ptr::copy_nonoverlapping(
                    self.sp.hg_evals.as_ptr() as *const u8,
                    ctx.pool.hg_evals.contents() as *mut u8,
                    total * BN254_ELEM_SIZE,
                );
            }
            let ge_ptr = ctx.pool.gate_exists.contents() as *mut u32;
            for i in 0..total {
                unsafe {
                    *ge_ptr.add(i) = if self.sp.gate_exists_5[i] { 1 } else { 0 };
                }
            }

            if !self.metal_fold_loop(
                ctx,
                input_var_num,
                &eq_simd,
                &eq_mpi,
                None,
                transcript,
                mpi_config,
                MetalRoundTarget::Rx,
            ) {
                return false;
            }
            true
        })
    }

    pub(crate) fn try_metal_ry_rounds<T: Transcript>(
        &mut self,
        transcript: &mut T,
        mpi_config: &impl MPIEngine,
    ) -> bool {
        use metal_accel::{metal_eq_eval_at, BN254_ELEM_SIZE};

        if std::mem::size_of::<F::Field>() != BN254_ELEM_SIZE {
            return false;
        }
        if std::mem::size_of::<F::ChallengeField>() != BN254_ELEM_SIZE {
            return false;
        }
        if !crate::metal_sumcheck::metal_available() {
            return false;
        }
        if self.input_var_num < 12 {
            return false;
        }

        let input_var_num = self.input_var_num;
        let Some(total) = 1usize.checked_shl(input_var_num as u32) else {
            return false;
        };
        if total.checked_mul(BN254_ELEM_SIZE).is_none() {
            return false;
        }
        if self.layer.input_vals.len() < total
            || self.sp.hg_evals.len() < total
            || self.sp.gate_exists_5.len() < total
            || self.sp.eq_evals_at_rx.len() < total
        {
            return false;
        }

        let mut v_rx_rsimd_rw = self.sp.mpi_var_v_evals[0];
        mpi_config.root_broadcast_f(&mut v_rx_rsimd_rw);

        self.sp.phase2_coef =
            EqPolynomial::<F::ChallengeField>::eq_vec(&self.challenge.r_mpi, &self.r_mpi_var)
                * self.sp.eq_evals_at_r_simd0[0]
                * v_rx_rsimd_rw;

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &self.r_mpi_var,
            &F::ChallengeField::ONE,
            &mut self.sp.eq_evals_at_r_mpi0,
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

        let phase2_coef = self.sp.phase2_coef;
        let eq_simd = self.sp.eq_evals_at_r_simd0.clone();
        let eq_mpi = self.sp.eq_evals_at_r_mpi0.clone();

        crate::metal_sumcheck::METAL_CTX.with(|cell| {
            let mut ctx_ref = cell.borrow_mut();
            let Some(ctx) = ctx_ref.as_mut() else {
                return false;
            };
            if ctx.pool.max_input_size() < total {
                return false;
            }

            let rx_limbs = Self::challenge_to_limbs(&self.rx);
            let eq_rx_len = 1usize << rx_limbs.len();
            if eq_rx_len != total {
                return false;
            }
            let one_limbs: [u64; 4] = unsafe { std::mem::transmute_copy(&F::ChallengeField::ONE) };
            metal_eq_eval_at(
                &ctx.accel,
                &rx_limbs,
                &one_limbs,
                &ctx.pool.eq_first_half,
                &ctx.pool.eq_second_half,
                &ctx.pool.eq_evals_rx,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.pool.eq_evals_rx.contents() as *const u8,
                    self.sp.eq_evals_at_rx.as_mut_ptr() as *mut u8,
                    total * BN254_ELEM_SIZE,
                );
            }

            let mul = &self.layer.mul;
            let eq_evals_at_rz0 = &self.sp.eq_evals_at_rz0;
            let eq_evals_at_rx = &self.sp.eq_evals_at_rx;
            let gate_exists = &mut self.sp.gate_exists_5;
            let hg_vals = &mut self.sp.hg_evals;
            unsafe {
                std::ptr::write_bytes(hg_vals.as_mut_ptr(), 0, total);
                std::ptr::write_bytes(gate_exists.as_mut_ptr(), 0, total);
            }
            for g in mul.iter() {
                hg_vals[g.i_ids[1]] +=
                    F::Field::from(eq_evals_at_rz0[g.o_id] * eq_evals_at_rx[g.i_ids[0]] * g.coef);
                gate_exists[g.i_ids[1]] = true;
            }

            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.layer.input_vals.as_ptr() as *const u8,
                    ctx.pool.v_evals.contents() as *mut u8,
                    total * BN254_ELEM_SIZE,
                );
                std::ptr::copy_nonoverlapping(
                    self.sp.hg_evals.as_ptr() as *const u8,
                    ctx.pool.hg_evals.contents() as *mut u8,
                    total * BN254_ELEM_SIZE,
                );
            }
            let ge_ptr = ctx.pool.gate_exists.contents() as *mut u32;
            for i in 0..total {
                unsafe {
                    *ge_ptr.add(i) = if self.sp.gate_exists_5[i] { 1 } else { 0 };
                }
            }

            if !self.metal_fold_loop(
                ctx,
                input_var_num,
                &eq_simd,
                &eq_mpi,
                Some(phase2_coef),
                transcript,
                mpi_config,
                MetalRoundTarget::Ry,
            ) {
                return false;
            }
            true
        })
    }
}
