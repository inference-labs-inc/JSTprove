//! This module implements helper functions for the prover side of the sumcheck protocol
//! to evaluate power gates

use std::ops::Mul;

use arith::{Field, SimdField};
use gkr_engine::FieldEngine;

pub(crate) struct SumcheckPowerGateHelper<const D: usize> {
    var_num: usize,
    sumcheck_var_idx: usize,
    cur_eval_size: usize,
}

impl<const D: usize> SumcheckPowerGateHelper<D> {
    pub(crate) fn new(var_num: usize) -> Self {
        SumcheckPowerGateHelper {
            var_num,
            sumcheck_var_idx: 0,
            cur_eval_size: 1 << var_num,
        }
    }

    // Function to interpolate a quadratic polynomial and update an array of points
    fn interpolate_3<F: FieldEngine>(p_add: &[F::Field; 3], p: &mut [F::Field; D]) {
        // Calculate coefficients for the interpolating polynomial
        let p_add_coef_0 = p_add[0];
        let p_add_coef_2 = (p_add[2] - p_add[1] - p_add[1] + p_add[0]) * F::CircuitField::INV_2;

        let p_add_coef_1 = p_add[1] - p_add_coef_0 - p_add_coef_2;

        // Update the p array by evaluating the interpolated polynomial at different points
        // and adding the results to the existing values
        p[0] += p_add_coef_0;
        p[1] += p_add_coef_0 + p_add_coef_1 + p_add_coef_2;
        p[2] += p_add_coef_0 + p_add_coef_1.double() + p_add_coef_2.double().double();
        p[3] += p_add_coef_0 + p_add_coef_1.mul_by_3() + p_add_coef_2.mul_by_3().mul_by_3();
        p[4] += p_add_coef_0
            + p_add_coef_1.double().double()
            + p_add_coef_2 * F::CircuitField::from(16);
        p[5] += p_add_coef_0 + p_add_coef_1.mul_by_5() + p_add_coef_2 * F::CircuitField::from(25);
        p[6] += p_add_coef_0
            + p_add_coef_1.mul_by_3().double()
            + p_add_coef_2 * F::CircuitField::from(36);
    }

    #[inline]
    fn evaluate<VF: Field, ChallengeF: Field, EvalF>(
        eval_size: usize,
        src_v: &[VF],
        bk_hg_5: &[ChallengeF],
        bk_hg_1: &[ChallengeF],
        gate_exists_5: &[bool],
        gate_exists_1: &[bool],
        p: &mut [EvalF],
    ) -> [EvalF; 3]
    where
        EvalF: Field + From<ChallengeF> + Mul<ChallengeF, Output = EvalF> + Mul<VF, Output = EvalF>,
    {
        log::trace!("Eval size: {eval_size}");
        for i in 0..eval_size {
            if !gate_exists_5[i * 2] && !gate_exists_5[i * 2 + 1] {
                continue;
            }
            let mut f_v = [VF::ZERO; D];
            let mut hg_v = [ChallengeF::ZERO; D];
            f_v[0] = src_v[i * 2];
            f_v[1] = src_v[i * 2 + 1];
            hg_v[0] = bk_hg_5[i * 2];
            hg_v[1] = bk_hg_5[i * 2 + 1];
            let delta_f = f_v[1] - f_v[0];
            let delta_hg = hg_v[1] - hg_v[0];

            for i in 2..D {
                f_v[i] = f_v[i - 1] + delta_f;
                hg_v[i] = hg_v[i - 1] + delta_hg;
            }
            for i in 0..D {
                let pow5 = f_v[i].square().square() * f_v[i];
                p[i] += EvalF::from(hg_v[i]) * pow5;
            }
        }

        let mut p_add = [EvalF::ZERO; 3];
        for i in 0..eval_size {
            if !gate_exists_1[i * 2] && !gate_exists_1[i * 2 + 1] {
                continue;
            }
            let mut f_v = [VF::ZERO; 3];
            let mut hg_v = [ChallengeF::ZERO; 3];
            f_v[0] = src_v[i * 2];
            f_v[1] = src_v[i * 2 + 1];
            hg_v[0] = bk_hg_1[i * 2];
            hg_v[1] = bk_hg_1[i * 2 + 1];
            p_add[0] += EvalF::from(hg_v[0]) * f_v[0];
            p_add[1] += EvalF::from(hg_v[1]) * f_v[1];
            p_add[2] += EvalF::from(hg_v[0] + hg_v[1]) * (f_v[0] + f_v[1]);
        }
        p_add[2] = p_add[1].mul_by_6() + p_add[0].mul_by_3() - p_add[2].double();

        // interpolate p_add into 7 points
        p_add
        // [p_add, p]
        // Self::interpolate_3(&p_add, &mut p);
        // p
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn poly_eval_at<F: FieldEngine>(
        &self,
        var_idx: usize,
        bk_f: &[F::Field],
        bk_hg_5: &[F::ChallengeField],
        bk_hg_1: &[F::ChallengeField],
        init_v: &[F::SimdCircuitField],
        gate_exists_5: &[bool],
        gate_exists_1: &[bool],
    ) -> [F::Field; D] {
        let mut p = [F::Field::zero(); D];
        log::trace!("bk_f: {:?}", &bk_f[..4]);
        log::trace!("bk_hg: {:?}", &bk_hg_5[..4]);
        log::trace!("init_v: {:?}", &init_v[..4]);

        let eval_size = 1 << (self.var_num - var_idx - 1);
        let p_add = {
            if var_idx == 0 {
                Self::evaluate(
                    eval_size,
                    init_v,
                    bk_hg_5,
                    bk_hg_1,
                    gate_exists_5,
                    gate_exists_1,
                    &mut p,
                )
            } else {
                Self::evaluate(
                    eval_size,
                    bk_f,
                    bk_hg_5,
                    bk_hg_1,
                    gate_exists_5,
                    gate_exists_1,
                    &mut p,
                )
            }
        };

        Self::interpolate_3::<F>(&p_add, &mut p);
        p
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn receive_challenge<F: FieldEngine>(
        &mut self,
        var_idx: usize,
        r: F::ChallengeField,
        bk_f: &mut [F::Field],
        bk_hg_5: &mut [F::ChallengeField],
        bk_hg_1: &mut [F::ChallengeField],
        init_v: &[F::SimdCircuitField],
        gate_exists_5: &mut [bool],
        gate_exists_1: &mut [bool],
    ) {
        assert_eq!(var_idx, self.sumcheck_var_idx);
        assert!(var_idx < self.var_num);
        log::trace!("challenge eval size: {}", self.cur_eval_size);

        self.cur_eval_size >>= 1;

        if var_idx == 0 {
            for i in 0..self.cur_eval_size {
                bk_f[i] = r * (init_v[2 * i + 1] - init_v[2 * i]) + init_v[2 * i];
            }
        }
        {
            for i in 0..self.cur_eval_size {
                bk_f[i] = bk_f[2 * i] + (bk_f[2 * i + 1] - bk_f[2 * i]).scale(&r);
            }
        }

        for i in 0..self.cur_eval_size {
            if !gate_exists_5[i * 2] && !gate_exists_5[i * 2 + 1] {
                gate_exists_5[i] = false;
                bk_hg_5[i] = F::ChallengeField::zero();
            } else {
                gate_exists_5[i] = true;
                bk_hg_5[i] = bk_hg_5[2 * i] + (bk_hg_5[2 * i + 1] - bk_hg_5[2 * i]) * r;
            }

            if !gate_exists_1[i * 2] && !gate_exists_1[i * 2 + 1] {
                gate_exists_1[i] = false;
                bk_hg_1[i] = F::ChallengeField::zero();
            } else {
                gate_exists_1[i] = true;
                bk_hg_1[i] = bk_hg_1[2 * i] + (bk_hg_1[2 * i + 1] - bk_hg_1[2 * i]) * r;
            }
        }

        self.sumcheck_var_idx += 1;
    }
}
