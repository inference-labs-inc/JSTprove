//! This module implements helper functions for the prover side of the sumcheck protocol
//! to evaluate Mul gates

use std::ops::Mul;

use arith::{ExtensionField, Field, SimdField};
use gkr_engine::{FieldEngine, FieldType};

pub(crate) struct SumcheckProductGateHelper {
    var_num: usize,
}

impl SumcheckProductGateHelper {
    pub(crate) fn new(var_num: usize) -> Self {
        SumcheckProductGateHelper { var_num }
    }

    #[inline]
    fn evaluate<VF: Field, EvalF>(
        eval_size: usize,
        bk_f: &[VF],
        bk_hg: &[EvalF],
        gate_exists: &[bool],
    ) -> [EvalF; 3]
    where
        EvalF: Field + Mul<VF, Output = EvalF>,
    {
        let mut p0 = EvalF::ZERO;
        let mut p1 = EvalF::ZERO;
        let mut p2 = EvalF::ZERO;
        for i in 0..eval_size {
            if !gate_exists[i * 2] && !gate_exists[i * 2 + 1] {
                continue;
            }

            let f_v_0 = bk_f[i * 2];
            let f_v_1 = bk_f[i * 2 + 1];
            let hg_v_0 = bk_hg[i * 2];
            let hg_v_1 = bk_hg[i * 2 + 1];
            p0 += hg_v_0 * f_v_0;
            log::trace!(
                "p0.v+= {:?} * {:?} =  {:?}",
                f_v_0,
                hg_v_0,
                hg_v_0 * f_v_0 + p1
            );
            p1 += hg_v_1 * f_v_1;
            p2 += (hg_v_0 + hg_v_1) * (f_v_0 + f_v_1);
        }
        [p0, p1, p2]
    }

    // Sumcheck the product of two multi-linear polynomials f and h_g
    //
    // Inputs:
    // - var_idx: the index of the variable to evaluate
    // - degree: the degree of the result univariate polynomial
    // - bk_f: bookkeeping table of f(x)
    // - bk_hg: bookkeeping table of h_g(x)
    // - init_v: input values; will be processed iff var_idex == 0
    // Output:
    // - the univariate polynomial that prover sends to the verifier
    #[inline]
    pub(crate) fn poly_eval_at<F: FieldEngine>(
        &self,
        var_idx: usize,
        degree: usize,
        bk_f: &[F::Field],
        bk_hg: &[F::Field],
        init_v: &[F::SimdCircuitField],
        gate_exists: &[bool],
    ) -> [F::Field; 3] {
        assert_eq!(degree, 2);

        log::trace!("bk_f: {:?}", &bk_f[..4]);
        log::trace!("bk_hg: {:?}", &bk_hg[..4]);
        log::trace!("init_v: {:?}", &init_v[..4]);

        let eval_size = 1 << (self.var_num - var_idx - 1);
        log::trace!("Eval size: {eval_size}");

        let [p0, p1, mut p2] = {
            if var_idx == 0 {
                Self::evaluate(eval_size, init_v, bk_hg, gate_exists)
            } else {
                Self::evaluate(eval_size, bk_f, bk_hg, gate_exists)
            }
        };

        if F::FIELD_TYPE == FieldType::GF2Ext128 {
            // over GF2_128, the three points are at 0, 1 and X
            let p2x = p2.mul_by_x();
            let p2x2 = p2x.mul_by_x();
            let linear_term = p1 + p0 + p2;
            p2 = p2x2 + linear_term.mul_by_x() + p0;
        } else {
            // when Field size > 2, the three points are 0, 1, -2
            p2 = p1.mul_by_6() + p0.mul_by_3() - p2.double();
        }
        [p0, p1, p2]
    }

    // process the challenge and update the bookkeeping tables for f and h_g accordingly
    #[inline]
    pub(crate) fn receive_challenge<F: FieldEngine>(
        &mut self,
        var_idx: usize,
        r: F::ChallengeField,
        bk_f: &mut [F::Field],
        bk_hg: &mut [F::Field],
        init_v: &[F::SimdCircuitField],
        gate_exists: &mut [bool],
    ) {
        assert!(var_idx < self.var_num);

        let eval_size = 1 << (self.var_num - var_idx - 1);

        if var_idx == 0 {
            for i in 0..eval_size {
                bk_f[i] = r * (init_v[2 * i + 1] - init_v[2 * i]) + init_v[2 * i];
            }
        } else {
            for i in 0..eval_size {
                bk_f[i] = bk_f[2 * i] + (bk_f[2 * i + 1] - bk_f[2 * i]).scale(&r);
            }
        }

        for i in 0..eval_size {
            if !gate_exists[i * 2] && !gate_exists[i * 2 + 1] {
                gate_exists[i] = false;
                bk_hg[i] = F::Field::zero();
            } else {
                gate_exists[i] = true;
                bk_hg[i] = bk_hg[2 * i] + (bk_hg[2 * i + 1] - bk_hg[2 * i]).scale(&r);
            }
        }
    }
}
