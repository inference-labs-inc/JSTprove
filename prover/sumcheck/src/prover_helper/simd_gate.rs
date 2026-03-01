//! This module implements helper functions for the prover side of the sumcheck protocol
//! to evaluate SIMD gates.
//! A SIMD gate is a gate that merges all elements in a SIMD into a single one.

use std::marker::PhantomData;

use arith::{ExtensionField, Field};
use gkr_engine::{FieldEngine, FieldType};

pub(crate) struct SumcheckSimdProdGateHelper<F: FieldEngine> {
    pub(crate) var_num: usize,
    field: PhantomData<F>,
}

// The logic is exactly the same as SumcheckProductGateHelper, but field types are different
impl<F: FieldEngine> SumcheckSimdProdGateHelper<F> {
    pub(crate) fn new(var_num: usize) -> Self {
        SumcheckSimdProdGateHelper {
            var_num,
            field: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn poly_eval_at(
        &self,
        var_idx: usize,
        degree: usize,
        bk_eq: &mut [F::ChallengeField],
        bk_f: &mut [F::ChallengeField],
        bk_hg: &mut [F::ChallengeField],
    ) -> [F::ChallengeField; 4] {
        assert_eq!(degree, 3);
        let mut p0 = F::ChallengeField::zero();
        let mut p1 = F::ChallengeField::zero();
        let mut p2 = F::ChallengeField::zero();
        let mut p3 = F::ChallengeField::zero();

        let eval_size = 1 << (self.var_num - var_idx - 1);

        if F::FIELD_TYPE == FieldType::GF2Ext128 {
            for i in 0..eval_size {
                let eq_v_0 = bk_eq[i * 2];
                let eq_v_1 = bk_eq[i * 2 + 1];
                let f_v_0 = bk_f[i * 2];
                let f_v_1 = bk_f[i * 2 + 1];
                let hg_v_0 = bk_hg[i * 2];
                let hg_v_1 = bk_hg[i * 2 + 1];

                p0 += eq_v_0 * f_v_0 * hg_v_0;
                p1 += eq_v_1 * f_v_1 * hg_v_1;

                let eq_linear = (eq_v_1 - eq_v_0).mul_by_x();
                let f_linear = (f_v_1 - f_v_0).mul_by_x();
                let hg_linear = (hg_v_1 - hg_v_0).mul_by_x();

                // evaluated at x and x^2 for p2 and p3
                p2 += (eq_linear + eq_v_0) * (f_linear + f_v_0) * (hg_linear + hg_v_0);
                p3 += (eq_linear.mul_by_x() + eq_v_0)
                    * (f_linear.mul_by_x() + f_v_0)
                    * (hg_linear.mul_by_x() + hg_v_0);
            }
        } else {
            for i in 0..eval_size {
                let eq_v_0 = bk_eq[i * 2];
                let eq_v_1 = bk_eq[i * 2 + 1];
                let f_v_0 = bk_f[i * 2];
                let f_v_1 = bk_f[i * 2 + 1];
                let hg_v_0 = bk_hg[i * 2];
                let hg_v_1 = bk_hg[i * 2 + 1];

                p0 += eq_v_0 * f_v_0 * hg_v_0;
                p1 += eq_v_1 * f_v_1 * hg_v_1;

                // evaluated at 2 and 3 for p2 and p3
                let tmp0 = eq_v_1 - eq_v_0;
                let tmp1 = f_v_1 - f_v_0;
                let tmp2 = hg_v_1 - hg_v_0;
                let tmp3 = eq_v_1 + tmp0;
                let tmp4 = f_v_1 + tmp1;
                let tmp5 = hg_v_1 + tmp2;

                p2 += tmp3 * tmp4 * tmp5;
                p3 += (tmp3 + tmp0) * (tmp4 + tmp1) * (tmp5 + tmp2);
            }
        }

        [p0, p1, p2, p3]
    }

    /// Evaluate the GKR2 sumcheck polynomial at a SIMD variable,
    /// after x-sumcheck rounds have fixed the x variables. The
    /// polynomial is degree (D-1) in the SIMD variables.
    pub(crate) fn gkr2_poly_eval_at<const D: usize>(
        &self,
        var_idx: usize,
        bk_eq: &[F::ChallengeField],
        bk_v_simd: &[F::ChallengeField],
        add_eval: F::ChallengeField,
        pow_5_eval: F::ChallengeField,
    ) -> [F::ChallengeField; D] {
        let mut p = [F::ChallengeField::zero(); D];
        let mut p_add = [F::ChallengeField::zero(); 3];
        let eval_size = 1 << (self.var_num - var_idx - 1);

        for i in 0..eval_size {
            // witness polynomial along current variable
            let mut f_v = [F::ChallengeField::zero(); D];
            // eq polynomial along current variable
            let mut eq_v = [F::ChallengeField::zero(); D];
            f_v[0] = bk_v_simd[i * 2];
            f_v[1] = bk_v_simd[i * 2 + 1];
            eq_v[0] = bk_eq[i * 2];
            eq_v[1] = bk_eq[i * 2 + 1];

            // Evaluate term eq(A, r_z) * Pow5(r_z, r_x) * V(A, r_x)^5
            let delta_f = f_v[1] - f_v[0];
            let delta_eq = eq_v[1] - eq_v[0];
            for i in 2..D {
                f_v[i] = f_v[i - 1] + delta_f;
                eq_v[i] = eq_v[i - 1] + delta_eq;
            }
            for i in 0..D {
                let pow5 = f_v[i].square().square() * f_v[i];
                p[i] += pow_5_eval * pow5 * eq_v[i];
            }

            // Evaluate term eq(A, r_z) * Add(r_z, r_x) * V(A, r_x)
            p_add[0] += add_eval * f_v[0] * eq_v[0];
            p_add[1] += add_eval * f_v[1] * eq_v[1];
            // Intermediate term for p_add[2]
            p_add[2] += add_eval * (f_v[0] + f_v[1]) * (eq_v[0] + eq_v[1]);
        }
        p_add[2] = p_add[1].mul_by_6() + p_add[0].mul_by_3() - p_add[2].double();

        // Interpolate p_add into 7 points, add to p
        Self::interpolate_3::<D>(&p_add, &mut p);
        p
    }

    // Function to interpolate a quadratic polynomial and update an array of points
    fn interpolate_3<const D: usize>(
        p_add: &[F::ChallengeField; 3],
        p: &mut [F::ChallengeField; D],
    ) {
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
    pub(crate) fn receive_challenge(
        &mut self,
        var_idx: usize,
        r: F::ChallengeField,
        bk_eq: &mut [F::ChallengeField],
        bk_f: &mut [F::ChallengeField],
        bk_hg: &mut [F::ChallengeField],
    ) {
        assert!(var_idx < self.var_num);

        let eval_size = 1 << (self.var_num - var_idx - 1);
        for i in 0..eval_size {
            bk_eq[i] = bk_eq[2 * i] + (bk_eq[2 * i + 1] - bk_eq[2 * i]) * r;
            bk_f[i] = bk_f[2 * i] + (bk_f[2 * i + 1] - bk_f[2 * i]) * r;
            bk_hg[i] = bk_hg[2 * i] + (bk_hg[2 * i + 1] - bk_hg[2 * i]) * r;
        }
    }
}
