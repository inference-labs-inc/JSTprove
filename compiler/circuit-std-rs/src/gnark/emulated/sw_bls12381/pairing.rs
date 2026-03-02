use super::g1::{G1Affine, G1};
use super::g2::{G2AffP, G2Affine, LineEvaluation, LineEvaluations, G2};
use super::point::AffinePoint;
use super::point::Curve;
use crate::gnark::element::value_of;
use crate::gnark::emparam::Bls12381Fp;
use crate::gnark::emparam::Bls12381Fr;
use crate::gnark::emparam::CurveParams;
use crate::gnark::emulated::field_bls12381::e12::{Ext12, GE12};
use crate::gnark::emulated::field_bls12381::e2::{CurveF, GE2};
use crate::gnark::emulated::field_bls12381::e6::GE6;
use expander_compiler::frontend::{Config, Error, RootAPI};
use num_bigint::BigInt;

const LOOP_COUNTER: [i8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
];
pub struct Pairing {
    pub ext12: Ext12,
    pub curve_f: CurveF,
    pub curve: Curve<Bls12381Fp, Bls12381Fr>,
    pub g1: G1,
    pub g2: G2,
    pub btwist: GE2,
}

impl Pairing {
    pub fn new<C: Config, B: RootAPI<C>>(native: &mut B) -> Self {
        let curve_f = CurveF::new(native, Bls12381Fp {});
        let ext12 = Ext12::new(native);
        let curve = Curve::new(native, &CurveParams::get_bls12381_params(), Bls12381Fp {});
        let btwist = GE2 {
            a0: value_of::<C, B, Bls12381Fp>(native, Box::new("4".to_string())),
            a1: value_of::<C, B, Bls12381Fp>(native, Box::new("4".to_string())),
        };

        let g1 = G1::new(native);
        let g2 = G2::new(native);

        Self {
            curve_f,
            ext12,
            curve,
            g1,
            g2,
            btwist,
        }
    }
    pub fn pairing_check<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &[G1Affine],
        q: &mut [G2Affine],
    ) -> Result<(), Error> {
        let f = self.miller_loop(native, p, q).unwrap();
        let buf = self.ext12.conjugate(native, &f);

        let buf = self.ext12.div(native, &buf, &f);
        let f = self.ext12.frobenius_square(native, &buf);
        let f = self.ext12.mul(native, &f, &buf);

        self.ext12.assert_final_exponentiation_is_one(native, &f);

        Ok(())
    }
    pub fn miller_loop<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &[G1Affine],
        q: &mut [G2Affine],
    ) -> Result<GE12, String> {
        let n = p.len();
        if n == 0 || n != q.len() {
            return Err("nvalid inputs sizes".to_string());
        }
        let mut lines = vec![];
        for cur_q in q {
            if cur_q.lines.is_empty() {
                let qlines = self.compute_lines_with_hint(native, &cur_q.p);
                cur_q.lines = qlines;
            }
            let line_evaluations = std::mem::take(&mut cur_q.lines);
            lines.push(line_evaluations);
        }
        self.miller_loop_lines_with_hint(native, p, lines)
    }
    pub fn miller_loop_lines_with_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &[G1Affine],
        lines: Vec<LineEvaluations>,
    ) -> Result<GE12, String> {
        let n = p.len();
        if n == 0 || n != lines.len() {
            return Err("invalid inputs sizes".to_string());
        }
        let mut y_inv = vec![];
        let mut x_neg_over_y = vec![];
        for cur_p in p.iter().take(n) {
            let y_inv_k = self.curve_f.inverse(native, &cur_p.y);
            let x_neg_over_y_k = self.curve_f.mul(native, &cur_p.x, &y_inv_k);
            let x_neg_over_y_k = self.curve_f.neg(native, &x_neg_over_y_k);
            y_inv.push(y_inv_k);
            x_neg_over_y.push(x_neg_over_y_k);
        }

        let mut res = self.ext12.one();

        if let Some(line_evaluation) = &lines[0].0[0][62] {
            let line = line_evaluation;
            res.c0.b0 = self
                .ext12
                .ext6
                .ext2
                .mul_by_element(native, &line.r1, &y_inv[0]);
            res.c0.b1 = self
                .ext12
                .ext6
                .ext2
                .mul_by_element(native, &line.r0, &x_neg_over_y[0]);
        } else {
            return Err("line evaluation is None".to_string());
        }
        res.c1.b1 = self.ext12.ext6.ext2.one();

        if let Some(line_evaluation) = &lines[0].0[1][62] {
            let line = line_evaluation;
            let tmp0 = self
                .ext12
                .ext6
                .ext2
                .mul_by_element(native, &line.r1, &y_inv[0]);
            let tmp1 = self
                .ext12
                .ext6
                .ext2
                .mul_by_element(native, &line.r0, &x_neg_over_y[0]);
            let prod_lines = self
                .ext12
                .mul_014_by_014(native, &tmp0, &tmp1, &res.c0.b0, &res.c0.b1);
            res = GE12 {
                c0: GE6 {
                    b0: prod_lines[0].clone(),
                    b1: prod_lines[1].clone(),
                    b2: prod_lines[2].clone(),
                },
                c1: GE6 {
                    b0: res.c1.b0.clone(),
                    b1: prod_lines[3].clone(),
                    b2: prod_lines[4].clone(),
                },
            };
        } else {
            return Err("line evaluation is None".to_string());
        }

        for k in 1..n {
            if let Some(line_evaluation) = &lines[k].0[0][62] {
                let line = line_evaluation;
                let tmp0 = self
                    .ext12
                    .ext6
                    .ext2
                    .mul_by_element(native, &line.r1, &y_inv[k]);
                let tmp1 = self
                    .ext12
                    .ext6
                    .ext2
                    .mul_by_element(native, &line.r0, &x_neg_over_y[k]);
                res = self.ext12.mul_by_014(native, &res, &tmp0, &tmp1);
            } else {
                return Err("line evaluation is None".to_string());
            }
            if let Some(line_evaluation) = &lines[k].0[1][62] {
                let line = line_evaluation;
                let tmp0 = self
                    .ext12
                    .ext6
                    .ext2
                    .mul_by_element(native, &line.r1, &y_inv[k]);
                let tmp1 = self
                    .ext12
                    .ext6
                    .ext2
                    .mul_by_element(native, &line.r0, &x_neg_over_y[k]);
                res = self.ext12.mul_by_014(native, &res, &tmp0, &tmp1);
            } else {
                return Err("line evaluation is None".to_string());
            }
        }

        let mut copy_res = self.ext12.copy(native, &res);

        for i in (0..=61).rev() {
            res = self.ext12.square(native, &copy_res);
            copy_res = self.ext12.copy(native, &res);
            for k in 0..n {
                if LOOP_COUNTER[i as usize] == 0 {
                    if let Some(line_evaluation) = &lines[k].0[0][i as usize] {
                        let line = line_evaluation;
                        let tmp0 = self
                            .ext12
                            .ext6
                            .ext2
                            .mul_by_element(native, &line.r1, &y_inv[k]);
                        let tmp1 =
                            self.ext12
                                .ext6
                                .ext2
                                .mul_by_element(native, &line.r0, &x_neg_over_y[k]);
                        res = self.ext12.mul_by_014(native, &copy_res, &tmp0, &tmp1);
                        copy_res = self.ext12.copy(native, &res);
                    } else {
                        return Err("line evaluation is None".to_string());
                    }
                } else {
                    if let Some(line_evaluation) = &lines[k].0[0][i as usize] {
                        let line = line_evaluation;
                        let tmp0 = self
                            .ext12
                            .ext6
                            .ext2
                            .mul_by_element(native, &line.r1, &y_inv[k]);
                        let tmp1 =
                            self.ext12
                                .ext6
                                .ext2
                                .mul_by_element(native, &line.r0, &x_neg_over_y[k]);
                        res = self.ext12.mul_by_014(native, &copy_res, &tmp0, &tmp1);
                        copy_res = self.ext12.copy(native, &res);
                    } else {
                        return Err("line evaluation is None".to_string());
                    }
                    if let Some(line_evaluation) = &lines[k].0[1][i as usize] {
                        let line = line_evaluation;
                        let tmp0 = self
                            .ext12
                            .ext6
                            .ext2
                            .mul_by_element(native, &line.r1, &y_inv[k]);
                        let tmp1 =
                            self.ext12
                                .ext6
                                .ext2
                                .mul_by_element(native, &line.r0, &x_neg_over_y[k]);
                        res = self.ext12.mul_by_014(native, &copy_res, &tmp0, &tmp1);
                        copy_res = self.ext12.copy(native, &res);
                    } else {
                        return Err("line evaluation is None".to_string());
                    }
                }
            }
        }
        res = self.ext12.conjugate(native, &copy_res);
        Ok(res)
    }
    pub fn compute_lines_with_hint<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G2AffP,
    ) -> LineEvaluations {
        // let mut c_lines = LineEvaluations::default();
        let mut c_lines: LineEvaluations = LineEvaluations::default();
        let q_acc = q;
        let mut copy_q_acc = self.copy_g2_aff_p(native, q_acc);
        let n = LOOP_COUNTER.len();
        let (q_acc, line1, line2) = self.triple_step(native, copy_q_acc);
        c_lines.0[0][n - 2] = line1;
        c_lines.0[1][n - 2] = line2;
        copy_q_acc = self.copy_g2_aff_p(native, &q_acc);
        for i in (1..=n - 3).rev() {
            if LOOP_COUNTER[i] == 0 {
                let (q_acc, c_lines_0_i) = self.double_step(native, copy_q_acc);
                copy_q_acc = self.copy_g2_aff_p(native, &q_acc);
                c_lines.0[0][i] = c_lines_0_i;
            } else {
                let (q_acc, c_lines_0_i, c_lines_1_i) =
                    self.double_and_add_step(native, copy_q_acc, q);
                copy_q_acc = self.copy_g2_aff_p(native, &q_acc);
                c_lines.0[0][i] = c_lines_0_i;
                c_lines.0[1][i] = c_lines_1_i;
            }
        }
        c_lines.0[0][0] = self.tangent_compute(native, copy_q_acc);
        c_lines
    }
    pub fn double_and_add_step<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p1: G2AffP,
        p2: &G2AffP,
    ) -> (
        G2AffP,
        Option<Box<LineEvaluation>>,
        Option<Box<LineEvaluation>>,
    ) {
        let n = self.ext12.ext6.ext2.sub(native, &p1.y, &p2.y);
        let d = self.ext12.ext6.ext2.sub(native, &p1.x, &p2.x);
        let λ1 = self.ext12.ext6.ext2.div(native, &n, &d);

        let xr = self.ext12.ext6.ext2.square(native, &λ1);
        let tmp = self.ext12.ext6.ext2.add(native, &p1.x, &p2.x);
        let xr = self.ext12.ext6.ext2.sub(native, &xr, &tmp);

        let r0 = λ1.clone();
        let mut r1 = self.ext12.ext6.ext2.mul(native, &λ1, &p1.x);
        r1 = self.ext12.ext6.ext2.sub(native, &r1, &p1.y);

        let line1 = Some(Box::new(LineEvaluation { r0, r1 }));

        let d = self.ext12.ext6.ext2.sub(native, &xr, &p1.x);
        let n = self.ext12.ext6.ext2.double(native, &p1.y);
        let λ2 = self.ext12.ext6.ext2.div(native, &n, &d);
        let λ2 = self.ext12.ext6.ext2.add(native, &λ2, &λ1);
        let λ2 = self.ext12.ext6.ext2.neg(native, &λ2);

        let x4 = self.ext12.ext6.ext2.square(native, &λ2);
        let tmp = self.ext12.ext6.ext2.add(native, &p1.x, &xr);
        let x4 = self.ext12.ext6.ext2.sub(native, &x4, &tmp);

        let y4 = self.ext12.ext6.ext2.sub(native, &p1.x, &x4);
        let y4 = self.ext12.ext6.ext2.mul(native, &λ2, &y4);
        let y4 = self.ext12.ext6.ext2.sub(native, &y4, &p1.y);

        let p = G2AffP { x: x4, y: y4 };

        let r0 = λ2.clone();
        let mut r1 = self.ext12.ext6.ext2.mul(native, &λ2, &p1.x);
        r1 = self.ext12.ext6.ext2.sub(native, &r1, &p1.y);

        let line2 = Some(Box::new(LineEvaluation { r0, r1 }));

        (p, line1, line2)
    }
    pub fn double_step<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p1: G2AffP,
    ) -> (G2AffP, Option<Box<LineEvaluation>>) {
        let n = self.ext12.ext6.ext2.square(native, &p1.x);
        let three = BigInt::from(3);
        let n = self
            .ext12
            .ext6
            .ext2
            .mul_by_const_element(native, &n, &three);
        let d = self.ext12.ext6.ext2.double(native, &p1.y);
        let λ = self.ext12.ext6.ext2.div(native, &n, &d);

        let xr = self.ext12.ext6.ext2.square(native, &λ);
        let tmp = self
            .ext12
            .ext6
            .ext2
            .mul_by_const_element(native, &p1.x, &BigInt::from(2));
        let xr = self.ext12.ext6.ext2.sub(native, &xr, &tmp);

        let pxr = self.ext12.ext6.ext2.sub(native, &p1.x, &xr);
        let λpxr = self.ext12.ext6.ext2.mul(native, &λ, &pxr);
        let yr = self.ext12.ext6.ext2.sub(native, &λpxr, &p1.y);

        let res = G2AffP { x: xr, y: yr };

        let r0 = λ.clone();
        let mut r1 = self.ext12.ext6.ext2.mul(native, &λ, &p1.x);
        r1 = self.ext12.ext6.ext2.sub(native, &r1, &p1.y);

        let line = Some(Box::new(LineEvaluation { r0, r1 }));

        (res, line)
    }
    pub fn triple_step<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p1: G2AffP,
    ) -> (
        G2AffP,
        Option<Box<LineEvaluation>>,
        Option<Box<LineEvaluation>>,
    ) {
        let n = self.ext12.ext6.ext2.square(native, &p1.x);
        let three = BigInt::from(3);
        let n = self
            .ext12
            .ext6
            .ext2
            .mul_by_const_element(native, &n, &three);
        let d = self.ext12.ext6.ext2.double(native, &p1.y);
        let λ1 = self.ext12.ext6.ext2.div(native, &n, &d);

        let r0 = λ1.clone();
        let mut r1 = self.ext12.ext6.ext2.mul(native, &λ1, &p1.x);
        r1 = self.ext12.ext6.ext2.sub(native, &r1, &p1.y);

        let line1 = Some(Box::new(LineEvaluation { r0, r1 }));

        let x2 = self.ext12.ext6.ext2.square(native, &λ1);
        let tmp = self
            .ext12
            .ext6
            .ext2
            .mul_by_const_element(native, &p1.x, &BigInt::from(2));
        let x2 = self.ext12.ext6.ext2.sub(native, &x2, &tmp);

        let x1x2 = self.ext12.ext6.ext2.sub(native, &p1.x, &x2);
        let λ2 = self.ext12.ext6.ext2.div(native, &d, &x1x2);
        let λ2 = self.ext12.ext6.ext2.sub(native, &λ2, &λ1);

        let r0 = λ2.clone();
        let mut r1 = self.ext12.ext6.ext2.mul(native, &λ2, &p1.x);
        r1 = self.ext12.ext6.ext2.sub(native, &r1, &p1.y);

        let line2 = Some(Box::new(LineEvaluation { r0, r1 }));

        let λ2λ2 = self.ext12.ext6.ext2.mul(native, &λ2, &λ2);
        let qxrx = self.ext12.ext6.ext2.add(native, &x2, &p1.x);
        let xr = self.ext12.ext6.ext2.sub(native, &λ2λ2, &qxrx);

        let pxrx = self.ext12.ext6.ext2.sub(native, &p1.x, &xr);
        let λ2pxrx = self.ext12.ext6.ext2.mul(native, &λ2, &pxrx);
        let yr = self.ext12.ext6.ext2.sub(native, &λ2pxrx, &p1.y);

        let res = G2AffP { x: xr, y: yr };

        (res, line1, line2)
    }
    pub fn tangent_compute<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p1: G2AffP,
    ) -> Option<Box<LineEvaluation>> {
        let n = self.ext12.ext6.ext2.square(native, &p1.x);
        let three = BigInt::from(3);
        let n = self
            .ext12
            .ext6
            .ext2
            .mul_by_const_element(native, &n, &three);
        let d = self.ext12.ext6.ext2.double(native, &p1.y);
        let λ = self.ext12.ext6.ext2.div(native, &n, &d);

        let r0 = λ.clone();
        let mut r1 = self.ext12.ext6.ext2.mul(native, &λ, &p1.x);
        r1 = self.ext12.ext6.ext2.sub(native, &r1, &p1.y);

        Some(Box::new(LineEvaluation { r0, r1 }))
    }
    pub fn copy_g2_aff_p<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G2AffP,
    ) -> G2AffP {
        let copy_q_acc_x = self.ext12.ext6.ext2.copy(native, &q.x);
        let copy_q_acc_y = self.ext12.ext6.ext2.copy(native, &q.y);
        G2AffP {
            x: copy_q_acc_x,
            y: copy_q_acc_y,
        }
    }

    pub fn assert_is_on_curve<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: G1Affine) {
        let p_affine = AffinePoint { x: p.x, y: p.y };
        self.curve.assert_is_on_curve(native, &p_affine);
    }

    pub fn assert_is_on_g1<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: &G1Affine) {
        // 1- Check P is on the curve
        self.assert_is_on_curve(native, p.clone());
        // 2- Check P has the right subgroup order
        // [x²]ϕ(P)
        let phi_p = self.g1.phi(native, p);
        let p_squared = self.g1.scalar_mul_by_seed_square(native, &phi_p);
        let neg_p_squared = self.curve.neg(native, &p_squared);
        self.g1.assert_is_equal(native, &neg_p_squared, &p_squared); // Neg returns an inverse of p. It doesn't modify p.);
    }

    // use p: G2Affine ?
    pub fn assert_is_on_g2<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, q: &G2AffP) {
        self.assert_is_on_twist(native, q.clone());
        let x_q = self.g2.scalar_mul_by_seed(native, q);
        let psi_q = self.g2.psi(native, q);

        // [r]Q == 0 <==>  ψ(Q) == [x₀]Q
        self.g2.assert_is_equal(native, &x_q, &psi_q);
    }

    // use p: G2Affine ?
    pub fn assert_is_on_twist<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, q: G2AffP) {
        let is_x_zero = self.ext12.ext6.ext2.is_zero(native, &q.x);
        let is_y_zero = self.ext12.ext6.ext2.is_zero(native, &q.y);
        let selector = native.and(is_x_zero, is_y_zero);
        let zero = self.ext12.ext6.ext2.zero();
        let b = self
            .ext12
            .ext6
            .ext2
            .select(native, selector, &zero, &self.btwist);

        let left = self.ext12.ext6.ext2.square(native, &q.y);
        let mut right = self.ext12.ext6.ext2.square(native, &q.x);
        right = self.ext12.ext6.ext2.mul(native, &right, &q.x);
        right = self.ext12.ext6.ext2.add(native, &right, &b);
        self.ext12.ext6.ext2.assert_isequal(native, &left, &right);
    }
}
