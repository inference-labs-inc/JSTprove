use std::fmt::{Debug, Formatter, Result};
use std::str::FromStr;

use crate::gnark::element::{value_of, Element};
use crate::gnark::emparam::Bls12381Fp;
use crate::gnark::emulated::field_bls12381::e2::Ext2;
use crate::gnark::emulated::field_bls12381::e2::GE2;
use crate::sha256::m31_utils::{big_less_than, from_binary};
use crate::utils::simple_select;

use expander_compiler::declare_circuit;
use expander_compiler::frontend::{Config, Define, M31Config, RootAPI, Variable};
use num_bigint::BigInt;
const M_COMPRESSED_SMALLEST: u8 = 0b100 << 5;
const M_COMPRESSED_LARGEST: u8 = 0b101 << 5;

#[derive(Default, Clone, Debug)]
pub struct G2AffP {
    pub x: GE2,
    pub y: GE2,
}

impl G2AffP {
    pub fn new(x: GE2, y: GE2) -> Self {
        Self { x, y }
    }
    pub fn from_vars(
        x0: Vec<Variable>,
        y0: Vec<Variable>,
        x1: Vec<Variable>,
        y1: Vec<Variable>,
    ) -> Self {
        Self {
            x: GE2::from_vars(x0, y0),
            y: GE2::from_vars(x1, y1),
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct LineEvaluation {
    pub r0: GE2,
    pub r1: GE2,
}

type LineEvaluationArray = [[Option<Box<LineEvaluation>>; 63]; 2];

#[derive(Clone, Debug)]
pub struct LineEvaluations(pub LineEvaluationArray);

impl Default for LineEvaluations {
    fn default() -> Self {
        LineEvaluations([[None; 63]; 2].map(|row: [Option<Bls12381Fp>; 63]| row.map(|_| None)))
    }
}
impl LineEvaluations {
    pub fn is_empty(&self) -> bool {
        self.0
            .iter()
            .all(|row| row.iter().all(|cell| cell.is_none()))
    }
}

#[derive(Clone)]
pub struct G2Affine {
    pub p: G2AffP,
    pub lines: LineEvaluations,
}

pub struct G2 {
    pub ext2: Ext2,
    pub u1: Element<Bls12381Fp>,
    pub v: GE2,
}

impl G2 {
    pub fn new<C: Config, B: RootAPI<C>>(native: &mut B) -> Self {
        let curve_f = Ext2::new(native);
        let u1 = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939437".to_string()));
        let v0 = value_of::<C, B, Bls12381Fp>(native, Box::new("2973677408986561043442465346520108879172042883009249989176415018091420807192182638567116318576472649347015917690530".to_string()));
        let v1 = value_of::<C, B, Bls12381Fp>(native, Box::new("1028732146235106349975324479215795277384839936929757896155643118032610843298655225875571310552543014690878354869257".to_string()));
        let v = GE2::from_vars(v0.limbs, v1.limbs);
        Self {
            ext2: curve_f,
            u1,
            v,
        }
    }
    pub fn neg<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: &G2AffP) -> G2AffP {
        let yr = self.ext2.neg(native, &p.y);
        G2AffP::new(p.x.clone(), yr)
    }
    pub fn copy_g2_aff_p<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G2AffP,
    ) -> G2AffP {
        let copy_q_acc_x = self.ext2.copy(native, &q.x);
        let copy_q_acc_y = self.ext2.copy(native, &q.y);
        G2AffP {
            x: copy_q_acc_x,
            y: copy_q_acc_y,
        }
    }
    pub fn double<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: &G2AffP) -> G2AffP {
        let xx3a = self.ext2.square(native, &p.x);
        let xx3a = self
            .ext2
            .mul_by_const_element(native, &xx3a, &BigInt::from(3));
        let y2 = self.ext2.double(native, &p.y);
        let λ = self.ext2.div(native, &xx3a, &y2);

        let x2 = self.ext2.double(native, &p.x);
        let λλ = self.ext2.square(native, &λ);
        let xr = self.ext2.sub(native, &λλ, &x2);

        let pxrx = self.ext2.sub(native, &p.x, &xr);
        let λpxrx = self.ext2.mul(native, &λ, &pxrx);
        let yr = self.ext2.sub(native, &λpxrx, &p.y);

        G2AffP::new(xr, yr)
    }

    pub fn double_and_add<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G2AffP,
        q: &G2AffP,
    ) -> G2AffP {
        // compute λ1 = (q.y-p.y)/(q.x-p.x)
        let yqyp = self.ext2.sub(native, &q.y, &p.y);
        let xqxp = self.ext2.sub(native, &q.x, &p.x);
        let λ1 = self.ext2.div(native, &yqyp, &xqxp);

        // compute x2 = λ1²-p.x-q.x
        let λ1λ1 = self.ext2.square(native, &λ1);
        let xqxp = self.ext2.add(native, &p.x, &q.x);
        let x2 = self.ext2.sub(native, &λ1λ1, &xqxp);
        // omit y2 computation
        // compute λ2 = -λ1-2*p.y/(x2-p.x)
        let pypy = self.ext2.add(native, &p.y, &p.y);
        let x2xp = self.ext2.sub(native, &x2, &p.x);
        let mut λ2 = self.ext2.div(native, &pypy, &x2xp);
        λ2 = self.ext2.add(native, &λ1, &λ2);
        λ2 = self.ext2.neg(native, &λ2);

        // compute x3 =λ2²-p.x-x3
        let λ2λ2 = self.ext2.square(native, &λ2);
        let mut x3 = self.ext2.sub(native, &λ2λ2, &p.x);
        x3 = self.ext2.sub(native, &x3, &x2);

        // compute y3 = λ2*(p.x - x3)-p.ys
        let mut y3 = self.ext2.sub(native, &p.x, &x3);
        y3 = self.ext2.mul(native, &λ2, &y3);
        y3 = self.ext2.sub(native, &y3, &p.y);

        G2AffP::new(x3, y3)
    }

    pub fn double_n<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G2AffP,
        n: usize,
    ) -> G2AffP {
        let mut pn = self.copy_g2_aff_p(native, p);
        for _ in 0..n {
            pn = self.double(native, &pn);
        }

        pn
    }

    pub fn triple<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: &G2AffP) -> G2AffP {
        // compute λ1 = (3p.x²)/2p.y
        let mut xx = self.ext2.square(native, &p.x);
        xx = self
            .ext2
            .mul_by_const_element(native, &xx, &BigInt::from(3));
        let y2 = self.ext2.double(native, &p.y);
        let λ1 = self.ext2.div(native, &xx, &y2);
        // xr = λ1²-2p.x
        let mut x2 = self
            .ext2
            .mul_by_const_element(native, &p.x, &BigInt::from(2));
        let λ1λ1 = self.ext2.square(native, &λ1);
        x2 = self.ext2.sub(native, &λ1λ1, &x2);

        let x1x2 = self.ext2.sub(native, &p.x, &x2);
        let mut λ2 = self.ext2.div(native, &y2, &x1x2);
        λ2 = self.ext2.sub(native, &λ2, &λ1);

        // xr = λ²-p.x-x2
        let λ2λ2 = self.ext2.square(native, &λ2);
        let qxrx = self.ext2.add(native, &x2, &p.x);
        let xr = self.ext2.sub(native, &λ2λ2, &qxrx);

        // yr = λ(p.x-xr) - p.y
        let pxrx = self.ext2.sub(native, &p.x, &xr);
        let λ2pxrx = self.ext2.mul(native, &λ2, &pxrx);
        let yr = self.ext2.sub(native, &λ2pxrx, &p.y);
        G2AffP::new(xr, yr)
    }

    pub fn assert_is_equal<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G2AffP,
        q: &G2AffP,
    ) {
        self.ext2.assert_isequal(native, &p.x, &q.x);
        self.ext2.assert_isequal(native, &p.y, &q.y);
    }
    pub fn add<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G2AffP,
        q: &G2AffP,
    ) -> G2AffP {
        let qypy = self.ext2.sub(native, &q.y, &p.y);
        let qxpx = self.ext2.sub(native, &q.x, &p.x);
        let λ = self.ext2.div(native, &qypy, &qxpx);

        let λλ = self.ext2.square(native, &λ);
        let qxpx = self.ext2.add(native, &p.x, &q.x);
        let xr = self.ext2.sub(native, &λλ, &qxpx);

        let pxrx = self.ext2.sub(native, &p.x, &xr);
        let λpxrx = self.ext2.mul(native, &λ, &pxrx);
        let yr = self.ext2.sub(native, &λpxrx, &p.y);

        G2AffP::new(xr, yr)
    }
    pub fn psi<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, q: &G2AffP) -> G2AffP {
        let x = self.ext2.mul_by_element(native, &q.x, &self.u1);
        let y = self.ext2.conjugate(native, &q.y);
        let y = self.ext2.mul(native, &y, &self.v);
        G2AffP::new(GE2::from_vars(x.a1.limbs, x.a0.limbs), y)
    }
    pub fn mul_windowed<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G2AffP,
        s: BigInt,
    ) -> G2AffP {
        let mut ops = [
            self.copy_g2_aff_p(native, q),
            self.copy_g2_aff_p(native, q),
            self.copy_g2_aff_p(native, q),
        ];
        ops[1] = self.double(native, &ops[1]);
        ops[2] = self.add(native, &ops[0], &ops[1]);
        let b = s.to_bytes_be();
        let b = &b.1[1..];
        let mut res = self.copy_g2_aff_p(native, &ops[2]);

        res = self.double(native, &res);
        res = self.double(native, &res);
        res = self.add(native, &res, &ops[0]);

        res = self.double(native, &res);
        res = self.double(native, &res);

        res = self.double(native, &res);
        res = self.double(native, &res);
        res = self.add(native, &res, &ops[1]);
        // let mut copy_res = self.copy_g2_aff_p(native, &res);
        for w in b {
            let mut mask = 0xc0;
            for j in 0..4 {
                res = self.double(native, &res);
                res = self.double(native, &res);
                let c = (w & mask) >> (6 - 2 * j);
                if c != 0 {
                    res = self.add(native, &res, &ops[(c - 1) as usize]);
                }
                mask >>= 2;
            }
        }
        res
    }
    pub fn clear_cofactor<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        p: &G2AffP,
    ) -> G2AffP {
        let p_neg = self.neg(native, p);
        let x_big = BigInt::from_str("15132376222941642752").expect("Invalid string for BigInt");

        let xg_neg = self.mul_windowed(native, p, x_big.clone());
        let xg = self.neg(native, &xg_neg);

        let xxg = self.mul_windowed(native, &xg, x_big.clone());
        let xxg = self.neg(native, &xxg);

        let mut res = self.add(native, &xxg, &xg_neg);
        res = self.add(native, &res, &p_neg);

        let mut t = self.add(native, &xg, &p_neg);
        t = self.psi(native, &t);

        res = self.add(native, &res, &t);

        let t_double = self.double(native, p);

        let third_root_one_g1 = value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string()));

        let mut t_double_mul = G2AffP::new(t_double.x.clone(), t_double.y.clone());
        t_double_mul.x = self
            .ext2
            .mul_by_element(native, &t_double_mul.x, &third_root_one_g1);
        t_double_mul = self.neg(native, &t_double_mul);

        self.add(native, &res, &t_double_mul)
    }
    pub fn map_to_curve2<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, in0: &GE2) -> G2AffP {
        let a = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(240)).limbs,
        );
        let b = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(1012)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(1012)).limbs,
        );

        let xi = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(-2i32)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(-1i32)).limbs,
        );

        let t_sq = self.ext2.square(native, in0);
        let xi_t_sq = self.ext2.mul(native, &t_sq, &xi);

        let xi_2_t_4 = self.ext2.square(native, &xi_t_sq);
        let num_den_common = self.ext2.add(native, &xi_2_t_4, &xi_t_sq);

        let a_neg = self.ext2.neg(native, &a);
        let x0_den = self.ext2.mul(native, &a_neg, &num_den_common);

        let x1_den = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(240)).limbs, value_of::<C, B, Bls12381Fp>(native, Box::new("4002409555221667392624310435006688643935503118305586438271171395842971157480381377015405980053539358417135540939436".to_string())).limbs,
        );

        let exception = self.ext2.is_zero(native, &x0_den);

        let one = self.ext2.one().clone();
        let num_den_common = self.ext2.add(native, &num_den_common, &one);
        let x0_num = self.ext2.mul(native, &num_den_common, &b);

        let denom = self.ext2.select(native, exception, &x1_den, &x0_den);

        let x0 = self.ext2.div(native, &x0_num, &denom);

        let x0_sq = self.ext2.square(native, &x0);
        let x0_cub = self.ext2.mul(native, &x0, &x0_sq);
        let x0_a = self.ext2.mul(native, &a, &x0);
        let g_x0_tmp = self.ext2.add(native, &x0_cub, &x0_a);
        let g_x0 = self.ext2.add(native, &g_x0_tmp, &b);

        let x1 = self.ext2.mul(native, &xi_t_sq, &x0);

        let xi_3_t_6_tmp = self.ext2.mul(native, &xi_t_sq, &xi_t_sq);
        let xi_3_t_6 = self.ext2.mul(native, &xi_3_t_6_tmp, &xi_t_sq);

        let g_x1 = self.ext2.mul(native, &xi_3_t_6, &g_x0);

        let inputs = vec![
            g_x0.a0.clone(),
            g_x0.a1.clone(),
            g_x1.a0.clone(),
            g_x1.a1.clone(),
            in0.a0.clone(),
            in0.a1.clone(),
        ];
        let output = self
            .ext2
            .curve_f
            .new_hint(native, "myhint.getsqrtx0x1fq2newhint", 3, inputs);
        let is_square = self.ext2.curve_f.is_zero(native, &output[0]); // is_square = 0 if g_x0 has not square root, 1 otherwise
        let y = GE2 {
            a0: output[1].clone(),
            a1: output[2].clone(),
        };

        let y_sq = self.ext2.square(native, &y);
        let expected = self.ext2.select(native, is_square, &g_x1, &g_x0);

        self.ext2.assert_isequal(native, &expected, &y_sq);

        let in_x0_zero = self.ext2.curve_f.is_zero(native, &in0.a0);
        let y_x0_zero = self.ext2.curve_f.is_zero(native, &y.a0);
        let sgn_in = self.ext2.get_e2_sign(native, in0, in_x0_zero);
        let sgn_y = self.ext2.get_e2_sign(native, &y, y_x0_zero);

        native.assert_is_equal(sgn_in, sgn_y);

        let out_b0 = self.ext2.select(native, is_square, &x1, &x0);
        let out_b1 = y.clone();
        G2AffP {
            x: out_b0,
            y: out_b1,
        }
    }

    pub fn eval_polynomial<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        monic: bool,
        coefficients: Vec<GE2>,
        x: &GE2,
    ) -> GE2 {
        let mut dst = coefficients[coefficients.len() - 1].clone();
        if monic {
            dst = self.ext2.add(native, &dst, x);
        }
        for i in (0..coefficients.len() - 1).rev() {
            dst = self.ext2.mul(native, &dst, x);
            dst = self.ext2.add(native, &dst, &coefficients[i]);
        }
        dst
    }
    pub fn isogeny_x_numerator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let coeff0 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new("889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235542".to_string())).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new("889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235542".to_string())).limbs,
        );
        let coeff1 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new("2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706522".to_string())).limbs,
        );
        let coeff2 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new("2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706526".to_string())).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new("1334136518407222464472596608578634718852294273313002628444019378708010550163612621480895876376338554679298090853261".to_string())).limbs,
        );
        let coeff3 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new("3557697382419259905260257622876359250272784728834673675850718343221361467102966990615722337003569479144794908942033".to_string())).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
        );
        self.eval_polynomial(native, false, vec![coeff0, coeff1, coeff2, coeff3], x)
    }
    pub fn isogeny_y_numerator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
        y: &GE2,
    ) -> GE2 {
        let coeff0 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new("3261222600550988246488569487636662646083386001431784202863158481286248011511053074731078808919938689216061999863558".to_string())).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new("3261222600550988246488569487636662646083386001431784202863158481286248011511053074731078808919938689216061999863558".to_string())).limbs,
        );
        let coeff1 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new("889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235518".to_string())).limbs,
        );
        let coeff2 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new("2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706524".to_string())).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new("1334136518407222464472596608578634718852294273313002628444019378708010550163612621480895876376338554679298090853263".to_string())).limbs,
        );
        let coeff3 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new("2816510427748580758331037284777117739799287910327449993381818688383577828123182200904113516794492504322962636245776".to_string())).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
        );
        let dst = self.eval_polynomial(native, false, vec![coeff0, coeff1, coeff2, coeff3], x);
        self.ext2.mul(native, &dst, y)
    }
    pub fn isogeny_x_denominator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let coeff0 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(-72)).limbs,
        );
        let coeff1 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(12)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(-12)).limbs,
        );
        self.eval_polynomial(native, true, vec![coeff0, coeff1], x)
    }
    pub fn isogeny_y_denominator<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        x: &GE2,
    ) -> GE2 {
        let coeff0 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(-432)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(-432)).limbs,
        );
        let coeff1 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(0)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(-216)).limbs,
        );
        let coeff2 = GE2::from_vars(
            value_of::<C, B, Bls12381Fp>(native, Box::new(18)).limbs,
            value_of::<C, B, Bls12381Fp>(native, Box::new(-18)).limbs,
        );
        self.eval_polynomial(native, true, vec![coeff0, coeff1, coeff2], x)
    }
    pub fn isogeny<C: Config, B: RootAPI<C>>(&mut self, native: &mut B, p: &G2AffP) -> G2AffP {
        let mut p = G2AffP {
            x: p.x.clone(),
            y: p.y.clone(),
        };
        let den1 = self.isogeny_y_denominator(native, &p.x);
        let den0 = self.isogeny_x_denominator(native, &p.x);
        p.y = self.isogeny_y_numerator(native, &p.x, &p.y);
        p.x = self.isogeny_x_numerator(native, &p.x);

        let den0 = self.ext2.inverse(native, &den0);
        let den1 = self.ext2.inverse(native, &den1);

        p.x = self.ext2.mul(native, &p.x, &den0);
        p.y = self.ext2.mul(native, &p.y, &den1);
        p
    }
    pub fn hash_to_fp<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        data: &[Variable],
    ) -> (GE2, GE2) {
        let u = self.ext2.curve_f.hash_to_fp(native, data, 2 * 2);
        (
            GE2::from_vars(u[0].clone().limbs, u[1].clone().limbs),
            GE2::from_vars(u[2].clone().limbs, u[3].clone().limbs),
        )
    }
    pub fn map_to_g2<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        in0: &GE2,
        in1: &GE2,
    ) -> G2AffP {
        let out0 = self.map_to_curve2(native, in0);
        let out1 = self.map_to_curve2(native, in1);
        let out = self.add(native, &out0, &out1);
        let new_out = self.isogeny(native, &out);
        self.clear_cofactor(native, &new_out)
    }

    pub fn uncompressed<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        bytes: &[Variable],
    ) -> G2AffP {
        let mut buf_x = bytes.to_vec();
        let buf0 = native.to_binary(buf_x[0], 8);
        let pad = vec![native.constant(0); 5];
        let m_data = from_binary(native, [pad, buf0[5..].to_vec()].concat()); //buf0 & mMask
        let buf0_and_non_mask = from_binary(native, buf0[..5].to_vec()); //buf0 & ^mMask
        buf_x[0] = buf0_and_non_mask;

        //get p.x
        let rev_buf = buf_x.iter().rev().cloned().collect::<Vec<_>>();
        let px = GE2::from_vars(rev_buf[0..48].to_vec(), rev_buf[48..].to_vec());

        //get YSquared
        let ysquared = self.ext2.square(native, &px);
        let ysquared = self.ext2.mul(native, &ysquared, &px);
        let b_curve_coeff = value_of::<C, B, Bls12381Fp>(native, Box::new(4));
        let b_twist_curve_coeff =
            GE2::from_vars(b_curve_coeff.clone().limbs, b_curve_coeff.clone().limbs);
        let ysquared = self.ext2.add(native, &ysquared, &b_twist_curve_coeff);

        let inputs = vec![ysquared.a0.clone(), ysquared.a1.clone()];
        let outputs = self
            .ext2
            .curve_f
            .new_hint(native, "myhint.gete2sqrthint", 3, inputs);

        //is_square should be one
        let is_square = outputs[0].clone();
        let one = self.ext2.curve_f.one_const.clone();
        self.ext2.curve_f.assert_is_equal(native, &is_square, &one);

        //get Y
        let y = GE2::from_vars(outputs[1].clone().limbs, outputs[2].clone().limbs);
        //y^2 = ysquared
        let y_squared = self.ext2.square(native, &y);
        self.ext2.assert_isequal(native, &y_squared, &ysquared);

        //if y is lexicographically largest
        let half_fp = BigInt::from_str("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787").unwrap() / 2;
        let half_fp_var = value_of::<C, B, Bls12381Fp>(native, Box::new(half_fp));
        let is_large_a1 = big_less_than(
            native,
            Bls12381Fp::bits_per_limb() as usize,
            Bls12381Fp::nb_limbs() as usize,
            &half_fp_var.limbs,
            &y.a1.limbs,
        );
        let is_zero_a1 = self.ext2.curve_f.is_zero(native, &y.a1);
        let is_large_a0 = big_less_than(
            native,
            Bls12381Fp::bits_per_limb() as usize,
            Bls12381Fp::nb_limbs() as usize,
            &half_fp_var.limbs,
            &y.a0.limbs,
        );
        let is_large = simple_select(native, is_zero_a1, is_large_a0, is_large_a1);

        //if Y > -Y --> check if mData == mCompressedSmallest
        //if Y <= -Y --> check if mData == mCompressedLargest
        let m_compressed_largest = native.constant(M_COMPRESSED_LARGEST as u32);
        let m_compressed_smallest = native.constant(M_COMPRESSED_SMALLEST as u32);
        let check_m_data = simple_select(
            native,
            is_large,
            m_compressed_smallest,
            m_compressed_largest,
        );

        let check_res = native.sub(m_data, check_m_data);
        let neg_flag = native.is_zero(check_res);

        let neg_y = self.ext2.neg(native, &y);

        let y = self.ext2.select(native, neg_flag, &neg_y, &y);

        //TBD: subgroup check, do we need to do that? Since we are pretty sure that the sig bytes are correct, its unmashalling must be on the right curve?
        G2AffP { x: px, y }
    }

    pub fn scalar_mul_by_seed<C: Config, B: RootAPI<C>>(
        &mut self,
        native: &mut B,
        q: &G2AffP,
    ) -> G2AffP {
        let mut z = self.triple(native, q);
        z = self.double(native, &z);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 2);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 8);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 31);
        z = self.double_and_add(native, &z, q);
        z = self.double_n(native, &z, 16);
        self.neg(native, &z)
    }
}

impl Debug for G2Affine {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("G2Affine")
            .field("P", &self.p)
            .field("lines", &self.lines)
            .finish()
    }
}

declare_circuit!(G2UncompressCircuit {
    x: [Variable; 96],
    y: [[[Variable; 48]; 2]; 2],
});

impl Define<M31Config> for G2UncompressCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut g2 = G2::new(builder);
        let g2_res = g2.uncompressed(builder, &self.x);
        let expected_g2 = G2AffP::from_vars(
            self.y[0][0].to_vec(),
            self.y[0][1].to_vec(),
            self.y[1][0].to_vec(),
            self.y[1][1].to_vec(),
        );
        g2.ext2.assert_isequal(builder, &g2_res.x, &expected_g2.x);
        g2.ext2.assert_isequal(builder, &g2_res.y, &expected_g2.y);
        g2.ext2.curve_f.check_mul(builder);
        g2.ext2.curve_f.table.final_check(builder);
        g2.ext2.curve_f.table.final_check(builder);
        g2.ext2.curve_f.table.final_check(builder);
    }
}

declare_circuit!(MapToG2Circuit {
    in0: [[Variable; 48]; 2],
    in1: [[Variable; 48]; 2],
    out: [[[Variable; 48]; 2]; 2],
});

impl Define<M31Config> for MapToG2Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut g2 = G2::new(builder);
        let in0 = GE2::from_vars(self.in0[0].to_vec(), self.in0[1].to_vec());
        let in1 = GE2::from_vars(self.in1[0].to_vec(), self.in1[1].to_vec());
        let res = g2.map_to_g2(builder, &in0, &in1);
        let target_out = G2AffP {
            x: GE2::from_vars(self.out[0][0].to_vec(), self.out[0][1].to_vec()),
            y: GE2::from_vars(self.out[1][0].to_vec(), self.out[1][1].to_vec()),
        };
        g2.assert_is_equal(builder, &res, &target_out);
        g2.ext2.curve_f.check_mul(builder);
        g2.ext2.curve_f.table.final_check(builder);
        g2.ext2.curve_f.table.final_check(builder);
        g2.ext2.curve_f.table.final_check(builder);
    }
}

declare_circuit!(HashToG2Circuit {
    msg: [Variable; 32],
    out: [[[Variable; 48]; 2]; 2],
});

impl Define<M31Config> for HashToG2Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut g2 = G2::new(builder);
        let (hm0, hm1) = g2.hash_to_fp(builder, &self.msg);
        let res = g2.map_to_g2(builder, &hm0, &hm1);
        let target_out = G2AffP {
            x: GE2::from_vars(self.out[0][0].to_vec(), self.out[0][1].to_vec()),
            y: GE2::from_vars(self.out[1][0].to_vec(), self.out[1][1].to_vec()),
        };
        g2.assert_is_equal(builder, &res, &target_out);
        g2.ext2.curve_f.check_mul(builder);
        g2.ext2.curve_f.table.final_check(builder);
        g2.ext2.curve_f.table.final_check(builder);
        g2.ext2.curve_f.table.final_check(builder);
    }
}

#[cfg(test)]
mod tests {
    use super::G2UncompressCircuit;
    use crate::gnark::emulated::sw_bls12381::g2::*;
    use crate::utils::register_hint;
    use expander_compiler::frontend::*;
    use expander_compiler::frontend::{HintRegistry, M31};
    use extra::debug_eval;
    use num_bigint::BigInt;
    use num_traits::Num;

    #[test]
    fn test_map_to_g2() {
        let mut hint_registry = HintRegistry::<M31>::new();
        register_hint(&mut hint_registry);
        let mut assignment = MapToG2Circuit::<M31> {
            in0: [[M31::from(0); 48]; 2],
            in1: [[M31::from(0); 48]; 2],
            out: [[[M31::from(0); 48]; 2]; 2],
        };
        let p1_x_bytes = [
            75, 240, 55, 239, 72, 231, 76, 188, 20, 26, 234, 236, 23, 166, 182, 159, 239, 165, 10,
            98, 220, 117, 40, 167, 160, 143, 63, 57, 113, 82, 97, 238, 36, 48, 226, 19, 210, 13,
            216, 163, 51, 199, 31, 228, 211, 18, 125, 25,
        ];
        let p1_y_bytes = [
            161, 161, 201, 159, 90, 241, 214, 89, 177, 71, 235, 130, 168, 37, 237, 255, 26, 105,
            22, 122, 136, 28, 83, 245, 117, 135, 212, 63, 208, 241, 109, 4, 109, 188, 74, 50, 63,
            41, 78, 174, 164, 121, 104, 77, 56, 23, 100, 5,
        ];
        let p2_x_bytes = [
            161, 152, 122, 79, 206, 47, 160, 114, 196, 82, 17, 183, 227, 115, 71, 7, 9, 141, 33,
            224, 127, 254, 158, 109, 69, 225, 184, 146, 239, 137, 146, 138, 224, 79, 56, 100, 184,
            236, 99, 77, 28, 117, 111, 179, 106, 181, 35, 21,
        ];
        let p2_y_bytes = [
            199, 231, 196, 205, 165, 5, 112, 203, 238, 82, 8, 79, 245, 151, 226, 80, 154, 146, 230,
            51, 79, 60, 20, 190, 9, 171, 34, 41, 131, 165, 60, 0, 10, 197, 177, 140, 108, 41, 99,
            113, 151, 51, 253, 219, 105, 227, 25, 24,
        ];
        let out0_x_bytes = [
            215, 186, 167, 113, 176, 255, 84, 123, 163, 0, 104, 202, 139, 197, 29, 119, 253, 35,
            206, 68, 130, 75, 218, 109, 179, 63, 65, 197, 67, 206, 64, 89, 30, 201, 95, 238, 5, 66,
            143, 94, 37, 238, 150, 113, 159, 165, 110, 3,
        ];
        let out0_y_bytes = [
            88, 110, 24, 185, 208, 195, 142, 173, 176, 12, 228, 155, 64, 223, 147, 25, 37, 234,
            200, 3, 123, 119, 193, 221, 234, 253, 199, 190, 120, 135, 32, 215, 32, 118, 55, 230,
            74, 204, 56, 12, 24, 221, 240, 188, 188, 76, 233, 20,
        ];
        let out1_x_bytes = [
            202, 105, 74, 230, 255, 158, 238, 160, 121, 234, 219, 154, 239, 176, 232, 81, 56, 53,
            154, 76, 221, 53, 156, 165, 215, 18, 148, 34, 124, 242, 154, 218, 243, 171, 88, 53, 13,
            182, 39, 84, 254, 161, 96, 192, 154, 242, 71, 15,
        ];
        let out1_y_bytes = [
            66, 124, 60, 101, 29, 246, 150, 109, 233, 119, 212, 23, 132, 79, 170, 0, 178, 98, 151,
            189, 214, 70, 171, 93, 2, 98, 194, 243, 38, 160, 178, 224, 91, 20, 11, 209, 190, 76,
            182, 253, 89, 144, 170, 191, 128, 66, 207, 1,
        ];

        for i in 0..48 {
            assignment.in0[0][i] = M31::from(p1_x_bytes[i]);
            assignment.in0[1][i] = M31::from(p1_y_bytes[i]);
            assignment.in1[0][i] = M31::from(p2_x_bytes[i]);
            assignment.in1[1][i] = M31::from(p2_y_bytes[i]);
            assignment.out[0][0][i] = M31::from(out0_x_bytes[i]);
            assignment.out[0][1][i] = M31::from(out0_y_bytes[i]);
            assignment.out[1][0][i] = M31::from(out1_x_bytes[i]);
            assignment.out[1][1][i] = M31::from(out1_y_bytes[i]);
        }

        debug_eval(&MapToG2Circuit::default(), &assignment, hint_registry);
    }

    #[test]
    fn test_hash_to_g2() {
        // compile_generic(&HashToG2Circuit::default(), CompileOptions::default()).unwrap();
        let mut hint_registry = HintRegistry::<M31>::new();
        register_hint(&mut hint_registry);
        let mut assignment = HashToG2Circuit::<M31> {
            msg: [M31::from(0); 32],
            out: [[[M31::from(0); 48]; 2]; 2],
        };
        let msg_bytes = [
            140, 148, 79, 140, 170, 85, 208, 7, 114, 138, 47, 198, 231, 255, 48, 104, 221, 225, 3,
            237, 99, 251, 57, 156, 89, 194, 79, 31, 130, 109, 228, 200,
        ];
        let out0_x_bytes = [
            215, 186, 167, 113, 176, 255, 84, 123, 163, 0, 104, 202, 139, 197, 29, 119, 253, 35,
            206, 68, 130, 75, 218, 109, 179, 63, 65, 197, 67, 206, 64, 89, 30, 201, 95, 238, 5, 66,
            143, 94, 37, 238, 150, 113, 159, 165, 110, 3,
        ];
        let out0_y_bytes = [
            88, 110, 24, 185, 208, 195, 142, 173, 176, 12, 228, 155, 64, 223, 147, 25, 37, 234,
            200, 3, 123, 119, 193, 221, 234, 253, 199, 190, 120, 135, 32, 215, 32, 118, 55, 230,
            74, 204, 56, 12, 24, 221, 240, 188, 188, 76, 233, 20,
        ];
        let out1_x_bytes = [
            202, 105, 74, 230, 255, 158, 238, 160, 121, 234, 219, 154, 239, 176, 232, 81, 56, 53,
            154, 76, 221, 53, 156, 165, 215, 18, 148, 34, 124, 242, 154, 218, 243, 171, 88, 53, 13,
            182, 39, 84, 254, 161, 96, 192, 154, 242, 71, 15,
        ];
        let out1_y_bytes = [
            66, 124, 60, 101, 29, 246, 150, 109, 233, 119, 212, 23, 132, 79, 170, 0, 178, 98, 151,
            189, 214, 70, 171, 93, 2, 98, 194, 243, 38, 160, 178, 224, 91, 20, 11, 209, 190, 76,
            182, 253, 89, 144, 170, 191, 128, 66, 207, 1,
        ];
        for i in 0..32 {
            assignment.msg[i] = M31::from(msg_bytes[i]);
        }
        for i in 0..48 {
            assignment.out[0][0][i] = M31::from(out0_x_bytes[i]);
            assignment.out[0][1][i] = M31::from(out0_y_bytes[i]);
            assignment.out[1][0][i] = M31::from(out1_x_bytes[i]);
            assignment.out[1][1][i] = M31::from(out1_y_bytes[i]);
        }

        debug_eval(&HashToG2Circuit::default(), &assignment, hint_registry);
    }

    #[test]
    fn test_uncompress_g2() {
        // compile_generic(&G2UncompressCircuit::default(), CompileOptions::default()).unwrap();
        let mut hint_registry = HintRegistry::<M31>::new();
        register_hint(&mut hint_registry);
        let mut assignment = G2UncompressCircuit::<M31> {
            x: [M31::default(); 96],
            y: [[[M31::default(); 48]; 2]; 2],
        };
        let x_bigint = BigInt::from_str_radix("aa79bf02bb1633716de959b5ed8ccf7548e6733d7ca11791f1f5d386afb6cebc7cf0339a791bd9187e5346185ace329402b641d106d783e7fe20e5c1cf5b3416590ad45004a0b396f66178511ce724c3df76c2fae61fb682a3ec2dde1ae5a359", 16).unwrap();

        let x_bytes = x_bigint.to_bytes_be();

        let y_b0_a0_bigint = BigInt::from_str_radix("417406042303837766676050444382954581819710384023930335899613364000243943316124744931107291428889984115562657456985", 10).unwrap();
        let y_b0_a1_bigint = BigInt::from_str_radix("1612337918776384379710682981548399375489832112491603419994252758241488024847803823620674751718035900645102653944468", 10).unwrap();
        let y_b1_a0_bigint = BigInt::from_str_radix("2138372746384454686692156684769748785619173944336480358459807585988147682623523096063056865298570471165754367761702", 10).unwrap();
        let y_b1_a1_bigint = BigInt::from_str_radix("2515621099638397509480666850964364949449167540660259026336903510150090825582288208580180650995842554224706524936338", 10).unwrap();

        let y_a0_bytes = y_b0_a0_bigint.to_bytes_le();
        let y_a1_bytes = y_b0_a1_bigint.to_bytes_le();
        let y_b0_bytes = y_b1_a0_bigint.to_bytes_le();
        let y_b1_bytes = y_b1_a1_bigint.to_bytes_le();

        for i in 0..48 {
            assignment.x[i] = M31::from(x_bytes.1[i] as u32);
            assignment.x[i + 48] = M31::from(x_bytes.1[i + 48] as u32);
            assignment.y[0][0][i] = M31::from(y_a0_bytes.1[i] as u32);
            assignment.y[0][1][i] = M31::from(y_a1_bytes.1[i] as u32);
            assignment.y[1][0][i] = M31::from(y_b0_bytes.1[i] as u32);
            assignment.y[1][1][i] = M31::from(y_b1_bytes.1[i] as u32);
        }

        debug_eval(&G2UncompressCircuit::default(), &assignment, hint_registry);
    }
}
