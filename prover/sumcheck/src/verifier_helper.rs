use arith::{ExtensionField, Field};
use circuit::{CircuitLayer, CoefType, GateAdd, GateConst, GateMul, GateUni};
use gkr_engine::{ExpanderDualVarChallenge, FieldEngine, FieldType};
use polynomials::EqPolynomial;

use crate::{scratch_pad::VerifierScratchPad, unpack_and_combine};

#[derive(Default)]
pub struct GKRVerifierHelper<F: FieldEngine> {
    phantom: std::marker::PhantomData<F>,
}

impl<F: FieldEngine> GKRVerifierHelper<F> {
    // This assumes we're verifying GKR layer by layer sequentially, and can reuse some of the
    // results from the previous layer, which is stored in the scratch pad.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn prepare_layer(
        layer: &CircuitLayer<F>,
        alpha: &Option<F::ChallengeField>,
        challenge: &ExpanderDualVarChallenge<F>,
        sp: &mut VerifierScratchPad<F>,
        is_output_layer: bool,
    ) {
        assert_eq!(alpha.is_none(), challenge.rz_1.is_none());

        if is_output_layer {
            EqPolynomial::<F::ChallengeField>::eq_eval_at(
                &challenge.rz_0,
                &F::ChallengeField::ONE,
                &mut sp.eq_evals_at_rz0,
                &mut sp.eq_evals_first_part,
                &mut sp.eq_evals_second_part,
            );
        } else {
            // use results from previous layer
            let output_len = 1 << challenge.rz_0.len();
            sp.eq_evals_at_rz0[..output_len].copy_from_slice(&sp.eq_evals_at_rx[..output_len]);
            if alpha.is_some() && challenge.rz_1.is_some() {
                let alpha = alpha.unwrap();
                for i in 0..(1usize << layer.output_var_num) {
                    sp.eq_evals_at_rz0[i] += alpha * sp.eq_evals_at_ry[i];
                }
            }
        }

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &challenge.r_simd,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_r_simd,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &challenge.r_mpi,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_r_mpi,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );

        sp.r_simd = challenge.r_simd.clone();
        sp.r_mpi = challenge.r_mpi.clone();
    }

    // This is for the case where we are verifying GKR layers in parallel, and we need to
    // recompute the evaluations for each layer independently.
    // This is less efficient than the sequential version, but allows for more parallelism.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn prepare_layer_non_sequential(
        layer: &CircuitLayer<F>,
        alpha: &Option<F::ChallengeField>,
        challenge: &ExpanderDualVarChallenge<F>,
        sp: &mut VerifierScratchPad<F>,
    ) {
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &challenge.rz_0,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_rz0,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );
        if alpha.is_some() && challenge.rz_1.is_some() {
            let alpha = alpha.unwrap();
            EqPolynomial::<F::ChallengeField>::eq_eval_at(
                challenge.rz_1.as_ref().unwrap(),
                &F::ChallengeField::ONE,
                &mut sp.eq_evals_at_rx,
                &mut sp.eq_evals_first_part,
                &mut sp.eq_evals_second_part,
            );
            for i in 0..(1usize << layer.output_var_num) {
                sp.eq_evals_at_rz0[i] += alpha * sp.eq_evals_at_rx[i];
            }
        }

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &challenge.r_simd,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_r_simd,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );

        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            &challenge.r_mpi,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_r_mpi,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );

        sp.r_simd = challenge.r_simd.clone();
        sp.r_mpi = challenge.r_mpi.clone();
    }

    #[inline(always)]
    pub fn eval_cst(
        cst_gates: &[GateConst<F>],
        public_input: &[F::SimdCircuitField],
        sp: &VerifierScratchPad<F>,
    ) -> F::ChallengeField {
        let mut v = F::ChallengeField::zero();

        let mpi_world_size = sp.eq_evals_at_r_mpi.len();
        let local_input_size = public_input.len() / mpi_world_size;

        for cst_gate in cst_gates {
            let tmp = match cst_gate.coef_type {
                CoefType::PublicInput(input_idx) => {
                    let mut input = vec![];
                    for i in 0..mpi_world_size {
                        input.push(public_input[i * local_input_size + input_idx]);
                    }

                    // mpi combined
                    let input_mpi_combined: F::Field = input
                        .iter()
                        .zip(&sp.eq_evals_at_r_mpi)
                        .map(|(v, c)| *c * *v)
                        .sum();

                    // simd combined
                    sp.eq_evals_at_rz0[cst_gate.o_id]
                        * unpack_and_combine::<F::Field>(
                            &input_mpi_combined,
                            &sp.eq_evals_at_r_simd,
                        )
                }
                _ => sp.eq_evals_at_rz0[cst_gate.o_id] * cst_gate.coef,
            };
            v += tmp;
        }

        v
    }

    #[inline(always)]
    pub fn eval_add(add_gates: &[GateAdd<F>], sp: &VerifierScratchPad<F>) -> F::ChallengeField {
        let mut v = F::ChallengeField::zero();
        for add_gate in add_gates {
            v += sp.eq_evals_at_rz0[add_gate.o_id]
                * sp.eq_evals_at_rx[add_gate.i_ids[0]]
                * add_gate.coef;
        }
        v * sp.eq_r_simd_r_simd_xy * sp.eq_r_mpi_r_mpi_xy
    }

    #[inline(always)]
    pub fn eval_mul(mul_gates: &[GateMul<F>], sp: &VerifierScratchPad<F>) -> F::ChallengeField {
        let mut v = F::ChallengeField::zero();
        for mul_gate in mul_gates {
            let tmp = sp.eq_evals_at_rx[mul_gate.i_ids[0]]
                * sp.eq_evals_at_ry[mul_gate.i_ids[1]]
                * mul_gate.coef;
            v += sp.eq_evals_at_rz0[mul_gate.o_id] * tmp;
        }
        v * sp.eq_r_simd_r_simd_xy * sp.eq_r_mpi_r_mpi_xy
    }

    /// GKR2 equivalent of `eval_add`. (Note that GKR2 uses pow1 gates instead of add gates)
    #[inline(always)]
    pub fn eval_pow_1(gates: &[GateUni<F>], sp: &VerifierScratchPad<F>) -> F::ChallengeField {
        let mut v = F::ChallengeField::zero();
        for gate in gates {
            // Gates of type 12346 represent an add gate
            if gate.gate_type == 12346 {
                v += sp.eq_evals_at_rz0[gate.o_id] * sp.eq_evals_at_rx[gate.i_ids[0]] * gate.coef;
            }
        }
        v * sp.eq_r_simd_r_simd_xy * sp.eq_r_mpi_r_mpi_xy
    }

    #[inline(always)]
    pub fn eval_pow_5(gates: &[GateUni<F>], sp: &VerifierScratchPad<F>) -> F::ChallengeField {
        let mut v = F::ChallengeField::zero();
        for gate in gates {
            // Gates of type 12345 represent a pow5 gate
            if gate.gate_type == 12345 {
                v += sp.eq_evals_at_rz0[gate.o_id] * sp.eq_evals_at_rx[gate.i_ids[0]] * gate.coef;
            }
        }
        v * sp.eq_r_simd_r_simd_xy * sp.eq_r_mpi_r_mpi_xy
    }

    #[inline(always)]
    pub fn set_rx(rx: &[F::ChallengeField], sp: &mut VerifierScratchPad<F>) {
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            rx,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_rx,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );
    }

    #[inline(always)]
    pub fn set_r_simd_xy(r_simd_xy: &[F::ChallengeField], sp: &mut VerifierScratchPad<F>) {
        sp.eq_r_simd_r_simd_xy = EqPolynomial::<F::ChallengeField>::eq_vec(&sp.r_simd, r_simd_xy);
    }

    #[inline(always)]
    pub fn set_r_mpi_xy(r_mpi_xy: &[F::ChallengeField], sp: &mut VerifierScratchPad<F>) {
        sp.eq_r_mpi_r_mpi_xy = EqPolynomial::<F::ChallengeField>::eq_vec(&sp.r_mpi, r_mpi_xy);
    }

    #[inline(always)]
    pub fn set_ry(ry: &[F::ChallengeField], sp: &mut VerifierScratchPad<F>) {
        EqPolynomial::<F::ChallengeField>::eq_eval_at(
            ry,
            &F::ChallengeField::ONE,
            &mut sp.eq_evals_at_ry,
            &mut sp.eq_evals_first_part,
            &mut sp.eq_evals_second_part,
        );
    }

    #[inline(always)]
    pub fn degree_2_eval(
        ps: &[F::ChallengeField],
        x: F::ChallengeField,
        sp: &VerifierScratchPad<F>,
    ) -> F::ChallengeField {
        assert_eq!(ps.len(), 3);
        let p0 = ps[0];
        let p1 = ps[1];
        let p2 = ps[2];

        if F::FIELD_TYPE == FieldType::GF2Ext128 {
            let tmp = p0 - p1;

            let c0 = p0;
            let c2 = (p2 - p0 + tmp.mul_by_x()) * sp.gf2_deg2_eval_coef;
            let c1 = tmp - c2;
            c0 + (c2 * x + c1) * x
        } else {
            let c0 = p0;
            let c2 = F::ChallengeField::INV_2 * (p2 - p1 - p1 + p0);
            let c1 = p1 - p0 - c2;
            c0 + (c2 * x + c1) * x
        }
    }

    #[inline(always)]
    pub fn degree_3_eval(
        vals: &[F::ChallengeField],
        x: F::ChallengeField,
        sp: &VerifierScratchPad<F>,
    ) -> F::ChallengeField {
        Self::lag_eval(vals, x, sp)
    }

    #[inline(always)]
    pub fn degree_6_eval(
        vals: &[F::ChallengeField],
        x: F::ChallengeField,
        sp: &VerifierScratchPad<F>,
    ) -> F::ChallengeField {
        Self::lag_eval(vals, x, sp)
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    fn lag_eval(
        vals: &[F::ChallengeField],
        x: F::ChallengeField,
        sp: &VerifierScratchPad<F>,
    ) -> F::ChallengeField {
        let (evals, lag_denoms_inv) = match vals.len() {
            4 => (sp.deg3_eval_at.to_vec(), sp.deg3_lag_denoms_inv.to_vec()),
            7 => (sp.deg6_eval_at.to_vec(), sp.deg6_lag_denoms_inv.to_vec()),
            _ => panic!("unsupported degree"),
        };

        let mut v = F::ChallengeField::ZERO;
        for i in 0..vals.len() {
            let mut numerator = F::ChallengeField::ONE;
            for j in 0..vals.len() {
                if j == i {
                    continue;
                }
                numerator *= x - evals[j];
            }
            v += numerator * lag_denoms_inv[i] * vals[i];
        }
        v
    }
}
