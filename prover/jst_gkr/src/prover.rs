use arith::Field;
use jst_gkr_engine::{FiatShamirTranscript, LayeredCircuit, Proof};

use crate::mle::{build_eq_table, evaluate_mle};
use crate::sumcheck::prove_sumcheck;

pub fn prove<F: Field, T: FiatShamirTranscript>(
    circuit: &LayeredCircuit<F>,
    witness: &[F],
    transcript: &mut T,
) -> Proof {
    let layer_vals = circuit.evaluate(witness);
    prove_with_evaluations(circuit, &layer_vals, transcript)
}

pub fn prove_with_evaluations<F: Field, T: FiatShamirTranscript>(
    circuit: &LayeredCircuit<F>,
    layer_vals: &[Vec<F>],
    transcript: &mut T,
) -> Proof {
    let depth = circuit.depth();

    let output_vals = &layer_vals[depth];

    let output_var_num = circuit.layers[0].output_var_num;
    let output_r: Vec<F> = (0..output_var_num)
        .map(|_| transcript.challenge_field_element())
        .collect();

    let claimed_v = evaluate_mle(output_vals, &output_r);
    transcript.append_field_element(&claimed_v);

    let mut rz_0 = output_r;
    let mut rz_1: Option<Vec<F>> = None;
    let mut alpha: Option<F> = None;
    let mut vx_claim = claimed_v;
    let mut vy_claim: Option<F> = None;

    for layer_idx in 0..depth {
        let layer = &circuit.layers[layer_idx];
        let next_vals = &layer_vals[depth - 1 - layer_idx];
        let input_var_num = layer.input_var_num;
        let input_size = 1 << input_var_num;

        let eq_rz0 = build_eq_table(&rz_0);
        let mut eq_combined = eq_rz0;
        if let (Some(ref rz1), Some(a)) = (&rz_1, alpha) {
            let eq_rz1 = build_eq_table(rz1);
            for i in 0..eq_combined.len() {
                eq_combined[i] += a * eq_rz1[i];
            }
        }

        let mut combined_claim = vx_claim;
        if let (Some(vy), Some(a)) = (vy_claim, alpha) {
            combined_claim += a * vy;
        }

        let mut const_contribution = F::ZERO;
        for g in &layer.const_gates {
            const_contribution += eq_combined[g.o_id] * g.coef;
        }

        let phase1_sum = combined_claim - const_contribution;

        let mut hg_x = vec![F::ZERO; input_size];
        for g in &layer.add_gates {
            hg_x[g.i_id] += eq_combined[g.o_id] * g.coef;
        }
        for g in &layer.mul_gates {
            hg_x[g.i_ids[0]] += eq_combined[g.o_id] * g.coef * next_vals[g.i_ids[1]];
        }

        let mut bk_f = next_vals[..input_size].to_vec();
        let mut bk_hg = hg_x;

        debug_assert_eq!(
            bk_hg
                .iter()
                .zip(bk_f.iter())
                .map(|(h, f)| *h * *f)
                .fold(F::ZERO, |a, b| a + b),
            phase1_sum,
        );

        let (_, challenges_x) = prove_sumcheck(&mut bk_f, &mut bk_hg, input_var_num, transcript);

        let vx = evaluate_mle(&next_vals[..input_size], &challenges_x);
        transcript.append_field_element(&vx);

        let has_mul = !layer.mul_gates.is_empty();
        if has_mul {
            let eq_rx = build_eq_table(&challenges_x);

            let mut hg_y = vec![F::ZERO; input_size];
            for g in &layer.mul_gates {
                hg_y[g.i_ids[1]] += eq_combined[g.o_id] * g.coef * eq_rx[g.i_ids[0]] * vx;
            }

            let mut bk_f_y = next_vals[..input_size].to_vec();
            let (_, challenges_y) =
                prove_sumcheck(&mut bk_f_y, &mut hg_y, input_var_num, transcript);

            let vy = evaluate_mle(&next_vals[..input_size], &challenges_y);
            transcript.append_field_element(&vy);

            let new_alpha: F = transcript.challenge_field_element();

            rz_0 = challenges_x;
            rz_1 = Some(challenges_y);
            alpha = Some(new_alpha);
            vx_claim = vx;
            vy_claim = Some(vy);
        } else {
            rz_0 = challenges_x;
            rz_1 = None;
            alpha = None;
            vx_claim = vx;
            vy_claim = None;
        }
    }

    transcript.finalize_proof()
}
