use arith::Field;
use jst_gkr_engine::{FiatShamirTranscript, LayeredCircuit, Proof};

use crate::mle::{build_eq_table, evaluate_mle};
use crate::sumcheck::{verify_sumcheck, SumcheckProof};

pub fn verify<F: Field, T: FiatShamirTranscript>(
    circuit: &LayeredCircuit<F>,
    witness: &[F],
    proof: &Proof,
    transcript: &mut T,
) -> bool {
    let depth = circuit.depth();

    let output_var_num = circuit.layers[0].output_var_num;
    let output_r: Vec<F> = (0..output_var_num)
        .map(|_| transcript.challenge_field_element())
        .collect();

    let mut offset = 0usize;

    let Some(claimed_v) = read_field::<F>(&proof.data, &mut offset) else {
        return false;
    };
    transcript.append_field_element(&claimed_v);

    let mut rz_0 = output_r;
    let mut rz_1: Option<Vec<F>> = None;
    let mut alpha: Option<F> = None;
    let mut vx_claim = claimed_v;
    let mut vy_claim: Option<F> = None;

    for layer_idx in 0..depth {
        let layer = &circuit.layers[layer_idx];
        let input_var_num = layer.input_var_num;

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

        let Some(sc1) = read_sumcheck_proof::<F>(&proof.data, &mut offset, input_var_num) else {
            return false;
        };
        let Some((final_val_1, challenges_x)) =
            verify_sumcheck(phase1_sum, &sc1, input_var_num, transcript)
        else {
            return false;
        };

        let Some(vx) = read_field::<F>(&proof.data, &mut offset) else {
            return false;
        };
        transcript.append_field_element(&vx);

        let has_mul = !layer.mul_gates.is_empty();
        if has_mul {
            let eq_rx = build_eq_table(&challenges_x);

            let mut add_at_rx = F::ZERO;
            for g in &layer.add_gates {
                add_at_rx += eq_combined[g.o_id] * g.coef * eq_rx[g.i_id];
            }

            let phase2_sum = final_val_1 - add_at_rx * vx;

            let Some(sc2) = read_sumcheck_proof::<F>(&proof.data, &mut offset, input_var_num)
            else {
                return false;
            };
            let Some((final_val_2, challenges_y)) =
                verify_sumcheck(phase2_sum, &sc2, input_var_num, transcript)
            else {
                return false;
            };

            let Some(vy) = read_field::<F>(&proof.data, &mut offset) else {
                return false;
            };
            transcript.append_field_element(&vy);

            let eq_ry = build_eq_table(&challenges_y);
            let mut weight_at_ry = F::ZERO;
            for g in &layer.mul_gates {
                weight_at_ry +=
                    eq_combined[g.o_id] * g.coef * eq_rx[g.i_ids[0]] * eq_ry[g.i_ids[1]];
            }
            if final_val_2 != weight_at_ry * vx * vy {
                return false;
            }

            let new_alpha: F = transcript.challenge_field_element();

            rz_0 = challenges_x;
            rz_1 = Some(challenges_y);
            alpha = Some(new_alpha);
            vx_claim = vx;
            vy_claim = Some(vy);
        } else {
            let eq_rx = build_eq_table(&challenges_x);
            let mut add_at_rx = F::ZERO;
            for g in &layer.add_gates {
                add_at_rx += eq_combined[g.o_id] * g.coef * eq_rx[g.i_id];
            }
            if final_val_1 != add_at_rx * vx {
                return false;
            }

            rz_0 = challenges_x;
            rz_1 = None;
            alpha = None;
            vx_claim = vx;
            vy_claim = None;
        }
    }

    let input_var_num = circuit.layers[depth - 1].input_var_num;
    let input_size = 1 << input_var_num;
    if witness.len() < input_size {
        return false;
    }

    let expected_vx = evaluate_mle(&witness[..input_size], &rz_0);
    if expected_vx != vx_claim {
        return false;
    }

    if let (Some(ref rz1), Some(vy)) = (&rz_1, vy_claim) {
        let expected_vy = evaluate_mle(&witness[..input_size], rz1);
        if expected_vy != vy {
            return false;
        }
    }

    true
}

fn read_field<F: Field>(data: &[u8], offset: &mut usize) -> Option<F> {
    if data.len() < *offset + F::SIZE {
        return None;
    }
    let mut element = F::ZERO;
    element.set_in_bytes(&data[*offset..*offset + F::SIZE]);
    *offset += F::SIZE;
    Some(element)
}

fn read_sumcheck_proof<F: Field>(
    data: &[u8],
    offset: &mut usize,
    num_vars: usize,
) -> Option<SumcheckProof<F>> {
    let mut round_polys = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let p0 = read_field::<F>(data, offset)?;
        let p1 = read_field::<F>(data, offset)?;
        let p2 = read_field::<F>(data, offset)?;
        round_polys.push([p0, p1, p2]);
    }
    Some(SumcheckProof { round_polys })
}
