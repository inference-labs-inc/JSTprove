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
    let depth = circuit.depth();

    let output_vals = &layer_vals[depth];

    let output_var_num = circuit.layers[0].output_var_num;
    let output_r: Vec<F> = (0..output_var_num)
        .map(|_| transcript.challenge_field_element())
        .collect();

    let claimed_v = evaluate_mle(output_vals, &output_r);
    transcript.append_field_element(&claimed_v);

    let mut r_x = output_r;
    let mut claimed_val = claimed_v;

    for layer_idx in 0..depth {
        let layer = &circuit.layers[layer_idx];
        let next_vals = &layer_vals[depth - 1 - layer_idx];
        let input_var_num = layer.input_var_num;
        let input_size = 1 << input_var_num;

        let eq_table = build_eq_table(&r_x);

        let mut hg_add = vec![F::ZERO; input_size];
        let mut hg_mul = vec![F::ZERO; input_size];

        for g in &layer.add_gates {
            hg_add[g.i_id] += eq_table[g.o_id] * g.coef;
        }

        for g in &layer.mul_gates {
            hg_mul[g.i_ids[0]] += eq_table[g.o_id] * g.coef * next_vals[g.i_ids[1]];
            hg_mul[g.i_ids[1]] += eq_table[g.o_id] * g.coef * next_vals[g.i_ids[0]];
        }

        let mut bk_hg: Vec<F> = (0..input_size).map(|i| hg_add[i] + hg_mul[i]).collect();
        let mut bk_f: Vec<F> = next_vals[..input_size].to_vec();

        let mut const_contribution = F::ZERO;
        for g in &layer.const_gates {
            const_contribution += eq_table[g.o_id] * g.coef;
        }

        transcript.append_field_element(&claimed_val);

        let (sc_proof, challenges) =
            prove_sumcheck(&mut bk_f, &mut bk_hg, input_var_num, transcript);

        let _ = sc_proof;

        let r_y = challenges;

        let vy = evaluate_mle(&next_vals[..input_size], &r_y);

        transcript.append_field_element(&vy);

        r_x = r_y;
        claimed_val = vy;
    }

    transcript.finalize_proof()
}
