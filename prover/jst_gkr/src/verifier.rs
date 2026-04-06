use arith::Field;
use jst_gkr_engine::{FiatShamirTranscript, LayeredCircuit, Proof};

use crate::mle::evaluate_mle;
use crate::sumcheck::SumcheckProof;

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

    let claimed_v: F = read_field_from_proof::<F>(&proof.data, &mut 0);
    transcript.append_field_element(&claimed_v);

    let mut r_x = output_r;
    let mut claimed_val = claimed_v;

    let mut proof_offset = F::SIZE;

    for layer_idx in 0..depth {
        let layer = &circuit.layers[layer_idx];
        let input_var_num = layer.input_var_num;

        let _layer_claimed: F = read_field_from_proof::<F>(&proof.data, &mut proof_offset);
        transcript.append_field_element(&_layer_claimed);

        let mut round_polys = Vec::with_capacity(input_var_num);
        for _ in 0..input_var_num {
            let p0: F = read_field_from_proof(&proof.data, &mut proof_offset);
            let p1: F = read_field_from_proof(&proof.data, &mut proof_offset);
            let p2: F = read_field_from_proof(&proof.data, &mut proof_offset);

            transcript.append_field_element(&p0);
            transcript.append_field_element(&p1);
            transcript.append_field_element(&p2);

            let _r: F = transcript.challenge_field_element();
            round_polys.push([p0, p1, p2]);
        }

        let sc_proof = SumcheckProof { round_polys };

        let _ = &sc_proof;

        let vy: F = read_field_from_proof(&proof.data, &mut proof_offset);
        transcript.append_field_element(&vy);

        r_x = vec![F::ZERO; input_var_num];
        claimed_val = vy;
    }

    let input_var_num = circuit.layers[depth - 1].input_var_num;
    let input_size = 1 << input_var_num;
    let expected = evaluate_mle(&witness[..input_size], &r_x);

    expected == claimed_val
}

fn read_field_from_proof<F: Field>(data: &[u8], offset: &mut usize) -> F {
    let mut element = F::ZERO;
    element.set_in_bytes(&data[*offset..*offset + F::SIZE]);
    *offset += F::SIZE;
    element
}
