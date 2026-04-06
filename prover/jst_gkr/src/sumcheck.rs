use arith::Field;
use jst_gkr_engine::FiatShamirTranscript;

pub struct SumcheckProof<F: Field> {
    pub round_polys: Vec<[F; 3]>,
}

pub fn prove_sumcheck<F: Field, T: FiatShamirTranscript>(
    bk_f: &mut Vec<F>,
    bk_hg: &mut Vec<F>,
    num_vars: usize,
    transcript: &mut T,
) -> (SumcheckProof<F>, Vec<F>) {
    let mut round_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        let eval_size = 1 << (num_vars - 1 - round);
        let mut p0 = F::ZERO;
        let mut p1 = F::ZERO;
        let mut p2 = F::ZERO;

        for i in 0..eval_size {
            let f_0 = bk_f[2 * i];
            let f_1 = bk_f[2 * i + 1];
            let hg_0 = bk_hg[2 * i];
            let hg_1 = bk_hg[2 * i + 1];

            p0 += hg_0 * f_0;
            p1 += hg_1 * f_1;

            let f_2 = f_1 + f_1 - f_0;
            let hg_2 = hg_1 + hg_1 - hg_0;
            p2 += hg_2 * f_2;
        }

        let round_poly = [p0, p1, p2];

        transcript.append_field_element(&round_poly[0]);
        transcript.append_field_element(&round_poly[1]);
        transcript.append_field_element(&round_poly[2]);

        let r: F = transcript.challenge_field_element();
        challenges.push(r);
        round_polys.push(round_poly);

        for i in 0..eval_size {
            bk_f[i] = bk_f[2 * i] * (F::ONE - r) + bk_f[2 * i + 1] * r;
            bk_hg[i] = bk_hg[2 * i] * (F::ONE - r) + bk_hg[2 * i + 1] * r;
        }
    }

    (SumcheckProof { round_polys }, challenges)
}

pub fn verify_sumcheck<F: Field, T: FiatShamirTranscript>(
    claimed_sum: F,
    proof: &SumcheckProof<F>,
    num_vars: usize,
    transcript: &mut T,
) -> Option<(F, Vec<F>)> {
    let mut current_sum = claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        let poly = &proof.round_polys[round];
        if poly[0] + poly[1] != current_sum {
            return None;
        }

        transcript.append_field_element(&poly[0]);
        transcript.append_field_element(&poly[1]);
        transcript.append_field_element(&poly[2]);

        let r: F = transcript.challenge_field_element();
        challenges.push(r);

        current_sum = interpolate_quadratic(poly, r);
    }

    Some((current_sum, challenges))
}

fn interpolate_quadratic<F: Field>(evals: &[F; 3], r: F) -> F {
    let e0 = evals[0];
    let e1 = evals[1];
    let e2 = evals[2];
    let two = F::from(2u32);
    let a = (e2 - two * e1 + e0) * F::INV_2;
    let b = (F::from(4u32) * e1 - F::from(3u32) * e0 - e2) * F::INV_2;
    let c = e0;
    a * r * r + b * r + c
}
