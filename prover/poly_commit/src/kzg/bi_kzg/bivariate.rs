use halo2curves::{
    ff::Field,
    group::{prime::PrimeCurveAffine, Curve, Group},
    msm,
    pairing::{MillerLoopResult, MultiMillerLoop},
    CurveAffine,
};
use itertools::izip;
use serdes::ExpSerde;

use crate::*;

#[inline(always)]
pub fn generate_coef_form_bi_kzg_local_srs_for_testing<E: MultiMillerLoop>(
    local_length: usize,
    distributed_parties: usize,
    party_rank: usize,
    mut rng: impl rand::RngCore,
) -> CoefFormBiKZGLocalSRS<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    assert!(local_length.is_power_of_two());
    assert!(distributed_parties.is_power_of_two());
    assert!(party_rank < distributed_parties);

    let tau_x = E::Fr::random(&mut rng);
    let tau_y = E::Fr::random(&mut rng);

    let g1 = E::G1Affine::generator();

    let tau_x_geometric_progression = powers_series(&tau_x, local_length);
    let tau_y_geometric_progression = powers_series(&tau_y, distributed_parties);

    let g1_prog = g1.to_curve();
    let x_coeff_bases = {
        let mut proj_bases = vec![g1_prog * tau_y_geometric_progression[party_rank]; local_length];
        izip!(&mut proj_bases, &tau_x_geometric_progression).for_each(|(b, tau_xi)| *b *= *tau_xi);

        let mut g_bases = vec![E::G1Affine::default(); local_length];
        E::G1::batch_normalize(&proj_bases, &mut g_bases);

        drop(proj_bases);
        g_bases
    };

    let tau_x_srs = CoefFormUniKZGSRS::<E> {
        powers_of_tau: x_coeff_bases,
        tau_g2: (E::G2Affine::generator() * tau_x).into(),
    };

    let y_coeff_bases = {
        let mut proj_bases = vec![g1_prog; distributed_parties];
        izip!(&mut proj_bases, &tau_y_geometric_progression).for_each(|(b, tau_yi)| *b *= *tau_yi);

        let mut g_bases = vec![E::G1Affine::default(); distributed_parties];
        E::G1::batch_normalize(&proj_bases, &mut g_bases);

        drop(proj_bases);
        g_bases
    };

    let tau_y_srs = CoefFormUniKZGSRS::<E> {
        powers_of_tau: y_coeff_bases,
        tau_g2: (E::G2Affine::generator() * tau_y).into(),
    };

    CoefFormBiKZGLocalSRS {
        tau_x_srs,
        tau_y_srs,
    }
}

#[inline(always)]
pub fn coeff_form_bi_kzg_open_leader<E: MultiMillerLoop>(
    srs: &CoefFormBiKZGLocalSRS<E>,
    evals_and_opens: &[(E::Fr, E::G1Affine)],
    beta: E::Fr,
) -> (E::Fr, BiKZGProof<E>)
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    assert_eq!(srs.tau_y_srs.powers_of_tau.len(), evals_and_opens.len());

    let x_open: E::G1 = evals_and_opens.iter().map(|(_, o)| o.to_curve()).sum();
    let gammas: Vec<E::Fr> = evals_and_opens.iter().map(|(e, _)| *e).collect();

    let (div, eval) = univariate_degree_one_quotient(&gammas, beta);

    let mut y_open = E::G1::generator() * E::Fr::ZERO;
    msm::multiexp_serial(&div, &srs.tau_y_srs.powers_of_tau[..div.len()], &mut y_open);

    (
        eval,
        BiKZGProof {
            quotient_x: x_open.into(),
            quotient_y: y_open.into(),
        },
    )
}

#[inline(always)]
pub fn coeff_form_bi_kzg_verify<E: MultiMillerLoop>(
    vk: BiKZGVerifierParam<E>,
    comm: E::G1Affine,
    alpha: E::Fr,
    beta: E::Fr,
    eval: E::Fr,
    opening: BiKZGProof<E>,
) -> bool
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    let g1_eval: E::G1Affine = (E::G1Affine::generator() * eval).into();
    let g2_alpha: E::G2 = E::G2Affine::generator() * alpha;
    let g2_beta: E::G2 = E::G2Affine::generator() * beta;

    let gt_result = E::multi_miller_loop(&[
        (
            &opening.quotient_x,
            &(vk.tau_x_g2.to_curve() - g2_alpha).to_affine().into(),
        ),
        (
            &opening.quotient_y,
            &(vk.tau_y_g2.to_curve() - g2_beta).to_affine().into(),
        ),
        (&(g1_eval - comm).into(), &E::G2Affine::generator().into()),
    ]);

    gt_result.final_exponentiation().is_identity().into()
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use halo2curves::{
        bn256::{Bn256, Fr, G1Affine, G1},
        ff::Field,
        group::{prime::PrimeCurveAffine, Curve},
    };
    use itertools::izip;

    use crate::*;

    #[test]
    fn test_coefficient_form_bivariate_kzg_e2e() {
        let x_degree = 15;
        let y_degree = 7;

        let party_srs: Vec<CoefFormBiKZGLocalSRS<Bn256>> = (0..=y_degree)
            .map(|rank| {
                let mut rng = test_rng();
                generate_coef_form_bi_kzg_local_srs_for_testing(
                    x_degree + 1,
                    y_degree + 1,
                    rank,
                    &mut rng,
                )
            })
            .collect();

        let mut rng = test_rng();
        let xy_coeffs: Vec<Vec<Fr>> = (0..=y_degree)
            .map(|_| (0..=x_degree).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let commitments: Vec<_> = izip!(&party_srs, &xy_coeffs)
            .map(|(srs, x_coeffs)| coeff_form_uni_kzg_commit(&srs.tau_x_srs, x_coeffs))
            .collect();

        let global_commitment_g1: G1 = commitments.iter().map(|c| c.to_curve()).sum::<G1>();
        let global_commitment: G1Affine = global_commitment_g1.to_affine();

        let alpha = Fr::random(&mut rng);
        let evals_and_opens: Vec<(Fr, G1Affine)> = izip!(&party_srs, &xy_coeffs)
            .map(|(srs, x_coeffs)| coeff_form_uni_kzg_open_eval(&srs.tau_x_srs, x_coeffs, alpha))
            .collect();

        let beta = Fr::random(&mut rng);
        let (final_eval, final_opening) =
            coeff_form_bi_kzg_open_leader(&party_srs[0], &evals_and_opens, beta);

        let vk: BiKZGVerifierParam<Bn256> = From::from(&party_srs[0]);

        assert!(coeff_form_bi_kzg_verify(
            vk,
            global_commitment,
            alpha,
            beta,
            final_eval,
            final_opening,
        ));
    }
}
