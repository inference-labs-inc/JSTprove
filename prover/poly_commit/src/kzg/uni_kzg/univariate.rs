use halo2curves::{
    ff::Field,
    group::{prime::PrimeCurveAffine, Curve, Group},
    msm,
    pairing::{MillerLoopResult, MultiMillerLoop},
    CurveAffine,
};
use rayon::prelude::*;
use serdes::ExpSerde;

use crate::*;

#[inline(always)]
pub(crate) fn generate_coef_form_uni_kzg_srs_for_testing<E: MultiMillerLoop>(
    length: usize,
    mut rng: impl rand::RngCore,
) -> CoefFormUniKZGSRS<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    assert!(length.is_power_of_two());

    let tau = E::Fr::random(&mut rng);
    let g1 = E::G1Affine::generator();

    let tau_geometric_progression = powers_series(&tau, length);

    let g1_prog = g1.to_curve();
    let coeff_bases = {
        let mut proj_bases = vec![g1_prog; length];
        proj_bases
            .par_iter_mut()
            .zip(tau_geometric_progression.par_iter())
            .for_each(|(b, tau_i)| *b *= tau_i);

        let mut g_bases = vec![E::G1Affine::default(); length];
        E::G1::batch_normalize(&proj_bases, &mut g_bases);

        drop(proj_bases);
        g_bases
    };

    CoefFormUniKZGSRS {
        powers_of_tau: coeff_bases,
        tau_g2: (E::G2Affine::generator() * tau).into(),
    }
}

#[inline(always)]
pub(crate) fn coeff_form_uni_kzg_commit<E: MultiMillerLoop>(
    srs: &CoefFormUniKZGSRS<E>,
    coeffs: &[E::Fr],
) -> E::G1Affine
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    assert!(srs.powers_of_tau.len() >= coeffs.len());

    let com = msm::best_multiexp(coeffs, &srs.powers_of_tau[..coeffs.len()]);

    com.into()
}

#[inline(always)]
pub fn coeff_form_uni_kzg_open_eval<E: MultiMillerLoop>(
    srs: &CoefFormUniKZGSRS<E>,
    coeffs: &[E::Fr],
    alpha: E::Fr,
) -> (E::Fr, E::G1Affine)
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    assert!(srs.powers_of_tau.len() >= coeffs.len());

    let (div, eval) = univariate_degree_one_quotient(coeffs, alpha);
    let opening = msm::best_multiexp(&div, &srs.powers_of_tau[..div.len()]);

    (eval, opening.into())
}

#[inline(always)]
pub(crate) fn coeff_form_uni_kzg_verify<E: MultiMillerLoop>(
    vk: &UniKZGVerifierParams<E>,
    comm: E::G1Affine,
    alpha: E::Fr,
    eval: E::Fr,
    opening: E::G1Affine,
) -> bool
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    E::G2Affine: ExpSerde,
{
    let g1_eval: E::G1Affine = (E::G1Affine::generator() * eval).into();
    let g2_alpha: E::G2 = E::G2Affine::generator() * alpha;

    let gt_result = E::multi_miller_loop(&[
        (
            &opening,
            &(vk.tau_g2.to_curve() - g2_alpha).to_affine().into(),
        ),
        (&(g1_eval - comm).into(), &E::G2Affine::generator().into()),
    ]);

    gt_result.final_exponentiation().is_identity().into()
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use halo2curves::bn256::{Bn256, Fr};

    use crate::*;

    #[test]
    fn test_coefficient_form_univariate_kzg_e2e() {
        // \prod_{i \in [1, 7]} (x + i)
        let poly = vec![
            Fr::from(5040u32),
            Fr::from(13068u64),
            Fr::from(13132u64),
            Fr::from(6769u64),
            Fr::from(1960u64),
            Fr::from(322u64),
            Fr::from(28u64),
            Fr::from(1u64),
        ];
        let alpha = Fr::from(3u64);
        let eval = Fr::from(604800u64);

        let mut rng = test_rng();
        let srs = generate_coef_form_uni_kzg_srs_for_testing::<Bn256>(8, &mut rng);
        let vk: UniKZGVerifierParams<Bn256> = From::from(&srs);
        let com = coeff_form_uni_kzg_commit(&srs, &poly);

        let (actual_eval, opening) = coeff_form_uni_kzg_open_eval(&srs, &poly, alpha);
        assert_eq!(actual_eval, eval);

        assert!(coeff_form_uni_kzg_verify(&vk, com, alpha, eval, opening))
    }

    #[test]
    fn test_coefficient_form_univariate_kzg_constant_e2e() {
        let poly = vec![Fr::from(100u64)];
        let alpha = Fr::from(3u64);
        let eval = Fr::from(100u64);

        let mut rng = test_rng();
        let srs = generate_coef_form_uni_kzg_srs_for_testing::<Bn256>(8, &mut rng);
        let vk: UniKZGVerifierParams<Bn256> = From::from(&srs);
        let com = coeff_form_uni_kzg_commit(&srs, &poly);

        let (actual_eval, opening) = coeff_form_uni_kzg_open_eval(&srs, &poly, alpha);
        assert_eq!(actual_eval, eval);

        assert!(coeff_form_uni_kzg_verify(&vk, com, alpha, eval, opening))
    }
}
