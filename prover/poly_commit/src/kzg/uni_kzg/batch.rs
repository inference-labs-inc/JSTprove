use arith::{ExtensionField, Field};
use gkr_engine::Transcript;
use halo2curves::group::Group;
use halo2curves::{group::Curve, msm::multiexp_serial, pairing::MultiMillerLoop, CurveAffine};
use polynomials::MultiLinearPoly;
use serdes::ExpSerde;

use crate::{
    coeff_form_uni_hyperkzg_open, coeff_form_uni_hyperkzg_verify, powers_series, CoefFormUniKZGSRS,
    HyperUniKZGOpening, UniKZGVerifierParams,
};

pub(crate) fn kzg_single_point_batch_open<E>(
    proving_key: &CoefFormUniKZGSRS<E>,
    polys: &[MultiLinearPoly<E::Fr>],
    x: &[E::Fr],
    transcript: &mut impl Transcript,
) -> (Vec<E::Fr>, HyperUniKZGOpening<E>)
where
    E: MultiMillerLoop,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
    E::Fr: ExtensionField,
{
    let rlc_randomness = transcript.generate_field_element::<E::Fr>();
    let num_poly = polys.len();
    let rlcs = powers_series(&rlc_randomness, num_poly);
    let mut buf = vec![E::Fr::default(); polys[0].coeffs.len()];

    let merged_poly = polys
        .iter()
        .zip(rlcs.iter())
        .skip(1)
        .fold(polys[0].clone(), |acc, (poly, r)| acc + &(poly * r));

    let mut evals = polys
        .iter()
        .map(|p| MultiLinearPoly::evaluate_with_buffer(p.coeffs.as_ref(), x, &mut buf))
        .collect::<Vec<_>>();

    let (_batch_eval, open) =
        coeff_form_uni_hyperkzg_open(proving_key, &merged_poly.coeffs, x, transcript);

    {
        // sanity check: the merged evaluation should match the batch evaluation
        // this step is not necessary if the performance is critical
        let mut merged_eval = evals[0];
        for (eval, r) in evals.iter_mut().zip(rlcs.iter()).skip(1) {
            merged_eval += *eval * r;
        }
        assert_eq!(_batch_eval, merged_eval);
    }

    (evals, open)
}

pub(crate) fn kzg_single_point_batch_verify<E>(
    verifying_key: &UniKZGVerifierParams<E>,
    commitments: &[E::G1Affine],
    x: &[E::Fr],
    evals: &[E::Fr],
    opening: &HyperUniKZGOpening<E>,
    transcript: &mut impl Transcript,
) -> bool
where
    E: MultiMillerLoop,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: ExpSerde,
    E::Fr: ExtensionField + ExpSerde,
{
    let rlc_randomness = transcript.generate_field_element::<E::Fr>();
    let num_poly = commitments.len();
    let rlcs = powers_series(&rlc_randomness, num_poly);

    // stay with single thread as the num_poly is usually small
    let mut merged_commitment = E::G1::identity();
    multiexp_serial(&rlcs, commitments, &mut merged_commitment);

    let merged_eval = evals
        .iter()
        .zip(rlcs.iter())
        .fold(E::Fr::zero(), |acc, (e, r)| acc + (*e * r));

    coeff_form_uni_hyperkzg_verify(
        verifying_key,
        merged_commitment.to_affine(),
        x,
        merged_eval,
        opening,
        transcript,
    )
}
