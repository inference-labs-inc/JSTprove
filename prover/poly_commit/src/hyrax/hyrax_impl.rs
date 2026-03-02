use arith::{ExtensionField, Field};
use gkr_engine::Transcript;
use halo2curves::{ff::PrimeField, group::UncompressedEncoding, msm, CurveAffine};
use polynomials::{
    EqPolynomial, MultilinearExtension, MutRefMultiLinearPoly, MutableMultilinearExtension,
    RefMultiLinearPoly,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serdes::ExpSerde;
use utils::timer::Timer;

use crate::batching::{prover_merge_points, verifier_merge_points};
use crate::traits::BatchOpening;
use crate::{
    hyrax::{
        pedersen::{pedersen_commit, pedersen_setup},
        PedersenParams,
    },
    powers_series,
};

use super::HyraxPCS;

pub(crate) fn hyrax_setup<C: CurveAffine + ExpSerde>(
    local_vars: usize,
    mpi_vars: usize,
    rng: impl rand::RngCore,
) -> PedersenParams<C>
where
    C::Scalar: PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let pedersen_vars = {
        let total_vars = mpi_vars + local_vars;
        let squared_row_var = total_vars.div_ceil(2);

        if mpi_vars + squared_row_var > total_vars {
            total_vars - mpi_vars
        } else {
            squared_row_var
        }
    };

    let pedersen_length = 1 << pedersen_vars;

    pedersen_setup(pedersen_length, rng)
}

#[derive(Clone, Debug, Default)]
pub struct HyraxCommitment<C>(pub Vec<C>)
where
    C: CurveAffine + ExpSerde + UncompressedEncoding;

/// Jutification: from AsRef Documentation:
///  Ideally, `AsRef` would be reflexive, i.e. there would be an `impl<T: ?Sized> AsRef<T> for T`
///  Such a blanket implementation is currently *not* provided due to technical restrictions of
///  Rust's type system
impl<C> AsRef<HyraxCommitment<C>> for HyraxCommitment<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
{
    fn as_ref(&self) -> &HyraxCommitment<C> {
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct HyraxOpening<C>(pub Vec<C::Scalar>)
where
    C: CurveAffine + ExpSerde + UncompressedEncoding;

impl<C> ExpSerde for HyraxCommitment<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
{
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> serdes::SerdeResult<()> {
        self.0.len().serialize_into(&mut writer)?;
        for c in self.0.iter() {
            let uncompressed = UncompressedEncoding::to_uncompressed(c);
            writer.write_all(uncompressed.as_ref())?;
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> serdes::SerdeResult<Self> {
        let num_elements = usize::deserialize_from(&mut reader)?;
        let mut uncompressed = <C as UncompressedEncoding>::Uncompressed::default();

        let mut elements = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            reader.read_exact(uncompressed.as_mut())?;
            elements.push(
                C::from_uncompressed_unchecked(&uncompressed)
                    .into_option()
                    .ok_or(serdes::SerdeError::DeserializeError)?,
            );
        }
        Ok(Self(elements))
    }
}

impl<C> ExpSerde for HyraxOpening<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExpSerde,
{
    fn serialize_into<W: std::io::Write>(&self, writer: W) -> serdes::SerdeResult<()> {
        self.0.serialize_into(writer)
    }

    fn deserialize_from<R: std::io::Read>(reader: R) -> serdes::SerdeResult<Self> {
        let buffer: Vec<C::Scalar> = <Vec<C::Scalar> as ExpSerde>::deserialize_from(reader)?;
        Ok(Self(buffer))
    }
}

pub(crate) fn hyrax_commit<C>(
    params: &PedersenParams<C>,
    mle_poly: &impl MultilinearExtension<C::Scalar>,
) -> HyraxCommitment<C>
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    if mle_poly.hypercube_basis_ref().len() < params.msm_len() {
        // usually the params should be smaller than mle_poly as we rearrange the polynomial as a
        // matrix, and the params are the number of columns.
        //
        // However, in the batch opening cases, it is possible that some of the polynomials are in
        // fact much smaller; whereas the params are determined according to the maximum
        // polynomial size of the batch.

        let mut scalars = mle_poly.hypercube_basis();
        scalars.resize(params.msm_len(), C::Scalar::zero());
        let commitment = pedersen_commit(params, scalars.as_ref());

        return HyraxCommitment(vec![commitment]);
    }

    let commitments: Vec<C> = mle_poly
        .hypercube_basis_ref()
        .chunks(params.msm_len())
        .map(|sub_hypercube| pedersen_commit(params, sub_hypercube))
        .collect();

    HyraxCommitment(commitments)
}

// NOTE(HS) the hyrax opening returns an eval and an opening against the eval_point on input.
pub(crate) fn hyrax_open<C>(
    params: &PedersenParams<C>,
    mle_poly: &impl MultilinearExtension<C::Scalar>,
    eval_point: &[C::Scalar],
) -> (C::Scalar, HyraxOpening<C>)
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let pedersen_len = params.msm_len();
    let pedersen_vars = pedersen_len.ilog2() as usize;

    let mut local_basis = mle_poly.hypercube_basis();
    let mut local_mle = MutRefMultiLinearPoly::from_ref(&mut local_basis);
    local_mle.fix_variables(&eval_point[pedersen_vars..]);

    let mut buffer = vec![C::Scalar::default(); local_mle.coeffs.len()];
    let final_eval = local_mle.evaluate_with_buffer(&eval_point[..pedersen_vars], &mut buffer);

    (final_eval, HyraxOpening(local_basis))
}

pub(crate) fn hyrax_verify<C>(
    params: &PedersenParams<C>,
    comm: &HyraxCommitment<C>,
    eval_point: &[C::Scalar],
    eval: C::Scalar,
    proof: &HyraxOpening<C>,
) -> bool
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let pedersen_len = params.msm_len();
    let pedersen_vars = pedersen_len.ilog2() as usize;

    let eq_combination: Vec<C::Scalar> = EqPolynomial::build_eq_x_r(&eval_point[pedersen_vars..]);
    let row_comm = msm::best_multiexp(&eq_combination, &comm.0);

    let pedersen_commitment = pedersen_commit(params, &proof.0);

    if pedersen_commitment != row_comm.into() {
        eprintln!("pedersen commitment not match",);
        return false;
    }

    let mut scratch = vec![C::Scalar::default(); proof.0.len()];
    let res = eval
        == RefMultiLinearPoly::from_ref(&proof.0)
            .evaluate_with_buffer(&eval_point[..pedersen_vars], &mut scratch);
    if !res {
        eprintln!("evaluation does not match");
    }

    res
}

// batch open a set of mle_polys at the same point
// returns a set of eval points and a signle opening
// NOTE: random linear combination is used to merge polynomials
pub(crate) fn hyrax_batch_open<C>(
    params: &PedersenParams<C>,
    mle_poly_list: &[impl MultilinearExtension<C::Scalar>],
    eval_point: &[C::Scalar],
    transcript: &mut impl Transcript,
) -> (Vec<C::Scalar>, HyraxOpening<C>)
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let len = mle_poly_list.len();
    let pedersen_len = params.msm_len();
    let pedersen_vars = pedersen_len.ilog2() as usize;

    let challenge = transcript.generate_field_element::<C::Scalar>();
    let challenge_power = powers_series(&challenge, len);

    // the opening is the random linearly combine all the polynomials
    let mut res = vec![C::Scalar::default(); 1 << pedersen_vars];
    let mut evals = vec![];
    let mut buffer = vec![C::Scalar::default(); 1 << pedersen_vars];

    for (mle_poly, challenge) in mle_poly_list.iter().zip(challenge_power.iter()) {
        let mut local_basis = mle_poly.hypercube_basis();
        let mut local_mle = MutRefMultiLinearPoly::from_ref(&mut local_basis);
        local_mle.fix_variables(&eval_point[pedersen_vars..]);

        evals.push(local_mle.evaluate_with_buffer(&eval_point[..pedersen_vars], &mut buffer));

        res.iter_mut()
            .zip(local_mle.coeffs.iter())
            .for_each(|(r, c)| {
                *r += *challenge * *c;
            });
    }
    (evals, HyraxOpening(res))
}

/// Batch verify a list of hyrax commitments/proofs that are opened at the same point.
pub(crate) fn hyrax_batch_verify<C>(
    params: &PedersenParams<C>,
    comm_list: &[HyraxCommitment<C>],
    eval_point: &[C::Scalar],
    eval_list: &[C::Scalar],
    batch_proof: &HyraxOpening<C>,
    transcript: &mut impl Transcript,
) -> bool
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let len = comm_list.len();
    assert_eq!(len, eval_list.len());

    let pedersen_len = params.msm_len();
    let pedersen_vars = pedersen_len.ilog2() as usize;

    let challenge = transcript.generate_field_element::<C::Scalar>();
    let challenge_power = powers_series(&challenge, len);

    // random linear combination of the commitments
    // for each i we want to do
    //  comm_i * challenge_i * eq_combination
    // we do the second mul first -- this is a field op
    // then we do a single multiexp to take advantage of Pippenger's algorithm

    let bases = comm_list
        .iter()
        .flat_map(|comm| comm.0.clone())
        .collect::<Vec<_>>();

    let mut scalars = vec![];
    let eq_combination: Vec<C::Scalar> = EqPolynomial::build_eq_x_r(&eval_point[pedersen_vars..]);
    for c in challenge_power.iter() {
        scalars.extend_from_slice(scale(&eq_combination, c).as_ref());
    }

    let row_comm = msm::best_multiexp(&scalars, &bases);

    if pedersen_commit(params, &batch_proof.0) != row_comm.into() {
        eprintln!("commitment not matching");
        return false;
    }

    // now we need to check the evaluations
    let eval_sum = eval_list
        .iter()
        .zip(challenge_power.iter())
        .map(|(eval, challenge)| *eval * *challenge)
        .sum::<C::Scalar>();

    let mut scratch = vec![C::Scalar::default(); batch_proof.0.len()];
    eval_sum
        == RefMultiLinearPoly::from_ref(&batch_proof.0)
            .evaluate_with_buffer(&eval_point[..pedersen_vars], &mut scratch)
}

#[inline(always)]
// scale a vector by a scalar
fn scale<F: Field>(base: &[F], scalar: &F) -> Vec<F> {
    base.iter().map(|x| *x * scalar).collect()
}

/// Open a set of polynomials at a multiple points.
/// Requires the length of the polys to be the same as points.
/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g_i(b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq_i(b) = eq(b, point_i)
/// 5. run sumcheck on \sum_i=1..k \tilde eq_i * \tilde g_i
/// 6. build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the sumcheck's point
/// 7. open g'(X) at point (a2)
///
/// Returns:
/// - the evaluations of the polynomials at their corresponding points
/// - the batch opening proof containing the sumcheck proof and the opening of g'(X)
#[allow(clippy::type_complexity)]
pub(crate) fn hyrax_multi_points_batch_open_internal<C>(
    proving_key: &PedersenParams<C>,
    polys: &[impl MultilinearExtension<C::Scalar>],
    points: &[Vec<C::Scalar>],
    transcript: &mut impl Transcript,
) -> (Vec<C::Scalar>, BatchOpening<C::Scalar, HyraxPCS<C>>)
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let timer = Timer::new("batch_opening", true);
    // generate evals for each polynomial at its corresponding point
    let eval_timer = Timer::new("eval all polys", true);
    let evals: Vec<C::Scalar> = polys
        .par_iter()
        .zip_eq(points.par_iter())
        .map(|(poly, point)| poly.evaluate(point))
        .collect();
    eval_timer.stop();

    let merger_timer = Timer::new("merging points", true);
    let (new_point, g_prime, proof) = prover_merge_points::<C>(polys, points, transcript);
    merger_timer.stop();

    // open g'(X) at point (a2)
    // g_prime is a MultiLinearPoly, so we can use the hyrax_open function
    let pcs_timer = Timer::new("hyrax_open", true);
    let (_g_prime_eval, g_prime_proof) = hyrax_open(proving_key, &g_prime, &new_point);
    pcs_timer.stop();

    timer.stop();
    (
        evals,
        BatchOpening {
            sum_check_proof: proof,
            g_prime_proof,
        },
    )
}

/// Verify the opening of a set of polynomials at a single point.
/// Steps:
/// 1. get challenge point t from transcript
/// 2. build g' commitment
/// 3. ensure \sum_i eq(a2, point_i) * eq(t, <i>) * f_i_evals matches the sum via SumCheck
///    verification
/// 4. verify commitment
pub(crate) fn hyrax_multi_points_batch_verify_internal<C>(
    verifying_key: &PedersenParams<C>,
    commitments: &[impl AsRef<HyraxCommitment<C>>],
    points: &[Vec<C::Scalar>],
    values: &[C::Scalar],
    batch_opening: &BatchOpening<C::Scalar, HyraxPCS<C>>,
    transcript: &mut impl Transcript,
) -> bool
where
    C: CurveAffine + ExpSerde + UncompressedEncoding,
    C::Scalar: ExtensionField + PrimeField,
    C::ScalarExt: ExtensionField + PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let a2 = batch_opening.sum_check_proof.export_point_to_expander();

    let commitments = commitments
        .iter()
        .map(|c| c.as_ref().0.clone())
        .collect::<Vec<_>>();

    let (sumcheck_verified, tilde_g_eval, g_prime_commit) = verifier_merge_points(
        &commitments,
        points,
        values,
        &batch_opening.sum_check_proof,
        transcript,
    );
    if !sumcheck_verified {
        return false;
    }

    let g_prime_commit = HyraxCommitment(g_prime_commit);

    // verify commitment
    hyrax_verify(
        verifying_key,
        &g_prime_commit,
        a2.as_ref(),
        tilde_g_eval,
        &batch_opening.g_prime_proof,
    )
}
