// NOTE(HS): the algorithm port for HyperKZG to "HyperBiKZG" is sketched here:
// https://drive.google.com/file/d/1NcRnqdwFLcLi77DvSZH28QwslTuBVyb4/

use std::{io::Cursor, iter};

use arith::ExtensionField;
use gkr_engine::{MPIEngine, Transcript};
use halo2curves::{
    ff::Field,
    group::{prime::PrimeCurveAffine, Curve, Group, GroupEncoding},
    pairing::MultiMillerLoop,
    CurveAffine,
};
use itertools::{chain, izip};
use polynomials::MultilinearExtension;
use serdes::ExpSerde;
use transcript::{transcript_root_broadcast, transcript_verifier_sync};

use crate::*;

pub fn coeff_form_hyper_bikzg_open<E>(
    srs: &CoefFormBiKZGLocalSRS<E>,
    mpi_engine: &impl MPIEngine,
    coeffs: &impl MultilinearExtension<E::Fr>,
    local_alphas: &[E::Fr],
    mpi_alphas: &[E::Fr],
    fs_transcript: &mut impl Transcript,
) -> Option<HyperBiKZGOpening<E>>
where
    E: MultiMillerLoop,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
    E::Fr: ExtensionField,
{
    // NOTE(HS) deteriorate to vanilla HyperKZG if mpi_alphas is empty, namely single party setting
    // since there is no other mpi variables, then the party running is the leader
    if mpi_alphas.is_empty() {
        let (_, hyperkzg_opening) = coeff_form_uni_hyperkzg_open(
            &srs.tau_x_srs,
            coeffs.hypercube_basis_ref(),
            local_alphas,
            fs_transcript,
        );

        let hyper_bikzg_opening: HyperBiKZGOpening<E> = hyperkzg_opening.into();
        return hyper_bikzg_opening.into();
    }

    //
    // Locally fold local variables, then commit to construct the poly oracles
    //

    let (local_folded_x_oracle_commits, local_folded_x_oracle_coeffs) =
        coeff_form_hyperkzg_local_poly_oracles(
            &srs.tau_x_srs,
            coeffs.hypercube_basis_ref(),
            local_alphas,
        );

    let local_final_eval_at_x = {
        let last_coeffs = local_folded_x_oracle_coeffs.last().unwrap().clone();
        let last_alpha = local_alphas[local_alphas.len() - 1];
        (E::Fr::ONE - last_alpha) * last_coeffs[0] + last_alpha * last_coeffs[1]
    };

    //
    // Leader party gathering evals and oracle commitments
    //

    let mut root_gathering_folded_oracle_commits: Vec<E::G1Affine> =
        vec![E::G1Affine::default(); mpi_engine.world_size() * local_folded_x_oracle_commits.len()];
    let mut final_evals_at_x: Vec<E::Fr> = vec![E::Fr::ZERO; mpi_engine.world_size()];

    mpi_engine.gather_vec(
        &local_folded_x_oracle_commits,
        &mut root_gathering_folded_oracle_commits,
    );
    mpi_engine.gather_vec(&[local_final_eval_at_x], &mut final_evals_at_x);

    //
    // Leader party collect oracle commitments, sum them up for folded oracles
    //

    let mut folded_x_oracle_commits: Vec<E::G1Affine> = Vec::new();
    let mut y_oracle_commit: E::G1Affine = E::G1Affine::default();

    if mpi_engine.is_root() {
        let g1_zero = E::G1Affine::default().to_curve();
        let mut folded_x_coms_g1 = vec![g1_zero; local_folded_x_oracle_commits.len()];

        root_gathering_folded_oracle_commits
            .chunks(local_folded_x_oracle_commits.len())
            .for_each(|folded_oracles| {
                izip!(&mut folded_x_coms_g1, folded_oracles)
                    .for_each(|(x_com_i, oracle_i)| *x_com_i += oracle_i.to_curve())
            });

        folded_x_oracle_commits = vec![E::G1Affine::default(); folded_x_coms_g1.len()];
        E::G1::batch_normalize(&folded_x_coms_g1, &mut folded_x_oracle_commits);

        y_oracle_commit = coeff_form_uni_kzg_commit(&srs.tau_y_srs, &final_evals_at_x);
    }

    //
    // The leader party continues on folding over "final_evals" over only y variables.
    //

    let mut folded_y_oracle_commits: Vec<E::G1Affine> = Vec::new();
    let mut folded_y_oracle_coeffs: Vec<Vec<E::Fr>> = Vec::new();

    if mpi_engine.is_root() {
        (folded_y_oracle_commits, folded_y_oracle_coeffs) =
            coeff_form_hyperkzg_local_poly_oracles(&srs.tau_y_srs, &final_evals_at_x, mpi_alphas);
    }

    //
    // The leader party feeds all folded oracles into RO, then sync party's transcript state
    //

    let mut folded_oracle_commitments: Vec<E::G1Affine> = Vec::new();

    if mpi_engine.is_root() {
        folded_oracle_commitments = {
            let mut temp = folded_x_oracle_commits.clone();
            temp.push(y_oracle_commit);
            temp.extend_from_slice(&folded_y_oracle_commits);
            temp
        };

        chain!(
            &folded_x_oracle_commits,
            iter::once(&y_oracle_commit),
            &folded_y_oracle_commits,
        )
        .for_each(|f| fs_transcript.append_u8_slice(f.to_bytes().as_ref()));
    }

    transcript_root_broadcast(fs_transcript, mpi_engine);

    let beta_x = fs_transcript.generate_field_element::<E::Fr>();
    let beta_y = fs_transcript.generate_field_element::<E::Fr>();

    //
    // Local parties run HyperKZG evals at beta_x, -beta_x, beta_x^2 over folded coeffs
    //

    let local_folded_x_evals: HyperKZGLocalEvals<E> = coeff_form_hyperkzg_local_evals(
        coeffs.hypercube_basis_ref(),
        &local_folded_x_oracle_coeffs,
        local_alphas,
        beta_x,
    );

    let local_exported_folded_x_evals: HyperKZGExportedLocalEvals<E> =
        local_folded_x_evals.clone().into();

    //
    // Collect all exported local folded evals at x to the leader party
    //

    let mut root_gathering_exported_folded_x_evals: Vec<HyperKZGExportedLocalEvals<E>> =
        vec![local_exported_folded_x_evals.clone(); mpi_engine.world_size()];
    let mut root_aggregated_x_evals = HyperKZGAggregatedEvals::<E>::default();
    let mut root_folded_y_evals = HyperKZGLocalEvals::<E>::default();

    {
        let mut local_exported_folded_x_evals_bytes: Vec<u8> = Vec::new();
        local_exported_folded_x_evals
            .serialize_into(&mut local_exported_folded_x_evals_bytes)
            .unwrap();

        let mut gathering_buffer =
            vec![0u8; mpi_engine.world_size() * local_exported_folded_x_evals_bytes.len()];

        mpi_engine.gather_vec(&local_exported_folded_x_evals_bytes, &mut gathering_buffer);

        if mpi_engine.is_root() {
            izip!(
                &mut root_gathering_exported_folded_x_evals,
                gathering_buffer.chunks(local_exported_folded_x_evals_bytes.len())
            )
            .for_each(|(es, bs)| {
                let mut cursor = Cursor::new(bs.to_vec());
                *es = HyperKZGExportedLocalEvals::<E>::deserialize_from(&mut cursor).unwrap();
            })
        }
    }

    //
    // Leader aggregates all local exported evaluations (at x) by evaluating at y
    // by three points: beta_y, -beta_y, beta_y^2, then fold the final evals at x,
    // which is degree 0 for variable x, along variable y.
    //

    if mpi_engine.is_root() {
        root_aggregated_x_evals = HyperKZGAggregatedEvals::new_from_exported_evals(
            &root_gathering_exported_folded_x_evals,
            beta_y,
        );

        root_folded_y_evals = coeff_form_hyperkzg_local_evals(
            &final_evals_at_x,
            &folded_y_oracle_coeffs,
            mpi_alphas,
            beta_y,
        );
    }

    //
    // The leader party feeds all evals into RO, then sync party's transcript state
    //

    if mpi_engine.is_root() {
        root_aggregated_x_evals.append_to_transcript(fs_transcript);
        root_folded_y_evals.append_to_transcript(fs_transcript);
    }

    transcript_root_broadcast(fs_transcript, mpi_engine);

    let gamma = fs_transcript.generate_field_element::<E::Fr>();

    //
    // The leader party linear combines folded coeffs at y with gamma,
    // then broadcast the coeffs back to local.
    //

    let mut leader_gamma_aggregated_y_coeffs: Vec<E::Fr> =
        vec![E::Fr::ZERO; mpi_engine.world_size()];

    if mpi_engine.is_root() {
        leader_gamma_aggregated_y_coeffs = {
            let gamma_n = gamma.pow_vartime([local_alphas.len() as u64]);
            let mut temp = coeff_form_hyperkzg_local_oracle_polys_aggregate::<E>(
                &final_evals_at_x,
                &folded_y_oracle_coeffs,
                gamma,
            );
            temp.iter_mut().for_each(|t| *t *= gamma_n);
            temp
        };
    }

    // TODO(HS) can be improved to broadcast a vec, returning a coeff to each party
    {
        let mut serialized_y_coeffs: Vec<u8> = Vec::new();
        leader_gamma_aggregated_y_coeffs
            .serialize_into(&mut serialized_y_coeffs)
            .unwrap();

        mpi_engine.root_broadcast_bytes(&mut serialized_y_coeffs);
        leader_gamma_aggregated_y_coeffs = {
            let mut cursor = Cursor::new(serialized_y_coeffs);
            Vec::deserialize_from(&mut cursor).unwrap()
        };
    }

    //
    // Local party compute the linear combined folded coeffs at x with gamma,
    // then the degree2 Lagrange over beta_x, -beta_x, beta_x^2,
    // then vanish the local aggregated x coeffs at the three points above,
    // and commit to the final quotient poly
    //

    let mut local_gamma_aggregated_x_coeffs = {
        let mut f_gamma_local = coeff_form_hyperkzg_local_oracle_polys_aggregate::<E>(
            coeffs.hypercube_basis_ref(),
            &local_folded_x_oracle_coeffs,
            gamma,
        );

        f_gamma_local[0] += leader_gamma_aggregated_y_coeffs[mpi_engine.world_rank()];
        f_gamma_local
    };

    let local_lagrange_degree2_at_x = {
        let mut local_degree_2 =
            local_folded_x_evals.interpolate_degree2_aggregated_evals(beta_x, gamma);

        local_degree_2[0] += leader_gamma_aggregated_y_coeffs[mpi_engine.world_rank()];
        local_degree_2
    };

    let local_gamma_aggregated_x_quotient = {
        let mut nom = local_gamma_aggregated_x_coeffs.clone();
        polynomial_add(&mut nom, -E::Fr::ONE, &local_lagrange_degree2_at_x);
        univariate_roots_quotient(nom, &[beta_x, -beta_x, beta_x * beta_x])
    };

    let local_gamma_aggregated_x_quotient_commitment_g1: E::G1 =
        coeff_form_uni_kzg_commit(&srs.tau_x_srs, &local_gamma_aggregated_x_quotient).to_curve();

    //
    // Leader collect all the quotient commitment at x, sum it up and feed it to RO,
    // then sync transcript state
    //

    let mut root_gathering_gamma_aggregated_x_quotient_commitment_g1s: Vec<E::G1> =
        vec![E::G1::generator(); mpi_engine.world_size()];
    mpi_engine.gather_vec(
        &[local_gamma_aggregated_x_quotient_commitment_g1],
        &mut root_gathering_gamma_aggregated_x_quotient_commitment_g1s,
    );

    let mut gamma_aggregated_x_quotient_commitment: E::G1Affine = E::G1Affine::default();

    if mpi_engine.is_root() {
        gamma_aggregated_x_quotient_commitment =
            root_gathering_gamma_aggregated_x_quotient_commitment_g1s
                .iter()
                .sum::<E::G1>()
                .to_affine();

        fs_transcript.append_u8_slice(gamma_aggregated_x_quotient_commitment.to_bytes().as_ref());
    }

    transcript_root_broadcast(fs_transcript, mpi_engine);

    let delta_x = fs_transcript.generate_field_element::<E::Fr>();

    //
    // Locally compute the Lagrange-degree2 interpolation at delta_x, pool at leader
    //

    let mut degree2_evals_at_delta_x: Vec<E::Fr> = vec![E::Fr::ZERO; mpi_engine.world_size()];

    let local_degree2_eval_at_delta_x = local_lagrange_degree2_at_x[0]
        + local_lagrange_degree2_at_x[1] * delta_x
        + local_lagrange_degree2_at_x[2] * delta_x * delta_x;

    mpi_engine.gather_vec(
        &[local_degree2_eval_at_delta_x],
        &mut degree2_evals_at_delta_x,
    );

    //
    // Leader does similar thing - quotient at beta_y, -beta_y, beta_y^2,
    // commit the quotient polynomial commitment at y, feed it to RO,
    // then sync transcript state
    //

    let mut leader_quotient_y_coeffs: Vec<E::Fr> = vec![E::Fr::ZERO; mpi_engine.world_size()];
    let mut leader_quotient_y_commitment: E::G1Affine = E::G1Affine::default();

    if mpi_engine.is_root() {
        let num_y_coeffs = mpi_engine.world_size();

        // NOTE(HS) interpolate at beta_y, beta_y2, -beta_y on lagrange_degree2_delta_x
        let lagrange_degree2_delta_y = {
            let pos_beta_y_pow_series = powers_series(&beta_y, num_y_coeffs);
            let neg_beta_y_pow_series = powers_series(&(-beta_y), num_y_coeffs);
            let beta_y2_pow_series = powers_series(&(beta_y * beta_y), num_y_coeffs);

            let at_beta_y = univariate_evaluate(&degree2_evals_at_delta_x, &pos_beta_y_pow_series);
            let at_neg_beta_y =
                univariate_evaluate(&degree2_evals_at_delta_x, &neg_beta_y_pow_series);
            let at_beta_y2 = univariate_evaluate(&degree2_evals_at_delta_x, &beta_y2_pow_series);

            coeff_form_degree2_lagrange(
                [beta_y, -beta_y, beta_y * beta_y],
                [at_beta_y, at_neg_beta_y, at_beta_y2],
            )
        };

        leader_quotient_y_coeffs = {
            let mut nom = degree2_evals_at_delta_x.clone();
            polynomial_add(&mut nom, -E::Fr::ONE, &lagrange_degree2_delta_y);
            univariate_roots_quotient(nom, &[beta_y, -beta_y, beta_y * beta_y])
        };

        leader_quotient_y_commitment =
            coeff_form_uni_kzg_commit(&srs.tau_y_srs, &leader_quotient_y_coeffs);

        fs_transcript.append_u8_slice(leader_quotient_y_commitment.to_bytes().as_ref());
    }

    transcript_root_broadcast(fs_transcript, mpi_engine);

    let delta_y = fs_transcript.generate_field_element::<E::Fr>();

    //
    // Leader send out the quotient on y coefficients back to local parties
    //

    // TODO(HS) can be better if the root only send corresponding coeffs to the parties
    {
        let mut serialized_y_quotient_coeffs: Vec<u8> = Vec::new();
        leader_quotient_y_coeffs
            .serialize_into(&mut serialized_y_quotient_coeffs)
            .unwrap();

        mpi_engine.root_broadcast_bytes(&mut serialized_y_quotient_coeffs);
        leader_quotient_y_coeffs = {
            let mut cursor = Cursor::new(serialized_y_quotient_coeffs);
            Vec::deserialize_from(&mut cursor).unwrap()
        };
        leader_quotient_y_coeffs.resize(mpi_engine.world_size(), E::Fr::ZERO);
    }

    //
    // Final step for local - trip off the prior quotients at x and y on \pm beta and beta^2
    //

    // NOTE(HS) f_gamma_s - (delta_x - beta_x) ... (delta_x - beta_x2) f_gamma_quotient_s
    //                    - (delta_y - beta_y) ... (delta_y - beta_y2) lagrange_quotient_y
    let delta_x_denom = (delta_x - beta_x) * (delta_x - beta_x * beta_x) * (delta_x + beta_x);
    let delta_y_denom = (delta_y - beta_y) * (delta_y - beta_y * beta_y) * (delta_y + beta_y);

    polynomial_add(
        &mut local_gamma_aggregated_x_coeffs,
        -delta_x_denom,
        &local_gamma_aggregated_x_quotient,
    );
    local_gamma_aggregated_x_coeffs[0] -=
        delta_y_denom * leader_quotient_y_coeffs[mpi_engine.world_rank()];

    //
    // BiKZG commit to the last bivariate poly
    //

    let mut gathered_eval_opens: Vec<(E::Fr, E::G1Affine)> =
        vec![(E::Fr::ZERO, E::G1Affine::default()); mpi_engine.world_size()];
    let local_eval_open =
        coeff_form_uni_kzg_open_eval(&srs.tau_x_srs, &local_gamma_aggregated_x_coeffs, delta_x);

    mpi_engine.gather_vec(&[local_eval_open], &mut gathered_eval_opens);

    if !mpi_engine.is_root() {
        return None;
    }

    let (_, final_opening) = coeff_form_bi_kzg_open_leader(srs, &gathered_eval_opens, delta_y);

    HyperBiKZGOpening {
        folded_oracle_commitments,
        aggregated_evals: root_aggregated_x_evals,
        leader_evals: root_folded_y_evals.into(),
        beta_x_commitment: gamma_aggregated_x_quotient_commitment,
        beta_y_commitment: leader_quotient_y_commitment,
        quotient_delta_x_commitment: final_opening.quotient_x,
        quotient_delta_y_commitment: final_opening.quotient_y,
    }
    .into()
}

#[allow(clippy::too_many_arguments)]
pub fn coeff_form_hyper_bikzg_verify<E, T>(
    vk: &BiKZGVerifierParam<E>,
    local_alphas: &[E::Fr],
    mpi_alphas: &[E::Fr],
    eval: E::Fr,
    commitment: E::G1Affine,
    opening: &HyperBiKZGOpening<E>,
    fs_transcript: &mut T,
) -> bool
where
    E: MultiMillerLoop,
    T: Transcript,
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1> + ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
    E::Fr: ExtensionField,
{
    // NOTE(HS) deteriorate to vanilla HyperKZG verify if mpi_alphas is empty
    if mpi_alphas.is_empty() {
        let hyper_bikzg_opening = opening.clone();
        let hyper_kzg_opening: HyperUniKZGOpening<E> = hyper_bikzg_opening.into();

        let what = coeff_form_uni_hyperkzg_verify(
            &vk.into(),
            commitment,
            local_alphas,
            eval,
            &hyper_kzg_opening,
            fs_transcript,
        );

        return what;
    }

    let mpi_world_size = 1 << mpi_alphas.len();

    opening
        .folded_oracle_commitments
        .iter()
        .for_each(|f| fs_transcript.append_u8_slice(f.to_bytes().as_ref()));

    // NOTE(HS) transcript MPI thing ...
    transcript_verifier_sync(fs_transcript, mpi_world_size);

    let beta_x = fs_transcript.generate_field_element::<E::Fr>();
    let beta_y = fs_transcript.generate_field_element::<E::Fr>();

    // dbg!(beta_x, beta_y);

    // NOTE(HS) evaluation checks

    let beta_y2_local = HyperKZGLocalEvals::new_from_exported_evals(
        &opening.aggregated_evals.beta_y2_evals,
        local_alphas,
        beta_x,
    );

    let pos_beta_y_local = HyperKZGLocalEvals::new_from_exported_evals(
        &opening.aggregated_evals.pos_beta_y_evals,
        local_alphas,
        beta_x,
    );

    let neg_beta_y_local = HyperKZGLocalEvals::new_from_exported_evals(
        &opening.aggregated_evals.neg_beta_y_evals,
        local_alphas,
        beta_x,
    );

    let beta_y2_final_eval = beta_y2_local.multilinear_final_eval();
    let pos_beta_y_final_eval = pos_beta_y_local.multilinear_final_eval();
    let neg_beta_y_final_eval = neg_beta_y_local.multilinear_final_eval();

    // dbg!(
    //     &beta_y2_final_eval,
    //     &pos_beta_y_final_eval,
    //     &neg_beta_y_final_eval
    // );

    // dbg!(
    //     &opening.leader_evals.beta_x2_eval,
    //     &opening.leader_evals.pos_beta_x_evals[0],
    //     &opening.leader_evals.neg_beta_x_evals[0]
    // );

    if beta_y2_final_eval != opening.leader_evals.beta_x2_eval {
        return false;
    }
    if pos_beta_y_final_eval != opening.leader_evals.pos_beta_x_evals[0] {
        return false;
    }
    if neg_beta_y_final_eval != opening.leader_evals.neg_beta_x_evals[0] {
        return false;
    }

    let local_final_eval =
        HyperKZGLocalEvals::new_from_exported_evals(&opening.leader_evals, mpi_alphas, beta_y);
    if eval != local_final_eval.multilinear_final_eval() {
        return false;
    }

    opening.aggregated_evals.append_to_transcript(fs_transcript);
    opening.leader_evals.append_to_transcript(fs_transcript);

    // NOTE(HS) transcript MPI thing ...
    transcript_verifier_sync(fs_transcript, mpi_world_size);

    let gamma = fs_transcript.generate_field_element::<E::Fr>();

    // dbg!(gamma);

    let aggregated_oracle_commitment: E::G1Affine = {
        let gamma_power_series = powers_series(&gamma, local_alphas.len() + mpi_alphas.len() + 1);

        let com_g1: E::G1 = izip!(
            iter::once(&commitment).chain(&opening.folded_oracle_commitments),
            &gamma_power_series
        )
        .map(|(com, g)| com.to_curve() * g)
        .sum();

        com_g1.into()
    };

    // NOTE(HS) aggregate lagrange degree 2 polys
    let (y_beta2, y_beta, y_neg_beta) = {
        let gamma_n = gamma.pow_vartime([local_alphas.len() as u64]);
        let (v_beta2, v_beta, v_neg_beta) = local_final_eval.gamma_aggregate_evals(gamma);

        (v_beta2 * gamma_n, v_beta * gamma_n, v_neg_beta * gamma_n)
    };

    let mut aggregated_beta_y2_locals =
        beta_y2_local.interpolate_degree2_aggregated_evals(beta_x, gamma);
    aggregated_beta_y2_locals[0] += y_beta2;

    let mut aggregated_pos_beta_y_locals =
        pos_beta_y_local.interpolate_degree2_aggregated_evals(beta_x, gamma);
    aggregated_pos_beta_y_locals[0] += y_beta;

    let mut aggregated_neg_beta_y_locals =
        neg_beta_y_local.interpolate_degree2_aggregated_evals(beta_x, gamma);
    aggregated_neg_beta_y_locals[0] += y_neg_beta;

    fs_transcript.append_u8_slice(opening.beta_x_commitment.to_bytes().as_ref());

    // NOTE(HS) transcript MPI thing ...
    transcript_verifier_sync(fs_transcript, mpi_world_size);

    let delta_x = fs_transcript.generate_field_element::<E::Fr>();

    // dbg!(delta_x);

    let delta_x_pow_series = powers_series(&delta_x, 3);
    let at_beta_y2 = univariate_evaluate(&aggregated_beta_y2_locals, &delta_x_pow_series);
    let at_beta_y = univariate_evaluate(&aggregated_pos_beta_y_locals, &delta_x_pow_series);
    let at_neg_beta_y = univariate_evaluate(&aggregated_neg_beta_y_locals, &delta_x_pow_series);

    // dbg!(at_beta_y2, at_beta_y, at_neg_beta_y);

    let lagrange_degree2_delta_y = coeff_form_degree2_lagrange(
        [beta_y, -beta_y, beta_y * beta_y],
        [at_beta_y, at_neg_beta_y, at_beta_y2],
    );

    // dbg!(lagrange_degree2_delta_y);

    fs_transcript.append_u8_slice(opening.beta_y_commitment.to_bytes().as_ref());

    // NOTE(HS) transcript MPI thing ...
    transcript_verifier_sync(fs_transcript, mpi_world_size);

    let delta_y = fs_transcript.generate_field_element::<E::Fr>();

    // dbg!(delta_y);

    let delta_y_pow_series = powers_series(&delta_y, 3);
    let degree_2_final_eval = univariate_evaluate(&lagrange_degree2_delta_y, &delta_y_pow_series);

    // dbg!(degree_2_final_eval);

    // NOTE(HS) f_gamma_s - (delta_x - beta_x) ... (delta_x - beta_x2) f_gamma_quotient_s
    //                    - (delta_y - beta_y) ... (delta_y - beta_y2) lagrange_quotient_y
    let delta_x_denom = (delta_x - beta_x) * (delta_x - beta_x * beta_x) * (delta_x + beta_x);
    let delta_y_denom = (delta_y - beta_y) * (delta_y - beta_y * beta_y) * (delta_y + beta_y);

    let com_r = aggregated_oracle_commitment.to_curve()
        - (opening.beta_x_commitment * delta_x_denom)
        - (opening.beta_y_commitment * delta_y_denom);

    // dbg!(com_r);

    let final_opening = BiKZGProof {
        quotient_x: opening.quotient_delta_x_commitment,
        quotient_y: opening.quotient_delta_y_commitment,
    };

    coeff_form_bi_kzg_verify(
        vk.clone(),
        com_r.to_affine(),
        delta_x,
        delta_y,
        degree_2_final_eval,
        final_opening,
    )
}
