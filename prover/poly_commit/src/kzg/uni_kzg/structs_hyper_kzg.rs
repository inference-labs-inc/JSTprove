use std::ops::{Index, IndexMut};

use arith::ExtensionField;
use derivative::Derivative;
use gkr_engine::Transcript;
use halo2curves::{ff::Field, pairing::Engine};
use itertools::izip;
use serdes::ExpSerde;

use crate::*;

#[derive(Clone, Debug, Derivative, ExpSerde)]
#[derivative(Default(bound = ""))]
pub struct HyperKZGExportedLocalEvals<E: Engine>
where
    E::Fr: ExpSerde,
{
    pub beta_x2_eval: E::Fr,
    pub pos_beta_x_evals: Vec<E::Fr>,
    pub neg_beta_x_evals: Vec<E::Fr>,
}

impl<E: Engine> HyperKZGExportedLocalEvals<E>
where
    E::Fr: ExpSerde,
{
    pub(crate) fn new(evals_num: usize) -> Self {
        Self {
            beta_x2_eval: E::Fr::default(),
            pos_beta_x_evals: vec![E::Fr::default(); evals_num],
            neg_beta_x_evals: vec![E::Fr::default(); evals_num],
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.pos_beta_x_evals.len() + self.neg_beta_x_evals.len() + 1
    }

    pub(crate) fn append_to_transcript<T>(&self, fs_transcript: &mut T)
    where
        T: Transcript,
        E::Fr: ExtensionField,
    {
        fs_transcript.append_field_element(&self.beta_x2_eval);
        izip!(&self.pos_beta_x_evals, &self.neg_beta_x_evals).for_each(
            |(beta_eval, neg_beta_eval)| {
                fs_transcript.append_field_element(beta_eval);
                fs_transcript.append_field_element(neg_beta_eval);
            },
        );
    }
}

#[derive(Clone, Debug, Derivative, ExpSerde)]
#[derivative(Default(bound = ""))]
pub struct HyperUniKZGOpening<E: Engine>
where
    E::Fr: ExpSerde,
    E::G1Affine: Default + ExpSerde,
{
    pub folded_oracle_commitments: Vec<E::G1Affine>,
    pub evals_at_x: HyperKZGExportedLocalEvals<E>,
    pub beta_x_commitment: E::G1Affine,
    pub quotient_delta_x_commitment: E::G1Affine,
}

#[derive(Clone, Debug, Derivative, ExpSerde)]
#[derivative(Default(bound = ""))]
pub(crate) struct HyperKZGLocalEvals<E: Engine>
where
    E::Fr: ExpSerde,
{
    pub(crate) beta2_evals: Vec<E::Fr>,
    pub(crate) pos_beta_evals: Vec<E::Fr>,
    pub(crate) neg_beta_evals: Vec<E::Fr>,
}

impl<E: Engine> HyperKZGLocalEvals<E>
where
    E::Fr: ExtensionField,
{
    pub(crate) fn new_from_beta2_evals(beta2_eval: E::Fr) -> Self {
        Self {
            beta2_evals: vec![beta2_eval],
            pos_beta_evals: Vec::new(),
            neg_beta_evals: Vec::new(),
        }
    }

    pub(crate) fn new_from_exported_evals(
        exported: &HyperKZGExportedLocalEvals<E>,
        alphas: &[E::Fr],
        beta: E::Fr,
    ) -> Self {
        let beta_inv = beta.invert().unwrap();
        let two_inv = E::Fr::ONE.double().invert().unwrap();

        let mut local_evals = Self::new_from_beta2_evals(exported.beta_x2_eval);

        izip!(
            &exported.pos_beta_x_evals,
            &exported.neg_beta_x_evals,
            alphas
        )
        .for_each(|(pos_beta_x_eval, neg_beta_x_eval, alpha)| {
            let beta2_eval = two_inv
                * ((*pos_beta_x_eval + *neg_beta_x_eval) * (E::Fr::ONE - alpha)
                    + (*pos_beta_x_eval - *neg_beta_x_eval) * beta_inv * alpha);

            local_evals.beta2_evals.push(beta2_eval);
            local_evals.pos_beta_evals.push(*pos_beta_x_eval);
            local_evals.neg_beta_evals.push(*neg_beta_x_eval);
        });

        local_evals
    }

    // NOTE(HS) introduce a gamma <- RO, s.t., we aggregate the evaluations
    // with the power series of gamma through linear combination.
    //
    // THE ASSUMPTION here is: pos/neg beta evals are from polynomial oracles
    // that are folded until degree being 1, i.e., 2 coefficients, while the
    // beta2 evals folds one step more, that the polynomial eventually has
    // degree 0, and thus we assume that beta2_evals is one element more than
    // pos/neg beta evals.
    //
    // The return order is the evals at beta2, beta, and -beta.
    pub(crate) fn gamma_aggregate_evals(&self, gamma: E::Fr) -> (E::Fr, E::Fr, E::Fr) {
        assert_eq!(self.pos_beta_evals.len(), self.neg_beta_evals.len());
        assert_eq!(self.pos_beta_evals.len() + 1, self.beta2_evals.len());

        let gamma_pow_series = powers_series(&gamma, self.pos_beta_evals.len());
        let v_beta2 = univariate_evaluate(&self.beta2_evals, &gamma_pow_series);
        let v_beta = univariate_evaluate(&self.pos_beta_evals, &gamma_pow_series);
        let v_neg_beta = univariate_evaluate(&self.neg_beta_evals, &gamma_pow_series);

        (v_beta2, v_beta, v_neg_beta)
    }

    pub(crate) fn interpolate_degree2_aggregated_evals(
        &self,
        beta: E::Fr,
        gamma: E::Fr,
    ) -> [E::Fr; 3] {
        let beta2 = beta * beta;
        let (v_beta2, v_beta, v_neg_beta) = self.gamma_aggregate_evals(gamma);
        coeff_form_degree2_lagrange([beta, -beta, beta2], [v_beta, v_neg_beta, v_beta2])
    }

    // NOTE(HS) the same assumption applies here, that last beta2 eval is the
    // multilinear polynomial eval, as it folds to univariate poly of degree 0.
    pub(crate) fn multilinear_final_eval(&self) -> E::Fr {
        self.beta2_evals[self.beta2_evals.len() - 1]
    }

    pub(crate) fn append_to_transcript<T>(&self, fs_transcript: &mut T)
    where
        T: Transcript,
    {
        fs_transcript.append_field_element(&self.beta2_evals[0]);
        izip!(&self.pos_beta_evals, &self.neg_beta_evals).for_each(|(beta_eval, neg_beta_eval)| {
            fs_transcript.append_field_element(beta_eval);
            fs_transcript.append_field_element(neg_beta_eval);
        });
    }
}

impl<E: Engine> Index<usize> for HyperKZGExportedLocalEvals<E>
where
    E::Fr: ExpSerde,
{
    type Output = E::Fr;

    fn index(&self, index: usize) -> &Self::Output {
        assert_eq!(self.pos_beta_x_evals.len(), self.neg_beta_x_evals.len());
        assert!(!self.pos_beta_x_evals.is_empty());

        let evals_len = self.pos_beta_x_evals.len();

        if index < evals_len {
            &self.pos_beta_x_evals[index]
        } else if index < 2 * evals_len {
            &self.neg_beta_x_evals[index - evals_len]
        } else if index == 2 * evals_len {
            &self.beta_x2_eval
        } else {
            unreachable!()
        }
    }
}

impl<E: Engine> IndexMut<usize> for HyperKZGExportedLocalEvals<E>
where
    E::Fr: ExpSerde,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert_eq!(self.pos_beta_x_evals.len(), self.neg_beta_x_evals.len());
        assert!(!self.pos_beta_x_evals.is_empty());

        let evals_len = self.pos_beta_x_evals.len();

        if index < evals_len {
            &mut self.pos_beta_x_evals[index]
        } else if index < 2 * evals_len {
            &mut self.neg_beta_x_evals[index - evals_len]
        } else if index == 2 * evals_len {
            &mut self.beta_x2_eval
        } else {
            unreachable!()
        }
    }
}

impl<E: Engine> From<HyperKZGLocalEvals<E>> for HyperKZGExportedLocalEvals<E>
where
    E::Fr: ExpSerde,
{
    fn from(value: HyperKZGLocalEvals<E>) -> Self {
        Self {
            beta_x2_eval: value.beta2_evals[0],
            pos_beta_x_evals: value.pos_beta_evals,
            neg_beta_x_evals: value.neg_beta_evals,
        }
    }
}

#[derive(Clone, Debug, Derivative, ExpSerde)]
#[derivative(Default(bound = ""))]
pub struct HyperKZGAggregatedEvals<E: Engine>
where
    E::Fr: ExpSerde,
{
    pub beta_y2_evals: HyperKZGExportedLocalEvals<E>,
    pub pos_beta_y_evals: HyperKZGExportedLocalEvals<E>,
    pub neg_beta_y_evals: HyperKZGExportedLocalEvals<E>,
}

impl<E: Engine> HyperKZGAggregatedEvals<E>
where
    E::Fr: ExtensionField,
{
    pub(crate) fn new_from_exported_evals(
        exported_evals: &[HyperKZGExportedLocalEvals<E>],
        beta_y: E::Fr,
    ) -> Self {
        let evals_len = exported_evals[0].pos_beta_x_evals.len();
        let num_local_evals = exported_evals[0].len();
        let num_parties = exported_evals.len();

        assert!(num_parties >= 2 && num_parties.is_power_of_two());

        let mut aggregated = Self {
            beta_y2_evals: HyperKZGExportedLocalEvals::new(evals_len),
            pos_beta_y_evals: HyperKZGExportedLocalEvals::new(evals_len),
            neg_beta_y_evals: HyperKZGExportedLocalEvals::new(evals_len),
        };

        let beta_y2 = beta_y * beta_y;
        let beta_y2_pow_series = powers_series(&beta_y2, num_parties);
        let pos_beta_y_pow_series = powers_series(&beta_y, num_parties);
        let neg_beta_y_pow_series = powers_series(&(-beta_y), num_parties);

        (0..num_local_evals).for_each(|i| {
            let y_poly: Vec<E::Fr> = exported_evals.iter().map(|e| e[i]).collect();

            aggregated.beta_y2_evals[i] = univariate_evaluate(&y_poly, &beta_y2_pow_series);
            aggregated.pos_beta_y_evals[i] = univariate_evaluate(&y_poly, &pos_beta_y_pow_series);
            aggregated.neg_beta_y_evals[i] = univariate_evaluate(&y_poly, &neg_beta_y_pow_series);
        });

        aggregated
    }

    pub(crate) fn append_to_transcript<T>(&self, fs_transcript: &mut T)
    where
        T: Transcript,
    {
        self.beta_y2_evals.append_to_transcript(fs_transcript);
        self.pos_beta_y_evals.append_to_transcript(fs_transcript);
        self.neg_beta_y_evals.append_to_transcript(fs_transcript);
    }
}
