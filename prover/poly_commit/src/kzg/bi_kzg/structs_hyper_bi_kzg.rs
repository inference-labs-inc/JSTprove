use derivative::Derivative;
use halo2curves::pairing::Engine;
use serdes::ExpSerde;

use crate::*;

#[derive(Debug, Clone, Derivative, ExpSerde)]
#[derivative(Default(bound = ""))]
pub struct HyperBiKZGOpening<E: Engine>
where
    E::Fr: ExpSerde,
    E::G1Affine: Default + ExpSerde,
{
    pub folded_oracle_commitments: Vec<E::G1Affine>,

    pub aggregated_evals: HyperKZGAggregatedEvals<E>,
    pub leader_evals: HyperKZGExportedLocalEvals<E>,

    pub beta_x_commitment: E::G1Affine,
    pub beta_y_commitment: E::G1Affine,

    pub quotient_delta_x_commitment: E::G1Affine,
    pub quotient_delta_y_commitment: E::G1Affine,
}

impl<E: Engine> From<HyperBiKZGOpening<E>> for HyperUniKZGOpening<E>
where
    E::Fr: ExpSerde,
    E::G1Affine: Default + ExpSerde,
{
    fn from(value: HyperBiKZGOpening<E>) -> Self {
        Self {
            folded_oracle_commitments: value.folded_oracle_commitments,
            evals_at_x: value.leader_evals,
            beta_x_commitment: value.beta_x_commitment,
            quotient_delta_x_commitment: value.quotient_delta_x_commitment,
        }
    }
}

impl<E: Engine> From<HyperUniKZGOpening<E>> for HyperBiKZGOpening<E>
where
    E::Fr: ExpSerde,
    E::G1Affine: Default + ExpSerde,
{
    fn from(value: HyperUniKZGOpening<E>) -> Self {
        Self {
            folded_oracle_commitments: value.folded_oracle_commitments,
            leader_evals: value.evals_at_x,
            beta_x_commitment: value.beta_x_commitment,
            quotient_delta_x_commitment: value.quotient_delta_x_commitment,

            ..Default::default()
        }
    }
}
