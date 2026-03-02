use derivative::Derivative;
use gkr_engine::StructuredReferenceString;
use halo2curves::{pairing::Engine, CurveAffine};
use serdes::{ExpSerde, SerdeResult};

use crate::{CoefFormUniKZGSRS, UniKZGVerifierParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Derivative)]
#[derivative(Default(bound = ""))]
pub struct BiKZGCommitment<E: Engine>(pub E::G1Affine)
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>;

// Derive macros does not work for associated types
impl<E: Engine> ExpSerde for BiKZGCommitment<E>
where
    E::G1Affine: ExpSerde + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn serialize_into<W: std::io::Write>(&self, writer: W) -> SerdeResult<()> {
        self.0.serialize_into(writer)
    }

    fn deserialize_from<R: std::io::Read>(reader: R) -> SerdeResult<Self> {
        Ok(Self(<E::G1Affine as ExpSerde>::deserialize_from(reader)?))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Derivative, ExpSerde)]
#[derivative(Default(bound = ""))]
pub struct CoefFormBiKZGLocalSRS<E: Engine>
where
    E::G1Affine: ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    pub tau_x_srs: CoefFormUniKZGSRS<E>,
    pub tau_y_srs: CoefFormUniKZGSRS<E>,
}

/// Bivariate KZG PCS verifier's params.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default, ExpSerde)]
pub struct BiKZGVerifierParam<E: Engine>
where
    E::G1Affine: ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    /// tau_x over G2.
    pub tau_x_g2: E::G2Affine,
    /// tau_y over G2.
    pub tau_y_g2: E::G2Affine,
}

impl<E: Engine> From<&CoefFormBiKZGLocalSRS<E>> for BiKZGVerifierParam<E>
where
    E::G1Affine: ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    fn from(srs: &CoefFormBiKZGLocalSRS<E>) -> Self {
        Self {
            tau_x_g2: srs.tau_x_srs.tau_g2,
            tau_y_g2: srs.tau_y_srs.tau_g2,
        }
    }
}

impl<E: Engine> StructuredReferenceString for CoefFormBiKZGLocalSRS<E>
where
    <E as Engine>::G1Affine: ExpSerde + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    <E as Engine>::G2Affine: ExpSerde + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2>,
{
    type PKey = CoefFormBiKZGLocalSRS<E>;
    type VKey = BiKZGVerifierParam<E>;

    fn into_keys(self) -> (Self::PKey, Self::VKey) {
        let vk: Self::VKey = From::from(&self);
        (self, vk)
    }
}

impl<E: Engine> From<&BiKZGVerifierParam<E>> for UniKZGVerifierParams<E>
where
    E::G1Affine: ExpSerde,
    E::G2Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G2> + ExpSerde,
{
    fn from(value: &BiKZGVerifierParam<E>) -> Self {
        Self {
            tau_g2: value.tau_x_g2,
        }
    }
}

/// Proof for Bi-KZG polynomial commitment scheme.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BiKZGProof<E: Engine> {
    pub quotient_x: E::G1Affine,
    pub quotient_y: E::G1Affine,
}
