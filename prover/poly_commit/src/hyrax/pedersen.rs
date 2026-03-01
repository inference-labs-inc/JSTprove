use gkr_engine::StructuredReferenceString;
use halo2curves::{
    ff::{Field, PrimeField},
    group::Curve,
    msm, CurveAffine,
};
use serdes::ExpSerde;

#[derive(Clone, Debug, Default)]
pub struct PedersenParams<C>
where
    C: CurveAffine,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    pub bases: Vec<C>,
    pub pre_bases: Vec<C::Curve>,
}

impl<C> ExpSerde for PedersenParams<C>
where
    C: CurveAffine,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> serdes::SerdeResult<()> {
        // basis
        {
            self.bases.len().serialize_into(&mut writer)?;
            for curve_element in self.bases.iter() {
                // We want to write both x and y coordinates.
                let coord = curve_element.coordinates().unwrap();
                // todo: this incurrs a cost to convert from Montogomery form.
                coord.x().to_repr().serialize_into(&mut writer)?;
                coord.y().to_repr().serialize_into(&mut writer)?;
            }
        }
        // pre-computed bases
        {
            self.pre_bases.len().serialize_into(&mut writer)?;

            let mut normalized_bases = vec![C::default(); self.pre_bases.len()];
            C::Curve::batch_normalize(&self.pre_bases, &mut normalized_bases);

            for curve_element in normalized_bases.iter() {
                // We want to write both x and y coordinates.
                let coord = curve_element.coordinates().unwrap();
                // todo: this incurrs a cost to convert from Montogomery form.
                coord.x().to_repr().serialize_into(&mut writer)?;
                coord.y().to_repr().serialize_into(&mut writer)?;
            }
        }
        Ok(())
    }

    fn deserialize_from<R: std::io::Read>(mut reader: R) -> serdes::SerdeResult<Self> {
        let mut buf = [0u8; 32];

        // bases
        let bases = {
            let bases_len = usize::deserialize_from(&mut reader)?;
            let mut bases = Vec::with_capacity(bases_len);
            for _ in 0..bases_len {
                reader.read_exact(&mut buf)?;
                let x = C::Base::from_repr(buf).unwrap();
                reader.read_exact(&mut buf)?;
                let y = C::Base::from_repr(buf).unwrap();
                bases.push(C::from_xy(x, y).unwrap())
            }
            bases
        };

        // pre-computed bases
        let pre_bases = {
            let pre_bases_len = usize::deserialize_from(&mut reader)?;
            let mut pre_bases = Vec::with_capacity(pre_bases_len);
            for _ in 0..pre_bases_len {
                reader.read_exact(&mut buf)?;
                let x = C::Base::from_repr(buf).unwrap();
                reader.read_exact(&mut buf)?;
                let y = C::Base::from_repr(buf).unwrap();
                pre_bases.push(C::from_xy(x, y).unwrap().to_curve())
            }
            pre_bases
        };

        Ok(Self { bases, pre_bases })
    }
}

impl<C> StructuredReferenceString for PedersenParams<C>
where
    C: CurveAffine,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    type PKey = Self;
    type VKey = Self;

    fn into_keys(self) -> (Self::PKey, Self::VKey) {
        (self.clone(), self)
    }
}

pub(crate) fn pedersen_setup<C>(length: usize, mut rng: impl rand::RngCore) -> PedersenParams<C>
where
    C: CurveAffine,
    C::Scalar: PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let proj_bases: Vec<C::Curve> = (0..length)
        .map(|_| {
            let scalar = C::Scalar::random(&mut rng);
            C::generator() * scalar
        })
        .collect();

    let mut bases = vec![C::default(); length];
    C::Curve::batch_normalize(&proj_bases, &mut bases);
    let pre_bases = msm::multiexp_precompute(&bases, 12);

    PedersenParams { bases, pre_bases }
}

pub(crate) fn pedersen_commit<C>(params: &PedersenParams<C>, coeffs: &[C::Scalar]) -> C
where
    C: CurveAffine,
    C::Scalar: PrimeField,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    let mut what = C::default().to_curve();

    msm::multiexp_precompute_serial::<C>(coeffs, &params.pre_bases, 12, &mut what);

    what.to_affine()
}

impl<C> PedersenParams<C>
where
    C: CurveAffine,
    C::Base: PrimeField<Repr = [u8; 32]>,
{
    pub(crate) fn msm_len(&self) -> usize {
        self.bases.len()
    }
}
