use std::marker::PhantomData;

use arith::{ExtensionField, Field, SimdField};
use gkr_engine::{StructuredReferenceString, Transcript};
use polynomials::{MultiLinearPoly, MultilinearExtension, RefMultiLinearPoly};

use crate::{
    orion::{
        simd_field_impl::{orion_commit_simd_field, orion_open_simd_field},
        verify::orion_verify,
        OrionCommitment, OrionProof, OrionSRS, OrionScratchPad, ORION_CODE_PARAMETER_INSTANCE,
    },
    PolynomialCommitmentScheme,
};

impl StructuredReferenceString for OrionSRS {
    type PKey = OrionSRS;
    type VKey = OrionSRS;

    fn into_keys(self) -> (Self::PKey, Self::VKey) {
        (self.clone(), self.clone())
    }
}

#[inline(always)]
fn pack_from_base<F, PackF>(es: &[F]) -> Vec<PackF>
where
    F: Field,
    PackF: SimdField<Scalar = F>,
{
    // NOTE: SIMD pack neighboring base field evals
    es.chunks(PackF::PACK_SIZE).map(PackF::pack).collect()
}

pub struct OrionBaseFieldPCS<F, EvalF, ComPackF, OpenPackF>
where
    F: Field,
    EvalF: ExtensionField<BaseField = F>,
    ComPackF: SimdField<Scalar = F>,
    OpenPackF: SimdField<Scalar = F>,
{
    _marker_f: PhantomData<F>,
    _marker_eval_f: PhantomData<EvalF>,
    _marker_commit_f: PhantomData<ComPackF>,
    _marker_open_f: PhantomData<OpenPackF>,
}

impl<F, EvalF, ComPackF, OpenPackF> PolynomialCommitmentScheme<EvalF>
    for OrionBaseFieldPCS<F, EvalF, ComPackF, OpenPackF>
where
    F: Field,
    EvalF: ExtensionField<BaseField = F>,
    ComPackF: SimdField<Scalar = F>,
    OpenPackF: SimdField<Scalar = F>,
{
    const NAME: &'static str = "OrionBaseFieldPCS";

    type Params = usize;
    type Poly = MultiLinearPoly<F>;
    type EvalPoint = Vec<EvalF>;
    type ScratchPad = OrionScratchPad;

    type SRS = OrionSRS;
    type Commitment = OrionCommitment;
    type Opening = OrionProof<EvalF>;

    fn gen_srs_for_testing(params: &Self::Params, rng: impl rand::RngCore) -> (Self::SRS, usize) {
        OrionSRS::from_random(
            1,
            *params,
            F::FIELD_SIZE,
            ComPackF::PACK_SIZE,
            ORION_CODE_PARAMETER_INSTANCE,
            rng,
        )
    }

    fn init_scratch_pad(_params: &Self::Params) -> Self::ScratchPad {
        OrionScratchPad::default()
    }

    fn commit(
        params: &Self::Params,
        pk: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        scratch_pad: &mut Self::ScratchPad,
    ) -> Self::Commitment {
        assert_eq!(*params, pk.num_vars);
        assert_eq!(poly.hypercube_size() % OpenPackF::PACK_SIZE, 0);

        let packed_evals: Vec<OpenPackF> = pack_from_base(poly.hypercube_basis_ref());
        let simd_poly = RefMultiLinearPoly::from_ref(&packed_evals);

        orion_commit_simd_field::<_, OpenPackF, ComPackF>(pk, &simd_poly, scratch_pad).unwrap()
    }

    fn open(
        params: &Self::Params,
        pk: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        x: &Self::EvalPoint,
        scratch_pad: &Self::ScratchPad,
        transcript: &mut impl Transcript,
    ) -> (EvalF, Self::Opening) {
        assert_eq!(*params, pk.num_vars);
        assert_eq!(poly.hypercube_size() % OpenPackF::PACK_SIZE, 0);

        let packed_evals: Vec<OpenPackF> = pack_from_base(poly.hypercube_basis_ref());
        let simd_poly = RefMultiLinearPoly::from_ref(&packed_evals);

        orion_open_simd_field::<_, OpenPackF, _, ComPackF>(
            pk,
            &simd_poly,
            x,
            transcript,
            scratch_pad,
        )
    }

    fn verify(
        params: &Self::Params,
        vk: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        x: &Self::EvalPoint,
        v: EvalF,
        opening: &Self::Opening,
        transcript: &mut impl Transcript,
    ) -> bool {
        assert_eq!(*params, vk.num_vars);
        orion_verify::<_, OpenPackF, _, ComPackF>(vk, commitment, x, &[], v, transcript, opening)
    }
}

pub struct OrionSIMDFieldPCS<F, SimdF, EvalF, ComPackF>
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    EvalF: ExtensionField<BaseField = F>,
    ComPackF: SimdField<Scalar = F>,
{
    _marker_f: PhantomData<F>,
    _marker_simd_f: PhantomData<SimdF>,
    _marker_eval_f: PhantomData<EvalF>,
    _marker_commit_f: PhantomData<ComPackF>,
}

impl<F, SimdF, EvalF, ComPackF> PolynomialCommitmentScheme<EvalF>
    for OrionSIMDFieldPCS<F, SimdF, EvalF, ComPackF>
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    EvalF: ExtensionField<BaseField = F>,
    ComPackF: SimdField<Scalar = F>,
{
    const NAME: &'static str = "OrionSIMDFieldPCS";

    type Params = usize;
    type Poly = MultiLinearPoly<SimdF>;
    type EvalPoint = Vec<EvalF>;
    type ScratchPad = OrionScratchPad;

    type SRS = OrionSRS;
    type Commitment = OrionCommitment;
    type Opening = OrionProof<EvalF>;

    // NOTE: here we say the number of variables is the sum of 2 following things:
    // - number of variables of the multilinear polynomial
    // - number of variables reside in the SIMD field - e.g., 3 vars for a SIMD 8 field
    fn gen_srs_for_testing(params: &Self::Params, rng: impl rand::RngCore) -> (Self::SRS, usize) {
        OrionSRS::from_random(
            1,
            *params,
            F::FIELD_SIZE,
            ComPackF::PACK_SIZE,
            ORION_CODE_PARAMETER_INSTANCE,
            rng,
        )
    }

    fn init_scratch_pad(_params: &Self::Params) -> Self::ScratchPad {
        OrionScratchPad::default()
    }

    fn commit(
        params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        scratch_pad: &mut Self::ScratchPad,
    ) -> Self::Commitment {
        assert_eq!(*params, proving_key.num_vars);
        assert_eq!(
            poly.get_num_vars(),
            proving_key.num_vars - SimdF::PACK_SIZE.ilog2() as usize
        );
        orion_commit_simd_field::<_, SimdF, ComPackF>(proving_key, poly, scratch_pad).unwrap()
    }

    fn open(
        params: &Self::Params,
        proving_key: &<Self::SRS as StructuredReferenceString>::PKey,
        poly: &Self::Poly,
        x: &Self::EvalPoint,
        scratch_pad: &Self::ScratchPad,
        transcript: &mut impl Transcript,
    ) -> (EvalF, Self::Opening) {
        assert_eq!(*params, proving_key.num_vars);
        assert_eq!(
            poly.get_num_vars(),
            proving_key.num_vars - SimdF::PACK_SIZE.ilog2() as usize
        );
        orion_open_simd_field::<F, SimdF, EvalF, ComPackF>(
            proving_key,
            poly,
            x,
            transcript,
            scratch_pad,
        )
    }

    fn verify(
        params: &Self::Params,
        vk: &<Self::SRS as StructuredReferenceString>::VKey,
        commitment: &Self::Commitment,
        x: &Self::EvalPoint,
        v: EvalF,
        opening: &Self::Opening,
        transcript: &mut impl Transcript,
    ) -> bool {
        assert_eq!(*params, vk.num_vars);
        assert_eq!(x.len(), vk.num_vars);
        orion_verify::<_, SimdF, _, ComPackF>(vk, commitment, x, &[], v, transcript, opening)
    }
}
