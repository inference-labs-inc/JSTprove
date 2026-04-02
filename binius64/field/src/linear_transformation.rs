// Copyright 2024-2025 Irreducible Inc.

use std::{iter, marker::PhantomData, ops::BitXor};

use rand::RngCore;

use crate::{
    BinaryField, BinaryField1b, ExtensionField, UnderlierWithBitOps, WithUnderlier,
    packed::PackedBinaryField,
    underlier::{Divisible, UnderlierType},
};

/// Generic transformation trait that is used both for scalars and packed fields
pub trait Transformation<Input, Output>: Sync {
    fn transform(&self, data: &Input) -> Output;
}

/// An $\mathbb{F}_2$-linear transformation on binary fields.
///
/// Stores transposed transformation matrix as a collection of field elements. `Data` is a generic
/// parameter because we want to be able both to have const instances that reference static arrays
/// and owning vector elements.
#[derive(Debug, Clone)]
pub struct FieldLinearTransformation<OF: BinaryField, Data: AsRef<[OF]> + Sync = &'static [OF]> {
    bases: Data,
    _pd: PhantomData<OF>,
}

impl<OF: BinaryField> FieldLinearTransformation<OF, &'static [OF]> {
    pub const fn new_const(bases: &'static [OF]) -> Self {
        assert!(bases.len() == OF::DEGREE);

        Self {
            bases,
            _pd: PhantomData,
        }
    }
}

impl<OF: BinaryField, Data: AsRef<[OF]> + Sync> FieldLinearTransformation<OF, Data> {
    pub fn new(bases: Data) -> Self {
        debug_assert_eq!(bases.as_ref().len(), OF::DEGREE);

        Self {
            bases,
            _pd: PhantomData,
        }
    }

    pub fn bases(&self) -> &[OF] {
        self.bases.as_ref()
    }
}

impl<IF: BinaryField, OF: BinaryField, Data: AsRef<[OF]> + Sync> Transformation<IF, OF>
    for FieldLinearTransformation<OF, Data>
{
    fn transform(&self, data: &IF) -> OF {
        assert_eq!(IF::DEGREE, OF::DEGREE);

        ExtensionField::<BinaryField1b>::iter_bases(data)
            .zip(self.bases.as_ref().iter())
            .fold(OF::ZERO, |acc, (scalar, &basis_elem)| {
                acc + basis_elem * scalar
            })
    }
}

impl<OF: BinaryField> FieldLinearTransformation<OF, Vec<OF>> {
    pub fn random(mut rng: impl RngCore) -> Self {
        Self {
            bases: (0..OF::DEGREE).map(|_| OF::random(&mut rng)).collect(),
            _pd: PhantomData,
        }
    }
}

const LOG_BITS_PER_BYTE: usize = 3;
const BITS_PER_BYTE: usize = 1 << LOG_BITS_PER_BYTE;

/// Linear transformation using precomputed byte-indexed lookup tables.
///
/// This implementation uses the [Method of Four Russians] to optimize the computation by
/// precomputing lookup tables for each byte position and using bitwise chunks of the words.
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
#[derive(Debug)]
pub struct BytewiseLookupTransformation<UIn, UOut> {
    lookup: Vec<[UOut; 1 << BITS_PER_BYTE]>,
    _uin_marker: PhantomData<UIn>,
}

impl<UIn, UOut> BytewiseLookupTransformation<UIn, UOut>
where
    UIn: UnderlierType + Divisible<u8>,
    UOut: UnderlierWithBitOps,
{
    pub fn new(cols: &[UOut]) -> Self {
        assert!(LOG_BITS_PER_BYTE <= UIn::LOG_BITS);
        assert_eq!(cols.len(), UIn::BITS);

        let lookup = cols
            .chunks(BITS_PER_BYTE)
            .map(|cols| {
                let cols: [_; BITS_PER_BYTE] = cols.try_into().expect(
                    "chunk size is BITS_PER_BYTE; \
					cols.len() is a multiple of BITS_PER_BYTE",
                );
                expand_subset_xors(cols)
            })
            .collect();

        Self {
            lookup,
            _uin_marker: PhantomData,
        }
    }
}

impl<UIn, UOut> Transformation<UIn, UOut> for BytewiseLookupTransformation<UIn, UOut>
where
    UIn: UnderlierType + Divisible<u8>,
    UOut: UnderlierWithBitOps,
{
    fn transform(&self, data: &UIn) -> UOut {
        Divisible::<u8>::ref_iter(data)
            .enumerate()
            .take(1 << (UIn::LOG_BITS - LOG_BITS_PER_BYTE))
            .map(|(i, byte)| {
                // Safety:
                // - lookup.len() == 2^(UIn::LOG_BITS - LOG_BITS_PER_BYTE) by struct invariant
                // - take limits iteration calls to 2^(UIn::LOG_BITS - LOG_BITS_PER_BYTE)
                let lookup = unsafe { self.lookup.get_unchecked(i) };
                lookup[byte as usize]
            })
            .reduce(BitXor::bitxor)
            .unwrap_or(UOut::ZERO)
    }
}

fn expand_subset_xors<U: UnderlierWithBitOps, const N: usize, const N_EXP2: usize>(
    elems: [U; N],
) -> [U; N_EXP2] {
    assert_eq!(N_EXP2, 1 << N);

    let mut expanded = [U::ZERO; N_EXP2];
    for (i, elem_i) in elems.into_iter().enumerate() {
        let span = &mut expanded[..1 << (i + 1)];
        let (lo_half, hi_half) = span.split_at_mut(1 << i);
        for (lo_half_i, hi_half_i) in iter::zip(lo_half, hi_half) {
            *hi_half_i = *lo_half_i ^ elem_i;
        }
    }
    expanded
}

/// Factory for creating bytewise lookup transformations.
#[derive(Debug)]
pub struct BytewiseLookupTransformationFactory;

/// Factory trait for creating linear transformations from column data.
pub trait LinearTransformationFactory<Input, Output> {
    type Transform: Transformation<Input, Output>;

    fn create(&self, cols: &[Output]) -> Self::Transform;
}

impl<UIn, UOut> LinearTransformationFactory<UIn, UOut> for BytewiseLookupTransformationFactory
where
    UIn: UnderlierType + Divisible<u8>,
    UOut: UnderlierWithBitOps,
{
    type Transform = BytewiseLookupTransformation<UIn, UOut>;

    fn create(&self, cols: &[UOut]) -> Self::Transform {
        BytewiseLookupTransformation::new(cols)
    }
}

/// Wraps a transformation on underliers to operate on types with underliers.
#[derive(Debug)]
pub struct OutputWrappingTransformation<Inner, Input, Output> {
    inner: Inner,
    _marker: PhantomData<(Input, Output)>,
}

impl<Inner, Input, Output> Transformation<Input, Output>
    for OutputWrappingTransformation<Inner, Input, Output>
where
    Inner: Transformation<Input, Output::Underlier>,
    Input: Sync,
    Output: WithUnderlier,
{
    #[inline]
    fn transform(&self, data: &Input) -> Output {
        Output::from_underlier(self.inner.transform(data))
    }
}

/// Factory that wraps an underlier transformation factory to work with types that have underliers.
#[derive(Debug)]
pub struct OutputWrappingTransformationFactory<Inner, Input, Output> {
    inner: Inner,
    _marker: PhantomData<(Input, Output)>,
}

impl<Inner, Input, Output> OutputWrappingTransformationFactory<Inner, Input, Output>
where
    Inner: LinearTransformationFactory<Input, Output::Underlier>,
    Input: Sync,
    Output: WithUnderlier,
{
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<Inner, Input, Output> LinearTransformationFactory<Input, Output>
    for OutputWrappingTransformationFactory<Inner, Input, Output>
where
    Inner: LinearTransformationFactory<Input, Output::Underlier>,
    Input: Sync,
    Output: WithUnderlier,
{
    type Transform = OutputWrappingTransformation<Inner::Transform, Input, Output>;

    #[inline]
    fn create(&self, cols: &[Output]) -> Self::Transform {
        OutputWrappingTransformation {
            inner: self.inner.create(Output::to_underliers_ref(cols)),
            _marker: PhantomData,
        }
    }
}

/// Wraps a transformation on underliers to accept inputs with underliers.
#[derive(Debug)]
pub struct InputWrappingTransformation<Inner, Input, Output> {
    inner: Inner,
    _marker: PhantomData<(Input, Output)>,
}

impl<Inner, Input, Output> Transformation<Input, Output>
    for InputWrappingTransformation<Inner, Input, Output>
where
    Inner: Transformation<Input::Underlier, Output>,
    Input: WithUnderlier,
    Output: Sync,
{
    #[inline]
    fn transform(&self, data: &Input) -> Output {
        self.inner.transform(&data.to_underlier())
    }
}

/// Factory that wraps an underlier transformation factory to accept inputs with underliers.
#[derive(Debug)]
pub struct InputWrappingTransformationFactory<Inner, Input, Output> {
    inner: Inner,
    _marker: PhantomData<(Input, Output)>,
}

impl<Inner, Input, Output> InputWrappingTransformationFactory<Inner, Input, Output>
where
    Inner: LinearTransformationFactory<Input::Underlier, Output>,
    Input: WithUnderlier,
    Output: Sync,
{
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }
}

impl<Inner, Input, Output> LinearTransformationFactory<Input, Output>
    for InputWrappingTransformationFactory<Inner, Input, Output>
where
    Inner: LinearTransformationFactory<Input::Underlier, Output>,
    Input: WithUnderlier,
    Output: Sync,
{
    type Transform = InputWrappingTransformation<Inner::Transform, Input, Output>;

    #[inline]
    fn create(&self, cols: &[Output]) -> Self::Transform {
        InputWrappingTransformation {
            inner: self.inner.create(cols),
            _marker: PhantomData,
        }
    }
}

/// Transformation that wraps both input and output, converting between types with underliers.
pub type WrappingTransformation<Inner, Input, Output> = OutputWrappingTransformation<
    InputWrappingTransformation<Inner, Input, <Output as WithUnderlier>::Underlier>,
    Input,
    Output,
>;

pub struct IDTransformation;

impl<OP: PackedBinaryField> Transformation<OP, OP> for IDTransformation {
    fn transform(&self, data: &OP) -> OP {
        *data
    }
}
