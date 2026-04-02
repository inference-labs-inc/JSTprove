// Copyright 2023-2025 Irreducible Inc.

use crate::{
    BinaryField, ExtensionField, Field, PackedField, arch::PackedPrimitiveType,
    underlier::WithUnderlier,
};

/// Trait represents a relationship between a packed struct of field elements and a packed struct
/// of elements from an extension field.
///
/// This trait guarantees that one packed type has the same
/// memory representation as the other, differing only in the scalar type and preserving the order
/// of smaller elements.
///
/// This trait relation guarantees that the following iterators yield the same sequence of scalar
/// elements:
///
/// ```
/// use binius_field::{ExtensionField, PackedExtension, PackedField, Field};
///
/// fn ext_then_bases<'a, F, PE>(packed: &'a PE) -> impl Iterator<Item=F> + 'a
///     where
///         PE: PackedField<Scalar: ExtensionField<F>>,
///         F: Field,
/// {
///     packed.iter().flat_map(|ext| ext.into_iter_bases())
/// }
///
/// fn cast_then_iter<'a, F, PE>(packed: &'a PE) -> impl Iterator<Item=F> + 'a
///     where
///         PE: PackedExtension<F>,
///         F: Field,
/// {
///     PE::cast_base_ref(packed).into_iter()
/// }
/// ```
///
/// # Safety
///
/// In order for the above relation to be guaranteed, the memory representation of
/// `PackedExtensionField` element must be the same as a slice of the underlying `PackedField`
/// element.
pub trait PackedExtension<FS: Field>: PackedField<Scalar: ExtensionField<FS>> {
    type PackedSubfield: PackedField<Scalar = FS>;

    fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield];
    fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield];

    fn cast_exts(packed: &[Self::PackedSubfield]) -> &[Self];
    fn cast_exts_mut(packed: &mut [Self::PackedSubfield]) -> &mut [Self];

    fn cast_base(self) -> Self::PackedSubfield;
    fn cast_base_ref(&self) -> &Self::PackedSubfield;
    fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield;

    fn cast_ext(base: Self::PackedSubfield) -> Self;
    fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self;
    fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self;

    #[inline(always)]
    fn cast_base_arr<const N: usize>(packed: [Self; N]) -> [Self::PackedSubfield; N] {
        packed.map(Self::cast_base)
    }

    #[inline(always)]
    fn cast_base_arr_ref<const N: usize>(packed: &[Self; N]) -> &[Self::PackedSubfield; N] {
        Self::cast_bases(packed)
            .try_into()
            .expect("array has size N")
    }

    #[inline(always)]
    fn cast_base_arr_mut<const N: usize>(packed: &mut [Self; N]) -> &mut [Self::PackedSubfield; N] {
        Self::cast_bases_mut(packed)
            .try_into()
            .expect("array has size N")
    }

    #[inline(always)]
    fn cast_ext_arr<const N: usize>(packed: [Self::PackedSubfield; N]) -> [Self; N] {
        packed.map(Self::cast_ext)
    }

    #[inline(always)]
    fn cast_ext_arr_ref<const N: usize>(packed: &[Self::PackedSubfield; N]) -> &[Self; N] {
        Self::cast_exts(packed)
            .try_into()
            .expect("array has size N")
    }

    #[inline(always)]
    fn cast_ext_arr_mut<const N: usize>(packed: &mut [Self::PackedSubfield; N]) -> &mut [Self; N] {
        Self::cast_exts_mut(packed)
            .try_into()
            .expect("array has size N")
    }
}

impl<PT, FS> PackedExtension<FS> for PT
where
    FS: BinaryField,
    PT: PackedField<Scalar: ExtensionField<FS>> + WithUnderlier,
    PackedPrimitiveType<PT::Underlier, FS>: PackedField<Scalar = FS>,
{
    type PackedSubfield = PackedPrimitiveType<PT::Underlier, FS>;

    fn cast_bases(packed: &[Self]) -> &[Self::PackedSubfield] {
        Self::PackedSubfield::from_underliers_ref(Self::to_underliers_ref(packed))
    }

    fn cast_bases_mut(packed: &mut [Self]) -> &mut [Self::PackedSubfield] {
        Self::PackedSubfield::from_underliers_ref_mut(Self::to_underliers_ref_mut(packed))
    }

    fn cast_exts(base: &[Self::PackedSubfield]) -> &[Self] {
        Self::from_underliers_ref(Self::PackedSubfield::to_underliers_ref(base))
    }

    fn cast_exts_mut(base: &mut [Self::PackedSubfield]) -> &mut [Self] {
        Self::from_underliers_ref_mut(Self::PackedSubfield::to_underliers_ref_mut(base))
    }

    fn cast_base(self) -> Self::PackedSubfield {
        Self::PackedSubfield::from_underlier(self.to_underlier())
    }

    fn cast_base_ref(&self) -> &Self::PackedSubfield {
        Self::PackedSubfield::from_underlier_ref(self.to_underlier_ref())
    }

    fn cast_base_mut(&mut self) -> &mut Self::PackedSubfield {
        Self::PackedSubfield::from_underlier_ref_mut(self.to_underlier_ref_mut())
    }

    fn cast_ext(base: Self::PackedSubfield) -> Self {
        Self::from_underlier(base.to_underlier())
    }

    fn cast_ext_ref(base: &Self::PackedSubfield) -> &Self {
        Self::from_underlier_ref(base.to_underlier_ref())
    }

    fn cast_ext_mut(base: &mut Self::PackedSubfield) -> &mut Self {
        Self::from_underlier_ref_mut(base.to_underlier_ref_mut())
    }
}

/// Convenient type alias that returns the packed field type for the scalar field `F` and packed
/// extension `P`.
pub type PackedSubfield<P, F> = <P as PackedExtension<F>>::PackedSubfield;
