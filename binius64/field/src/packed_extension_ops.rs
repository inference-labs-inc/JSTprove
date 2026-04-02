// Copyright 2024-2025 Irreducible Inc.

use binius_utils::rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{ExtensionField, Field, PackedExtension, PackedField};

pub fn ext_base_mul<PE: PackedExtension<F>, F: Field>(lhs: &mut [PE], rhs: &[PE::PackedSubfield]) {
    ext_base_op(lhs, rhs, |_, lhs, broadcasted_rhs| {
        PE::cast_ext(lhs.cast_base() * broadcasted_rhs)
    })
}

pub fn ext_base_mul_par<PE: PackedExtension<F>, F: Field>(
    lhs: &mut [PE],
    rhs: &[PE::PackedSubfield],
) {
    ext_base_op_par(lhs, rhs, |_, lhs, broadcasted_rhs| {
        PE::cast_ext(lhs.cast_base() * broadcasted_rhs)
    })
}

/// # Safety
///
/// Width of PackedSubfield is >= the width of the field implementing PackedExtension.
pub unsafe fn get_packed_subfields_at_pe_idx<PE: PackedExtension<F>, F: Field>(
    packed_subfields: &[PE::PackedSubfield],
    i: usize,
) -> PE::PackedSubfield {
    let bottom_most_scalar_idx = i * PE::WIDTH;
    let bottom_most_scalar_idx_in_subfield_arr = bottom_most_scalar_idx / PE::PackedSubfield::WIDTH;
    let bottom_most_scalar_idx_within_packed_subfield =
        bottom_most_scalar_idx % PE::PackedSubfield::WIDTH;
    let block_idx = bottom_most_scalar_idx_within_packed_subfield / PE::WIDTH;

    unsafe {
        packed_subfields
            .get_unchecked(bottom_most_scalar_idx_in_subfield_arr)
            .spread_unchecked(PE::LOG_WIDTH, block_idx)
    }
}

/// Refer to the functions above for examples of closures to pass
/// Func takes in the following parameters
///
/// Note that this function overwrites the lhs buffer, copy that data before
/// invoking this function if you need to use it elsewhere
///
/// lhs: PE::WIDTH extension field scalars
///
/// broadcasted_rhs: a broadcasted version of PE::WIDTH subfield scalars
/// with each one occurring PE::PackedSubfield::WIDTH/PE::WIDTH times in  a row
/// such that the bits of the broadcasted scalars align with the lhs scalars
///
/// # Preconditions
///
/// * `lhs.len()` must equal `rhs.len() * PE::Scalar::DEGREE`.
pub fn ext_base_op<PE, F, Func>(lhs: &mut [PE], rhs: &[PE::PackedSubfield], op: Func)
where
    PE: PackedExtension<F>,
    F: Field,
    Func: Fn(usize, PE, PE::PackedSubfield) -> PE,
{
    assert!(
        lhs.len() == rhs.len() * PE::Scalar::DEGREE,
        "lhs.len() ({}) must equal rhs.len() * PE::Scalar::DEGREE ({})",
        lhs.len(),
        rhs.len() * PE::Scalar::DEGREE
    );

    lhs.iter_mut().enumerate().for_each(|(i, lhs_elem)| {
        // SAFETY: Width of PackedSubfield is always >= the width of the field implementing
        // PackedExtension
        let broadcasted_rhs = unsafe { get_packed_subfields_at_pe_idx::<PE, F>(rhs, i) };

        *lhs_elem = op(i, *lhs_elem, broadcasted_rhs);
    });
}

/// A multithreaded version of the function directly above, use for long arrays
/// on the prover side
///
/// # Preconditions
///
/// * `lhs.len()` must equal `rhs.len() * PE::Scalar::DEGREE`.
pub fn ext_base_op_par<PE, F, Func>(lhs: &mut [PE], rhs: &[PE::PackedSubfield], op: Func)
where
    PE: PackedExtension<F>,
    F: Field,
    Func: Fn(usize, PE, PE::PackedSubfield) -> PE + std::marker::Sync,
{
    assert!(
        lhs.len() == rhs.len() * PE::Scalar::DEGREE,
        "lhs.len() ({}) must equal rhs.len() * PE::Scalar::DEGREE ({})",
        lhs.len(),
        rhs.len() * PE::Scalar::DEGREE
    );

    lhs.par_iter_mut().enumerate().for_each(|(i, lhs_elem)| {
        // SAFETY: Width of PackedSubfield is always >= the width of the field implementing
        // PackedExtension
        let broadcasted_rhs = unsafe { get_packed_subfields_at_pe_idx::<PE, F>(rhs, i) };

        *lhs_elem = op(i, *lhs_elem, broadcasted_rhs);
    });
}
