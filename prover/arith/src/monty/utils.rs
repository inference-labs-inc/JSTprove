//! credit: https://github.com/Plonky3/Plonky3/blob/main/monty-31/src/utils.rs
#![allow(unused)]
use crate::{FieldParameters, MontyParameters};

/// Convert a u32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
pub(crate) const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

/// Convert an i32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
pub(crate) const fn to_monty_signed<MP: MontyParameters>(x: i32) -> u32 {
    let red = (((x as i64) << MP::MONTY_BITS) % MP::PRIME as i64) as i32;
    if red >= 0 {
        red as u32
    } else {
        MP::PRIME.wrapping_add_signed(red)
    }
}

/// Convert a u64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
pub(crate) const fn to_monty_64<MP: MontyParameters>(x: u64) -> u32 {
    (((x as u128) << MP::MONTY_BITS) % MP::PRIME as u128) as u32
}

/// Convert an i64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
pub(crate) const fn to_monty_64_signed<MP: MontyParameters>(x: i64) -> u32 {
    let red = (((x as i128) << MP::MONTY_BITS) % MP::PRIME as i128) as i32;
    if red >= 0 {
        red as u32
    } else {
        MP::PRIME.wrapping_add_signed(red)
    }
}

/// Convert a u32 out of MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range [0, P).
#[inline]
#[must_use]
pub(crate) const fn from_monty<MP: MontyParameters>(x: u32) -> u32 {
    monty_reduce::<MP>(x as u64)
}

/// Given an element x from a 31 bit field F_P compute x/2.
/// The input must be in [0, P).
/// The output will also be in [0, P).
#[inline]
pub(crate) const fn halve_u32<FP: FieldParameters>(input: u32) -> u32 {
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + FP::HALF_P_PLUS_1;
    if lo_bit == 0 {
        shr
    } else {
        shr_corr
    }
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
/// the input must be in [0, MONTY * P).
/// the output will be in [0, P).
#[inline]
#[must_use]
pub(crate) const fn monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);
    let u = t * (MP::PRIME as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}
