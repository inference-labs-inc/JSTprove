//! This module provides common utilities, traits and structures for group and
//! field arithmetic.
//!
//! This module is temporary, and the extension traits defined here are expected to be
//! upstreamed into the `ff` and `group` crates after some refactoring.

use crate::CurveExt;

pub(crate) struct EndoParameters {
    pub(crate) gamma1: [u64; 4],
    pub(crate) gamma2: [u64; 4],
    pub(crate) b1: [u64; 4],
    pub(crate) b2: [u64; 4],
}

pub trait CurveEndo: CurveExt {
    fn decompose_scalar(e: &Self::ScalarExt) -> (u128, bool, u128, bool);
}

/// Compute a + b + carry, returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + (b as u128) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a - (b + borrow), returning the result and the new borrow.
#[inline(always)]
pub(crate) const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let ret = (a as u128).wrapping_sub((b as u128) + ((borrow >> 63) as u128));
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a + (b * c) + carry, returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
    let ret = (a as u128) + ((b as u128) * (c as u128)) + (carry as u128);
    (ret as u64, (ret >> 64) as u64)
}

/// Compute a + (b * c), returning the result and the new carry over.
#[inline(always)]
pub(crate) const fn macx(a: u64, b: u64, c: u64) -> (u64, u64) {
    let res = (a as u128) + ((b as u128) * (c as u128));
    (res as u64, (res >> 64) as u64)
}

/// Returns a >= b
#[inline(always)]
pub(crate) const fn bigint_geq(a: &[u64; 4], b: &[u64; 4]) -> bool {
    if a[3] > b[3] {
        return true;
    } else if a[3] < b[3] {
        return false;
    }
    if a[2] > b[2] {
        return true;
    } else if a[2] < b[2] {
        return false;
    }
    if a[1] > b[1] {
        return true;
    } else if a[1] < b[1] {
        return false;
    }
    if a[0] >= b[0] {
        return true;
    }
    false
}

/// Compute a * b, returning the result.
#[inline(always)]
pub(crate) fn mul_512(a: [u64; 4], b: [u64; 4]) -> [u64; 8] {
    let (r0, carry) = macx(0, a[0], b[0]);
    let (r1, carry) = macx(carry, a[0], b[1]);
    let (r2, carry) = macx(carry, a[0], b[2]);
    let (r3, carry_out) = macx(carry, a[0], b[3]);

    let (r1, carry) = macx(r1, a[1], b[0]);
    let (r2, carry) = mac(r2, a[1], b[1], carry);
    let (r3, carry) = mac(r3, a[1], b[2], carry);
    let (r4, carry_out) = mac(carry_out, a[1], b[3], carry);

    let (r2, carry) = macx(r2, a[2], b[0]);
    let (r3, carry) = mac(r3, a[2], b[1], carry);
    let (r4, carry) = mac(r4, a[2], b[2], carry);
    let (r5, carry_out) = mac(carry_out, a[2], b[3], carry);

    let (r3, carry) = macx(r3, a[3], b[0]);
    let (r4, carry) = mac(r4, a[3], b[1], carry);
    let (r5, carry) = mac(r5, a[3], b[2], carry);
    let (r6, carry_out) = mac(carry_out, a[3], b[3], carry);

    [r0, r1, r2, r3, r4, r5, r6, carry_out]
}
