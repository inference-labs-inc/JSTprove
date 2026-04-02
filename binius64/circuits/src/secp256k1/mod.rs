// Copyright 2025 Irreducible Inc.
//! Implementation of secp256k1 elliptic curve group operations.
//!
//! Points are represented either in affine form as a `(x, y)` coordinate tuple
//! satisfying `y^2 = x^3 + 7`, or in Jacobian weighted projective coordinate system
//! where a tuple `(x, y, z)` corresponds to a tuple `(x/z^2, y/z^3)` in affine form.
mod common;
mod curve;
mod point;

#[cfg(test)]
mod tests;

pub use common::{N_LIMBS, coord_lambda, coord_zero};
pub use curve::Secp256k1;
pub use point::{Secp256k1Affine, select};
