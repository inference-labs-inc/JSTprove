// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire, util::all_true};

use super::scalar_mul::shamirs_trick_endomorphism;
use crate::{
    bignum::{BigUint, biguint_lt},
    secp256k1::{Secp256k1, Secp256k1Affine, coord_zero},
};

/// EcRecover - an "Ethereum-style" verification of ECDSA signatures over secp256k1.
///
/// # Arguments
/// * `z`         - hash of the signed message as an integer
/// * `r`         - R part of the signature, the x coordinate of the nonce point
/// * `s`         - S part of the signature
/// * `recid_odd` - parity flag of the y coordinate of the assumed nonce point R with R.x = r; note
///   that we do not support `r` being greater or equal than the scalar field modulus, and thus only
///   need parity; some implementations assume 0-3 bitmask which encodes both y parity and r scalar
///   field overflow, but that's not needed for Ethereum.
///
/// # Outputs
/// The recovered public key `pk` in affine form.
pub fn ecrecover(
    b: &CircuitBuilder,
    z: &BigUint,
    r: &BigUint,
    s: &BigUint,
    recid_odd: Wire,
) -> Secp256k1Affine {
    let curve = Secp256k1::new(b);

    let nonce = curve.recover(b, r, recid_odd);
    let nonce_not_pai = b.bnot(nonce.is_point_at_infinity);

    let f_scalar = curve.f_scalar();
    let valid_r = b.band(b.bnot(r.is_zero(b)), biguint_lt(b, r, f_scalar.modulus()));
    let valid_s = b.band(b.bnot(s.is_zero(b)), biguint_lt(b, s, f_scalar.modulus()));

    let r_inverse = f_scalar.inverse(b, r, valid_r);
    let u1 = f_scalar.sub(b, &coord_zero(b), &f_scalar.mul(b, z, &r_inverse));
    let u2 = f_scalar.mul(b, s, &r_inverse);

    let recovered_pk = shamirs_trick_endomorphism(b, &curve, &u1, &u2, nonce);

    let conditions = [valid_r, valid_s, nonce_not_pai];
    recovered_pk.pai_unless(b, all_true(b, conditions))
}
