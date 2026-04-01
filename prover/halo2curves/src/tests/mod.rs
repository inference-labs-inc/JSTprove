use ff::PrimeField;
use num_bigint::BigUint;
use pasta_curves::arithmetic::CurveAffine;

pub mod curve;
pub mod field;

pub(crate) fn hex_to_bytes(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks(2)
        .map(|chunk| u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap())
        .collect()
}

pub(crate) fn hex_to_field<F: PrimeField>(hex: &str) -> F {
    let bytes = hex_to_bytes(hex);
    let mut repr = F::Repr::default();
    repr.as_mut().copy_from_slice(&bytes);
    repr.as_mut().reverse();
    F::from_repr(repr).unwrap()
}

pub(crate) fn point_from_hex<C: CurveAffine>(x: &str, y: &str) -> C {
    let x = crate::tests::hex_to_field(x);
    let y = crate::tests::hex_to_field(y);
    C::from_xy(x, y).unwrap()
}

pub(crate) fn fe_to_big<F: PrimeField>(fe: &F) -> BigUint {
    BigUint::from_bytes_le(fe.to_repr().as_ref())
}

pub(crate) fn modulus<F: PrimeField>() -> BigUint {
    fe_to_big(&-F::ONE) + 1usize
}
