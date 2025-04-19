use expander_compiler::frontend::*;
use ethnum::U256;
use std::ops::Neg;

/// Convert signed 64-bit integer to field element (not Variable)
pub fn i64_to_field<C: Config>(x: i64) -> C::CircuitField {
    if x < 0 {
        C::CircuitField::from_u256(U256::from(x.unsigned_abs())).neg()
    } else {
        C::CircuitField::from_u256(U256::from(x as u64))
    }
}

/// Convert 2D i64 matrix to 2D matrix of field elements
pub fn matrix_i64_to_field<C: Config>(mat: Vec<Vec<i64>>) -> Vec<Vec<C::CircuitField>> {
    mat.into_iter()
        .map(|row| row.into_iter().map(i64_to_field::<C>).collect())
        .collect()
}
