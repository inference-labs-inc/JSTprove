use std::ops::Mul;

use crate::Field;

/// Configurations for Extension Field over
/// - either the Binomial polynomial x^DEGREE - W
/// - or the AES polynomial x^128 + x^7 + x^2 + x + 1
//
pub trait ExtensionField: Mul<Self::BaseField> + From<Self::BaseField> + Field {
    /// Degree of the Extension
    const DEGREE: usize;

    /// constant term if the extension field is represented as a binomial polynomial
    const W: u32;

    /// x, i.e, 0 + x + 0 x^2 + 0 x^3 + ...
    const X: Self;

    /// Base field for the extension
    type BaseField: Field + Send;

    /// Multiply the extension field with the base field
    fn mul_by_base_field(&self, base: &Self::BaseField) -> Self;

    /// Add the extension field with the base field
    fn add_by_base_field(&self, base: &Self::BaseField) -> Self;

    /// Multiply the extension field element by x, i.e, 0 + x + 0 x^2 + 0 x^3 + ...
    fn mul_by_x(&self) -> Self;

    /// Extract polynomial field coefficients from the extension field instance
    /// The order should be from coefficients with low degree to coefficients with high degrees
    fn to_limbs(&self) -> Vec<Self::BaseField>;

    /// Construct a new instance of extension field from coefficients
    /// The order should be from coefficients with low degree to coefficients with high degrees
    fn from_limbs(limbs: &[Self::BaseField]) -> Self;
}
