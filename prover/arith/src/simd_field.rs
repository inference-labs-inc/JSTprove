use crate::Field;

/// Configurations for the SimdField.
pub trait SimdField: From<Self::Scalar> + Field {
    /// Field for the challenge. Can be self.
    type Scalar: Field + Send;

    /// Pack size (width) for the SIMD instruction
    const PACK_SIZE: usize;

    /// scale self with the challenge
    fn scale(&self, challenge: &Self::Scalar) -> Self;

    /// pack a single scalar field into self by cloning the
    /// scalar field element into all slots
    fn pack_full(base: &Self::Scalar) -> Self;

    /// pack a vec of scalar field into self
    fn pack(base_vec: &[Self::Scalar]) -> Self;

    /// unpack into a vector.
    fn unpack(&self) -> Vec<Self::Scalar>;

    /// horizontally sum all packed scalars
    fn horizontal_sum(&self) -> Self::Scalar {
        self.unpack().iter().sum()
    }
}
