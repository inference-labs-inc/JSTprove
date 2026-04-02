// Copyright 2024-2025 Irreducible Inc.

use std::{
    iter::Sum,
    marker::PhantomData,
    mem,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use binius_field::{ExtensionField, Field};

use crate::inner_product::inner_product;

/// An element of the tensor algebra defined as the tensor product of `FE` and `FE` as fields.
///
/// A tensor algebra element is a length $D$ vector of `FE` field elements, where $D$ is the degree
/// of `FE` as an extension of `F`. The algebra has a "vertical subring" and a "horizontal subring",
/// which are both isomorphic to `FE` as a field.
///
/// See [DP24] Section 2 for further details.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    pub elems: Vec<FE>,
    _marker: PhantomData<F>,
}

impl<F, FE> Default for TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    fn default() -> Self {
        Self {
            elems: vec![FE::default(); FE::DEGREE],
            _marker: PhantomData,
        }
    }
}

impl<F, FE> TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    /// Constructs an element from a vector of vertical subring elements.
    ///
    /// ## Preconditions
    ///
    /// * `elems` must have length `FE::DEGREE`, otherwise this will pad or truncate.
    pub fn new(mut elems: Vec<FE>) -> Self {
        elems.resize(FE::DEGREE, FE::ZERO);
        Self {
            elems,
            _marker: PhantomData,
        }
    }

    /// Returns $\kappa$, the base-2 logarithm of the extension degree.
    pub const fn kappa() -> usize {
        FE::LOG_DEGREE
    }

    /// Returns the byte size of an element.
    pub const fn byte_size() -> usize {
        mem::size_of::<FE>() << Self::kappa()
    }

    /// Returns the multiplicative identity element, one.
    pub fn one() -> Self {
        let mut one = Self::default();
        one.elems[0] = FE::ONE;
        one
    }

    /// Returns a slice of the vertical subfield elements composing the tensor algebra element.
    pub fn vertical_elems(&self) -> &[FE] {
        &self.elems
    }

    /// Tensor product of a vertical subring element and a horizontal subring element.
    pub fn tensor(vertical: FE, horizontal: FE) -> Self {
        let elems = horizontal
            .iter_bases()
            .map(|base| vertical * base)
            .collect();
        Self {
            elems,
            _marker: PhantomData,
        }
    }

    /// Constructs a [`TensorAlgebra`] in the vertical subring.
    pub fn from_vertical(x: FE) -> Self {
        let mut elems = vec![FE::ZERO; FE::DEGREE];
        elems[0] = x;
        Self {
            elems,
            _marker: PhantomData,
        }
    }

    /// If the algebra element lives in the vertical subring, this returns it as a field element.
    pub fn try_extract_vertical(&self) -> Option<FE> {
        self.elems
            .iter()
            .skip(1)
            .all(|&elem| elem == FE::ZERO)
            .then_some(self.elems[0])
    }

    /// Multiply by an element from the vertical subring.
    pub fn scale_vertical(mut self, scalar: FE) -> Self {
        for elem_i in &mut self.elems {
            *elem_i *= scalar;
        }
        self
    }
}

impl<F: Field, FE: ExtensionField<F>> TensorAlgebra<F, FE> {
    /// Multiply by an element from the vertical subring.
    ///
    /// Internally, this performs a transpose, vertical scaling, then transpose sequence. If
    /// multiple horizontal scaling operations are required and performance is a concern, it may be
    /// better for the caller to do the transposes directly and amortize their cost.
    pub fn scale_horizontal(self, scalar: FE) -> Self {
        self.transpose().scale_vertical(scalar).transpose()
    }

    /// Transposes the algebra element.
    ///
    /// A transpose flips the vertical and horizontal subring elements.
    pub fn transpose(mut self) -> Self {
        FE::square_transpose(&mut self.elems);
        self
    }

    /// Fold the tensor algebra element into a field element by scaling the rows and accumulating.
    ///
    /// ## Preconditions
    ///
    /// * `coeffs` must have length $2^\kappa$
    pub fn fold_vertical(self, coeffs: &[FE]) -> FE {
        inner_product(self.transpose().elems, coeffs.iter().copied())
    }
}

impl<F, FE> Add<&Self> for TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self {
        self.add_assign(rhs);
        self
    }
}

impl<F, FE> Sub<&Self> for TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self {
        self.sub_assign(rhs);
        self
    }
}

impl<F, FE> AddAssign<&Self> for TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    fn add_assign(&mut self, rhs: &Self) {
        for (self_i, rhs_i) in self.elems.iter_mut().zip(rhs.elems.iter()) {
            *self_i += *rhs_i;
        }
    }
}

impl<F, FE> SubAssign<&Self> for TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        for (self_i, rhs_i) in self.elems.iter_mut().zip(rhs.elems.iter()) {
            *self_i -= *rhs_i;
        }
    }
}

impl<'a, F, FE> Sum<&'a Self> for TensorAlgebra<F, FE>
where
    F: Field,
    FE: ExtensionField<F>,
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |sum, item| sum + item)
    }
}

#[cfg(test)]
mod tests {
    use binius_field::{BinaryField1b as B1, Random};
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::test_utils::B128;

    #[test]
    fn test_tensor_product() {
        type F = B1;
        type FE = B128;

        let mut rng = StdRng::seed_from_u64(0);

        let vert = FE::random(&mut rng);
        let hztl = FE::random(&mut rng);

        let expected = TensorAlgebra::<F, _>::from_vertical(vert).scale_horizontal(hztl);
        assert_eq!(TensorAlgebra::tensor(vert, hztl), expected);
    }

    #[test]
    fn test_try_extract_vertical() {
        type F = B1;
        type FE = B128;

        let mut rng = StdRng::seed_from_u64(0);

        let vert = FE::random(&mut rng);
        let elem = TensorAlgebra::<F, _>::from_vertical(vert);
        assert_eq!(elem.try_extract_vertical(), Some(vert));

        // Scale horizontally by an extension element, and we should no longer be vertical.
        let hztl = FE::new(1111);
        let elem = elem.scale_horizontal(hztl);
        assert_eq!(elem.try_extract_vertical(), None);

        // Scale back by the inverse to get back to the vertical subring.
        let hztl_inv = hztl.invert().unwrap();
        let elem = elem.scale_horizontal(hztl_inv);
        assert_eq!(elem.try_extract_vertical(), Some(vert));

        // If we scale horizontally by an F element, we should remain in the vertical subring.
        let hztl_subfield = FE::from(F::ONE);
        let elem = elem.scale_horizontal(hztl_subfield);
        assert_eq!(elem.try_extract_vertical(), Some(vert * hztl_subfield));
    }
}
