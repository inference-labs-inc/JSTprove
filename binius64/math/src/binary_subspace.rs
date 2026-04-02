// Copyright 2024-2025 Irreducible Inc.

use std::ops::Deref;

use binius_field::{BinaryField, BinaryField1b};

/// An $F_2$-linear subspace of a binary field.
///
/// The subspace is defined by a basis of elements from a binary field. The basis elements are
/// ordered, which implies an ordering on the subspace elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySubspace<F, Data: Deref<Target = [F]> = Vec<F>> {
    basis: Data,
}

impl<F: BinaryField, Data: Deref<Target = [F]>> BinarySubspace<F, Data> {
    /// Creates a new subspace from a vector of ordered basis elements.
    ///
    /// This constructor does not check that the basis elements are linearly independent.
    pub const fn new_unchecked(basis: Data) -> Self {
        Self { basis }
    }

    /// Creates a new subspace isomorphic to the given one.
    pub fn isomorphic<FIso>(&self) -> BinarySubspace<FIso>
    where
        FIso: BinaryField + From<F>,
    {
        BinarySubspace {
            basis: self.basis.iter().copied().map(Into::into).collect(),
        }
    }

    /// Returns the dimension of the subspace.
    pub fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Returns the slice of ordered basis elements.
    pub fn basis(&self) -> &[F] {
        &self.basis
    }

    pub fn get(&self, index: usize) -> F {
        assert!(
            index < 1 << self.dim(),
            "precondition: index must be less than 2^dim"
        );
        self.basis
            .iter()
            .enumerate()
            .map(|(i, &basis_i)| basis_i * BinaryField1b::from((index >> i) & 1 == 1))
            .sum()
    }

    /// Returns an iterator over all elements of the subspace in order.
    ///
    /// This has a limitation that the iterator only yields the first `2^usize::BITS` elements.
    pub fn iter(&self) -> BinarySubspaceIterator<'_, F> {
        BinarySubspaceIterator::new(&self.basis)
    }
}

impl<F: BinaryField> BinarySubspace<F> {
    /// Creates a new subspace of this binary subspace with the given dimension.
    ///
    /// This creates a new sub-subspace using a prefix of the default $\mathbb{F}_2$ basis elements
    /// of the field.
    ///
    /// ## Preconditions
    ///
    /// * `dim` must be at most `F::DEGREE`
    pub fn with_dim(dim: usize) -> Self {
        assert!(
            dim <= F::DEGREE,
            "precondition: dim must be at most F::DEGREE"
        );
        let basis = (0..dim).map(|i| F::basis(i)).collect();
        Self { basis }
    }

    /// Creates a new subspace of this binary subspace with reduced dimension.
    ///
    /// This creates a new sub-subspace using a prefix of the ordered basis elements.
    ///
    /// ## Preconditions
    ///
    /// * `dim` must be at most this subspace's dimension
    pub fn reduce_dim(&self, dim: usize) -> Self {
        assert!(
            dim <= self.dim(),
            "precondition: dim must be at most this subspace's dimension"
        );
        Self {
            basis: self.basis[..dim].to_vec(),
        }
    }
}

/// Iterator over all elements of a binary subspace.
///
/// Each element is computed as a subset sum (XOR) of the basis elements.
/// The iterator supports efficient `nth` operation without computing intermediate values.
#[derive(Debug, Clone)]
pub struct BinarySubspaceIterator<'a, F> {
    basis: &'a [F],
    index: usize,
    next: Option<F>,
}

impl<'a, F: BinaryField> BinarySubspaceIterator<'a, F> {
    pub fn new(basis: &'a [F]) -> Self {
        assert!(basis.len() < usize::BITS as usize);
        Self {
            basis,
            index: 0,
            next: Some(F::ZERO),
        }
    }
}

impl<'a, F: BinaryField> Iterator for BinarySubspaceIterator<'a, F> {
    type Item = F;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.next?;

        let mut next = ret;
        let mut i = 0;
        while (self.index >> i) & 1 == 1 {
            next -= self.basis[i];
            i += 1;
        }
        self.next = self.basis.get(i).map(|&basis_i| next + basis_i);

        self.index += 1;
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let last = 1 << self.basis.len();
        let remaining = last - self.index;
        (remaining, Some(remaining))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match self.index.checked_add(n) {
            Some(new_index) if new_index < 1 << self.basis.len() => {
                let new_next = BinarySubspace::new_unchecked(self.basis).get(new_index);

                self.index = new_index;
                self.next = Some(new_next);
            }
            _ => {
                self.index = 1 << self.basis.len();
                self.next = None;
            }
        }

        self.next()
    }
}

impl<'a, F: BinaryField> ExactSizeIterator for BinarySubspaceIterator<'a, F> {
    fn len(&self) -> usize {
        let last = 1 << self.basis.len();
        last - self.index
    }
}

impl<'a, F: BinaryField> std::iter::FusedIterator for BinarySubspaceIterator<'a, F> {}

impl<F: BinaryField> Default for BinarySubspace<F> {
    fn default() -> Self {
        let basis = (0..F::DEGREE).map(|i| F::basis(i)).collect();
        Self { basis }
    }
}

#[cfg(test)]
mod tests {
    use binius_field::{AESTowerField8b as B8, BinaryField128bGhash as B128, Field};

    use super::*;

    #[test]
    fn test_default_binary_subspace_iterates_elements() {
        let subspace = BinarySubspace::<B8>::default();
        for i in 0..=255 {
            assert_eq!(subspace.get(i), B8::new(i as u8));
        }
    }

    #[test]
    #[should_panic(expected = "precondition")]
    fn test_binary_subspace_range_error() {
        let subspace = BinarySubspace::<B8>::default();
        let _ = subspace.get(256);
    }

    #[test]
    fn test_default_binary_subspace() {
        let subspace = BinarySubspace::<B8>::default();
        assert_eq!(subspace.dim(), 8);
        assert_eq!(subspace.basis().len(), 8);

        assert_eq!(
            subspace.basis(),
            [
                B8::new(0b00000001),
                B8::new(0b00000010),
                B8::new(0b00000100),
                B8::new(0b00001000),
                B8::new(0b00010000),
                B8::new(0b00100000),
                B8::new(0b01000000),
                B8::new(0b10000000)
            ]
        );

        let expected_elements: [u8; 256] = (0..=255).collect::<Vec<_>>().try_into().unwrap();

        for (i, &expected) in expected_elements.iter().enumerate() {
            assert_eq!(subspace.get(i), B8::new(expected));
        }
    }

    #[test]
    fn test_with_dim_valid() {
        let subspace = BinarySubspace::<B8>::with_dim(3);
        assert_eq!(subspace.dim(), 3);
        assert_eq!(subspace.basis().len(), 3);

        assert_eq!(
            subspace.basis(),
            [B8::new(0b001), B8::new(0b010), B8::new(0b100)]
        );

        let expected_elements: [u8; 8] = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];

        for (i, &expected) in expected_elements.iter().enumerate() {
            assert_eq!(subspace.get(i), B8::new(expected));
        }
    }

    #[test]
    #[should_panic(expected = "precondition")]
    fn test_with_dim_invalid() {
        let _ = BinarySubspace::<B8>::with_dim(10);
    }

    #[test]
    fn test_reduce_dim_valid() {
        let subspace = BinarySubspace::<B8>::with_dim(6);
        let reduced = subspace.reduce_dim(4);
        assert_eq!(reduced.dim(), 4);
        assert_eq!(reduced.basis().len(), 4);

        assert_eq!(
            reduced.basis(),
            [
                B8::new(0b0001),
                B8::new(0b0010),
                B8::new(0b0100),
                B8::new(0b1000)
            ]
        );

        let expected_elements: [u8; 16] = (0..16).collect::<Vec<_>>().try_into().unwrap();

        for (i, &expected) in expected_elements.iter().enumerate() {
            assert_eq!(reduced.get(i), B8::new(expected));
        }
    }

    #[test]
    #[should_panic(expected = "precondition")]
    fn test_reduce_dim_invalid() {
        let subspace = BinarySubspace::<B8>::with_dim(4);
        let _ = subspace.reduce_dim(6);
    }

    #[test]
    fn test_isomorphic_conversion() {
        let subspace = BinarySubspace::<B8>::with_dim(3);
        let iso_subspace: BinarySubspace<B128> = subspace.isomorphic();
        assert_eq!(iso_subspace.dim(), 3);
        assert_eq!(iso_subspace.basis().len(), 3);

        assert_eq!(
            iso_subspace.basis(),
            [
                B128::from(B8::new(0b001)),
                B128::from(B8::new(0b010)),
                B128::from(B8::new(0b100)),
            ]
        );
    }

    #[test]
    fn test_iterate_subspace() {
        let subspace = BinarySubspace::<B8>::with_dim(3);
        let elements: Vec<_> = subspace.iter().collect();
        assert_eq!(elements.len(), 8);

        let expected_elements: [u8; 8] = [0b000, 0b001, 0b010, 0b011, 0b100, 0b101, 0b110, 0b111];

        for (i, &expected) in expected_elements.iter().enumerate() {
            assert_eq!(elements[i], B8::new(expected));
        }
    }

    #[test]
    fn test_iterator_matches_get() {
        let subspace = BinarySubspace::<B8>::with_dim(5);

        // Test that iterator produces same elements as get()
        for (i, elem) in subspace.iter().enumerate() {
            assert_eq!(elem, subspace.get(i), "Mismatch at index {}", i);
        }
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn test_iterator_nth() {
        let subspace = BinarySubspace::<B8>::with_dim(4);

        // Test nth with various positions
        let mut iter = subspace.iter();
        assert_eq!(iter.nth(0), Some(subspace.get(0)));
        assert_eq!(iter.nth(0), Some(subspace.get(1)));
        assert_eq!(iter.nth(2), Some(subspace.get(4)));
        assert_eq!(iter.nth(5), Some(subspace.get(10)));

        // Test nth at the end
        let mut iter = subspace.iter();
        assert_eq!(iter.nth(15), Some(subspace.get(15)));
        assert_eq!(iter.nth(0), None);
    }

    #[test]
    fn test_iterator_nth_skips_efficiently() {
        let subspace = BinarySubspace::<B8>::with_dim(6);

        // Test that we can jump directly to any position
        let mut iter = subspace.iter();
        assert_eq!(iter.nth(30), Some(subspace.get(30)));
        assert_eq!(iter.next(), Some(subspace.get(31)));

        // Test large skip
        let mut iter = subspace.iter();
        assert_eq!(iter.nth(50), Some(subspace.get(50)));
    }

    #[test]
    fn test_iterator_size_hint() {
        let subspace = BinarySubspace::<B8>::with_dim(3);
        let mut iter = subspace.iter();

        assert_eq!(iter.size_hint(), (8, Some(8)));
        iter.next();
        assert_eq!(iter.size_hint(), (7, Some(7)));
        iter.nth(3);
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iterator_exact_size() {
        let subspace = BinarySubspace::<B8>::with_dim(4);
        let mut iter = subspace.iter();

        assert_eq!(iter.len(), 16);
        iter.next();
        assert_eq!(iter.len(), 15);
        iter.nth(5);
        assert_eq!(iter.len(), 9);
    }

    #[test]
    fn test_iterator_empty_subspace() {
        let subspace = BinarySubspace::<B8>::with_dim(0);
        let mut iter = subspace.iter();

        // Subspace of dimension 0 has only one element: zero
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some(B8::ZERO));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_full_iteration() {
        let subspace = BinarySubspace::<B8>::default();
        let collected: Vec<_> = subspace.iter().collect();

        assert_eq!(collected.len(), 256);
        for (i, elem) in collected.iter().enumerate() {
            assert_eq!(*elem, subspace.get(i));
        }
    }

    #[test]
    fn test_iterator_partial_then_nth() {
        let subspace = BinarySubspace::<B8>::with_dim(5);
        let mut iter = subspace.iter();

        // Iterate a few elements
        assert_eq!(iter.next(), Some(subspace.get(0)));
        assert_eq!(iter.next(), Some(subspace.get(1)));
        assert_eq!(iter.next(), Some(subspace.get(2)));

        // Then skip ahead
        assert_eq!(iter.nth(5), Some(subspace.get(8)));
        assert_eq!(iter.next(), Some(subspace.get(9)));
    }

    #[test]
    fn test_iterator_clone() {
        let subspace = BinarySubspace::<B8>::with_dim(3);
        let mut iter1 = subspace.iter();

        iter1.next();
        iter1.next();

        let mut iter2 = iter1.clone();

        // Both iterators should produce the same remaining elements
        assert_eq!(iter1.next(), iter2.next());
        assert_eq!(iter1.collect::<Vec<_>>(), iter2.collect::<Vec<_>>());
    }
}
