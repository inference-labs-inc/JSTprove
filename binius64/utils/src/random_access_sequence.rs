// Copyright 2025 Irreducible Inc.

/// A trait for a collection that allows indexed access by value.
/// This trait is used to abstract over different types of collections - scalar slices,
/// slices of packed field elements including subranges of collections.
pub trait RandomAccessSequence<T: Copy> {
    fn len(&self) -> usize;

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    fn get(&self, index: usize) -> T {
        assert!(index < self.len(), "Index out of bounds");
        unsafe { self.get_unchecked(index) }
    }

    /// Returns a copy of the element at the given index.
    ///
    /// # Safety
    /// The caller must ensure that the `index` < `self.len()`.
    unsafe fn get_unchecked(&self, index: usize) -> T;
}

/// A trait for a mutable access to a collection of scalars.
pub trait RandomAccessSequenceMut<T: Copy>: RandomAccessSequence<T> {
    #[inline(always)]
    fn set(&mut self, index: usize, value: T) {
        assert!(index < self.len(), "Index out of bounds");
        unsafe { self.set_unchecked(index, value) }
    }

    /// Sets the element at the given index to the given value.
    ///
    /// # Safety
    /// The caller must ensure that the `index` < `self.len()`.
    unsafe fn set_unchecked(&mut self, index: usize, value: T);
}

impl<T: Copy> RandomAccessSequence<T> for &[T] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    #[inline(always)]
    fn get(&self, index: usize) -> T {
        self[index]
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> T {
        unsafe { *<[T]>::get_unchecked(self, index) }
    }
}

impl<T: Copy> RandomAccessSequence<T> for &mut [T] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    #[inline(always)]
    fn get(&self, index: usize) -> T {
        self[index]
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> T {
        unsafe { *<[T]>::get_unchecked(self, index) }
    }
}

impl<T: Copy> RandomAccessSequenceMut<T> for &mut [T] {
    #[inline(always)]
    fn set(&mut self, index: usize, value: T) {
        self[index] = value;
    }

    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: T) {
        unsafe {
            *<[T]>::get_unchecked_mut(self, index) = value;
        }
    }
}

/// A subrange adapter of a collection of scalars.
#[derive(Clone)]
pub struct SequenceSubrange<'a, T: Copy, Inner: RandomAccessSequence<T>> {
    inner: &'a Inner,
    offset: usize,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Copy, Inner: RandomAccessSequence<T>> SequenceSubrange<'a, T, Inner> {
    #[inline(always)]
    pub fn new(inner: &'a Inner, offset: usize, len: usize) -> Self {
        assert!(offset + len <= inner.len(), "subrange out of bounds");

        Self {
            inner,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Copy, Inner: RandomAccessSequence<T>> RandomAccessSequence<T>
    for SequenceSubrange<'_, T, Inner>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> T {
        unsafe { self.inner.get_unchecked(index + self.offset) }
    }
}

/// A subrange adapter of a mutable collection of scalars.
pub struct SequenceSubrangeMut<'a, T: Copy, Inner: RandomAccessSequenceMut<T>> {
    inner: &'a mut Inner,
    offset: usize,
    len: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T: Copy, Inner: RandomAccessSequenceMut<T>> SequenceSubrangeMut<'a, T, Inner> {
    #[inline(always)]
    pub fn new(inner: &'a mut Inner, offset: usize, len: usize) -> Self {
        assert!(offset + len <= inner.len(), "subrange out of bounds");

        Self {
            inner,
            offset,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}
impl<T: Copy, Inner: RandomAccessSequenceMut<T>> RandomAccessSequence<T>
    for SequenceSubrangeMut<'_, T, Inner>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> T {
        unsafe { self.inner.get_unchecked(index + self.offset) }
    }
}
impl<T: Copy, Inner: RandomAccessSequenceMut<T>> RandomAccessSequenceMut<T>
    for SequenceSubrangeMut<'_, T, Inner>
{
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: T) {
        unsafe {
            self.inner.set_unchecked(index + self.offset, value);
        }
    }
}

/// Power-of-two aligned vertical slice of a sequence when viewed as a row-major matrix.
/// This is useful access pattern for algorithms like 4-step NTT or switchover.
#[derive(Clone)]
pub struct MatrixVertSliceSubrange<'a, T: Copy, Inner: RandomAccessSequence<T>> {
    inner: &'a Inner,
    log_cols: usize,
    log_slice: usize,
    slice_index: usize,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Copy, Inner: RandomAccessSequence<T>> MatrixVertSliceSubrange<'a, T, Inner> {
    /// Rearrange the `inner` sequence into a row-major matrix with sides `2^log_rows` and
    /// `2^log_cols`, then take an aligned vertical slice of size `2^log_slice` with index
    /// `slice_index` (which ranges from 0 to `2^(log_cols - log_slice)`, non-inclusive), and
    /// present that slice as a view.
    #[inline(always)]
    pub fn new(
        inner: &'a Inner,
        log_rows: usize,
        log_cols: usize,
        log_slice: usize,
        slice_index: usize,
    ) -> Self {
        assert_eq!(
            1 << (log_rows + log_cols),
            inner.len(),
            "matrix dimensions do not match inner sequence"
        );
        assert!(log_slice <= log_cols && slice_index < 1 << (log_cols - log_slice));

        let len = 1 << (log_slice + log_rows);

        Self {
            inner,
            log_cols,
            log_slice,
            slice_index,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Copy, Inner: RandomAccessSequence<T>> RandomAccessSequence<T>
    for MatrixVertSliceSubrange<'_, T, Inner>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> T {
        let row = index >> self.log_slice;
        let col = index ^ (row << self.log_slice);
        let inner_index = row << self.log_cols | self.slice_index << self.log_slice | col;
        unsafe { self.inner.get_unchecked(inner_index) }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    fn check_collection<T: Copy + Eq + Debug>(
        collection: &impl RandomAccessSequence<T>,
        expected: &[T],
    ) {
        assert_eq!(collection.len(), expected.len());

        for (i, v) in expected.iter().enumerate() {
            assert_eq!(&collection.get(i), v);
            assert_eq!(&unsafe { collection.get_unchecked(i) }, v);
        }
    }

    fn check_collection_get_set<T: Eq + Copy + Debug>(
        collection: &mut impl RandomAccessSequenceMut<T>,
        random: &mut impl FnMut() -> T,
    ) {
        for i in 0..collection.len() {
            let value = random();
            collection.set(i, value);
            assert_eq!(collection.get(i), value);
            assert_eq!(unsafe { collection.get_unchecked(i) }, value);
        }
    }

    #[test]
    fn check_slice() {
        let slice: &[usize] = &[];
        check_collection::<usize>(&slice, slice);

        let slice: &[usize] = &[1usize, 2, 3];
        check_collection(&slice, slice);
    }

    #[test]
    fn check_slice_mut() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut random = || -> usize { rng.random::<u64>() as usize };

        let mut slice: &mut [usize] = &mut [];

        check_collection(&slice, slice);
        check_collection_get_set(&mut slice, &mut random);

        let mut slice: &mut [usize] = &mut [1, 2, 3];
        check_collection(&slice, slice);
        check_collection_get_set(&mut slice, &mut random);
    }

    #[test]
    fn test_subrange() {
        let slice: &[usize] = &[1, 2, 3, 4, 5];
        let subrange = SequenceSubrange::new(&slice, 1, 3);
        check_collection(&subrange, &[2, 3, 4]);
    }

    #[test]
    fn test_subrange_mut() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut random = || -> usize { rng.random::<u64>() as usize };

        let mut slice: &mut [usize] = &mut [1, 2, 3, 4, 5];
        let values = slice[1..4].to_vec();
        let mut subrange = SequenceSubrangeMut::new(&mut slice, 1, 3);
        check_collection(&subrange, &values);
        check_collection_get_set(&mut subrange, &mut random);
    }
}
