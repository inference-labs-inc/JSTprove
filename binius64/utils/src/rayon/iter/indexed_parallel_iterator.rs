// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

use super::{
    IntoParallelIterator, ParallelIterator, parallel_iterator::ParallelIteratorInner,
    parallel_wrapper::ParallelWrapper,
};

/// The reason why we need this trait is because `IndexedParallelIterator` contains
/// methods that overlaps with `std::iterator::Iterator` trait. So we implement it separately for
/// different `std::iter::Iterator` types while `IndexedParallelIterator` is implemented for
/// `ParallelWrapper<I>` where `I` is `IndexedParallelIteratorInner`.
///
/// Currently only those methods are implemented that are used in the `binius` code base. All other
/// methods can be implemented upon request.
pub(crate) trait IndexedParallelIteratorInner:
    ParallelIteratorInner + ExactSizeIterator
{
    #[inline(always)]
    fn with_min_len(self, _min: usize) -> Self
    where
        Self: Sized,
    {
        self
    }

    #[inline(always)]
    fn with_max_len(self, _max: usize) -> Self
    where
        Self: Sized,
    {
        self
    }

    #[inline]
    fn enumerate(self) -> impl IndexedParallelIteratorInner<Item = (usize, Self::Item)>
    where
        Self: Sized,
    {
        Iterator::enumerate(self.into_iter())
    }

    #[inline]
    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        target.clear();
        target.extend(self);
    }

    #[inline]
    fn zip<Z>(self, zip_op: Z) -> std::iter::Zip<Self, Z>
    where
        Z: IndexedParallelIteratorInner,
    {
        Iterator::zip(self, zip_op)
    }

    #[inline]
    fn zip_eq<Z>(self, zip_op: Z) -> itertools::ZipEq<Self, Z>
    where
        Z: IndexedParallelIteratorInner,
    {
        itertools::Itertools::zip_eq(self, zip_op)
    }

    #[inline]
    fn step_by(self, step: usize) -> std::iter::StepBy<Self>
    where
        Self: Sized,
    {
        Iterator::step_by(self, step)
    }

    #[inline]
    fn chunks(self, chunk_size: usize) -> impl IndexedParallelIteratorInner<Item = Vec<Self::Item>>
    where
        Self: Sized,
    {
        Chunks {
            inner: self,
            chunk_size,
        }
    }

    #[inline]
    fn take(self, n: usize) -> impl IndexedParallelIteratorInner<Item = Self::Item>
    where
        Self: Sized,
    {
        Iterator::take(self, n)
    }
}

struct Chunks<I> {
    inner: I,
    chunk_size: usize,
}

impl<I: ExactSizeIterator> Iterator for Chunks<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        for _ in 0..self.chunk_size {
            match self.inner.next() {
                Some(item) => chunk.push(item),
                None => break,
            }
        }
        if chunk.is_empty() { None } else { Some(chunk) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.inner.len().div_ceil(self.chunk_size);
        (size, Some(size))
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Chunks<I> {}

impl<I: ExactSizeIterator> IndexedParallelIteratorInner for Chunks<I> {}

/// Wrapper around std::iter::Chain that implements ExactSizeIterator.
pub struct Chain<I1, I2> {
    pub(super) inner: std::iter::Chain<I1, I2>,
}

impl<I1, I2> Iterator for Chain<I1, I2>
where
    I1: Iterator,
    I2: Iterator<Item = I1::Item>,
{
    type Item = I1::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<I1, I2> ExactSizeIterator for Chain<I1, I2>
where
    I1: ExactSizeIterator,
    I2: ExactSizeIterator<Item = I1::Item>,
{
}

impl<I1, I2> IndexedParallelIteratorInner for Chain<I1, I2>
where
    I1: IndexedParallelIteratorInner,
    I2: IndexedParallelIteratorInner<Item = I1::Item>,
{
}

// Implement `IndexedParallelIteratorInner` for different `std::iter::Iterator` types.
// Unfortunately, we can't implement it for all `std::iter::Iterator` types because of the
// collisions with generic implementation for tuples (see `multizip_impls!` macro in
// `parallel_iterator.rs`). If you need to implement it for some other type, please add
// implementation here.
impl<Idx> IndexedParallelIteratorInner for std::ops::Range<Idx> where
    Self: ExactSizeIterator<Item = Idx>
{
}
impl<T> IndexedParallelIteratorInner for std::slice::IterMut<'_, T> {}
impl<L: IndexedParallelIteratorInner, R: IndexedParallelIteratorInner> IndexedParallelIteratorInner
    for std::iter::Zip<L, R>
{
}
impl<I: IndexedParallelIteratorInner> IndexedParallelIteratorInner for std::iter::Enumerate<I> {}
impl<I: IndexedParallelIteratorInner> IndexedParallelIteratorInner for std::iter::StepBy<I> {}
impl<I: IndexedParallelIteratorInner, R, F: FnMut(I::Item) -> R> IndexedParallelIteratorInner
    for std::iter::Map<I, F>
{
}
impl<I: IndexedParallelIteratorInner> IndexedParallelIteratorInner for std::iter::Take<I> {}
impl<T> IndexedParallelIteratorInner for std::vec::IntoIter<T> {}
impl<T, const N: usize> IndexedParallelIteratorInner for std::array::IntoIter<T, N> {}
impl<T: Clone> IndexedParallelIteratorInner for std::iter::RepeatN<T> {}
impl<I: IndexedParallelIteratorInner> IndexedParallelIteratorInner for std::iter::Skip<I> {}
impl<L: IndexedParallelIteratorInner, R: IndexedParallelIteratorInner<Item = L::Item>>
    IndexedParallelIteratorInner for itertools::Either<L, R>
{
}

#[allow(private_bounds)]
pub trait IndexedParallelIterator:
    ParallelIterator<Inner: IndexedParallelIteratorInner<Item = Self::Item>>
{
    #[inline(always)]
    fn len(&self) -> usize {
        ParallelIterator::as_inner(self).len()
    }

    #[inline(always)]
    fn with_min_len(self, min: usize) -> impl IndexedParallelIterator<Item = Self::Item>
    where
        Self: Sized,
    {
        ParallelWrapper::new(ParallelIterator::into_inner(self).with_min_len(min))
    }

    #[inline(always)]
    fn with_max_len(self, max: usize) -> impl IndexedParallelIterator<Item = Self::Item>
    where
        Self: Sized,
    {
        ParallelWrapper::new(ParallelIterator::into_inner(self).with_max_len(max))
    }

    #[inline]
    fn enumerate(self) -> impl IndexedParallelIterator<Item = (usize, Self::Item)>
    where
        Self: Sized,
    {
        ParallelWrapper::new(IndexedParallelIteratorInner::enumerate(
            ParallelIterator::into_inner(self),
        ))
    }

    #[inline]
    fn collect_into_vec(self, target: &mut Vec<Self::Item>) {
        ParallelIterator::into_inner(self).collect_into_vec(target)
    }

    #[inline]
    fn zip<Z>(
        self,
        zip_op: Z,
    ) -> ParallelWrapper<
        std::iter::Zip<
            <Self as ParallelIterator>::Inner,
            <<Z as IntoParallelIterator>::Iter as ParallelIterator>::Inner,
        >,
    >
    where
        Z: IntoParallelIterator<Iter: IndexedParallelIterator>,
    {
        ParallelWrapper::new(IndexedParallelIteratorInner::zip(
            ParallelIterator::into_inner(self),
            ParallelIterator::into_inner(zip_op.into_par_iter()),
        ))
    }

    #[inline]
    fn zip_eq<Z>(
        self,
        zip_op: Z,
    ) -> ParallelWrapper<
        itertools::ZipEq<
            <Self as ParallelIterator>::Inner,
            <<Z as IntoParallelIterator>::Iter as ParallelIterator>::Inner,
        >,
    >
    where
        Z: IntoParallelIterator<Iter: IndexedParallelIterator>,
    {
        ParallelWrapper::new(IndexedParallelIteratorInner::zip_eq(
            ParallelIterator::into_inner(self),
            ParallelIterator::into_inner(zip_op.into_par_iter()),
        ))
    }

    #[inline]
    fn step_by(
        self,
        step: usize,
    ) -> ParallelWrapper<std::iter::StepBy<<Self as ParallelIterator>::Inner>>
    where
        Self: Sized,
    {
        ParallelWrapper::new(IndexedParallelIteratorInner::step_by(
            ParallelIterator::into_inner(self),
            step,
        ))
    }

    #[inline]
    fn chunks(self, chunk_size: usize) -> impl IndexedParallelIterator<Item = Vec<Self::Item>> {
        assert!(chunk_size != 0, "chunk_size must not be zero");

        ParallelWrapper::new(ParallelIterator::into_inner(self).chunks(chunk_size))
    }

    #[inline]
    fn take(self, n: usize) -> impl IndexedParallelIterator<Item = Self::Item>
    where
        Self: Sized,
    {
        ParallelWrapper::new(IndexedParallelIteratorInner::take(
            ParallelIterator::into_inner(self),
            n,
        ))
    }
}

impl<I: IndexedParallelIteratorInner> IndexedParallelIterator for ParallelWrapper<I> {}

impl<L, R> IndexedParallelIterator for itertools::Either<L, R>
where
    L: IndexedParallelIterator,
    R: IndexedParallelIterator<Item = L::Item>,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_zip() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        let result = a.into_par_iter().zip(b.into_par_iter()).collect::<Vec<_>>();
        assert_eq!(result, vec![(1, 4), (2, 5), (3, 6)]);
    }

    #[test]
    fn check_step_by() {
        let a = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let result = a.into_par_iter().step_by(2).collect::<Vec<_>>();
        assert_eq!(result, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn check_step_by_range() {
        let a = 1..10;

        let result = a.into_par_iter().step_by(2).collect::<Vec<_>>();
        assert_eq!(result, vec![1, 3, 5, 7, 9]);
    }

    #[test]
    fn check_map() {
        let a = &[1, 2, 3];

        let result = a.into_par_iter().map(|x| x * 2).collect::<Vec<_>>();
        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn check_chain() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        let result = a
            .into_par_iter()
            .chain(b.into_par_iter())
            .collect::<Vec<_>>();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn check_chain_with_map() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6];

        let result = ParallelIterator::chain(
            a.into_par_iter().map(|x| x * 2),
            b.into_par_iter().map(|x| x * 3),
        )
        .collect::<Vec<_>>();
        assert_eq!(result, vec![2, 4, 6, 12, 15, 18]);
    }

    #[test]
    fn check_chain_len() {
        let a = &[1, 2, 3];
        let b = &[4, 5, 6, 7];

        let chained = ParallelIterator::chain(a.into_par_iter(), b.into_par_iter());
        assert_eq!(chained.len(), 7);
    }

    #[test]
    fn check_chain_returns_indexed_parallel_iterator() {
        // Test that chaining two IndexedParallelIterators returns an IndexedParallelIterator
        let a = &[1, 2, 3];
        let b = &[4, 5, 6, 7];

        // This should compile because chain returns an IndexedParallelIterator
        let chained = ParallelIterator::chain(a.into_par_iter(), b.into_par_iter());

        // We can call len() which is only available on IndexedParallelIterator
        assert_eq!(chained.len(), 7);

        // We can also chain again, which shows it's still indexed
        let c = &[8, 9];
        let triple_chain = ParallelIterator::chain(chained, c.into_par_iter());
        assert_eq!(triple_chain.len(), 9);

        // And collect the results
        let result = triple_chain.collect::<Vec<_>>();
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
