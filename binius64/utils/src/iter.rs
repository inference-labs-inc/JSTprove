// Copyright 2024-2025 Irreducible Inc.

use std::iter::FusedIterator;

pub trait IterExtensions: Iterator + Sized {
    fn map_skippable<R, F>(self, f: F) -> SkippableMap<Self, F>
    where
        F: Fn(Self::Item) -> R,
    {
        SkippableMap::new(self, f)
    }
}

impl<T: Iterator + Sized> IterExtensions for T {}

/// A map iterator that skips values when `nth` is called.
///
/// `std::iter::Map` guarantees that the function will be called for every value of the inner
/// iterator. However, it makes it impossible to implement `nth` in efficient way. `SkippableMap`
/// pretty much follows the interface of the `std::iter::Map` except that F is required to be `Fn`
/// instead of `FnMut`.
#[derive(Debug, Clone)]
pub struct SkippableMap<I, F> {
    iter: I,
    func: F,
}

impl<I, F> SkippableMap<I, F> {
    const fn new(iter: I, func: F) -> Self {
        Self { iter, func }
    }
}

impl<B, I: Iterator, F> Iterator for SkippableMap<I, F>
where
    F: Fn(I::Item) -> B,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().map(&self.func)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn fold<Acc, G>(self, init: Acc, mut g: G) -> Acc
    where
        G: FnMut(Acc, Self::Item) -> Acc,
    {
        let func = self.func;
        self.iter.fold(init, move |acc, elt| g(acc, func(elt)))
    }

    // Consider `advance_by` once it gets stabilised (https://github.com/rust-lang/rust/issues/77404)
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n).map(&self.func)
    }
}

impl<B, I: ExactSizeIterator, F> ExactSizeIterator for SkippableMap<I, F>
where
    F: Fn(I::Item) -> B,
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<B, I: FusedIterator, F> FusedIterator for SkippableMap<I, F> where F: Fn(I::Item) -> B {}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[test]
    fn test_map_skippable() {
        // Test that obervable behaviour is equivalent to `map`
        let vals = [1, 2, 3, 4, 5];
        assert_eq!(
            vals.iter().map(|v| v * v).collect::<Vec<_>>(),
            vals.iter().map_skippable(|v| v * v).collect::<Vec<_>>()
        );
        assert_eq!(
            vals.iter().map(|v| v * v).fold(0, |l, r| l + 2 * r),
            vals.iter()
                .map_skippable(|v| v * v)
                .fold(0, |l, r| l + 2 * r)
        );
        assert_eq!(
            vals.iter().size_hint(),
            vals.iter().map_skippable(|v| v * v).size_hint()
        );

        // Test that `nth` skips values
        let count = RefCell::new(0);
        let mut iter = vals.iter().map_skippable(|i| {
            *count.borrow_mut() += 1;
            i * i
        });
        assert_eq!(iter.nth(3), Some(16));
        assert_eq!(*count.borrow(), 1);
        assert_eq!(iter.nth(2), None);
        assert_eq!(*count.borrow(), 1);
    }
}
