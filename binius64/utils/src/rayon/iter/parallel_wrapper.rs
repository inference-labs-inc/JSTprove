// Copyright 2025 Irreducible Inc.

/// We use this struct to wrap some inner iterator trate (e.g. `ParallelIteratorInner` or
/// `IndexedParallelIteratorInner`) and implement the outer iterator trait (e.g. `ParallelIterator`
/// or `IndexedParallelIterator`) for it. That allows us avoid collisions with the methods of the
/// `std::iterator::Iterator` for the methods which have different signatures.
pub struct ParallelWrapper<I>(pub(crate) I);

impl<I> ParallelWrapper<I> {
    #[inline(always)]
    pub(crate) const fn new(iter: I) -> Self {
        Self(iter)
    }
}
