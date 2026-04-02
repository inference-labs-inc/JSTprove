// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

use crate::rayon::iter::{IndexedParallelIteratorInner, ParallelWrapper};

pub trait ParallelSliceMut<T: Sync> {
    fn as_parallel_slice_mut(&mut self) -> &mut [T];

    #[inline(always)]
    fn par_split_mut<P>(&mut self, separator: P) -> ParallelWrapper<std::slice::SplitMut<'_, T, P>>
    where
        P: Fn(&T) -> bool + Sync + Send,
    {
        ParallelWrapper::new(self.as_parallel_slice_mut().split_mut(separator))
    }

    #[inline(always)]
    fn par_split_inclusive_mut<P>(
        &mut self,
        separator: P,
    ) -> ParallelWrapper<std::slice::SplitInclusiveMut<'_, T, P>>
    where
        P: Fn(&T) -> bool + Sync + Send,
    {
        ParallelWrapper::new(self.as_parallel_slice_mut().split_inclusive_mut(separator))
    }

    #[inline(always)]
    fn par_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> ParallelWrapper<std::slice::ChunksMut<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice_mut().chunks_mut(chunk_size))
    }

    #[inline(always)]
    fn par_chunks_exact_mut(
        &mut self,
        chunk_size: usize,
    ) -> ParallelWrapper<std::slice::ChunksExactMut<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice_mut().chunks_exact_mut(chunk_size))
    }

    #[inline(always)]
    fn par_rchunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> ParallelWrapper<std::slice::RChunksMut<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice_mut().rchunks_mut(chunk_size))
    }

    #[inline(always)]
    fn par_rchunks_exact_mut(
        &mut self,
        chunk_size: usize,
    ) -> ParallelWrapper<std::slice::RChunksExactMut<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice_mut().rchunks_exact_mut(chunk_size))
    }

    #[inline(always)]
    fn par_sort(&mut self)
    where
        T: Ord,
    {
        self.as_parallel_slice_mut().sort()
    }

    #[inline(always)]
    fn par_sort_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> std::cmp::Ordering + Sync,
    {
        self.as_parallel_slice_mut().sort_by(compare)
    }

    #[inline(always)]
    fn par_sort_by_key<K, F>(&mut self, f: F)
    where
        K: Ord,
        F: Fn(&T) -> K + Sync,
    {
        self.as_parallel_slice_mut().sort_by_key(f)
    }

    #[inline(always)]
    fn par_sort_by_cached_key<K, F>(&mut self, f: F)
    where
        F: Fn(&T) -> K + Sync,
        K: Ord + Send,
    {
        self.as_parallel_slice_mut().sort_by_cached_key(f)
    }

    #[inline(always)]
    fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.as_parallel_slice_mut().sort_unstable()
    }

    #[inline(always)]
    fn par_sort_unstable_by<F>(&mut self, compare: F)
    where
        F: Fn(&T, &T) -> std::cmp::Ordering + Sync,
    {
        self.as_parallel_slice_mut().sort_unstable_by(compare)
    }

    #[inline(always)]
    fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        K: Ord,
        F: Fn(&T) -> K + Sync,
    {
        self.as_parallel_slice_mut().sort_unstable_by_key(f)
    }

    #[inline(always)]
    fn par_chunk_by_mut<F>(&mut self, pred: F) -> ParallelWrapper<std::slice::ChunkByMut<'_, T, F>>
    where
        F: Fn(&T, &T) -> bool + Send + Sync,
    {
        ParallelWrapper::new(self.as_parallel_slice_mut().chunk_by_mut(pred))
    }
}

impl<T: Sync> ParallelSliceMut<T> for [T] {
    #[inline(always)]
    fn as_parallel_slice_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::ChunksExactMut<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::ChunksMut<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::RChunksMut<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::RChunksExactMut<'a, T> {}
