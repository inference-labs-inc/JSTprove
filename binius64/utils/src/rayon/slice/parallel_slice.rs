// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

use crate::rayon::iter::{IndexedParallelIteratorInner, ParallelWrapper};

pub trait ParallelSlice<T: Sync> {
    fn as_parallel_slice(&self) -> &[T];

    #[inline(always)]
    fn par_split<P>(&self, separator: P) -> ParallelWrapper<std::slice::Split<'_, T, P>>
    where
        P: Fn(&T) -> bool + Sync + Send,
    {
        ParallelWrapper::new(self.as_parallel_slice().split(separator))
    }

    #[inline(always)]
    fn par_split_inclusive<P>(
        &self,
        separator: P,
    ) -> ParallelWrapper<std::slice::SplitInclusive<'_, T, P>>
    where
        P: Fn(&T) -> bool + Sync + Send,
    {
        ParallelWrapper::new(self.as_parallel_slice().split_inclusive(separator))
    }

    #[inline(always)]
    fn par_windows(&self, window_size: usize) -> ParallelWrapper<std::slice::Windows<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice().windows(window_size))
    }

    #[inline(always)]
    fn par_chunks(&self, chunk_size: usize) -> ParallelWrapper<std::slice::Chunks<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice().chunks(chunk_size))
    }

    #[inline(always)]
    fn par_chunks_exact(
        &self,
        chunk_size: usize,
    ) -> ParallelWrapper<std::slice::ChunksExact<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice().chunks_exact(chunk_size))
    }

    #[inline(always)]
    fn par_rchunks(&self, chunk_size: usize) -> ParallelWrapper<std::slice::RChunks<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice().rchunks(chunk_size))
    }

    #[inline(always)]
    fn par_rchunks_exact(
        &self,
        chunk_size: usize,
    ) -> ParallelWrapper<std::slice::RChunksExact<'_, T>> {
        ParallelWrapper::new(self.as_parallel_slice().rchunks_exact(chunk_size))
    }

    #[inline(always)]
    fn par_chunk_by<F>(&self, pred: F) -> ParallelWrapper<std::slice::ChunkBy<'_, T, F>>
    where
        F: Fn(&T, &T) -> bool + Send + Sync,
    {
        ParallelWrapper::new(self.as_parallel_slice().chunk_by(pred))
    }
}

impl<T: Sync> ParallelSlice<T> for [T] {
    #[inline(always)]
    fn as_parallel_slice(&self) -> &[T] {
        self
    }
}

impl<T> IndexedParallelIteratorInner for std::slice::Iter<'_, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::Chunks<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::Windows<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::ChunksExact<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::RChunks<'a, T> {}
impl<'a, T: 'a> IndexedParallelIteratorInner for std::slice::RChunksExact<'a, T> {}
