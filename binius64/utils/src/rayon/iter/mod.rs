// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

mod from_parallel_iterator;
mod indexed_parallel_iterator;
mod into_parallel_iterator;
mod par_bridge;
mod parallel_iterator;
mod parallel_wrapper;

pub use core::iter::{empty, once, repeat, repeat_n as repeatn};

pub use from_parallel_iterator::FromParallelIterator;
pub use indexed_parallel_iterator::IndexedParallelIterator;
pub(super) use indexed_parallel_iterator::IndexedParallelIteratorInner;
pub use into_parallel_iterator::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
};
pub use itertools::Either;
pub use par_bridge::ParallelBridge;
pub use parallel_iterator::ParallelIterator;
pub(super) use parallel_wrapper::ParallelWrapper;
