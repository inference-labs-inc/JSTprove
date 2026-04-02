// Copyright 2024-2025 Irreducible Inc.
//
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

//! This crate provides a subset of the `rayon` API to allow conditional
//! compilation without `rayon`.
//! This is useful for profiling single-threaded code, as it simplifies call stacks significantly.
//! The initial code was taken from the `maybe-rayon` crate, but many changes were made to
//! support the usage of `ParallelIterator` and `IndexedParallelIterator` methods, which have
//! different signatures from `std::iter::Iterator`. Some of these changes may be potentially
//! backward-incompatible, and given the absence of tests in the original crate, it is very unlikely
//! that it is possible to commit the changes back to the original crate.

cfg_if::cfg_if! {
    if #[cfg(any(not(feature = "rayon"), all(target_arch="wasm32", not(target_feature = "atomics"))))] {
        use std::marker::PhantomData;

        pub mod iter;
        pub mod slice;

        pub mod prelude {
            pub use super::{iter::*, slice::*};
        }

        #[derive(Default)]
        pub struct ThreadPoolBuilder();
        impl ThreadPoolBuilder {
            #[inline(always)]
            pub const fn new() -> Self {
                Self()
            }

            #[inline(always)]
            pub const fn build(self) -> Result<ThreadPool, ::core::convert::Infallible> {
                Ok(ThreadPool())
            }

            #[inline(always)]
            pub const fn num_threads(self, _num_threads: usize) -> Self {
                Self()
            }

            #[inline(always)]
            pub const fn use_current_thread(self) -> Self {
                Self()
            }

            #[inline(always)]
            pub const fn build_global(self) -> Result<(), ThreadPoolBuildError> {
                Ok(())
            }
        }

        #[derive(Debug)]
        pub struct ThreadPool();
        impl ThreadPool {
            #[inline(always)]
            pub fn install<OP, R>(&self, op: OP) -> R
            where
                OP: FnOnce() -> R + Send,
                R: Send,
            {
                op()
            }
        }

        #[derive(Debug, Default)]
        pub struct ThreadPoolBuildError;
        impl std::fmt::Display for ThreadPoolBuildError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "ThreadPoolBuildError")
            }
        }
        impl std::error::Error for ThreadPoolBuildError {
            fn description(&self) -> &str {
                "Error building thread pool"
            }
        }

        #[inline(always)]
        pub const fn current_num_threads() -> usize {
            1
        }

        #[inline(always)]
        pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
        where
            A: FnOnce() -> RA + Send,
            B: FnOnce() -> RB + Send,
            RA: Send,
            RB: Send,
        {
            (oper_a(), oper_b())
        }

        pub struct Scope<'scope> {
            #[allow(clippy::type_complexity)]
            marker: PhantomData<Box<dyn FnOnce(&Scope<'scope>) + Send + Sync + 'scope>>,
        }

        impl<'scope> Scope<'scope> {
            #[inline(always)]
            pub fn spawn<BODY>(&self, body: BODY)
            where
                BODY: FnOnce(&Self) + Send + 'scope,
            {
                body(self)
            }
        }

        #[inline(always)]
        pub fn scope<'scope, OP, R>(op: OP) -> R
        where
            OP: for<'s> FnOnce(&'s Scope<'scope>) -> R + 'scope + Send,
            R: Send,
        {
            op(&Scope {
                marker: PhantomData,
            })
        }
    } else {
        pub use rayon::*;
    }
}

pub mod config;
