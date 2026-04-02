// Copyright 2025 Irreducible Inc.

use super::{ParallelIterator, ParallelWrapper};

pub trait ParallelBridge: Sized + Iterator {
    fn par_bridge(self) -> impl ParallelIterator<Item = Self::Item>;
}

impl<I: Iterator> ParallelBridge for I {
    fn par_bridge(self) -> impl ParallelIterator<Item = Self::Item> {
        ParallelWrapper::new(self)
    }
}
