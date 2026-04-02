// Copyright 2024-2025 Irreducible Inc.

use std::ops::Mul;

use crate::{
    arch::ReuseMultiplyStrategy,
    arithmetic_traits::{TaggedMulAlpha, TaggedSquare},
};

impl<T> TaggedSquare<ReuseMultiplyStrategy> for T
where
    T: Mul<Self, Output = Self> + Copy,
{
    fn square(self) -> Self {
        self * self
    }
}

pub trait Alpha {
    fn alpha() -> Self;
}

impl<T> TaggedMulAlpha<ReuseMultiplyStrategy> for T
where
    T: Mul<Self, Output = Self> + Alpha,
{
    #[inline]
    fn mul_alpha(self) -> Self {
        self * Self::alpha()
    }
}
