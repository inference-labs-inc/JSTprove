// Copyright 2024-2025 Irreducible Inc.

use rand::distr::{Distribution, StandardUniform};

/// A value that can be randomly generated
pub trait Random {
    /// Generate random value
    fn random(rng: impl rand::Rng) -> Self;
}

impl<T> Random for T
where
    StandardUniform: Distribution<T>,
{
    fn random(mut rng: impl rand::Rng) -> Self {
        rng.random()
    }
}
