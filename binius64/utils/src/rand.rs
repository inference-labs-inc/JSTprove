// Copyright 2026 The Binius Developers

//! Parallel random number generation utilities.

use rand::{Rng, SeedableRng};

use crate::rayon::prelude::*;

/// Generates random values in parallel using a deterministic per-index seeding scheme.
///
/// Creates a base seed from the provided RNG, then for each index `i` in `0..n`,
/// XORs the index bytes into the seed to create a unique but deterministic seed
/// for that index's RNG.
pub fn par_rand<InnerR, T, F>(
    n: usize,
    mut rng: impl Rng,
    f: F,
) -> impl IndexedParallelIterator<Item = T>
where
    InnerR: SeedableRng,
    InnerR::Seed: Send + Sync,
    T: Send,
    F: Fn(InnerR) -> T + Sync + Send,
{
    let mut base_seed = InnerR::Seed::default();
    rng.fill_bytes(base_seed.as_mut());

    (0..n).into_par_iter().map(move |i| {
        let mut seed = base_seed.clone();
        let seed_bytes = seed.as_mut();

        let index_bytes = i.to_le_bytes();
        for (seed_byte, &index_byte) in seed_bytes.iter_mut().zip(index_bytes.iter()) {
            *seed_byte ^= index_byte;
        }

        f(InnerR::from_seed(seed))
    })
}
