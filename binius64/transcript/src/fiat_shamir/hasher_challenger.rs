// Copyright 2024-2025 Irreducible Inc.

use std::{cmp::min, mem};

use bytes::{Buf, BufMut, buf::UninitSlice};
use digest::{
    Digest, FixedOutputReset, Output,
    core_api::{Block, BlockSizeUser},
};

use super::Challenger;

/// Challenger type which implements `[Buf]` that has similar functionality as `[CanSample]`
#[derive(Debug, Default, Clone)]
pub struct HasherSampler<H: Digest> {
    index: usize,
    buffer: Output<H>,
    hasher: H,
}

/// Challenger type which implements `[BufMut]` that has similar functionality as `[CanObserve]`
#[derive(Debug, Default, Clone)]
pub struct HasherObserver<H: Digest + BlockSizeUser> {
    index: usize,
    buffer: Block<H>,
    hasher: H,
}

/// Challenger interface over hashes that implement `[Digest]` trait,
///
/// This challenger works over bytes instead of Field elements
#[derive(Debug, Clone)]
pub enum HasherChallenger<H: Digest + BlockSizeUser> {
    Observer(HasherObserver<H>),
    Sampler(HasherSampler<H>),
}

impl<H> HasherChallenger<H>
where
    H: Digest + BlockSizeUser,
{
    fn new(initial_digest: Output<H>) -> Self {
        let mut hasher = H::new();
        Digest::update(&mut hasher, &initial_digest);

        Self::Sampler(HasherSampler {
            hasher,
            index: 0,
            buffer: initial_digest,
        })
    }
}

impl<H> Default for HasherChallenger<H>
where
    H: Digest + BlockSizeUser + FixedOutputReset,
{
    fn default() -> Self {
        Self::new(H::digest([]))
    }
}

impl<H: Digest + BlockSizeUser + FixedOutputReset + Default> Challenger for HasherChallenger<H> {
    /// This returns the inner challenger which implements `[BufMut]`
    fn observer(&mut self) -> &mut impl BufMut {
        match self {
            Self::Observer(observer) => observer,
            Self::Sampler(sampler) => {
                *self = Self::Observer(mem::take(sampler).into_observer());
                match self {
                    Self::Observer(observer) => observer,
                    _ => unreachable!(),
                }
            }
        }
    }

    /// This returns the inner challenger which implements [`Buf`].
    fn sampler(&mut self) -> &mut impl Buf {
        match self {
            Self::Sampler(sampler) => sampler,
            Self::Observer(observer) => {
                *self = Self::Sampler(mem::take(observer).into_sampler());
                match self {
                    Self::Sampler(sampler) => sampler,
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl<H> HasherSampler<H>
where
    H: Digest + Default + BlockSizeUser,
{
    fn into_observer(mut self) -> HasherObserver<H> {
        Digest::update(&mut self.hasher, self.index.to_le_bytes());

        HasherObserver {
            hasher: self.hasher,
            index: 0,
            buffer: Block::<H>::default(),
        }
    }
}

impl<H> HasherSampler<H>
where
    H: Digest + FixedOutputReset,
{
    fn fill_buffer(&mut self) {
        let digest = self.hasher.finalize_reset();

        // feed forward to the empty state
        Digest::update(&mut self.hasher, &digest);

        self.buffer = digest;
        self.index = 0
    }
}

impl<H> HasherObserver<H>
where
    H: Digest + BlockSizeUser + Default,
{
    fn into_sampler(mut self) -> HasherSampler<H> {
        self.flush();
        HasherSampler {
            hasher: self.hasher,
            index: <H as Digest>::output_size(),
            buffer: Output::<H>::default(),
        }
    }
}

impl<H> HasherObserver<H>
where
    H: Digest + BlockSizeUser,
{
    fn flush(&mut self) {
        self.hasher.update(&self.buffer[..self.index]);
        self.index = 0
    }
}

impl<H> Buf for HasherSampler<H>
where
    H: Digest + FixedOutputReset + Default,
{
    fn remaining(&self) -> usize {
        usize::MAX
    }

    fn chunk(&self) -> &[u8] {
        &self.buffer[self.index..]
    }

    fn advance(&mut self, mut cnt: usize) {
        // Must handle the case when `cnt` is 0
        if self.index == <H as Digest>::output_size() {
            self.fill_buffer();
        }

        while cnt > 0 {
            let remaining = min(<H as Digest>::output_size() - self.index, cnt);
            if remaining == 0 {
                self.fill_buffer();
                continue;
            }
            cnt -= remaining;
            self.index += remaining;
        }
    }
}

unsafe impl<H> BufMut for HasherObserver<H>
where
    H: Digest + BlockSizeUser,
{
    fn remaining_mut(&self) -> usize {
        usize::MAX
    }

    unsafe fn advance_mut(&mut self, mut cnt: usize) {
        while cnt > 0 {
            let remaining = min(<H as BlockSizeUser>::block_size() - self.index, cnt);
            cnt -= remaining;
            self.index += remaining;
            if self.index == <H as BlockSizeUser>::block_size() {
                self.flush();
            }
        }
    }

    fn chunk_mut(&mut self) -> &mut UninitSlice {
        let buffer = &mut self.buffer[self.index..];
        buffer.into()
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use sha2::Sha256;

    use super::*;

    #[test]
    fn test_starting_sampler() {
        let empty_string_digest = Sha256::digest([]);

        let mut hasher = {
            let mut hasher = Sha256::default();
            Digest::update(&mut hasher, empty_string_digest);
            hasher
        };

        let mut challenger = HasherChallenger::<Sha256>::default();

        // first sampling
        let mut out = [0u8; 8];
        challenger.sampler().copy_to_slice(&mut out);

        let first_hash_out = empty_string_digest;

        assert_eq!(first_hash_out[0..8], out);

        // second sampling
        let mut out = [0u8; 24];
        challenger.sampler().copy_to_slice(&mut out);
        assert_eq!(first_hash_out[8..], out);

        // first observing
        challenger.observer().put_slice(&[0x48, 0x55]);
        hasher.update([32, 0, 0, 0, 0, 0, 0, 0]);
        hasher.update([0x48, 0x55]);

        // third sampling
        let mut out_after_observe = [0u8; 2];
        challenger.sampler().copy_to_slice(&mut out_after_observe);

        let second_hash_out = hasher.finalize_reset();

        assert_eq!(out_after_observe, second_hash_out[..2]);
    }

    #[test]
    fn test_starting_observer() {
        let empty_string_digest = Sha256::digest([]);

        let mut hasher = {
            let mut hasher = Sha256::default();
            Digest::update(&mut hasher, empty_string_digest);
            hasher
        };

        let mut challenger = HasherChallenger::<Sha256>::default();

        let mut rng = StdRng::seed_from_u64(0);

        // first observing
        let mut observable = [0u8; 1019];
        rng.fill_bytes(&mut observable);
        challenger.observer().put_slice(&observable[..39]);
        challenger.observer().put_slice(&observable[39..300]);
        challenger.observer().put_slice(&observable[300..987]);
        challenger.observer().put_slice(&observable[987..]);
        hasher.update([0, 0, 0, 0, 0, 0, 0, 0]);
        hasher.update(observable);

        // first sampling
        let mut out = [0u8; 7];
        challenger.sampler().copy_to_slice(&mut out);

        let first_hash_out = hasher.finalize_reset();
        hasher.update(first_hash_out);

        assert_eq!(first_hash_out[..7], out);

        // second observing
        rng.fill_bytes(&mut observable);
        challenger.observer().put_slice(&observable[..128]);
        hasher.update([7, 0, 0, 0, 0, 0, 0, 0]);

        hasher.update(&observable[..128]);

        // second sampling
        let mut out = [0u8; 32];
        challenger.sampler().copy_to_slice(&mut out);

        let second_hasher_out = hasher.finalize_reset();
        hasher.update(second_hasher_out);

        assert_eq!(second_hasher_out[..], out);

        // third observing
        challenger.observer().put_slice(&observable[128..]);
        hasher.update([32, 0, 0, 0, 0, 0, 0, 0]);
        hasher.update(&observable[128..]);

        // third sampling
        let mut out_again = [0u8; 7];
        challenger.sampler().copy_to_slice(&mut out_again);

        let final_hasher_out = hasher.finalize_reset();
        assert_eq!(final_hasher_out[..7], out_again);
    }
}
