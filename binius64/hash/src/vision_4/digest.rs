// Copyright 2025 Irreducible Inc.

use binius_field::{BinaryField128bGhash as Ghash, Field};
use binius_utils::{DeserializeBytes, SerializeBytes};
use digest::{
    FixedOutput, FixedOutputReset, HashMarker, OutputSizeUser, Reset, Update, consts::U32,
    core_api::BlockSizeUser,
};

use super::{constants::M, permutation::permutation};

pub const RATE_AS_U128: usize = 2;
pub const RATE_AS_U8: usize = RATE_AS_U128 * std::mem::size_of::<u128>();

const PADDING_START: u8 = 0x80;
const PADDING_END: u8 = 0x01;

pub const PADDING_BLOCK: [u8; RATE_AS_U8] = {
    let mut block = [0; RATE_AS_U8];
    block[0] = PADDING_START;
    block[RATE_AS_U8 - 1] |= PADDING_END;
    block
};

/// Fill the data using Keccak padding scheme.
#[inline(always)]
pub fn fill_padding(data: &mut [u8]) {
    debug_assert!(!data.is_empty() && data.len() <= RATE_AS_U8);

    data.fill(0);
    data[0] |= PADDING_START;
    data[data.len() - 1] |= PADDING_END;
}

/// An implementation of the Vision permutation with 4 Ghash elements for state.
#[derive(Clone)]
pub struct VisionHasherDigest {
    state: [Ghash; M],
    buffer: [u8; RATE_AS_U8],
    filled_bytes: usize,
}

impl Default for VisionHasherDigest {
    fn default() -> Self {
        Self {
            state: [Ghash::ZERO; M],
            buffer: [0; RATE_AS_U8],
            filled_bytes: 0,
        }
    }
}

impl VisionHasherDigest {
    pub fn permute(state: &mut [Ghash; M], data: &[u8]) {
        debug_assert_eq!(data.len(), RATE_AS_U8);

        // Overwrite first RATE_AS_U128 elements of state with data
        for i in 0..RATE_AS_U128 {
            state[i] = Ghash::deserialize(&data[i * 16..]).expect("data len checked");
        }

        permutation(state);
    }

    fn finalize(&mut self, out: &mut digest::Output<Self>) {
        if self.filled_bytes != 0 {
            fill_padding(&mut self.buffer[self.filled_bytes..]);
            Self::permute(&mut self.state, &self.buffer);
        } else {
            Self::permute(&mut self.state, &PADDING_BLOCK);
        }

        // Serialize first two state elements to output (32 bytes total)
        let (state0, state1) = out.as_mut_slice().split_at_mut(16);
        self.state[0].serialize(state0).expect("fits in 16 bytes");
        self.state[1].serialize(state1).expect("fits in 16 bytes");
    }
}

impl HashMarker for VisionHasherDigest {}

impl Update for VisionHasherDigest {
    fn update(&mut self, mut data: &[u8]) {
        if self.filled_bytes != 0 {
            let to_copy = std::cmp::min(data.len(), RATE_AS_U8 - self.filled_bytes);
            self.buffer[self.filled_bytes..self.filled_bytes + to_copy]
                .copy_from_slice(&data[..to_copy]);
            data = &data[to_copy..];
            self.filled_bytes += to_copy;

            if self.filled_bytes == RATE_AS_U8 {
                Self::permute(&mut self.state, &self.buffer);
                self.filled_bytes = 0;
            }
        }

        let mut chunks = data.chunks_exact(RATE_AS_U8);
        for chunk in &mut chunks {
            Self::permute(&mut self.state, chunk);
        }

        let remaining = chunks.remainder();
        if !remaining.is_empty() {
            self.buffer[..remaining.len()].copy_from_slice(remaining);
            self.filled_bytes = remaining.len();
        }
    }
}

impl OutputSizeUser for VisionHasherDigest {
    type OutputSize = U32;
}

impl BlockSizeUser for VisionHasherDigest {
    type BlockSize = U32;
}

impl FixedOutput for VisionHasherDigest {
    fn finalize_into(mut self, out: &mut digest::Output<Self>) {
        Self::finalize(&mut self, out);
    }
}

impl FixedOutputReset for VisionHasherDigest {
    fn finalize_into_reset(&mut self, out: &mut digest::Output<Self>) {
        Self::finalize(self, out);
        Reset::reset(self);
    }
}

impl Reset for VisionHasherDigest {
    fn reset(&mut self) {
        self.state = [Ghash::ZERO; M];
        self.buffer = [0; RATE_AS_U8];
        self.filled_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use digest::Digest;

    use super::VisionHasherDigest;

    const INPUT: &[u8] = "One part of the mysterious existence of Captain Nemo had been unveiled and, if his identity had not been recognised, at least, the nations united against him were no longer hunting a chimerical creature, but a man who had vowed a deadly hatred against them".as_bytes();

    #[test]
    fn test_multi_block_aligned() {
        let mut hasher = VisionHasherDigest::default();

        hasher.update(INPUT);
        let out = hasher.finalize();

        let mut hasher = VisionHasherDigest::default();
        let input_as_b = INPUT;
        hasher.update(&input_as_b[0..63]);
        hasher.update(&input_as_b[63..128]);
        hasher.update(&input_as_b[128..163]);
        hasher.update(&input_as_b[163..]);

        assert_eq!(out, hasher.finalize());
    }

    #[test]
    fn test_multi_block_unaligned() {
        let mut hasher = VisionHasherDigest::default();
        hasher.update(INPUT);
        let out = hasher.finalize();

        let mut hasher = VisionHasherDigest::default();
        let input_as_b = INPUT;
        hasher.update(&input_as_b[0..1]);
        hasher.update(&input_as_b[1..120]);
        hasher.update(&input_as_b[120..120]);
        hasher.update(&input_as_b[120..]);

        assert_eq!(out, hasher.finalize());
    }
}
