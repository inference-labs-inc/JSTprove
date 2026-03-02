use tiny_keccak::{Hasher, Keccak};

use crate::FiatShamirHasher;

#[derive(Clone, Default, Debug)]
pub struct Keccak256hasher {}

impl FiatShamirHasher for Keccak256hasher {
    const NAME: &'static str = "Keccak256 Hasher";

    const DIGEST_SIZE: usize = 32;

    #[inline]
    fn new() -> Keccak256hasher {
        Keccak256hasher {}
    }

    #[inline]
    fn hash(&self, output: &mut [u8], input: &[u8]) {
        let mut hasher = Keccak::v256();
        hasher.update(input);
        hasher.finalize(output);
    }

    #[inline]
    fn hash_inplace(&self, buffer: &mut [u8]) {
        let mut hasher = Keccak::v256();
        hasher.update(&*buffer);
        hasher.finalize(buffer);
    }
}
