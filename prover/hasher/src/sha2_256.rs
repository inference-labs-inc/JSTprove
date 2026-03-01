use sha2::{digest::Output, Digest, Sha256};

use super::FiatShamirHasher;

#[derive(Debug, Clone, Default)]
pub struct SHA256hasher;

impl FiatShamirHasher for SHA256hasher {
    const NAME: &'static str = "SHA256 Hasher";

    const DIGEST_SIZE: usize = 32;

    #[inline]
    fn new() -> SHA256hasher {
        SHA256hasher
    }

    #[inline]
    fn hash(&self, output: &mut [u8], input: &[u8]) {
        let mut hasher = Sha256::new();

        hasher.update(input);
        hasher.finalize_into_reset(Output::<Sha256>::from_mut_slice(output));
    }

    #[inline]
    fn hash_inplace(&self, buffer: &mut [u8]) {
        let mut hasher = Sha256::new();
        hasher.update(&*buffer);
        hasher.finalize_into_reset(Output::<Sha256>::from_mut_slice(buffer));
    }
}
