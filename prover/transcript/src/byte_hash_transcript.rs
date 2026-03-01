use gkr_engine::{Proof, Transcript};
use gkr_hashers::FiatShamirHasher;

// When appending the initial commitment, we hash the commitment bytes
// for sufficient number of times, so that the FS hash has a sufficient circuit depth

#[cfg(not(feature = "recursion"))]
const PCS_DIGEST_LOOP: usize = 1000;

#[derive(Clone, Default, Debug, PartialEq)]
pub struct BytesHashTranscript<H: FiatShamirHasher> {
    hasher: H,

    /// The digest bytes.
    pub digest: Vec<u8>,

    /// The proof bytes.
    proof: Proof,

    /// The pointer to the proof bytes indicating where the hash starts.
    hash_start_index: usize,

    /// locking point
    proof_locked: bool,
    proof_locked_at: usize,
}

impl<H: FiatShamirHasher> BytesHashTranscript<H> {
    /// When appending the initial commitment, we hash the commitment bytes
    /// for sufficient number of times, so that the FS hash has a sufficient circuit depth
    #[cfg(not(feature = "recursion"))]
    #[inline(always)]
    fn hash_init_commitment(&mut self, commitment_bytes: &[u8]) -> Vec<u8> {
        let mut digest = vec![0u8; H::DIGEST_SIZE];
        self.hasher.hash(&mut digest, commitment_bytes);
        for _ in 0..PCS_DIGEST_LOOP {
            self.hasher.hash_inplace(&mut digest);
        }
        digest
    }
}

impl<H: FiatShamirHasher> Transcript for BytesHashTranscript<H> {
    fn new() -> Self {
        Self {
            hasher: H::new(),
            digest: vec![0u8; H::DIGEST_SIZE],
            proof: Proof::default(),
            hash_start_index: 0,
            proof_locked: false,
            proof_locked_at: 0,
        }
    }

    #[inline]
    fn append_commitment(&mut self, commitment_bytes: &[u8]) {
        self.append_u8_slice(commitment_bytes);

        #[cfg(not(feature = "recursion"))]
        {
            let digest = self.hash_init_commitment(commitment_bytes);
            self.set_state(&digest);
        }
    }

    /// Append a byte slice to the transcript.
    #[inline(always)]
    fn append_u8_slice(&mut self, buffer: &[u8]) {
        self.proof.bytes.extend_from_slice(buffer);
    }

    #[inline]
    fn generate_u8_slice(&mut self, n_bytes: usize) -> Vec<u8> {
        let mut ret = vec![0u8; n_bytes];
        let mut cur_n_bytes = 0usize;

        while cur_n_bytes < n_bytes {
            self.refresh_digest();
            let digest_len = H::DIGEST_SIZE.min(n_bytes - cur_n_bytes);
            ret[cur_n_bytes..cur_n_bytes + digest_len].copy_from_slice(&self.digest[..digest_len]);
            cur_n_bytes += digest_len;
        }

        ret
    }

    #[inline(always)]
    fn finalize_and_get_proof(&mut self) -> Proof {
        if self.proof_locked {
            self.unlock_proof();
        }
        self.proof.clone()
    }

    #[inline(always)]
    fn hash_and_return_state(&mut self) -> Vec<u8> {
        self.refresh_digest();
        self.digest.clone()
    }

    #[inline(always)]
    fn set_state(&mut self, state: &[u8]) {
        self.hash_start_index = self.proof.bytes.len(); // discard unhashed data
        assert!(state.len() == H::DIGEST_SIZE);
        self.digest = state.to_vec();
    }

    #[inline(always)]
    fn lock_proof(&mut self) {
        assert!(!self.proof_locked);
        self.proof_locked = true;
        self.proof_locked_at = self.proof.bytes.len();
    }

    #[inline(always)]
    fn unlock_proof(&mut self) {
        assert!(self.proof_locked);
        self.proof_locked = false;
        if self.hash_start_index < self.proof.bytes.len() {
            self.refresh_digest();
        }
        self.proof.bytes.resize(self.proof_locked_at, 0);
        self.hash_start_index = self.proof.bytes.len();
    }

    #[inline]
    fn refresh_digest(&mut self) {
        let hash_end_index = self.proof.bytes.len();
        if hash_end_index > self.hash_start_index {
            let hash_inputs = {
                let mut res = self.digest.clone();
                res.extend_from_slice(&self.proof.bytes[self.hash_start_index..hash_end_index]);
                res
            };

            self.hasher.hash(&mut self.digest, &hash_inputs);
            self.hash_start_index = hash_end_index;
        } else {
            self.hasher.hash_inplace(&mut self.digest);
        }
    }
}
