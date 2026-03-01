use std::{fmt::Debug, str::FromStr};

use arith::Field;
use serdes::ExpSerde;

use crate::ExpErrors;

use super::Proof;

/// A trait for transcript generation over the challenge field
/// The associated field is the challenge field, i.e., M31Ext3
/// The challenge field is not SIMD enabled
pub trait Transcript: Clone + Debug {
    /// Create a new transcript.
    fn new() -> Self;

    /// Append a polynomial commitment to the transcript
    /// called by the prover
    fn append_commitment(&mut self, commitment_bytes: &[u8]);

    /// Append a field element to the transcript.
    #[inline]
    fn append_field_element<F: Field>(&mut self, f: &F) {
        let mut buf = vec![];
        f.serialize_into(&mut buf).unwrap();
        self.append_u8_slice(&buf);
    }

    /// Append a slice of bytes
    fn append_u8_slice(&mut self, buffer: &[u8]);

    fn append_serializable_data<T: ExpSerde>(&mut self, data: &T) {
        let mut buf = vec![];
        data.serialize_into(&mut buf).unwrap();
        self.append_u8_slice(&buf);
    }

    /*
        self.proof.bytes.extend_from_slice(buffer);
    */

    /// Generate a slice of random bytes.
    fn generate_u8_slice(&mut self, n_bytes: usize) -> Vec<u8>;

    fn generate_usize_vector(&mut self, n: usize) -> Vec<usize> {
        let mut res: Vec<usize> = vec![0; n];
        let mut buf = [0u8; 8];
        for e in res.iter_mut() {
            buf.copy_from_slice(&self.generate_u8_slice(8));
            *e = usize::from_le_bytes(buf);
        }
        res
    }

    /// Generate a field element.
    #[inline(always)]
    fn generate_field_element<F: Field>(&mut self) -> F {
        F::from_uniform_bytes(&self.generate_u8_slice(F::SIZE))
    }

    /// Generate a field element vector.
    #[inline(always)]
    fn generate_field_elements<F: Field>(&mut self, n: usize) -> Vec<F> {
        let mut res = vec![F::ZERO; n];
        for e in res.iter_mut() {
            *e = self.generate_field_element();
        }
        res
    }

    /// Produce the proof
    /// It is not recommended to append/challenge after calling this function
    fn finalize_and_get_proof(&mut self) -> Proof;

    /// Return current state of the transcript
    /// Note: this may incur an additional hash to shrink the state
    fn hash_and_return_state(&mut self) -> Vec<u8>;
    /*
        self.refresh_digest();
        self.digets.clone();
    */

    /// Set the state
    /// Note: Any unhashed data will be discarded
    fn set_state(&mut self, state: &[u8]);
    /*
        self.hash_start_index = self.proof.bytes.len(); // discard unhashed data
        assert!(state.len() == H::DIGEST_SIZE);
        self.digest = state.to_vec();
    */

    /// lock proof, no changes will be made to the proof until unlock is called
    fn lock_proof(&mut self);
    /*
        assert!(!self.proof_locked);
        self.proof_locked = true;
        self.proof_locked_at = self.proof.bytes.len();
    */

    /// unlock proof
    fn unlock_proof(&mut self);

    fn refresh_digest(&mut self);
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum FiatShamirHashType {
    #[default]
    SHA256,
    Keccak256,
    Poseidon,
    Animoe,
    MIMC5, // Note: use MIMC5 for bn254 ONLY
}

impl FromStr for FiatShamirHashType {
    type Err = ExpErrors;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SHA256" => Ok(FiatShamirHashType::SHA256),
            "Keccak256" => Ok(FiatShamirHashType::Keccak256),
            "Poseidon" => Ok(FiatShamirHashType::Poseidon),
            "Animoe" => Ok(FiatShamirHashType::Animoe),
            "MIMC5" => Ok(FiatShamirHashType::MIMC5),
            _ => Err(ExpErrors::FiatShamirHashTypeError(s.to_string())),
        }
    }
}
