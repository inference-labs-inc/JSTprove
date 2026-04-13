use arith::Field;
use jst_gkr_engine::{FiatShamirTranscript, Proof};
use sha2::{Digest, Sha256};

#[derive(Clone, Default)]
pub struct Sha256Transcript {
    state: Vec<u8>,
    proof_data: Vec<u8>,
}

impl FiatShamirTranscript for Sha256Transcript {
    fn append_field_element<F: Field>(&mut self, element: &F) {
        let mut buf = vec![0u8; F::SIZE];
        element.to_bytes(&mut buf);
        self.state.extend_from_slice(&buf);
        self.proof_data.extend_from_slice(&buf);
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.state.extend_from_slice(bytes);
        self.proof_data.extend_from_slice(bytes);
    }

    fn challenge_field_element<F: Field>(&mut self) -> F {
        let mut expanded = Vec::with_capacity(64);
        for counter in 0u8..2 {
            let mut hasher = Sha256::new();
            hasher.update(&self.state);
            hasher.update([counter]);
            expanded.extend_from_slice(&hasher.finalize());
        }
        self.state = expanded[..32].to_vec();
        F::from_uniform_bytes(&expanded)
    }

    fn finalize_proof(&self) -> Proof {
        Proof {
            data: self.proof_data.clone(),
        }
    }

    fn parse_proof(proof: &Proof) -> Self {
        Self {
            state: Vec::new(),
            proof_data: proof.data.clone(),
        }
    }
}
