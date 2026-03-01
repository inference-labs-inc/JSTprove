use arith::{ExtensionField, Field};
use gkr_engine::Transcript;

/// A transcript that uses a random tape to generate challenges.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct RandomTape<ChallengeF: ExtensionField> {
    /// The random tape used to generate the transcript.
    pub tape: Vec<ChallengeF>,
    /// The current position in the random tape.
    pub position: usize,
}

impl<ChallengeF: ExtensionField> RandomTape<ChallengeF> {
    /// Creates a new `RandomTape` with the given random tape.
    pub fn new_with_tape(tape: Vec<ChallengeF>) -> Self {
        Self { tape, position: 0 }
    }
}

impl<ChallengeF: ExtensionField> Transcript for RandomTape<ChallengeF> {
    fn new() -> Self {
        Self {
            tape: vec![],
            position: 0,
        }
    }

    /// Generate a field element.
    /// In the case of a random tape, we actually require F to be the same as ChallengeF.
    #[inline(always)]
    fn generate_field_element<F: Field>(&mut self) -> F {
        if self.position >= self.tape.len() {
            panic!("Random tape exhausted");
        }
        let element = self.tape[self.position];
        self.position += 1;
        let mut element_to_return = F::ZERO;
        assert!(F::NAME == ChallengeF::NAME);
        unsafe {
            std::ptr::copy_nonoverlapping(
                &element as *const ChallengeF as *const u8,
                &mut element_to_return as *mut F as *mut u8,
                F::SIZE,
            );
        }
        element_to_return
    }

    fn append_commitment(&mut self, _commitment_bytes: &[u8]) {
        unimplemented!()
    }

    // Do nothing, randomness are already stored in the tape
    fn append_u8_slice(&mut self, _buffer: &[u8]) {}

    fn generate_u8_slice(&mut self, _n_bytes: usize) -> Vec<u8> {
        unimplemented!()
    }

    fn finalize_and_get_proof(&mut self) -> gkr_engine::Proof {
        unimplemented!()
    }

    fn hash_and_return_state(&mut self) -> Vec<u8> {
        unimplemented!()
    }

    fn set_state(&mut self, _state: &[u8]) {
        unimplemented!()
    }

    fn lock_proof(&mut self) {
        unimplemented!()
    }

    fn unlock_proof(&mut self) {
        unimplemented!()
    }

    fn refresh_digest(&mut self) {
        unimplemented!()
    }
}
