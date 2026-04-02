// Copyright 2025 Irreducible Inc.
//! Hint system.
//!
//! Hints are deterministic computations that happen on the prover side.
//!
//! They can be used for operations that require many constraints to compute but few constraints
//! to verify.

use binius_core::Word;

mod big_uint_divide;
mod big_uint_mod_pow;
mod mod_inverse;
mod secp256k1_endosplit;

pub use big_uint_divide::BigUintDivideHint;
pub use big_uint_mod_pow::BigUintModPowHint;
pub use mod_inverse::ModInverseHint;
pub use secp256k1_endosplit::Secp256k1EndosplitHint;

pub type HintId = u32;

/// Hint handler trait for extensible operations
pub trait Hint: Send + Sync {
    /// Execute the hint with given inputs, writing outputs
    fn execute(&self, dimensions: &[usize], inputs: &[Word], outputs: &mut [Word]);

    /// Get the shape of this hint (n_inputs, n_outputs)
    #[allow(unused)]
    fn shape(&self, dimensions: &[usize]) -> (usize, usize);
}

/// Registry for hint handlers
pub struct HintRegistry {
    handlers: Vec<Box<dyn Hint>>,
}

impl HintRegistry {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    pub fn register(&mut self, handler: Box<dyn Hint>) -> HintId {
        let id = self.handlers.len() as HintId;
        self.handlers.push(handler);
        id
    }

    pub fn execute(
        &self,
        hint_id: usize,
        dimensions: &[usize],
        inputs: &[Word],
        outputs: &mut [Word],
    ) {
        self.handlers[hint_id].execute(dimensions, inputs, outputs);
    }
}

impl Default for HintRegistry {
    fn default() -> Self {
        Self::new()
    }
}
