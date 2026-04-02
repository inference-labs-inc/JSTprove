// Copyright 2025 Irreducible Inc.

use binius_verifier::protocols::shift::{BITAND_ARITY, INTMUL_ARITY, SHIFT_VARIANT_COUNT};

mod error;
mod key_collection;
mod monster;
mod phase_1;
mod phase_2;
mod prove;

pub use error::Error;
pub use key_collection::{KeyCollection, build_key_collection};
pub use prove::{OperatorData, PreparedOperatorData, prove};
