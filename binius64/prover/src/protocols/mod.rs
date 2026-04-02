// Copyright 2025 Irreducible Inc.

mod inout_check;
pub mod intmul;
pub mod shift;

// Re-export from binius-iop-prover for backward compatibility
pub use binius_iop_prover::basefold;
// Re-export from binius-ip-prover for backward compatibility
pub use binius_ip_prover::{fracaddcheck, prodcheck, sumcheck};
pub use inout_check::InOutCheckProver;
