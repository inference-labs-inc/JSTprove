// Copyright 2025 Irreducible Inc.
//! SIMD binary field operations for the Rijndael field.
//!
//! This module implements arithmetic operations in the GF(2^8) binary field used by
//! the AES/Rijndael S-Box. This is the finite field with 256 elements defined by the
//! reduction polynomial x^8 + x^4 + x^3 + x + 1.
//!
//! The Rijndael field is fundamental to AES encryption, where it's used in the SubBytes
//! transformation (S-Box) to provide non-linearity in the cipher.

pub mod gfni;
#[cfg(target_arch = "aarch64")]
pub mod vmull;
