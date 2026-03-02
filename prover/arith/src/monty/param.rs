use std::{fmt::Debug, hash::Hash};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use super::avx512::PackedMontyParameters;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
use super::avx256::PackedMontyParameters;

#[cfg(target_arch = "aarch64")]
use super::neon::PackedMontyParameters;
use super::MontyField31;

/// MontyParameters contains the prime P along with constants needed to convert elements into and
/// out of MONTY form. The MONTY constant is assumed to be a power of 2.
pub trait MontyParameters:
    Copy + Clone + Default + Debug + Eq + PartialEq + Sync + Send + Hash + 'static
{
    // A 31-bit prime.
    const PRIME: u32;

    // The log_2 of our MONTY constant.
    const MONTY_BITS: u32;

    // We define MONTY_MU = PRIME^-1 (mod 2^MONTY_BITS). This is different from the usual convention
    // (MONTY_MU = -PRIME^-1 (mod 2^MONTY_BITS)) but it avoids a carry.
    const MONTY_MU: u32;

    const MONTY_MASK: u32 = ((1u64 << Self::MONTY_BITS) - 1) as u32;
}

/// FieldParameters contains constants and methods needed to imply PrimeCharacteristicRing, Field
/// and PrimeField32 for MontyField31.
pub trait FieldParameters: PackedMontyParameters + Sized {
    // Simple field constants.
    const MONTY_ZERO: MontyField31<Self> = MontyField31::new(0);
    const MONTY_ONE: MontyField31<Self> = MontyField31::new(1);
    const MONTY_TWO: MontyField31<Self> = MontyField31::new(2);
    const MONTY_NEG_ONE: MontyField31<Self> = MontyField31::new(Self::PRIME - 1);

    // A generator of the fields multiplicative group. Needs to be given in Monty Form.
    const MONTY_GEN: MontyField31<Self>;

    const HALF_P_PLUS_1: u32 = (Self::PRIME + 1) >> 1;

    fn try_inverse(a: &MontyField31<Self>) -> Option<MontyField31<Self>>;

    const TWO_ADICITY: usize;

    const TWO_ADICITY_GENERATOR_IN_U32: u32;
}
