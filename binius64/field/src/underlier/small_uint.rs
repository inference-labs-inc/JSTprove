// Copyright 2024-2025 Irreducible Inc.

use std::{
    fmt::{Debug, Display, LowerHex},
    hash::{Hash, Hasher},
    ops::{Not, Shl, Shr},
};

use binius_utils::{
    SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
    checked_arithmetics::checked_log_2,
    serialization::DeserializeBytes,
};
use bytemuck::{NoUninit, Zeroable};
use derive_more::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use super::{UnderlierType, underlier_with_bit_ops::UnderlierWithBitOps};
use crate::arch::{interleave_mask_even, interleave_with_mask};

/// Unsigned type with a size strictly less than 8 bits.
#[derive(
    Default,
    Zeroable,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    BitXor,
    BitXorAssign,
)]
#[repr(transparent)]
pub struct SmallU<const N: usize>(u8);

impl<const N: usize> SmallU<N> {
    const _CHECK_SIZE: () = {
        assert!(N < 8);
    };

    /// All bits set to one.
    pub const ONES: Self = Self((1u8 << N) - 1);

    #[inline(always)]
    pub const fn new(val: u8) -> Self {
        Self(val & Self::ONES.0)
    }

    #[inline(always)]
    pub const fn new_unchecked(val: u8) -> Self {
        Self(val)
    }

    #[inline(always)]
    pub const fn val(&self) -> u8 {
        self.0
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        self.val()
            .checked_add(rhs.val())
            .and_then(|value| (value < Self::ONES.0).then_some(Self(value)))
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let a = self.val();
        let b = rhs.val();
        (b > a).then_some(Self(b - a))
    }
}

impl<const N: usize> Debug for SmallU<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.val(), f)
    }
}

impl<const N: usize> Display for SmallU<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.val(), f)
    }
}

impl<const N: usize> LowerHex for SmallU<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LowerHex::fmt(&self.0, f)
    }
}
impl<const N: usize> Hash for SmallU<N> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val().hash(state);
    }
}

impl<const N: usize> Distribution<SmallU<N>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SmallU<N> {
        SmallU(rng.random_range(0..1u8 << N))
    }
}

impl<const N: usize> Shr<usize> for SmallU<N> {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        Self(self.val() >> rhs)
    }
}

impl<const N: usize> Shl<usize> for SmallU<N> {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.val() << rhs) & Self::ONES
    }
}

impl<const N: usize> Not for SmallU<N> {
    type Output = Self;

    fn not(self) -> Self::Output {
        self ^ Self::ONES
    }
}

unsafe impl<const N: usize> NoUninit for SmallU<N> {}

impl<const N: usize> UnderlierType for SmallU<N> {
    const LOG_BITS: usize = checked_log_2(N);
}

impl UnderlierWithBitOps for U1 {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const ONES: Self = Self(1);

    fn interleave(self, _other: Self, _log_block_len: usize) -> (Self, Self) {
        panic!("interleave not supported for U1")
    }
}

impl UnderlierWithBitOps for U2 {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const ONES: Self = Self(0b11);

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[U2] = &[U2::new(interleave_mask_even!(u8, 0))];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

impl UnderlierWithBitOps for U4 {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const ONES: Self = Self(0b1111);

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[U4] = &[
            U4::new(interleave_mask_even!(u8, 0)),
            U4::new(interleave_mask_even!(u8, 1)),
        ];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

impl<const N: usize> From<SmallU<N>> for u8 {
    #[inline(always)]
    fn from(value: SmallU<N>) -> Self {
        value.val()
    }
}

impl<const N: usize> From<SmallU<N>> for u16 {
    #[inline(always)]
    fn from(value: SmallU<N>) -> Self {
        u8::from(value) as _
    }
}

impl<const N: usize> From<SmallU<N>> for u32 {
    #[inline(always)]
    fn from(value: SmallU<N>) -> Self {
        u8::from(value) as _
    }
}

impl<const N: usize> From<SmallU<N>> for u64 {
    #[inline(always)]
    fn from(value: SmallU<N>) -> Self {
        u8::from(value) as _
    }
}

impl<const N: usize> From<SmallU<N>> for usize {
    #[inline(always)]
    fn from(value: SmallU<N>) -> Self {
        u8::from(value) as _
    }
}

impl<const N: usize> From<SmallU<N>> for u128 {
    #[inline(always)]
    fn from(value: SmallU<N>) -> Self {
        u8::from(value) as _
    }
}

impl From<SmallU<1>> for SmallU<2> {
    #[inline(always)]
    fn from(value: SmallU<1>) -> Self {
        Self(value.val())
    }
}

impl From<SmallU<1>> for SmallU<4> {
    #[inline(always)]
    fn from(value: SmallU<1>) -> Self {
        Self(value.val())
    }
}

impl From<SmallU<2>> for SmallU<4> {
    #[inline(always)]
    fn from(value: SmallU<2>) -> Self {
        Self(value.val())
    }
}

pub type U1 = SmallU<1>;
pub type U2 = SmallU<2>;
pub type U4 = SmallU<4>;

impl From<bool> for U1 {
    fn from(value: bool) -> Self {
        Self::new_unchecked(value as u8)
    }
}

impl From<U1> for bool {
    fn from(value: U1) -> Self {
        value == U1::ONE
    }
}

impl<const N: usize> SerializeBytes for SmallU<N> {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.val().serialize(write_buf)
    }
}

impl<const N: usize> DeserializeBytes for SmallU<N> {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Ok(Self::new(DeserializeBytes::deserialize(read_buf)?))
    }
}

#[cfg(test)]
impl<const N: usize> proptest::arbitrary::Arbitrary for SmallU<N> {
    type Parameters = ();
    type Strategy = proptest::strategy::BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        use proptest::strategy::Strategy;

        (0u8..(1u8 << N)).prop_map(Self::new_unchecked).boxed()
    }
}
