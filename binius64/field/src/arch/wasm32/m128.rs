// Copyright 2025 Irreducible Inc.

use std::{
    arch::wasm32::*,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
    DeserializeBytes, SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
    serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable};
use derive_more::{From, Into};
use rand::{Rng, RngCore};

use crate::{
    BinaryField, Random,
    arch::portable::{packed::PackedPrimitiveType, packed_arithmetic::interleave_mask_even},
    underlier::{
        NumCast, SmallU, U1, U2, U4, UnderlierType, UnderlierWithBitOps, WithUnderlier,
        impl_divisible_bitmask, impl_divisible_memcast,
    },
};

#[derive(Copy, Clone, From, Into)]
#[repr(transparent)]
pub struct M128(pub v128);

impl M128 {
    #[inline]
    pub(super) const fn from_u128(value: u128) -> Self {
        Self(u64x2(value as u64, (value >> 64) as u64))
    }

    #[inline]
    pub fn from_lanes_u64(lo: u64, hi: u64) -> Self {
        Self(u64x2(lo, hi))
    }

    #[inline]
    pub fn split_lanes_u64(self) -> (u64, u64) {
        (
            u64x2_extract_lane::<0>(self.0),
            u64x2_extract_lane::<1>(self.0),
        )
    }
}

impl Default for M128 {
    #[inline]
    fn default() -> Self {
        Self::zeroed()
    }
}

impl PartialEq for M128 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        i8x16_all_true(i8x16_eq(self.0, other.0))
    }
}

impl Eq for M128 {}

impl PartialOrd for M128 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for M128 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        u128::from(*self).cmp(&u128::from(*other))
    }
}

unsafe impl Pod for M128 {}

unsafe impl Zeroable for M128 {
    #[inline]
    fn zeroed() -> Self {
        Self(u64x2(0, 0))
    }
}

impl From<u128> for M128 {
    #[inline]
    fn from(value: u128) -> Self {
        debug_assert_eq!(std::mem::align_of::<u128>(), std::mem::align_of::<v128>());

        Self(unsafe { v128_load(&raw const value as *const v128) })
    }
}

impl From<M128> for u128 {
    #[inline]
    fn from(m: M128) -> Self {
        debug_assert_eq!(std::mem::align_of::<u128>(), std::mem::align_of::<v128>());

        let mut value = 0u128;
        unsafe {
            v128_store(&raw mut value as *mut v128, m.0);
        }
        value
    }
}

impl From<u64> for M128 {
    #[inline]
    fn from(value: u64) -> Self {
        Self(u64x2(value, 0))
    }
}
impl From<u32> for M128 {
    #[inline]
    fn from(value: u32) -> Self {
        Self::from(value as u64)
    }
}
impl From<u16> for M128 {
    #[inline]
    fn from(value: u16) -> Self {
        Self::from(value as u64)
    }
}
impl From<u8> for M128 {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from(value as u64)
    }
}

impl<const N: usize> From<SmallU<N>> for M128 {
    #[inline]
    fn from(value: SmallU<N>) -> Self {
        Self::from(value.val() as u64)
    }
}

impl<U: NumCast<u128>> NumCast<M128> for U {
    #[inline]
    fn num_cast_from(val: M128) -> Self {
        Self::num_cast_from(val.into())
    }
}

impl SerializeBytes for M128 {
    #[inline]
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

        write_buf.put_u128_le((*self).into());

        Ok(())
    }
}

impl DeserializeBytes for M128 {
    #[inline]
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;

        Ok(Self::from(read_buf.get_u128_le()))
    }
}

impl_divisible_memcast!(M128, u128, u64, u32, u16, u8);
impl_divisible_bitmask!(M128, 1, 2, 4);

impl BitAnd for M128 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(v128_and(self.0, rhs.0))
    }
}

impl BitAndAssign for M128 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs
    }
}

impl BitOr for M128 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(v128_or(self.0, rhs.0))
    }
}

impl BitOrAssign for M128 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs
    }
}

impl BitXor for M128 {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(v128_xor(self.0, rhs.0))
    }
}

impl BitXorAssign for M128 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl Not for M128 {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(v128_not(self.0))
    }
}

impl Shl<usize> for M128 {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        // Perform a 128-bit shift by converting to u128, shifting, and converting back
        let val: u128 = self.into();
        Self::from(val << rhs)
    }
}

impl Shr<usize> for M128 {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        // Perform a 128-bit shift by converting to u128, shifting, and converting back
        let val: u128 = self.into();
        Self::from(val >> rhs)
    }
}

impl Random for M128 {
    fn random(mut rng: impl RngCore) -> Self {
        let val: u128 = rng.random();
        val.into()
    }
}

impl std::fmt::Display for M128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data: u128 = unsafe {
            let mut value = 0u128;
            v128_store(&raw mut value as *mut v128, self.0);
            value
        };
        write!(f, "{data:032X}")
    }
}

impl std::fmt::Debug for M128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data: u128 = unsafe {
            let mut value = 0u128;
            v128_store(&raw mut value as *mut v128, self.0);
            value
        };
        write!(f, "M128({data:032X})")
    }
}

impl UnderlierType for M128 {
    const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
    const ZERO: Self = { Self(u64x2(0, 0)) };
    const ONE: Self = { Self(u64x2(1, 0)) };
    const ONES: Self = { Self(u64x2(u64::MAX, u64::MAX)) };

    #[inline(always)]
    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: [M128; 3] = [
            M128::from_u128(interleave_mask_even!(u128, 0)),
            M128::from_u128(interleave_mask_even!(u128, 1)),
            M128::from_u128(interleave_mask_even!(u128, 2)),
        ];

        match log_block_len {
            // Bitwise/masked interleave
            0..=2 => {
                let a: v128 = self.into();
                let b: v128 = other.into();
                let mask: v128 = MASKS[log_block_len].into();
                let shift_amt = 1 << log_block_len;
                let t = v128_and(v128_xor(i64x2_shr(a, shift_amt as u32), b), mask);
                let c = v128_xor(a, i64x2_shl(t, shift_amt as u32));
                let d = v128_xor(b, t);
                (c.into(), d.into())
            }

            // 8-bit interleaving
            3 => {
                let a: v128 = self.into();
                let b: v128 = other.into();
                let c = i8x16_shuffle::<0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30>(
                    a, b,
                );
                let d = i8x16_shuffle::<1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31>(
                    a, b,
                );
                (c.into(), d.into())
            }

            // 16-bit interleaving
            4 => {
                let a: v128 = self.into();
                let b: v128 = other.into();
                let c =
                    i8x16_shuffle::<0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29>(a, b);
                let d = i8x16_shuffle::<2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31>(
                    a, b,
                );
                (c.into(), d.into())
            }

            // 32-bit interleaving
            5 => {
                let a: v128 = self.into();
                let b: v128 = other.into();
                let c =
                    i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27>(a, b);
                let d = i8x16_shuffle::<4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31>(
                    a, b,
                );
                (c.into(), d.into())
            }

            // 64-bit interleaving
            6 => {
                let a: v128 = self.into();
                let b: v128 = other.into();
                let c =
                    i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(a, b);
                let d = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(
                    a, b,
                );
                (c.into(), d.into())
            }

            _ => panic!("Unsupported block length"),
        }
    }
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
    #[inline]
    fn from(value: u128) -> Self {
        Self::from(M128::from(value))
    }
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for u128 {
    #[inline]
    fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
        value.to_underlier().into()
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    fn check_interleave(lhs: M128, rhs: M128, log_block_len: usize) {
        let (interleaved_lhs, interleaved_rhs) = lhs.interleave(rhs, log_block_len);
        let (u128_lhs, u128_rhs) = u128::from(lhs).interleave(u128::from(rhs), log_block_len);

        assert_eq!(u128::from(interleaved_lhs), u128_lhs);
        assert_eq!(u128::from(interleaved_rhs), u128_rhs);
    }

    #[test]
    fn test_interleave() {
        let mut rng = StdRng::from_seed([0; 32]);
        let lhs = M128::random(&mut rng);
        let rhs = M128::random(&mut rng);

        for log_block_len in 0..=6 {
            check_interleave(lhs, rhs, log_block_len);
            check_interleave(rhs, lhs, log_block_len);
        }
    }

    #[test]
    fn test_from_into_roundtrip() {
        let mut rng = StdRng::from_seed([0; 32]);
        let value: u128 = rng.random();
        let m128: M128 = value.into();
        let m128_2 = M128::from_u128(value);
        assert_eq!(m128, m128_2, "Conversion from u128 to M128 failed");

        let roundtrip_value: u128 = m128.into();
        assert_eq!(value, roundtrip_value, "Roundtrip conversion failed");
    }
}
