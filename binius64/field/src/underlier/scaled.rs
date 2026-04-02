// Copyright 2024-2025 Irreducible Inc.

use std::{
    array, mem,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
    DeserializeBytes, SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
    checked_arithmetics::checked_log_2,
};
use bytemuck::{Pod, Zeroable};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use super::{Divisible, NumCast, UnderlierType, UnderlierWithBitOps, mapget};
use crate::Random;

/// A type that represents N elements of the same underlier type.
/// Used as an underlier for 256-bit and 512-bit packed fields in the portable implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ScaledUnderlier<U, const N: usize>(pub [U; N]);

impl<U: Default, const N: usize> Default for ScaledUnderlier<U, N> {
    fn default() -> Self {
        Self(array::from_fn(|_| U::default()))
    }
}

impl<U: Random, const N: usize> Distribution<ScaledUnderlier<U, N>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, mut rng: &mut R) -> ScaledUnderlier<U, N> {
        ScaledUnderlier(array::from_fn(|_| U::random(&mut rng)))
    }
}

impl<U, const N: usize> From<ScaledUnderlier<U, N>> for [U; N] {
    fn from(val: ScaledUnderlier<U, N>) -> Self {
        val.0
    }
}

impl<T, U: From<T>, const N: usize> From<[T; N]> for ScaledUnderlier<U, N> {
    fn from(value: [T; N]) -> Self {
        Self(value.map(U::from))
    }
}

impl<T: Copy, U: From<[T; 2]>> From<[T; 4]> for ScaledUnderlier<U, 2> {
    fn from(value: [T; 4]) -> Self {
        Self([[value[0], value[1]], [value[2], value[3]]].map(Into::into))
    }
}

unsafe impl<U: Zeroable, const N: usize> Zeroable for ScaledUnderlier<U, N> {}
unsafe impl<U: Pod, const N: usize> Pod for ScaledUnderlier<U, N> {}

impl<U: UnderlierType + Pod, const N: usize> UnderlierType for ScaledUnderlier<U, N> {
    const LOG_BITS: usize = U::LOG_BITS + checked_log_2(N);
}

impl<U: BitAnd<Output = U> + Copy, const N: usize> BitAnd for ScaledUnderlier<U, N> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self.0[i] & rhs.0[i]))
    }
}

impl<U: BitAndAssign + Copy, const N: usize> BitAndAssign for ScaledUnderlier<U, N> {
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] &= rhs.0[i];
        }
    }
}

impl<U: BitOr<Output = U> + Copy, const N: usize> BitOr for ScaledUnderlier<U, N> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self.0[i] | rhs.0[i]))
    }
}

impl<U: BitOrAssign + Copy, const N: usize> BitOrAssign for ScaledUnderlier<U, N> {
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] |= rhs.0[i];
        }
    }
}

impl<U: BitXor<Output = U> + Copy, const N: usize> BitXor for ScaledUnderlier<U, N> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }
}

impl<U: BitXorAssign + Copy, const N: usize> BitXorAssign for ScaledUnderlier<U, N> {
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] ^= rhs.0[i];
        }
    }
}

impl<U: UnderlierWithBitOps, const N: usize> Shr<usize> for ScaledUnderlier<U, N> {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        let mut result = Self::default();

        let shift_in_items = rhs / U::BITS;
        for i in 0..N.saturating_sub(shift_in_items.saturating_sub(1)) {
            if i + shift_in_items < N {
                result.0[i] |= self.0[i + shift_in_items] >> (rhs % U::BITS);
            }
            if i + shift_in_items + 1 < N && !rhs.is_multiple_of(U::BITS) {
                result.0[i] |= self.0[i + shift_in_items + 1] << (U::BITS - (rhs % U::BITS));
            }
        }

        result
    }
}

impl<U: UnderlierWithBitOps, const N: usize> Shl<usize> for ScaledUnderlier<U, N> {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        let mut result = Self::default();

        let shift_in_items = rhs / U::BITS;
        for i in shift_in_items.saturating_sub(1)..N {
            if i >= shift_in_items {
                result.0[i] |= self.0[i - shift_in_items] << (rhs % U::BITS);
            }
            if i > shift_in_items && !rhs.is_multiple_of(U::BITS) {
                result.0[i] |= self.0[i - shift_in_items - 1] >> (U::BITS - (rhs % U::BITS));
            }
        }

        result
    }
}

impl<U: Not<Output = U>, const N: usize> Not for ScaledUnderlier<U, N> {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(self.0.map(U::not))
    }
}

impl<U: UnderlierWithBitOps + Pod, const N: usize> UnderlierWithBitOps for ScaledUnderlier<U, N> {
    const ZERO: Self = Self([U::ZERO; N]);
    const ONE: Self = {
        let mut arr = [U::ZERO; N];
        arr[0] = U::ONE;
        Self(arr)
    };
    const ONES: Self = Self([U::ONES; N]);

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        if log_block_len < U::LOG_BITS {
            // Case 1: Delegate to element-wise interleave
            let pairs: [(U, U); N] =
                array::from_fn(|i| self.0[i].interleave(other.0[i], log_block_len));
            (
                Self(array::from_fn(|i| pairs[i].0)),
                Self(array::from_fn(|i| pairs[i].1)),
            )
        } else {
            // Case 2: Interleave at element level by swapping array elements
            // Each super-block of 2*block_len elements gets transposed as a 2x2 matrix of blocks
            let block_len = 1 << (log_block_len - U::LOG_BITS);

            let mut a = self.0;
            let mut b = other.0;
            for super_block in 0..(N / (2 * block_len)) {
                let base = super_block * 2 * block_len;
                for offset in 0..block_len {
                    mem::swap(&mut a[base + block_len + offset], &mut b[base + offset]);
                }
            }

            (Self(a), Self(b))
        }
    }
}

impl<U: UnderlierType, const N: usize> NumCast<ScaledUnderlier<U, N>> for u8
where
    Self: NumCast<U>,
{
    fn num_cast_from(val: ScaledUnderlier<U, N>) -> Self {
        Self::num_cast_from(val.0[0])
    }
}

impl<U, const N: usize> From<u8> for ScaledUnderlier<U, N>
where
    U: From<u8>,
{
    fn from(val: u8) -> Self {
        Self(array::from_fn(|_| U::from(val)))
    }
}

impl<U, T, const N: usize> Divisible<T> for ScaledUnderlier<U, N>
where
    U: Divisible<T> + Pod + Send + Sync,
    T: Send + 'static,
{
    const LOG_N: usize = <U as Divisible<T>>::LOG_N + checked_log_2(N);

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = T> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> T {
        let u_index = index >> <U as Divisible<T>>::LOG_N;
        let sub_index = index & (<U as Divisible<T>>::N - 1);
        Divisible::<T>::get(self.0[u_index], sub_index)
    }

    #[inline]
    fn set(self, index: usize, val: T) -> Self {
        let u_index = index >> <U as Divisible<T>>::LOG_N;
        let sub_index = index & (<U as Divisible<T>>::N - 1);
        let mut arr = self.0;
        arr[u_index] = Divisible::<T>::set(arr[u_index], sub_index, val);
        Self(arr)
    }

    #[inline]
    fn broadcast(val: T) -> Self {
        Self([Divisible::<T>::broadcast(val); N])
    }

    #[inline]
    fn from_iter(mut iter: impl Iterator<Item = T>) -> Self {
        Self(array::from_fn(|_| Divisible::<T>::from_iter(&mut iter)))
    }
}

impl<U: SerializeBytes, const N: usize> SerializeBytes for ScaledUnderlier<U, N> {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.0.serialize(write_buf)
    }
}

impl<U: DeserializeBytes, const N: usize> DeserializeBytes for ScaledUnderlier<U, N> {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError> {
        <[U; N]>::deserialize(read_buf).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shr() {
        let val = ScaledUnderlier::<u8, 4>([0, 1, 2, 3]);
        assert_eq!(
            val >> 1,
            ScaledUnderlier::<u8, 4>([0b10000000, 0b00000000, 0b10000001, 0b00000001])
        );
        assert_eq!(
            val >> 2,
            ScaledUnderlier::<u8, 4>([0b01000000, 0b10000000, 0b11000000, 0b00000000])
        );
        assert_eq!(
            val >> 8,
            ScaledUnderlier::<u8, 4>([0b00000001, 0b00000010, 0b00000011, 0b00000000])
        );
        assert_eq!(
            val >> 9,
            ScaledUnderlier::<u8, 4>([0b00000000, 0b10000001, 0b00000001, 0b00000000])
        );
    }

    #[test]
    fn test_shl() {
        let val = ScaledUnderlier::<u8, 4>([0, 1, 2, 3]);
        assert_eq!(val << 1, ScaledUnderlier::<u8, 4>([0, 2, 4, 6]));
        assert_eq!(val << 2, ScaledUnderlier::<u8, 4>([0, 4, 8, 12]));
        assert_eq!(val << 8, ScaledUnderlier::<u8, 4>([0, 0, 1, 2]));
        assert_eq!(val << 9, ScaledUnderlier::<u8, 4>([0, 0, 2, 4]));
    }

    #[test]
    fn test_interleave_within_element() {
        // Test case 1: log_block_len < U::LOG_BITS
        // ScaledUnderlier<u8, 4> has LOG_BITS = 5 (32 bits total)
        // u8 has LOG_BITS = 3
        let a = ScaledUnderlier::<u8, 4>([0b01010101, 0b11110000, 0b00001111, 0b10101010]);
        let b = ScaledUnderlier::<u8, 4>([0b10101010, 0b00001111, 0b11110000, 0b01010101]);

        // At log_block_len = 0 (1-bit blocks), should delegate to u8::interleave
        let (c, d) = a.interleave(b, 0);

        // Verify element-wise interleave occurred
        for i in 0..4 {
            let (expected_c, expected_d) = a.0[i].interleave(b.0[i], 0);
            assert_eq!(c.0[i], expected_c);
            assert_eq!(d.0[i], expected_d);
        }
    }

    #[test]
    fn test_interleave_across_elements() {
        // Test case 2: log_block_len >= U::LOG_BITS
        let a = ScaledUnderlier::<u8, 4>([0, 1, 2, 3]);
        let b = ScaledUnderlier::<u8, 4>([4, 5, 6, 7]);

        // At log_block_len = 3 (8-bit blocks = 1 element), swap individual elements
        let (c, d) = a.interleave(b, 3);
        assert_eq!(c.0, [0, 4, 2, 6]);
        assert_eq!(d.0, [1, 5, 3, 7]);

        // At log_block_len = 4 (16-bit blocks = 2 elements), swap pairs
        let (c, d) = a.interleave(b, 4);
        assert_eq!(c.0, [0, 1, 4, 5]);
        assert_eq!(d.0, [2, 3, 6, 7]);
    }
}
