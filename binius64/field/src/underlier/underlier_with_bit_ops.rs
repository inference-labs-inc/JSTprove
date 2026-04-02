// Copyright 2024-2025 Irreducible Inc.

use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use binius_utils::checked_arithmetics::checked_int_div;

use super::{
    U1, U2, U4,
    underlier_type::{NumCast, UnderlierType},
};
use crate::Divisible;

/// Underlier type that supports bit arithmetic.
pub trait UnderlierWithBitOps:
    UnderlierType
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + Shr<usize, Output = Self>
    + Shl<usize, Output = Self>
    + Not<Output = Self>
    + Divisible<U1>
{
    const ZERO: Self;
    const ONE: Self;
    const ONES: Self;

    /// Fill value with the given bit
    /// `val` must be 0 or 1.
    fn fill_with_bit(val: u8) -> Self {
        Self::broadcast_subvalue(U1::new(val))
    }

    /// Interleave with the given bit size
    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self);

    /// Transpose with the given bit size
    fn transpose(mut self, mut other: Self, log_block_len: usize) -> (Self, Self) {
        assert!(log_block_len < Self::LOG_BITS);

        for log_block_len in (log_block_len..Self::LOG_BITS).rev() {
            (self, other) = self.interleave(other, log_block_len);
        }

        (self, other)
    }

    #[inline]
    fn from_fn<T>(f: impl FnMut(usize) -> T) -> Self
    where
        T: UnderlierType,
        Self: Divisible<T>,
    {
        Self::from_iter((0..<Self as Divisible<T>>::N).map(f))
    }

    /// Broadcast subvalue to fill `Self`.
    /// `Self::BITS/T::BITS` is supposed to be a power of 2.
    #[inline]
    fn broadcast_subvalue<T>(value: T) -> Self
    where
        T: UnderlierType,
        Self: Divisible<T>,
    {
        Divisible::<T>::broadcast(value)
    }

    /// Gets the subvalue from the given position.
    /// Function panics in case when index is out of range.
    ///
    /// # Safety
    /// `i` must be less than `Self::BITS/T::BITS`.
    #[inline]
    unsafe fn get_subvalue<T>(&self, i: usize) -> T
    where
        T: UnderlierType,
        Self: Divisible<T>,
    {
        debug_assert!(
            i < checked_int_div(Self::BITS, T::BITS),
            "i: {} Self::BITS: {}, T::BITS: {}",
            i,
            Self::BITS,
            T::BITS
        );
        Divisible::<T>::get(*self, i)
    }

    /// Sets the subvalue in the given position.
    /// Function panics in case when index is out of range.
    ///
    /// # Safety
    /// `i` must be less than `Self::BITS/T::BITS`.
    #[inline]
    unsafe fn set_subvalue<T>(&mut self, i: usize, val: T)
    where
        T: UnderlierWithBitOps,
        Self: Divisible<T>,
    {
        debug_assert!(i < checked_int_div(Self::BITS, T::BITS));
        *self = (*self).set(i, val);
    }

    /// Spread takes a block of sub_elements of `T` type within the current value and
    /// repeats them to the full underlier width.
    ///
    /// # Safety
    /// `log_block_len + T::LOG_BITS` must be less than or equal to `Self::LOG_BITS`.
    /// `block_idx` must be less than `1 << (Self::LOG_BITS - log_block_len)`.
    #[inline]
    unsafe fn spread<T>(self, log_block_len: usize, block_idx: usize) -> Self
    where
        T: UnderlierWithBitOps,
        Self: Divisible<T>,
    {
        unsafe { spread_fallback::<Self, T>(self, log_block_len, block_idx) }
    }
}

/// Fallback implementation of `spread` method.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
/// `block_idx` must be less than `1 << (U::LOG_BITS - log_block_len)`.
pub(crate) unsafe fn spread_fallback<U, T>(value: U, log_block_len: usize, block_idx: usize) -> U
where
    U: UnderlierWithBitOps + Divisible<T>,
    T: UnderlierWithBitOps,
{
    debug_assert!(
        log_block_len + T::LOG_BITS <= U::LOG_BITS,
        "log_block_len: {}, U::BITS: {}, T::BITS: {}",
        log_block_len,
        U::BITS,
        T::BITS
    );
    debug_assert!(
        block_idx < 1 << (U::LOG_BITS - log_block_len),
        "block_idx: {}, U::BITS: {}, log_block_len: {}",
        block_idx,
        U::BITS,
        log_block_len
    );

    let mut result = U::ZERO;
    let block_offset = block_idx << log_block_len;
    let log_repeat = U::LOG_BITS - T::LOG_BITS - log_block_len;
    for i in 0..1 << log_block_len {
        unsafe {
            result.set_subvalue(i << log_repeat, value.get_subvalue::<T>(block_offset + i));
        }
    }

    for i in 0..log_repeat {
        result |= result << (1 << (T::LOG_BITS + i));
    }

    result
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn single_element_mask_bits<T: UnderlierWithBitOps>(bits_count: usize) -> T {
    use binius_utils::checked_arithmetics::checked_log_2;

    if bits_count == T::BITS {
        !T::ZERO
    } else {
        let mut result = T::ONE;
        for height in 0..checked_log_2(bits_count) {
            result |= result << (1 << height)
        }

        result
    }
}

/// Value that can be spread to a single u8
pub(crate) trait SpreadToByte {
    fn spread_to_byte(self) -> u8;
}

impl SpreadToByte for U1 {
    #[inline(always)]
    fn spread_to_byte(self) -> u8 {
        u8::fill_with_bit(self.val())
    }
}

impl SpreadToByte for U2 {
    #[inline(always)]
    fn spread_to_byte(self) -> u8 {
        let mut result = self.val();
        result |= result << 2;
        result |= result << 4;

        result
    }
}

impl SpreadToByte for U4 {
    #[inline(always)]
    fn spread_to_byte(self) -> u8 {
        let mut result = self.val();
        result |= result << 4;

        result
    }
}

/// A helper functions for implementing `UnderlierWithBitOps::spread_unchecked` for SIMD types.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
#[allow(unused)]
#[inline(always)]
pub(crate) unsafe fn get_block_values<U, T, const BLOCK_LEN: usize>(
    value: U,
    block_idx: usize,
) -> [T; BLOCK_LEN]
where
    U: UnderlierWithBitOps + From<T> + Divisible<T>,
    T: UnderlierType + NumCast<U>,
{
    std::array::from_fn(|i| unsafe { value.get_subvalue::<T>(block_idx * BLOCK_LEN + i) })
}

/// A helper functions for implementing `UnderlierWithBitOps::spread_unchecked` for SIMD types.
///
/// # Safety
/// `log_block_len + T::LOG_BITS` must be less than or equal to `U::LOG_BITS`.
#[allow(unused)]
#[inline(always)]
pub(crate) unsafe fn get_spread_bytes<U, T, const BLOCK_LEN: usize>(
    value: U,
    block_idx: usize,
) -> [u8; BLOCK_LEN]
where
    U: UnderlierWithBitOps + From<T> + Divisible<T>,
    T: UnderlierType + SpreadToByte + NumCast<U>,
{
    unsafe { get_block_values::<U, T, BLOCK_LEN>(value, block_idx) }
        .map(SpreadToByte::spread_to_byte)
}

#[cfg(test)]
mod tests {
    use proptest::{arbitrary::any, bits, proptest};

    use super::{
        super::small_uint::{U1, U2, U4},
        *,
    };

    #[test]
    fn test_from_fn() {
        assert_eq!(u32::from_fn(|_| U1::new(0)), 0);
        assert_eq!(u32::from_fn(|i| U1::new((i % 2) as u8)), 0xaaaaaaaa);
        assert_eq!(u32::from_fn(|_| U1::new(1)), u32::MAX);

        assert_eq!(u32::from_fn(|_| U2::new(0)), 0);
        assert_eq!(u32::from_fn(|_| U2::new(1)), 0x55555555);
        assert_eq!(u32::from_fn(|_| U2::new(2)), 0xaaaaaaaa);
        assert_eq!(u32::from_fn(|_| U2::new(3)), u32::MAX);
        assert_eq!(u32::from_fn(|i| U2::new((i % 4) as u8)), 0xe4e4e4e4);

        assert_eq!(u32::from_fn(|_| U4::new(0)), 0);
        assert_eq!(u32::from_fn(|_| U4::new(1)), 0x11111111);
        assert_eq!(u32::from_fn(|_| U4::new(8)), 0x88888888);
        assert_eq!(u32::from_fn(|_| U4::new(31)), 0xffffffff);
        assert_eq!(u32::from_fn(|i| U4::new(i as u8)), 0x76543210);

        assert_eq!(u32::from_fn(|_| 0u8), 0);
        assert_eq!(u32::from_fn(|_| 0xabu8), 0xabababab);
        assert_eq!(u32::from_fn(|_| 255u8), 0xffffffff);
        assert_eq!(u32::from_fn(|i| i as u8), 0x03020100);
    }

    #[test]
    fn test_broadcast_subvalue() {
        assert_eq!(u32::broadcast_subvalue(U1::new(0)), 0);
        assert_eq!(u32::broadcast_subvalue(U1::new(1)), u32::MAX);

        assert_eq!(u32::broadcast_subvalue(U2::new(0)), 0);
        assert_eq!(u32::broadcast_subvalue(U2::new(1)), 0x55555555);
        assert_eq!(u32::broadcast_subvalue(U2::new(2)), 0xaaaaaaaa);
        assert_eq!(u32::broadcast_subvalue(U2::new(3)), u32::MAX);

        assert_eq!(u32::broadcast_subvalue(U4::new(0)), 0);
        assert_eq!(u32::broadcast_subvalue(U4::new(1)), 0x11111111);
        assert_eq!(u32::broadcast_subvalue(U4::new(8)), 0x88888888);
        assert_eq!(u32::broadcast_subvalue(U4::new(31)), 0xffffffff);

        assert_eq!(u32::broadcast_subvalue(0u8), 0);
        assert_eq!(u32::broadcast_subvalue(0xabu8), 0xabababab);
        assert_eq!(u32::broadcast_subvalue(255u8), 0xffffffff);
    }

    #[test]
    fn test_get_subvalue() {
        let value = 0xab12cd34u32;

        unsafe {
            assert_eq!(value.get_subvalue::<U1>(0), U1::new(0));
            assert_eq!(value.get_subvalue::<U1>(1), U1::new(0));
            assert_eq!(value.get_subvalue::<U1>(2), U1::new(1));
            assert_eq!(value.get_subvalue::<U1>(31), U1::new(1));

            assert_eq!(value.get_subvalue::<U2>(0), U2::new(0));
            assert_eq!(value.get_subvalue::<U2>(1), U2::new(1));
            assert_eq!(value.get_subvalue::<U2>(2), U2::new(3));
            assert_eq!(value.get_subvalue::<U2>(15), U2::new(2));

            assert_eq!(value.get_subvalue::<U4>(0), U4::new(4));
            assert_eq!(value.get_subvalue::<U4>(1), U4::new(3));
            assert_eq!(value.get_subvalue::<U4>(2), U4::new(13));
            assert_eq!(value.get_subvalue::<U4>(7), U4::new(10));

            assert_eq!(value.get_subvalue::<u8>(0), 0x34u8);
            assert_eq!(value.get_subvalue::<u8>(1), 0xcdu8);
            assert_eq!(value.get_subvalue::<u8>(2), 0x12u8);
            assert_eq!(value.get_subvalue::<u8>(3), 0xabu8);
        }
    }

    proptest! {
        #[test]
        fn test_set_subvalue_1b(mut init_val in any::<u32>(), i in 0usize..31, val in bits::u8::masked(1)) {
            unsafe {
                init_val.set_subvalue(i, U1::new(val));
                assert_eq!(init_val.get_subvalue::<U1>(i), U1::new(val));
            }
        }

        #[test]
        fn test_set_subvalue_2b(mut init_val in any::<u32>(), i in 0usize..15, val in bits::u8::masked(3)) {
            unsafe {
                init_val.set_subvalue(i, U2::new(val));
                assert_eq!(init_val.get_subvalue::<U2>(i), U2::new(val));
            }
        }

        #[test]
        fn test_set_subvalue_4b(mut init_val in any::<u32>(), i in 0usize..7, val in bits::u8::masked(7)) {
            unsafe {
                init_val.set_subvalue(i, U4::new(val));
                assert_eq!(init_val.get_subvalue::<U4>(i), U4::new(val));
            }
        }

        #[test]
        fn test_set_subvalue_8b(mut init_val in any::<u32>(), i in 0usize..3, val in bits::u8::masked(15)) {
            unsafe {
                init_val.set_subvalue(i, val);
                assert_eq!(init_val.get_subvalue::<u8>(i), val);
            }
        }
    }
}
