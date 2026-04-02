// Copyright 2025 Irreducible Inc.
use std::fmt::Debug;

use rand::RngCore;

/// A type that supports bitwise operations and has a known bit width.
///
/// Underliers are the basic building blocks for field arithmetic operations,
/// providing bitwise AND and XOR operations that can be implemented efficiently
/// on various architectures.
///
/// Since [`Underlier`] is implemented on external types (eg. native unsigned integers and SIMD
/// register types), we use trait methods in place of standard Rust traits like Eq, BitAnd, etc.
pub trait Underlier: Sized + Clone + Copy + Debug {
    /// The number of bits in this underlier type.
    const BITS: usize;

    /// Performs bitwise AND operation.
    fn and(a: Self, b: Self) -> Self;

    /// Performs AND assignment operation.
    #[inline]
    fn and_assign(&mut self, other: Self) {
        *self = Self::and(*self, other);
    }

    /// Performs bitwise XOR operation.
    fn xor(a: Self, b: Self) -> Self;

    /// Performs XOR assignment operation.
    #[inline]
    fn xor_assign(&mut self, other: Self) {
        *self = Self::xor(*self, other);
    }

    /// Returns the zero value for this underlier type.
    fn zero() -> Self;

    /// Checks if two underlier values are equal.
    fn is_equal(a: Self, b: Self) -> bool;

    /// Generates a random value of this underlier type using the provided rng.
    fn random(rng: impl RngCore) -> Self;
}

impl<U: Underlier, const N: usize> Underlier for [U; N] {
    const BITS: usize = U::BITS * N;

    #[inline]
    fn and(a: Self, b: Self) -> Self {
        let mut result = a;
        for i in 0..N {
            result[i] = U::and(a[i], b[i]);
        }
        result
    }

    #[inline]
    fn xor(a: Self, b: Self) -> Self {
        let mut result = a;
        for i in 0..N {
            result[i] = U::xor(a[i], b[i]);
        }
        result
    }

    #[inline]
    fn zero() -> Self {
        std::array::from_fn(|_| U::zero())
    }

    #[inline]
    fn is_equal(a: Self, b: Self) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| U::is_equal(*x, *y))
    }

    fn random(mut rng: impl RngCore) -> Self {
        std::array::from_fn(|_| U::random(&mut rng))
    }
}

/// An [`Underlier`] that can represent a packed vector of smaller underlier values.
///
/// This trait models SIMD-style packed vectors where a larger underlier type
/// contains multiple values of a smaller type. For example, a 256-bit SIMD
/// register can pack four 64-bit values or two 128-bit values.
///
/// # Example
///
/// ```
/// # #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
/// # {
/// use std::arch::x86_64::__m256i;
/// use binius_arith_bench::{PackedUnderlier, Underlier};
///
/// // __m256i can pack 4 u64 values
/// let packed = <__m256i as PackedUnderlier<u64>>::broadcast(0x1234567890ABCDEF);
/// assert!(packed.get(0) == 0x1234567890ABCDEF);
/// assert!(packed.get(1) == 0x1234567890ABCDEF);
/// assert!(packed.get(2) == 0x1234567890ABCDEF);
/// assert!(packed.get(3) == 0x1234567890ABCDEF);
/// # }
/// ```
pub trait PackedUnderlier<Inner>: Underlier {
    /// The logarithm base 2 of the number of packed elements.
    ///
    /// The total number of packed elements is `2^LOG_WIDTH`.
    const LOG_WIDTH: usize;

    /// Gets the inner value at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if index is greater than or equal to `2^LOG_WIDTH`.
    fn get(self, index: usize) -> Inner;

    /// Sets the inner value at the specified index and returns the modified packed value.
    ///
    /// # Panics
    ///
    /// Panics if index is greater than or equal to `2^LOG_WIDTH`.
    fn set(self, index: usize, val: Inner) -> Self;

    /// Creates a packed value with all elements set to `val`.
    fn broadcast(val: Inner) -> Self;
}

pub trait OpsGfni {
    /// Performs a multiplication in GF(2^8) on the packed bytes.
    /// The field is in polynomial representation with the reduction polynomial
    ///  x^8 + x^4 + x^3 + x + 1.
    ///
    /// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_gf2p8mul_epi8)
    fn gf2p8mul(a: Self, b: Self) -> Self;

    /// Performs an affine transformation on the packed bytes in x.
    /// That is computes a*x+b over the Galois Field 2^8 for each packed byte with a being a 8x8 bit
    /// matrix and b being a constant 8-bit immediate value.
    /// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
    ///
    /// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_gf2p8affine_epi64_epi8)
    fn gf2p8affine<const B: i32>(x: Self, a: Self) -> Self;

    /// Performs an affine transformation on the inverted packed bytes in x.
    /// That is computes a*inv(x)+b over the Galois Field 2^8 for each packed byte with a being a
    /// 8x8 bit matrix and b being a constant 8-bit immediate value.
    /// The inverse of a byte is defined with respect to the reduction polynomial x^8+x^4+x^3+x+1.
    /// The inverse of 0 is 0.
    /// Each pack of 8 bytes in x is paired with the 64-bit word at the same position in a.
    ///
    /// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_gf2p8affineinv_epi64_epi8)
    fn gf2p8affineinv<const B: i32>(x: Self, a: Self) -> Self;
}

#[allow(dead_code)]
pub trait OpsClmul {
    fn clmulepi64<const IMM8: i32>(a: Self, b: Self) -> Self;

    fn duplicate_hi_64(a: Self) -> Self;
    fn swap_hi_lo_64(a: Self) -> Self;

    fn extract_hi_lo_64(a: Self, b: Self) -> Self;

    fn unpacklo_epi64(a: Self, b: Self) -> Self;

    fn unpackhi_epi64(a: Self, b: Self) -> Self;

    /// Shifts 128-bit value left by IMM8 bytes while shifting in zeros.
    ///
    /// For 256-bit values, this operates on each 128-bit lane independently.
    fn slli_si128<const IMM8: i32>(a: Self) -> Self;

    /// Shifts each packed 64-bit integer left by IMM8 bits while shifting in zeros.
    ///
    /// For 128-bit values, this shifts two 64-bit integers independently.
    /// For 256-bit values, this shifts four 64-bit integers independently.
    fn slli_epi64<const IMM8: i32>(a: Self) -> Self;

    /// Creates a SIMD mask from the sign bit (bit 63) of each 64-bit lane.
    ///
    /// Returns a SIMD value where each 64-bit element contains either all 1s (0xFFFFFFFFFFFFFFFF)
    /// or all 0s (0x0000000000000000) based on the sign bit of the corresponding 64-bit lane.
    fn movepi64_mask(a: Self) -> Self;
}

macro_rules! impl_underlier_for_native_uint {
    ($type:ty) => {
        impl Underlier for $type {
            const BITS: usize = <$type>::BITS as usize;

            #[inline]
            fn and(a: Self, b: Self) -> Self {
                a & b
            }

            #[inline]
            fn xor(a: Self, b: Self) -> Self {
                a ^ b
            }

            #[inline]
            fn zero() -> Self {
                0
            }

            #[inline]
            fn is_equal(a: Self, b: Self) -> bool {
                a == b
            }

            #[inline]
            fn random(mut rng: impl RngCore) -> Self {
                use rand::Rng;

                rng.random()
            }
        }
    };
}

impl_underlier_for_native_uint!(u8);
impl_underlier_for_native_uint!(u16);
impl_underlier_for_native_uint!(u32);
impl_underlier_for_native_uint!(u64);
impl_underlier_for_native_uint!(u128);

macro_rules! impl_packed_underlier_for_native_uint {
	($inner:ty, $outer:ty) => {
		impl PackedUnderlier<$inner> for $outer {
			const LOG_WIDTH: usize = (<$outer>::BITS / <$inner>::BITS).trailing_zeros() as usize;

			#[inline]
			fn get(self, index: usize) -> $inner {
				let width = <$inner>::BITS as usize;
				let count = (<$outer>::BITS / <$inner>::BITS) as usize;
				assert!(index < count, "index out of bounds");
				((self >> (index * width)) & (<$inner>::MAX as $outer)) as $inner
			}

			#[inline]
			fn set(self, index: usize, val: $inner) -> Self {
				let width = <$inner>::BITS as usize;
				let count = (<$outer>::BITS / <$inner>::BITS) as usize;
				assert!(index < count, "index out of bounds");
				let mask = !(<$inner>::MAX as $outer << (index * width));
				(self & mask) | ((val as $outer) << (index * width))
			}

			#[inline]
			fn broadcast(val: $inner) -> Self {
				let width = <$inner>::BITS;
				let mut pattern = val as $outer;
				let mut current_width = width;
				while current_width < <$outer>::BITS {
					pattern |= pattern << current_width;
					current_width *= 2;
				}
				pattern
			}
		}
	};
}

// u8 packed into larger types
impl_packed_underlier_for_native_uint!(u8, u16);
impl_packed_underlier_for_native_uint!(u8, u32);
impl_packed_underlier_for_native_uint!(u8, u64);
impl_packed_underlier_for_native_uint!(u8, u128);

// u16 packed into larger types
impl_packed_underlier_for_native_uint!(u16, u32);
impl_packed_underlier_for_native_uint!(u16, u64);
impl_packed_underlier_for_native_uint!(u16, u128);

// u32 packed into larger types
impl_packed_underlier_for_native_uint!(u32, u64);
impl_packed_underlier_for_native_uint!(u32, u128);

// u64 packed into u128
impl_packed_underlier_for_native_uint!(u64, u128);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_u8_in_u64() {
        // Test broadcast
        let packed = <u64 as PackedUnderlier<u8>>::broadcast(0x42);
        assert_eq!(packed, 0x4242424242424242);

        // Test get
        for i in 0..8 {
            assert_eq!(<u64 as PackedUnderlier<u8>>::get(packed, i), 0x42);
        }

        // Test set
        let mut packed = 0u64;
        for i in 0..8 {
            packed = <u64 as PackedUnderlier<u8>>::set(packed, i, (i as u8) + 0x10);
        }
        assert_eq!(packed, 0x1716151413121110);

        // Verify get after set
        for i in 0..8 {
            assert_eq!(
                <u64 as PackedUnderlier<u8>>::get(packed, i),
                (i as u8) + 0x10
            );
        }
    }

    #[test]
    fn test_packed_u16_in_u64() {
        // Test broadcast
        let packed = <u64 as PackedUnderlier<u16>>::broadcast(0x1234);
        assert_eq!(packed, 0x1234123412341234);

        // Test get
        for i in 0..4 {
            assert_eq!(<u64 as PackedUnderlier<u16>>::get(packed, i), 0x1234);
        }

        // Test set
        let mut packed = 0u64;
        packed = <u64 as PackedUnderlier<u16>>::set(packed, 0, 0xABCD);
        packed = <u64 as PackedUnderlier<u16>>::set(packed, 1, 0xEF01);
        packed = <u64 as PackedUnderlier<u16>>::set(packed, 2, 0x2345);
        packed = <u64 as PackedUnderlier<u16>>::set(packed, 3, 0x6789);
        assert_eq!(packed, 0x67892345EF01ABCD);
    }

    #[test]
    fn test_log_width() {
        assert_eq!(<u16 as PackedUnderlier<u8>>::LOG_WIDTH, 1); // 2^1 = 2 elements
        assert_eq!(<u32 as PackedUnderlier<u8>>::LOG_WIDTH, 2); // 2^2 = 4 elements
        assert_eq!(<u64 as PackedUnderlier<u8>>::LOG_WIDTH, 3); // 2^3 = 8 elements
        assert_eq!(<u128 as PackedUnderlier<u8>>::LOG_WIDTH, 4); // 2^4 = 16 elements

        assert_eq!(<u32 as PackedUnderlier<u16>>::LOG_WIDTH, 1); // 2^1 = 2 elements
        assert_eq!(<u64 as PackedUnderlier<u16>>::LOG_WIDTH, 2); // 2^2 = 4 elements
        assert_eq!(<u128 as PackedUnderlier<u16>>::LOG_WIDTH, 3); // 2^3 = 8 elements

        assert_eq!(<u64 as PackedUnderlier<u32>>::LOG_WIDTH, 1); // 2^1 = 2 elements
        assert_eq!(<u128 as PackedUnderlier<u32>>::LOG_WIDTH, 2); // 2^2 = 4 elements

        assert_eq!(<u128 as PackedUnderlier<u64>>::LOG_WIDTH, 1); // 2^1 = 2 elements
    }
}
