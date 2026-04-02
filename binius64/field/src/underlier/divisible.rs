// Copyright 2024-2025 Irreducible Inc.

use std::mem::size_of;

/// Divides an underlier type into smaller underliers in memory and iterates over them.
///
/// [`Divisible`] provides iteration over the subdivisions of an underlier type, guaranteeing that
/// iteration proceeds from the least significant bits to the most significant bits, regardless of
/// the CPU architecture's endianness.
///
/// # Endianness Handling
///
/// To ensure consistent LSB-to-MSB iteration order across all platforms:
/// - On little-endian systems: elements are naturally ordered LSB-to-MSB in memory, so iteration
///   proceeds forward through the array
/// - On big-endian systems: elements are ordered MSB-to-LSB in memory, so iteration is reversed to
///   achieve LSB-to-MSB order
///
/// This abstraction allows code to work with subdivided underliers in a platform-independent way
/// while maintaining the invariant that the first element always represents the least significant
/// portion of the value.
pub trait Divisible<T>: Copy {
    /// The log2 of the number of `T` elements that fit in `Self`.
    const LOG_N: usize;

    /// The number of `T` elements that fit in `Self`.
    const N: usize = 1 << Self::LOG_N;

    /// Returns an iterator over subdivisions of this underlier value, ordered from LSB to MSB.
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = T> + Send + Clone;

    /// Returns an iterator over subdivisions of this underlier reference, ordered from LSB to MSB.
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_;

    /// Returns an iterator over subdivisions of a slice of underliers, ordered from LSB to MSB.
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = T> + Send + Clone + '_;

    /// Get element at index (LSB-first ordering).
    ///
    /// # Panics
    ///
    /// Panics if `index >= Self::N`.
    fn get(self, index: usize) -> T;

    /// Set element at index (LSB-first ordering), returning modified value.
    ///
    /// # Panics
    ///
    /// Panics if `index >= Self::N`.
    fn set(self, index: usize, val: T) -> Self;

    /// Create a value with `val` broadcast to all `N` positions.
    fn broadcast(val: T) -> Self;

    /// Construct a value from an iterator of elements.
    ///
    /// Consumes at most `N` elements from the iterator. If the iterator
    /// yields fewer than `N` elements, remaining positions are filled with zeros.
    fn from_iter(iter: impl Iterator<Item = T>) -> Self;
}

/// Helper functions for Divisible implementations using bytemuck memory casting.
///
/// These functions handle the endianness-aware iteration over subdivisions of an underlier type.
pub mod memcast {
    use bytemuck::{Pod, Zeroable};

    /// Returns an iterator over subdivisions of a value, ordered from LSB to MSB.
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn value_iter<Big, Small, const N: usize>(
        value: Big,
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone
    where
        Big: Pod,
        Small: Pod + Send,
    {
        bytemuck::must_cast::<Big, [Small; N]>(value).into_iter()
    }

    /// Returns an iterator over subdivisions of a value, ordered from LSB to MSB.
    #[cfg(target_endian = "big")]
    #[inline]
    pub fn value_iter<Big, Small, const N: usize>(
        value: Big,
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone
    where
        Big: Pod,
        Small: Pod + Send,
    {
        bytemuck::must_cast::<Big, [Small; N]>(value)
            .into_iter()
            .rev()
    }

    /// Returns an iterator over subdivisions of a reference, ordered from LSB to MSB.
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn ref_iter<Big, Small, const N: usize>(
        value: &Big,
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
    where
        Big: Pod,
        Small: Pod + Send + Sync,
    {
        bytemuck::must_cast_ref::<Big, [Small; N]>(value)
            .iter()
            .copied()
    }

    /// Returns an iterator over subdivisions of a reference, ordered from LSB to MSB.
    #[cfg(target_endian = "big")]
    #[inline]
    pub fn ref_iter<Big, Small, const N: usize>(
        value: &Big,
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
    where
        Big: Pod,
        Small: Pod + Send + Sync,
    {
        bytemuck::must_cast_ref::<Big, [Small; N]>(value)
            .iter()
            .rev()
            .copied()
    }

    /// Returns an iterator over subdivisions of a slice, ordered from LSB to MSB.
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn slice_iter<Big, Small>(
        slice: &[Big],
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
    where
        Big: Pod,
        Small: Pod + Send + Sync,
    {
        bytemuck::must_cast_slice::<Big, Small>(slice)
            .iter()
            .copied()
    }

    /// Returns an iterator over subdivisions of a slice, ordered from LSB to MSB.
    ///
    /// For big-endian: iterate through the raw slice, but for each element's
    /// subdivisions, reverse the index to maintain LSB-first ordering.
    #[cfg(target_endian = "big")]
    #[inline]
    pub fn slice_iter<Big, Small, const LOG_N: usize>(
        slice: &[Big],
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
    where
        Big: Pod,
        Small: Pod + Send + Sync,
    {
        const N: usize = 1 << LOG_N;
        let raw_slice = bytemuck::must_cast_slice::<Big, Small>(slice);
        (0..raw_slice.len()).map(move |i| {
            let element_idx = i >> LOG_N;
            let sub_idx = i & (N - 1);
            let reversed_sub_idx = N - 1 - sub_idx;
            let raw_idx = element_idx * N + reversed_sub_idx;
            raw_slice[raw_idx]
        })
    }

    /// Get element at index (LSB-first ordering).
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn get<Big, Small, const N: usize>(value: &Big, index: usize) -> Small
    where
        Big: Pod,
        Small: Pod,
    {
        bytemuck::must_cast_ref::<Big, [Small; N]>(value)[index]
    }

    /// Get element at index (LSB-first ordering).
    #[cfg(target_endian = "big")]
    #[inline]
    pub fn get<Big, Small, const N: usize>(value: &Big, index: usize) -> Small
    where
        Big: Pod,
        Small: Pod,
    {
        bytemuck::must_cast_ref::<Big, [Small; N]>(value)[N - 1 - index]
    }

    /// Set element at index (LSB-first ordering), returning modified value.
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn set<Big, Small, const N: usize>(value: &Big, index: usize, val: Small) -> Big
    where
        Big: Pod,
        Small: Pod,
    {
        let mut arr = *bytemuck::must_cast_ref::<Big, [Small; N]>(value);
        arr[index] = val;
        bytemuck::must_cast(arr)
    }

    /// Set element at index (LSB-first ordering), returning modified value.
    #[cfg(target_endian = "big")]
    #[inline]
    pub fn set<Big, Small, const N: usize>(value: &Big, index: usize, val: Small) -> Big
    where
        Big: Pod,
        Small: Pod,
    {
        let mut arr = *bytemuck::must_cast_ref::<Big, [Small; N]>(value);
        arr[N - 1 - index] = val;
        bytemuck::must_cast(arr)
    }

    /// Broadcast a value to all positions.
    #[inline]
    pub fn broadcast<Big, Small, const N: usize>(val: Small) -> Big
    where
        Big: Pod,
        Small: Pod + Copy,
    {
        bytemuck::must_cast::<[Small; N], Big>([val; N])
    }

    /// Construct a value from an iterator of elements.
    #[cfg(target_endian = "little")]
    #[inline]
    pub fn from_iter<Big, Small, const N: usize>(iter: impl Iterator<Item = Small>) -> Big
    where
        Big: Pod,
        Small: Pod,
    {
        let mut arr: [Small; N] = Zeroable::zeroed();
        for (i, val) in iter.take(N).enumerate() {
            arr[i] = val;
        }
        bytemuck::must_cast(arr)
    }

    /// Construct a value from an iterator of elements.
    #[cfg(target_endian = "big")]
    #[inline]
    pub fn from_iter<Big, Small, const N: usize>(iter: impl Iterator<Item = Small>) -> Big
    where
        Big: Pod,
        Small: Pod,
    {
        let mut arr: [Small; N] = Zeroable::zeroed();
        for (i, val) in iter.take(N).enumerate() {
            arr[N - 1 - i] = val;
        }
        bytemuck::must_cast(arr)
    }
}

/// Helper functions for Divisible implementations using bitmask operations on sub-byte elements.
///
/// These functions work on any type that implements `Divisible<u8>` by extracting
/// and modifying sub-byte elements through the byte interface.
pub mod bitmask {
    use super::{Divisible, SmallU};

    /// Get a sub-byte element at index (LSB-first ordering).
    #[inline]
    pub fn get<Big, const BITS: usize>(value: Big, index: usize) -> SmallU<BITS>
    where
        Big: Divisible<u8>,
    {
        let elems_per_byte = 8 / BITS;
        let byte_index = index / elems_per_byte;
        let sub_index = index % elems_per_byte;
        let byte = Divisible::<u8>::get(value, byte_index);
        let shift = sub_index * BITS;
        SmallU::<BITS>::new(byte >> shift)
    }

    /// Set a sub-byte element at index (LSB-first ordering), returning modified value.
    #[inline]
    pub fn set<Big, const BITS: usize>(value: Big, index: usize, val: SmallU<BITS>) -> Big
    where
        Big: Divisible<u8>,
    {
        let elems_per_byte = 8 / BITS;
        let byte_index = index / elems_per_byte;
        let sub_index = index % elems_per_byte;
        let byte = Divisible::<u8>::get(value, byte_index);
        let shift = sub_index * BITS;
        let mask = (1u8 << BITS) - 1;
        let new_byte = (byte & !(mask << shift)) | (val.val() << shift);
        Divisible::<u8>::set(value, byte_index, new_byte)
    }
}

/// Helper functions for Divisible implementations using the get method.
///
/// These functions create iterators by mapping indices through `Divisible::get`,
/// useful for SIMD types where extract intrinsics provide efficient element access.
pub mod mapget {
    use binius_utils::iter::IterExtensions;

    use super::Divisible;

    /// Create an iterator over subdivisions by mapping get over indices.
    #[inline]
    pub fn value_iter<Big, Small>(value: Big) -> impl ExactSizeIterator<Item = Small> + Send + Clone
    where
        Big: Divisible<Small> + Send,
        Small: Send,
    {
        (0..Big::N).map_skippable(move |i| Divisible::<Small>::get(value, i))
    }

    /// Create a slice iterator by computing global index and using get.
    #[inline]
    pub fn slice_iter<Big, Small>(
        slice: &[Big],
    ) -> impl ExactSizeIterator<Item = Small> + Send + Clone + '_
    where
        Big: Divisible<Small> + Send + Sync,
        Small: Send,
    {
        let total = slice.len() * Big::N;
        (0..total).map_skippable(move |global_idx| {
            let elem_idx = global_idx / Big::N;
            let sub_idx = global_idx % Big::N;
            Divisible::<Small>::get(slice[elem_idx], sub_idx)
        })
    }
}

/// Iterator for dividing an underlier into sub-byte elements (ie. [`SmallU`]).
///
/// This iterator wraps a byte iterator and extracts sub-byte elements from each byte.
/// Generic over the byte iterator type `I`.
#[derive(Clone)]
pub struct SmallUDivisIter<I, const N: usize> {
    byte_iter: I,
    current_byte: Option<u8>,
    sub_idx: usize,
}

impl<I: Iterator<Item = u8>, const N: usize> SmallUDivisIter<I, N> {
    const ELEMS_PER_BYTE: usize = 8 / N;

    pub fn new(mut byte_iter: I) -> Self {
        let current_byte = byte_iter.next();
        Self {
            byte_iter,
            current_byte,
            sub_idx: 0,
        }
    }
}

impl<I: ExactSizeIterator<Item = u8>, const N: usize> Iterator for SmallUDivisIter<I, N> {
    type Item = SmallU<N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let byte = self.current_byte?;
        let shift = self.sub_idx * N;
        let result = SmallU::<N>::new(byte >> shift);

        self.sub_idx += 1;
        if self.sub_idx >= Self::ELEMS_PER_BYTE {
            self.sub_idx = 0;
            self.current_byte = self.byte_iter.next();
        }

        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_in_current = if self.current_byte.is_some() {
            Self::ELEMS_PER_BYTE - self.sub_idx
        } else {
            0
        };
        let remaining_bytes = self.byte_iter.len();
        let total = remaining_in_current + remaining_bytes * Self::ELEMS_PER_BYTE;
        (total, Some(total))
    }
}

impl<I: ExactSizeIterator<Item = u8>, const N: usize> ExactSizeIterator for SmallUDivisIter<I, N> {}

/// Implements `Divisible` trait using bytemuck memory casting.
///
/// This macro generates `Divisible` implementations for a big type over smaller types.
/// The implementations use the helper functions in the `memcast` module.
macro_rules! impl_divisible_memcast {
	($big:ty, $($small:ty),+) => {
		$(
			impl $crate::underlier::Divisible<$small> for $big {
				const LOG_N: usize = (size_of::<$big>() / size_of::<$small>()).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::value_iter::<$big, $small, N>(value)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::ref_iter::<$big, $small, N>(value)
				}

				#[inline]
				#[cfg(target_endian = "little")]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					$crate::underlier::memcast::slice_iter::<$big, $small>(slice)
				}

				#[inline]
				#[cfg(target_endian = "big")]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $small> + Send + Clone + '_ {
					const LOG_N: usize = (size_of::<$big>() / size_of::<$small>()).ilog2() as usize;
					$crate::underlier::memcast::slice_iter::<$big, $small, LOG_N>(slice)
				}

				#[inline]
				fn get(self, index: usize) -> $small {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::get::<$big, $small, N>(&self, index)
				}

				#[inline]
				fn set(self, index: usize, val: $small) -> Self {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::set::<$big, $small, N>(&self, index, val)
				}

				#[inline]
				fn broadcast(val: $small) -> Self {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::broadcast::<$big, $small, N>(val)
				}

				#[inline]
				fn from_iter(iter: impl Iterator<Item = $small>) -> Self {
					const N: usize = size_of::<$big>() / size_of::<$small>();
					$crate::underlier::memcast::from_iter::<$big, $small, N>(iter)
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divisible_memcast;

/// Implements `Divisible` trait for SmallU types using bitmask operations.
///
/// This macro generates `Divisible<SmallU<BITS>>` implementations for a big type
/// by wrapping byte iteration with bitmasking to extract sub-byte elements.
macro_rules! impl_divisible_bitmask {
	// Special case for u8: operates directly on the byte without needing Divisible::<u8>
	(u8, $($bits:expr),+) => {
		$(
			impl $crate::underlier::Divisible<$crate::underlier::SmallU<$bits>> for u8 {
				const LOG_N: usize = (8usize / $bits).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone {
					$crate::underlier::SmallUDivisIter::new(std::iter::once(value))
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(std::iter::once(*value))
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(slice.iter().copied())
				}

				#[inline]
				fn get(self, index: usize) -> $crate::underlier::SmallU<$bits> {
					let shift = index * $bits;
					$crate::underlier::SmallU::<$bits>::new(self >> shift)
				}

				#[inline]
				fn set(self, index: usize, val: $crate::underlier::SmallU<$bits>) -> Self {
					let shift = index * $bits;
					let mask = (1u8 << $bits) - 1;
					(self & !(mask << shift)) | (val.val() << shift)
				}

				#[inline]
				fn broadcast(val: $crate::underlier::SmallU<$bits>) -> Self {
					if $bits == 1 {
						// For 1-bit values: 0 -> 0x00, 1 -> 0xFF
						val.val().wrapping_neg()
					} else {
						let mut result = val.val();
						// Self-replicate to fill the byte
						let mut current_bits = $bits;
						while current_bits < 8 {
							result |= result << current_bits;
							current_bits *= 2;
						}
						result
					}
				}

				#[inline]
				fn from_iter(iter: impl Iterator<Item = $crate::underlier::SmallU<$bits>>) -> Self {
					const N: usize = 8 / $bits;
					let mut result: Self = 0;
					for (i, val) in iter.take(N).enumerate() {
						result = $crate::underlier::Divisible::<$crate::underlier::SmallU<$bits>>::set(result, i, val);
					}
					result
				}
			}
		)+
	};

	// General case for types larger than u8: wraps byte iteration
	($big:ty, $($bits:expr),+) => {
		$(
			impl $crate::underlier::Divisible<$crate::underlier::SmallU<$bits>> for $big {
				const LOG_N: usize = (8 * size_of::<$big>() / $bits).ilog2() as usize;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone {
					$crate::underlier::SmallUDivisIter::new(
						$crate::underlier::Divisible::<u8>::value_iter(value)
					)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(
						$crate::underlier::Divisible::<u8>::ref_iter(value)
					)
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $crate::underlier::SmallU<$bits>> + Send + Clone + '_ {
					$crate::underlier::SmallUDivisIter::new(
						$crate::underlier::Divisible::<u8>::slice_iter(slice)
					)
				}

				#[inline]
				fn get(self, index: usize) -> $crate::underlier::SmallU<$bits> {
					$crate::underlier::bitmask::get::<Self, $bits>(self, index)
				}

				#[inline]
				fn set(self, index: usize, val: $crate::underlier::SmallU<$bits>) -> Self {
					$crate::underlier::bitmask::set::<Self, $bits>(self, index, val)
				}

				#[inline]
				fn broadcast(val: $crate::underlier::SmallU<$bits>) -> Self {
					// First splat to u8, then splat the byte to fill Self
					let byte = $crate::underlier::Divisible::<$crate::underlier::SmallU<$bits>>::broadcast(val);
					$crate::underlier::Divisible::<u8>::broadcast(byte)
				}

				#[inline]
				fn from_iter(iter: impl Iterator<Item = $crate::underlier::SmallU<$bits>>) -> Self {
					const N: usize = 8 * size_of::<$big>() / $bits;
					let mut result: Self = bytemuck::Zeroable::zeroed();
					for (i, val) in iter.take(N).enumerate() {
						result = $crate::underlier::Divisible::<$crate::underlier::SmallU<$bits>>::set(result, i, val);
					}
					result
				}
			}
		)+
	};
}

#[allow(unused)]
pub(crate) use impl_divisible_bitmask;

use super::small_uint::SmallU;

// Implement Divisible using memcast for primitive types
impl_divisible_memcast!(u128, u64, u32, u16, u8);
impl_divisible_memcast!(u64, u32, u16, u8);
impl_divisible_memcast!(u32, u16, u8);
impl_divisible_memcast!(u16, u8);

// Implement Divisible using bitmask for SmallU types
impl_divisible_bitmask!(u8, 1, 2, 4);
impl_divisible_bitmask!(u16, 1, 2, 4);
impl_divisible_bitmask!(u32, 1, 2, 4);
impl_divisible_bitmask!(u64, 1, 2, 4);
impl_divisible_bitmask!(u128, 1, 2, 4);

// Divisible for SmallU types that subdivide into smaller SmallU types
impl Divisible<SmallU<1>> for SmallU<2> {
    const LOG_N: usize = 1;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> SmallU<1> {
        SmallU::<1>::new(self.val() >> index)
    }

    #[inline]
    fn set(self, index: usize, val: SmallU<1>) -> Self {
        let mask = 1u8 << index;
        SmallU::<2>::new((self.val() & !mask) | (val.val() << index))
    }

    #[inline]
    fn broadcast(val: SmallU<1>) -> Self {
        // 0b0 -> 0b00, 0b1 -> 0b11
        let v = val.val();
        SmallU::<2>::new(v | (v << 1))
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = SmallU<1>>) -> Self {
        iter.chain(std::iter::repeat(SmallU::<1>::new(0)))
            .take(2)
            .enumerate()
            .fold(SmallU::<2>::new(0), |acc, (i, val)| acc.set(i, val))
    }
}

impl Divisible<SmallU<1>> for SmallU<4> {
    const LOG_N: usize = 2;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = SmallU<1>> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> SmallU<1> {
        SmallU::<1>::new(self.val() >> index)
    }

    #[inline]
    fn set(self, index: usize, val: SmallU<1>) -> Self {
        let mask = 1u8 << index;
        SmallU::<4>::new((self.val() & !mask) | (val.val() << index))
    }

    #[inline]
    fn broadcast(val: SmallU<1>) -> Self {
        // 0b0 -> 0b0000, 0b1 -> 0b1111
        let mut v = val.val();
        v |= v << 1;
        v |= v << 2;
        SmallU::<4>::new(v)
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = SmallU<1>>) -> Self {
        iter.chain(std::iter::repeat(SmallU::<1>::new(0)))
            .take(4)
            .enumerate()
            .fold(SmallU::<4>::new(0), |acc, (i, val)| acc.set(i, val))
    }
}

impl Divisible<SmallU<2>> for SmallU<4> {
    const LOG_N: usize = 1;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = SmallU<2>> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = SmallU<2>> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = SmallU<2>> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> SmallU<2> {
        SmallU::<2>::new(self.val() >> (index * 2))
    }

    #[inline]
    fn set(self, index: usize, val: SmallU<2>) -> Self {
        let shift = index * 2;
        let mask = 0b11u8 << shift;
        SmallU::<4>::new((self.val() & !mask) | (val.val() << shift))
    }

    #[inline]
    fn broadcast(val: SmallU<2>) -> Self {
        // 0bXX -> 0bXXXX
        let v = val.val();
        SmallU::<4>::new(v | (v << 2))
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = SmallU<2>>) -> Self {
        iter.chain(std::iter::repeat(SmallU::<2>::new(0)))
            .take(2)
            .enumerate()
            .fold(SmallU::<4>::new(0), |acc, (i, val)| acc.set(i, val))
    }
}

/// Implements reflexive `Divisible<Self>` for a type (dividing into itself once).
macro_rules! impl_divisible_self {
	($($ty:ty),+) => {
		$(
			impl Divisible<$ty> for $ty {
				const LOG_N: usize = 0;

				#[inline]
				fn value_iter(value: Self) -> impl ExactSizeIterator<Item = $ty> + Send + Clone {
					std::iter::once(value)
				}

				#[inline]
				fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = $ty> + Send + Clone + '_ {
					std::iter::once(*value)
				}

				#[inline]
				fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = $ty> + Send + Clone + '_ {
					slice.iter().copied()
				}

				#[inline]
				fn get(self, index: usize) -> $ty {
					debug_assert_eq!(index, 0);
					self
				}

				#[inline]
				fn set(self, index: usize, val: $ty) -> Self {
					debug_assert_eq!(index, 0);
					val
				}

				#[inline]
				fn broadcast(val: $ty) -> Self {
					val
				}

				#[inline]
				fn from_iter(mut iter: impl Iterator<Item = $ty>) -> Self {
					iter.next().unwrap_or_else(bytemuck::Zeroable::zeroed)
				}
			}
		)+
	};
}

impl_divisible_self!(u8, u16, u32, u64, u128, SmallU<1>, SmallU<2>, SmallU<4>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::underlier::small_uint::{U1, U2, U4};

    #[test]
    fn test_divisible_u8_u4() {
        let val: u8 = 0x34;

        // Test get - LSB first: nibbles
        assert_eq!(Divisible::<U4>::get(val, 0), U4::new(0x4));
        assert_eq!(Divisible::<U4>::get(val, 1), U4::new(0x3));

        // Test set
        let modified = Divisible::<U4>::set(val, 0, U4::new(0xF));
        assert_eq!(modified, 0x3F);
        let modified = Divisible::<U4>::set(val, 1, U4::new(0xA));
        assert_eq!(modified, 0xA4);

        // Test ref_iter
        let parts: Vec<U4> = Divisible::<U4>::ref_iter(&val).collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], U4::new(0x4));
        assert_eq!(parts[1], U4::new(0x3));

        // Test value_iter
        let parts: Vec<U4> = Divisible::<U4>::value_iter(val).collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], U4::new(0x4));
        assert_eq!(parts[1], U4::new(0x3));

        // Test slice_iter
        let vals = [0x34u8, 0x56u8];
        let parts: Vec<U4> = Divisible::<U4>::slice_iter(&vals).collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], U4::new(0x4));
        assert_eq!(parts[1], U4::new(0x3));
        assert_eq!(parts[2], U4::new(0x6));
        assert_eq!(parts[3], U4::new(0x5));
    }

    #[test]
    fn test_divisible_u16_u4() {
        let val: u16 = 0x1234;

        // Test get - LSB first: nibbles
        assert_eq!(Divisible::<U4>::get(val, 0), U4::new(0x4));
        assert_eq!(Divisible::<U4>::get(val, 1), U4::new(0x3));
        assert_eq!(Divisible::<U4>::get(val, 2), U4::new(0x2));
        assert_eq!(Divisible::<U4>::get(val, 3), U4::new(0x1));

        // Test set
        let modified = Divisible::<U4>::set(val, 1, U4::new(0xF));
        assert_eq!(modified, 0x12F4);

        // Test ref_iter
        let parts: Vec<U4> = Divisible::<U4>::ref_iter(&val).collect();
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], U4::new(0x4));
        assert_eq!(parts[3], U4::new(0x1));
    }

    #[test]
    fn test_divisible_u16_u2() {
        // 0b1011_0010_1101_0011 = 0xB2D3
        let val: u16 = 0b1011001011010011;

        // Test get - LSB first: 2-bit chunks
        assert_eq!(Divisible::<U2>::get(val, 0), U2::new(0b11)); // bits 0-1
        assert_eq!(Divisible::<U2>::get(val, 1), U2::new(0b00)); // bits 2-3
        assert_eq!(Divisible::<U2>::get(val, 7), U2::new(0b10)); // bits 14-15

        // Test ref_iter
        let parts: Vec<U2> = Divisible::<U2>::ref_iter(&val).collect();
        assert_eq!(parts.len(), 8);
        assert_eq!(parts[0], U2::new(0b11));
        assert_eq!(parts[7], U2::new(0b10));
    }

    #[test]
    fn test_divisible_u16_u1() {
        // 0b1010_1100_0011_0101 = 0xAC35
        let val: u16 = 0b1010110000110101;

        // Test get - LSB first: individual bits
        assert_eq!(Divisible::<U1>::get(val, 0), U1::new(1)); // bit 0
        assert_eq!(Divisible::<U1>::get(val, 1), U1::new(0)); // bit 1
        assert_eq!(Divisible::<U1>::get(val, 15), U1::new(1)); // bit 15

        // Test set
        let modified = Divisible::<U1>::set(val, 0, U1::new(0));
        assert_eq!(modified, 0b1010110000110100);

        // Test ref_iter
        let parts: Vec<U1> = Divisible::<U1>::ref_iter(&val).collect();
        assert_eq!(parts.len(), 16);
        assert_eq!(parts[0], U1::new(1));
        assert_eq!(parts[15], U1::new(1));
    }

    #[test]
    fn test_divisible_u64_u4() {
        let val: u64 = 0x123456789ABCDEF0;

        // Test get - LSB first: nibbles
        assert_eq!(Divisible::<U4>::get(val, 0), U4::new(0x0));
        assert_eq!(Divisible::<U4>::get(val, 1), U4::new(0xF));
        assert_eq!(Divisible::<U4>::get(val, 15), U4::new(0x1));

        // Test ref_iter
        let parts: Vec<U4> = Divisible::<U4>::ref_iter(&val).collect();
        assert_eq!(parts.len(), 16);
    }

    #[test]
    fn test_divisible_u32_u8_slice() {
        let vals: [u32; 2] = [0x04030201, 0x08070605];

        // Test slice_iter
        let parts: Vec<u8> = Divisible::<u8>::slice_iter(&vals).collect();
        assert_eq!(parts.len(), 8);
        // LSB-first ordering within each u32
        assert_eq!(parts[0], 0x01);
        assert_eq!(parts[1], 0x02);
        assert_eq!(parts[2], 0x03);
        assert_eq!(parts[3], 0x04);
        assert_eq!(parts[4], 0x05);
        assert_eq!(parts[5], 0x06);
        assert_eq!(parts[6], 0x07);
        assert_eq!(parts[7], 0x08);
    }

    #[test]
    fn test_broadcast_u32_u8() {
        let result: u32 = Divisible::<u8>::broadcast(0xAB);
        assert_eq!(result, 0xABABABAB);
    }

    #[test]
    fn test_broadcast_u64_u16() {
        let result: u64 = Divisible::<u16>::broadcast(0x1234);
        assert_eq!(result, 0x1234123412341234);
    }

    #[test]
    fn test_broadcast_u128_u32() {
        let result: u128 = Divisible::<u32>::broadcast(0xDEADBEEF);
        assert_eq!(result, 0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF);
    }

    #[test]
    fn test_broadcast_u8_u4() {
        let result: u8 = Divisible::<U4>::broadcast(U4::new(0x5));
        assert_eq!(result, 0x55);
    }

    #[test]
    fn test_broadcast_u16_u4() {
        let result: u16 = Divisible::<U4>::broadcast(U4::new(0xA));
        assert_eq!(result, 0xAAAA);
    }

    #[test]
    fn test_broadcast_u8_u2() {
        let result: u8 = Divisible::<U2>::broadcast(U2::new(0b11));
        assert_eq!(result, 0xFF);
        let result: u8 = Divisible::<U2>::broadcast(U2::new(0b01));
        assert_eq!(result, 0x55);
    }

    #[test]
    fn test_broadcast_u8_u1() {
        let result: u8 = Divisible::<U1>::broadcast(U1::new(0));
        assert_eq!(result, 0x00);
        let result: u8 = Divisible::<U1>::broadcast(U1::new(1));
        assert_eq!(result, 0xFF);
    }

    #[test]
    fn test_broadcast_smallu2_from_smallu1() {
        let result: SmallU<2> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(0));
        assert_eq!(result.val(), 0b00);
        let result: SmallU<2> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(1));
        assert_eq!(result.val(), 0b11);
    }

    #[test]
    fn test_broadcast_smallu4_from_smallu1() {
        let result: SmallU<4> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(0));
        assert_eq!(result.val(), 0b0000);
        let result: SmallU<4> = Divisible::<SmallU<1>>::broadcast(SmallU::<1>::new(1));
        assert_eq!(result.val(), 0b1111);
    }

    #[test]
    fn test_broadcast_smallu4_from_smallu2() {
        let result: SmallU<4> = Divisible::<SmallU<2>>::broadcast(SmallU::<2>::new(0b10));
        assert_eq!(result.val(), 0b1010);
    }

    #[test]
    fn test_broadcast_reflexive() {
        let result: u64 = Divisible::<u64>::broadcast(0x123456789ABCDEF0);
        assert_eq!(result, 0x123456789ABCDEF0);
    }

    #[test]
    fn test_from_iter_full() {
        let result: u32 = Divisible::<u8>::from_iter([0x01, 0x02, 0x03, 0x04].into_iter());
        assert_eq!(result, 0x04030201);
    }

    #[test]
    fn test_from_iter_partial() {
        // Only 2 elements, remaining should be 0
        let result: u32 = Divisible::<u8>::from_iter([0xAB, 0xCD].into_iter());
        assert_eq!(result, 0x0000CDAB);
    }

    #[test]
    fn test_from_iter_empty() {
        let result: u32 = Divisible::<u8>::from_iter(std::iter::empty());
        assert_eq!(result, 0);
    }

    #[test]
    fn test_from_iter_excess() {
        // More than N elements, only first 4 should be consumed
        let result: u32 =
            Divisible::<u8>::from_iter([0x01, 0x02, 0x03, 0x04, 0x05, 0x06].into_iter());
        assert_eq!(result, 0x04030201);
    }

    #[test]
    fn test_from_iter_u64_u16() {
        let result: u64 = Divisible::<u16>::from_iter([0x1234, 0x5678, 0x9ABC].into_iter());
        // Only 3 elements provided, 4th should be 0
        assert_eq!(result, 0x0000_9ABC_5678_1234);
    }

    #[test]
    fn test_from_iter_smallu() {
        let result: u8 = Divisible::<U4>::from_iter([U4::new(0xA), U4::new(0xB)].into_iter());
        assert_eq!(result, 0xBA);
    }
}
