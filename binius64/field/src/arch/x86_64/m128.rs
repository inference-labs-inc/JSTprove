// Copyright 2024-2025 Irreducible Inc.

use std::{
    arch::x86_64::*,
    array,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr},
};

use binius_utils::{
    DeserializeBytes, SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
    serialization::{assert_enough_data_for, assert_enough_space_for},
};
use bytemuck::{Pod, Zeroable};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use seq_macro::seq;

use crate::{
    BinaryField,
    arch::portable::packed::PackedPrimitiveType,
    underlier::{
        Divisible, NumCast, SmallU, SpreadToByte, U2, U4, UnderlierType, UnderlierWithBitOps,
        impl_divisible_bitmask, mapget, spread_fallback,
    },
};

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M128(pub(super) __m128i);

impl M128 {
    #[inline(always)]
    pub const fn from_u128(val: u128) -> Self {
        let mut result = Self::ZERO;
        unsafe {
            result.0 = std::mem::transmute_copy(&val);
        }

        result
    }
}

impl From<__m128i> for M128 {
    #[inline(always)]
    fn from(value: __m128i) -> Self {
        Self(value)
    }
}

impl From<u128> for M128 {
    fn from(value: u128) -> Self {
        Self(unsafe {
            // Safety: u128 is 16-byte aligned
            assert_eq!(align_of::<u128>(), 16);
            _mm_load_si128(&raw const value as *const __m128i)
        })
    }
}

impl From<u64> for M128 {
    fn from(value: u64) -> Self {
        Self::from(value as u128)
    }
}

impl From<u32> for M128 {
    fn from(value: u32) -> Self {
        Self::from(value as u128)
    }
}

impl From<u16> for M128 {
    fn from(value: u16) -> Self {
        Self::from(value as u128)
    }
}

impl From<u8> for M128 {
    fn from(value: u8) -> Self {
        Self::from(value as u128)
    }
}

impl<const N: usize> From<SmallU<N>> for M128 {
    fn from(value: SmallU<N>) -> Self {
        Self::from(value.val() as u128)
    }
}

impl From<M128> for u128 {
    fn from(value: M128) -> Self {
        let mut result = 0u128;
        unsafe {
            // Safety: u128 is 16-byte aligned
            assert_eq!(align_of::<u128>(), 16);
            _mm_store_si128(&raw mut result as *mut __m128i, value.0)
        };
        result
    }
}

impl From<M128> for __m128i {
    #[inline(always)]
    fn from(value: M128) -> Self {
        value.0
    }
}

impl SerializeBytes for M128 {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

        let raw_value: u128 = (*self).into();

        write_buf.put_u128_le(raw_value);
        Ok(())
    }
}

impl DeserializeBytes for M128 {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;

        let raw_value = read_buf.get_u128_le();

        Ok(Self::from(raw_value))
    }
}

impl_divisible_bitmask!(M128, 1, 2, 4);

impl<U: NumCast<u128>> NumCast<M128> for U {
    #[inline(always)]
    fn num_cast_from(val: M128) -> Self {
        Self::num_cast_from(u128::from(val))
    }
}

impl Default for M128 {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }
}

impl BitAnd for M128 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_and_si128(self.0, rhs.0) })
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
        Self(unsafe { _mm_or_si128(self.0, rhs.0) })
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
        Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
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

    fn not(self) -> Self::Output {
        const ONES: __m128i = m128_from_u128!(u128::MAX);

        self ^ Self(ONES)
    }
}

/// `std::cmp::max` isn't const, so we need our own implementation
pub(crate) const fn max_i32(left: i32, right: i32) -> i32 {
    if left > right { left } else { right }
}

/// This solution shows 4X better performance.
/// We have to use macro because parameter `count` in _mm_slli_epi64/_mm_srli_epi64 should be passed
/// as constant and Rust currently doesn't allow passing expressions (`count - 64`) where variable
/// is a generic constant parameter. Source: <https://stackoverflow.com/questions/34478328/the-best-way-to-shift-a-m128i/34482688#34482688>
macro_rules! bitshift_128b {
	($val:expr, $shift:ident, $byte_shift:ident, $bit_shift_64:ident, $bit_shift_64_opposite:ident, $or:ident) => {
		unsafe {
			let carry = $byte_shift($val, 8);
			seq!(N in 64..128 {
				if $shift == N {
					return $bit_shift_64(
						carry,
						crate::arch::x86_64::m128::max_i32((N - 64) as i32, 0) as _,
					).into();
				}
			});
			seq!(N in 0..64 {
				if $shift == N {
					let carry = $bit_shift_64_opposite(
						carry,
						crate::arch::x86_64::m128::max_i32((64 - N) as i32, 0) as _,
					);

					let val = $bit_shift_64($val, N);
					return $or(val, carry).into();
				}
			});

			return Default::default()
		}
	};
}

pub(crate) use bitshift_128b;

impl Shr<usize> for M128 {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        // This implementation is effective when `rhs` is known at compile-time.
        // In our code this is always the case.
        bitshift_128b!(
            self.0,
            rhs,
            _mm_bsrli_si128,
            _mm_srli_epi64,
            _mm_slli_epi64,
            _mm_or_si128
        )
    }
}

impl Shl<usize> for M128 {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        // This implementation is effective when `rhs` is known at compile-time.
        // In our code this is always the case.
        bitshift_128b!(
            self.0,
            rhs,
            _mm_bslli_si128,
            _mm_slli_epi64,
            _mm_srli_epi64,
            _mm_or_si128
        );
    }
}

impl PartialEq for M128 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let neq = _mm_xor_si128(self.0, other.0);
            _mm_test_all_zeros(neq, neq) == 1
        }
    }
}

impl Eq for M128 {}

impl PartialOrd for M128 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for M128 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        u128::from(*self).cmp(&u128::from(*other))
    }
}

impl Distribution<M128> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> M128 {
        M128(rng.random())
    }
}

impl std::fmt::Display for M128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data: u128 = (*self).into();
        write!(f, "{data:02X?}")
    }
}

impl std::fmt::Debug for M128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "M128({self})")
    }
}

#[repr(align(16))]
pub struct AlignedData(pub [u128; 1]);

macro_rules! m128_from_u128 {
    ($val:expr) => {{
        let aligned_data = $crate::arch::x86_64::m128::AlignedData([$val]);
        unsafe { *(aligned_data.0.as_ptr() as *const core::arch::x86_64::__m128i) }
    }};
}

pub(super) use m128_from_u128;

impl UnderlierType for M128 {
    const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
    const ZERO: Self = { Self(m128_from_u128!(0)) };
    const ONE: Self = { Self(m128_from_u128!(1)) };
    const ONES: Self = { Self(m128_from_u128!(u128::MAX)) };

    #[inline(always)]
    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        unsafe {
            let (c, d) = interleave_bits(
                Into::<Self>::into(self).into(),
                Into::<Self>::into(other).into(),
                log_block_len,
            );
            (Self::from(c), Self::from(d))
        }
    }

    #[inline(always)]
    unsafe fn spread<T>(self, log_block_len: usize, block_idx: usize) -> Self
    where
        T: UnderlierWithBitOps,
        Self: Divisible<T>,
    {
        match T::LOG_BITS {
            0 => match log_block_len {
                0 => Self::fill_with_bit(((u128::from(self) >> block_idx) & 1) as _),
                1 => unsafe {
                    let bits: [u8; 2] =
                        array::from_fn(|i| ((u128::from(self) >> (block_idx * 2 + i)) & 1) as _);

                    _mm_set_epi64x(
                        u64::fill_with_bit(bits[1]) as i64,
                        u64::fill_with_bit(bits[0]) as i64,
                    )
                    .into()
                },
                2 => unsafe {
                    let bits: [u8; 4] =
                        array::from_fn(|i| ((u128::from(self) >> (block_idx * 4 + i)) & 1) as _);

                    _mm_set_epi32(
                        u32::fill_with_bit(bits[3]) as i32,
                        u32::fill_with_bit(bits[2]) as i32,
                        u32::fill_with_bit(bits[1]) as i32,
                        u32::fill_with_bit(bits[0]) as i32,
                    )
                    .into()
                },
                3 => unsafe {
                    let bits: [u8; 8] =
                        array::from_fn(|i| ((u128::from(self) >> (block_idx * 8 + i)) & 1) as _);

                    _mm_set_epi16(
                        u16::fill_with_bit(bits[7]) as i16,
                        u16::fill_with_bit(bits[6]) as i16,
                        u16::fill_with_bit(bits[5]) as i16,
                        u16::fill_with_bit(bits[4]) as i16,
                        u16::fill_with_bit(bits[3]) as i16,
                        u16::fill_with_bit(bits[2]) as i16,
                        u16::fill_with_bit(bits[1]) as i16,
                        u16::fill_with_bit(bits[0]) as i16,
                    )
                    .into()
                },
                4 => unsafe {
                    let bits: [u8; 16] =
                        array::from_fn(|i| ((u128::from(self) >> (block_idx * 16 + i)) & 1) as _);

                    _mm_set_epi8(
                        u8::fill_with_bit(bits[15]) as i8,
                        u8::fill_with_bit(bits[14]) as i8,
                        u8::fill_with_bit(bits[13]) as i8,
                        u8::fill_with_bit(bits[12]) as i8,
                        u8::fill_with_bit(bits[11]) as i8,
                        u8::fill_with_bit(bits[10]) as i8,
                        u8::fill_with_bit(bits[9]) as i8,
                        u8::fill_with_bit(bits[8]) as i8,
                        u8::fill_with_bit(bits[7]) as i8,
                        u8::fill_with_bit(bits[6]) as i8,
                        u8::fill_with_bit(bits[5]) as i8,
                        u8::fill_with_bit(bits[4]) as i8,
                        u8::fill_with_bit(bits[3]) as i8,
                        u8::fill_with_bit(bits[2]) as i8,
                        u8::fill_with_bit(bits[1]) as i8,
                        u8::fill_with_bit(bits[0]) as i8,
                    )
                    .into()
                },
                _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
            },
            1 => match log_block_len {
                0 => unsafe {
                    let value =
                        U2::new((u128::from(self) >> (block_idx * 2)) as _).spread_to_byte();

                    _mm_set1_epi8(value as i8).into()
                },
                1 => {
                    let bytes: [u8; 2] = array::from_fn(|i| {
                        U2::new((u128::from(self) >> (block_idx * 4 + i * 2)) as _).spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| bytes[i / 8])
                }
                2 => {
                    let bytes: [u8; 4] = array::from_fn(|i| {
                        U2::new((u128::from(self) >> (block_idx * 8 + i * 2)) as _).spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| bytes[i / 4])
                }
                3 => {
                    let bytes: [u8; 8] = array::from_fn(|i| {
                        U2::new((u128::from(self) >> (block_idx * 16 + i * 2)) as _)
                            .spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| bytes[i / 2])
                }
                4 => {
                    let bytes: [u8; 16] = array::from_fn(|i| {
                        U2::new((u128::from(self) >> (block_idx * 32 + i * 2)) as _)
                            .spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| bytes[i])
                }
                _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
            },
            2 => match log_block_len {
                0 => {
                    let value =
                        U4::new((u128::from(self) >> (block_idx * 4)) as _).spread_to_byte();

                    unsafe { _mm_set1_epi8(value as i8).into() }
                }
                1 => {
                    let values: [u8; 2] = array::from_fn(|i| {
                        U4::new((u128::from(self) >> (block_idx * 8 + i * 4)) as _).spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| values[i / 8])
                }
                2 => {
                    let values: [u8; 4] = array::from_fn(|i| {
                        U4::new((u128::from(self) >> (block_idx * 16 + i * 4)) as _)
                            .spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| values[i / 4])
                }
                3 => {
                    let values: [u8; 8] = array::from_fn(|i| {
                        U4::new((u128::from(self) >> (block_idx * 32 + i * 4)) as _)
                            .spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| values[i / 2])
                }
                4 => {
                    let values: [u8; 16] = array::from_fn(|i| {
                        U4::new((u128::from(self) >> (block_idx * 64 + i * 4)) as _)
                            .spread_to_byte()
                    });

                    Self::from_fn::<u8>(|i| values[i])
                }
                _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
            },
            3 => match log_block_len {
                0 => unsafe { _mm_shuffle_epi8(self.0, LOG_B8_0[block_idx].0).into() },
                1 => unsafe { _mm_shuffle_epi8(self.0, LOG_B8_1[block_idx].0).into() },
                2 => unsafe { _mm_shuffle_epi8(self.0, LOG_B8_2[block_idx].0).into() },
                3 => unsafe { _mm_shuffle_epi8(self.0, LOG_B8_3[block_idx].0).into() },
                4 => self,
                _ => panic!("unsupported block length"),
            },
            4 => match log_block_len {
                0 => {
                    let value = (u128::from(self) >> (block_idx * 16)) as u16;

                    unsafe { _mm_set1_epi16(value as i16).into() }
                }
                1 => {
                    let values: [u16; 2] =
                        array::from_fn(|i| (u128::from(self) >> (block_idx * 32 + i * 16)) as u16);

                    Self::from_fn::<u16>(|i| values[i / 4])
                }
                2 => {
                    let values: [u16; 4] =
                        array::from_fn(|i| (u128::from(self) >> (block_idx * 64 + i * 16)) as u16);

                    Self::from_fn::<u16>(|i| values[i / 2])
                }
                3 => self,
                _ => panic!("unsupported block length"),
            },
            5 => match log_block_len {
                0 => unsafe {
                    let value = (u128::from(self) >> (block_idx * 32)) as u32;

                    _mm_set1_epi32(value as i32).into()
                },
                1 => {
                    let values: [u32; 2] =
                        array::from_fn(|i| (u128::from(self) >> (block_idx * 64 + i * 32)) as u32);

                    Self::from_fn::<u32>(|i| values[i / 2])
                }
                2 => self,
                _ => panic!("unsupported block length"),
            },
            6 => match log_block_len {
                0 => unsafe {
                    let value = (u128::from(self) >> (block_idx * 64)) as u64;

                    _mm_set1_epi64x(value as i64).into()
                },
                1 => self,
                _ => panic!("unsupported block length"),
            },
            7 => self,
            _ => panic!("unsupported bit length"),
        }
    }
}

unsafe impl Zeroable for M128 {}

unsafe impl Pod for M128 {}

unsafe impl Send for M128 {}

unsafe impl Sync for M128 {}

static LOG_B8_0: [M128; 16] = precompute_spread_mask::<16>(0, 3);
static LOG_B8_1: [M128; 8] = precompute_spread_mask::<8>(1, 3);
static LOG_B8_2: [M128; 4] = precompute_spread_mask::<4>(2, 3);
static LOG_B8_3: [M128; 2] = precompute_spread_mask::<2>(3, 3);

const fn precompute_spread_mask<const BLOCK_IDX_AMOUNT: usize>(
    log_block_len: usize,
    t_log_bits: usize,
) -> [M128; BLOCK_IDX_AMOUNT] {
    let element_log_width = t_log_bits - 3;

    let element_width = 1 << element_log_width;

    let block_size = 1 << (log_block_len + element_log_width);
    let repeat = 1 << (4 - element_log_width - log_block_len);
    let mut masks = [[0u8; 16]; BLOCK_IDX_AMOUNT];

    let mut block_idx = 0;

    while block_idx < BLOCK_IDX_AMOUNT {
        let base = block_idx * block_size;
        let mut j = 0;
        while j < 16 {
            masks[block_idx][j] =
                (base + ((j / element_width) / repeat) * element_width + j % element_width) as u8;
            j += 1;
        }
        block_idx += 1;
    }
    let mut m128_masks = [M128::ZERO; BLOCK_IDX_AMOUNT];

    let mut block_idx = 0;

    while block_idx < BLOCK_IDX_AMOUNT {
        m128_masks[block_idx] = M128::from_u128(u128::from_le_bytes(masks[block_idx]));
        block_idx += 1;
    }

    m128_masks
}

impl<Scalar: BinaryField> From<__m128i> for PackedPrimitiveType<M128, Scalar> {
    fn from(value: __m128i) -> Self {
        M128::from(value).into()
    }
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
    fn from(value: u128) -> Self {
        M128::from(value).into()
    }
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for __m128i {
    fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
        value.to_underlier().into()
    }
}

#[inline]
unsafe fn interleave_bits(a: __m128i, b: __m128i, log_block_len: usize) -> (__m128i, __m128i) {
    match log_block_len {
        0 => unsafe {
            let mask = _mm_set1_epi8(0x55i8);
            interleave_bits_imm::<1>(a, b, mask)
        },
        1 => unsafe {
            let mask = _mm_set1_epi8(0x33i8);
            interleave_bits_imm::<2>(a, b, mask)
        },
        2 => unsafe {
            let mask = _mm_set1_epi8(0x0fi8);
            interleave_bits_imm::<4>(a, b, mask)
        },
        3 => unsafe {
            let shuffle = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
            let a = _mm_shuffle_epi8(a, shuffle);
            let b = _mm_shuffle_epi8(b, shuffle);
            let a_prime = _mm_unpacklo_epi8(a, b);
            let b_prime = _mm_unpackhi_epi8(a, b);
            (a_prime, b_prime)
        },
        4 => unsafe {
            let shuffle = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
            let a = _mm_shuffle_epi8(a, shuffle);
            let b = _mm_shuffle_epi8(b, shuffle);
            let a_prime = _mm_unpacklo_epi16(a, b);
            let b_prime = _mm_unpackhi_epi16(a, b);
            (a_prime, b_prime)
        },
        5 => unsafe {
            let shuffle = _mm_set_epi8(15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
            let a = _mm_shuffle_epi8(a, shuffle);
            let b = _mm_shuffle_epi8(b, shuffle);
            let a_prime = _mm_unpacklo_epi32(a, b);
            let b_prime = _mm_unpackhi_epi32(a, b);
            (a_prime, b_prime)
        },
        6 => unsafe {
            let a_prime = _mm_unpacklo_epi64(a, b);
            let b_prime = _mm_unpackhi_epi64(a, b);
            (a_prime, b_prime)
        },
        _ => panic!("unsupported block length"),
    }
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: i32>(
    a: __m128i,
    b: __m128i,
    mask: __m128i,
) -> (__m128i, __m128i) {
    unsafe {
        let t = _mm_and_si128(_mm_xor_si128(_mm_srli_epi64::<BLOCK_LEN>(a), b), mask);
        let a_prime = _mm_xor_si128(a, _mm_slli_epi64::<BLOCK_LEN>(t));
        let b_prime = _mm_xor_si128(b, t);
        (a_prime, b_prime)
    }
}

// Divisible implementations using SIMD extract/insert intrinsics

impl Divisible<u128> for M128 {
    const LOG_N: usize = 0;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = u128> + Send + Clone {
        std::iter::once(u128::from(value))
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = u128> + Send + Clone + '_ {
        std::iter::once(u128::from(*value))
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = u128> + Send + Clone + '_ {
        slice.iter().map(|&v| u128::from(v))
    }

    #[inline]
    fn get(self, index: usize) -> u128 {
        assert!(index == 0, "index out of bounds");
        u128::from(self)
    }

    #[inline]
    fn set(self, index: usize, val: u128) -> Self {
        assert!(index == 0, "index out of bounds");
        Self::from(val)
    }

    #[inline]
    fn broadcast(val: u128) -> Self {
        Self::from(val)
    }

    #[inline]
    fn from_iter(mut iter: impl Iterator<Item = u128>) -> Self {
        iter.next().map(Self::from).unwrap_or(Self::ZERO)
    }
}

impl Divisible<u64> for M128 {
    const LOG_N: usize = 1;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = u64> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = u64> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = u64> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> u64 {
        unsafe {
            match index {
                0 => _mm_extract_epi64(self.0, 0) as u64,
                1 => _mm_extract_epi64(self.0, 1) as u64,
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u64) -> Self {
        unsafe {
            match index {
                0 => Self(_mm_insert_epi64(self.0, val as i64, 0)),
                1 => Self(_mm_insert_epi64(self.0, val as i64, 1)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u64) -> Self {
        unsafe { Self(_mm_set1_epi64x(val as i64)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u64>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u64; 2] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(2).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u32> for M128 {
    const LOG_N: usize = 2;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = u32> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = u32> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = u32> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> u32 {
        unsafe {
            match index {
                0 => _mm_extract_epi32(self.0, 0) as u32,
                1 => _mm_extract_epi32(self.0, 1) as u32,
                2 => _mm_extract_epi32(self.0, 2) as u32,
                3 => _mm_extract_epi32(self.0, 3) as u32,
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u32) -> Self {
        unsafe {
            match index {
                0 => Self(_mm_insert_epi32(self.0, val as i32, 0)),
                1 => Self(_mm_insert_epi32(self.0, val as i32, 1)),
                2 => Self(_mm_insert_epi32(self.0, val as i32, 2)),
                3 => Self(_mm_insert_epi32(self.0, val as i32, 3)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u32) -> Self {
        unsafe { Self(_mm_set1_epi32(val as i32)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u32>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u32; 4] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(4).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u16> for M128 {
    const LOG_N: usize = 3;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = u16> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = u16> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = u16> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> u16 {
        unsafe {
            match index {
                0 => _mm_extract_epi16(self.0, 0) as u16,
                1 => _mm_extract_epi16(self.0, 1) as u16,
                2 => _mm_extract_epi16(self.0, 2) as u16,
                3 => _mm_extract_epi16(self.0, 3) as u16,
                4 => _mm_extract_epi16(self.0, 4) as u16,
                5 => _mm_extract_epi16(self.0, 5) as u16,
                6 => _mm_extract_epi16(self.0, 6) as u16,
                7 => _mm_extract_epi16(self.0, 7) as u16,
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u16) -> Self {
        unsafe {
            match index {
                0 => Self(_mm_insert_epi16(self.0, val as i32, 0)),
                1 => Self(_mm_insert_epi16(self.0, val as i32, 1)),
                2 => Self(_mm_insert_epi16(self.0, val as i32, 2)),
                3 => Self(_mm_insert_epi16(self.0, val as i32, 3)),
                4 => Self(_mm_insert_epi16(self.0, val as i32, 4)),
                5 => Self(_mm_insert_epi16(self.0, val as i32, 5)),
                6 => Self(_mm_insert_epi16(self.0, val as i32, 6)),
                7 => Self(_mm_insert_epi16(self.0, val as i32, 7)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u16) -> Self {
        unsafe { Self(_mm_set1_epi16(val as i16)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u16>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u16; 8] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(8).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u8> for M128 {
    const LOG_N: usize = 4;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = u8> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = u8> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = u8> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> u8 {
        unsafe {
            match index {
                0 => _mm_extract_epi8(self.0, 0) as u8,
                1 => _mm_extract_epi8(self.0, 1) as u8,
                2 => _mm_extract_epi8(self.0, 2) as u8,
                3 => _mm_extract_epi8(self.0, 3) as u8,
                4 => _mm_extract_epi8(self.0, 4) as u8,
                5 => _mm_extract_epi8(self.0, 5) as u8,
                6 => _mm_extract_epi8(self.0, 6) as u8,
                7 => _mm_extract_epi8(self.0, 7) as u8,
                8 => _mm_extract_epi8(self.0, 8) as u8,
                9 => _mm_extract_epi8(self.0, 9) as u8,
                10 => _mm_extract_epi8(self.0, 10) as u8,
                11 => _mm_extract_epi8(self.0, 11) as u8,
                12 => _mm_extract_epi8(self.0, 12) as u8,
                13 => _mm_extract_epi8(self.0, 13) as u8,
                14 => _mm_extract_epi8(self.0, 14) as u8,
                15 => _mm_extract_epi8(self.0, 15) as u8,
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u8) -> Self {
        unsafe {
            match index {
                0 => Self(_mm_insert_epi8(self.0, val as i32, 0)),
                1 => Self(_mm_insert_epi8(self.0, val as i32, 1)),
                2 => Self(_mm_insert_epi8(self.0, val as i32, 2)),
                3 => Self(_mm_insert_epi8(self.0, val as i32, 3)),
                4 => Self(_mm_insert_epi8(self.0, val as i32, 4)),
                5 => Self(_mm_insert_epi8(self.0, val as i32, 5)),
                6 => Self(_mm_insert_epi8(self.0, val as i32, 6)),
                7 => Self(_mm_insert_epi8(self.0, val as i32, 7)),
                8 => Self(_mm_insert_epi8(self.0, val as i32, 8)),
                9 => Self(_mm_insert_epi8(self.0, val as i32, 9)),
                10 => Self(_mm_insert_epi8(self.0, val as i32, 10)),
                11 => Self(_mm_insert_epi8(self.0, val as i32, 11)),
                12 => Self(_mm_insert_epi8(self.0, val as i32, 12)),
                13 => Self(_mm_insert_epi8(self.0, val as i32, 13)),
                14 => Self(_mm_insert_epi8(self.0, val as i32, 14)),
                15 => Self(_mm_insert_epi8(self.0, val as i32, 15)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u8) -> Self {
        unsafe { Self(_mm_set1_epi8(val as i8)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u8>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u8; 16] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(16).enumerate() {
            arr[i] = val;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use binius_utils::bytes::BytesMut;
    use proptest::{arbitrary::any, proptest};
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    fn check_roundtrip<T>(val: M128)
    where
        T: From<M128>,
        M128: From<T>,
    {
        assert_eq!(M128::from(T::from(val)), val);
    }

    #[test]
    fn test_constants() {
        assert_eq!(M128::default(), M128::ZERO);
        assert_eq!(M128::from(0u128), M128::ZERO);
        assert_eq!(M128::from(1u128), M128::ONE);
    }

    fn get(value: M128, log_block_len: usize, index: usize) -> M128 {
        (value >> (index << log_block_len)) & M128::from(1u128 << log_block_len)
    }

    proptest! {
        #[test]
        fn test_conversion(a in any::<u128>()) {
            check_roundtrip::<u128>(a.into());
            check_roundtrip::<__m128i>(a.into());
        }

        #[test]
        fn test_binary_bit_operations(a in any::<u128>(), b in any::<u128>()) {
            assert_eq!(M128::from(a & b), M128::from(a) & M128::from(b));
            assert_eq!(M128::from(a | b), M128::from(a) | M128::from(b));
            assert_eq!(M128::from(a ^ b), M128::from(a) ^ M128::from(b));
        }

        #[test]
        fn test_negate(a in any::<u128>()) {
            assert_eq!(M128::from(!a), !M128::from(a))
        }

        #[test]
        fn test_shifts(a in any::<u128>(), b in 0..128usize) {
            assert_eq!(M128::from(a << b), M128::from(a) << b);
            assert_eq!(M128::from(a >> b), M128::from(a) >> b);
        }

        #[test]
        fn test_interleave_bits(a in any::<u128>(), b in any::<u128>(), height in 0usize..7) {
            let a = M128::from(a);
            let b = M128::from(b);

            let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
            let (c, d) = (M128::from(c), M128::from(d));

            for i in (0..128>>height).step_by(2) {
                assert_eq!(get(c, height, i), get(a, height, i));
                assert_eq!(get(c, height, i+1), get(b, height, i));
                assert_eq!(get(d, height, i), get(a, height, i+1));
                assert_eq!(get(d, height, i+1), get(b, height, i+1));
            }
        }
    }

    #[test]
    fn test_fill_with_bit() {
        assert_eq!(M128::fill_with_bit(1), M128::from(u128::MAX));
        assert_eq!(M128::fill_with_bit(0), M128::from(0u128));
    }

    #[test]
    fn test_eq() {
        let a = M128::from(0u128);
        let b = M128::from(42u128);
        let c = M128::from(u128::MAX);

        assert_eq!(a, a);
        assert_eq!(b, b);
        assert_eq!(c, c);

        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn test_serialize_and_deserialize_m128() {
        let mut rng = StdRng::from_seed([0; 32]);

        let original_value = M128::from(rng.random::<u128>());

        let mut buf = BytesMut::new();
        original_value.serialize(&mut buf).unwrap();

        let deserialized_value = M128::deserialize(buf.freeze()).unwrap();

        assert_eq!(original_value, deserialized_value);
    }
}
