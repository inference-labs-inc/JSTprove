// Copyright 2024-2025 Irreducible Inc.

use std::{
    arch::x86_64::*,
    mem::transmute_copy,
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

use crate::{
    BinaryField,
    arch::{
        portable::packed::PackedPrimitiveType,
        x86_64::{m128::M128, m256::M256},
    },
    underlier::{
        Divisible, NumCast, SmallU, U1, U2, U4, UnderlierType, UnderlierWithBitOps,
        get_block_values, get_spread_bytes, impl_divisible_bitmask, mapget, spread_fallback,
    },
};

/// 512-bit value that is used for 512-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M512(pub(super) __m512i);

impl M512 {
    pub const fn from_equal_u128s(val: u128) -> Self {
        unsafe { transmute_copy(&AlignedData([val, val, val, val])) }
    }
}

impl From<__m512i> for M512 {
    #[inline(always)]
    fn from(value: __m512i) -> Self {
        Self(value)
    }
}

impl From<[u128; 4]> for M512 {
    fn from(value: [u128; 4]) -> Self {
        Self(unsafe {
            _mm512_set_epi64(
                (value[3] >> 64) as i64,
                value[3] as i64,
                (value[2] >> 64) as i64,
                value[2] as i64,
                (value[1] >> 64) as i64,
                value[1] as i64,
                (value[0] >> 64) as i64,
                value[0] as i64,
            )
        })
    }
}

impl From<u128> for M512 {
    fn from(value: u128) -> Self {
        Self::from([value, 0, 0, 0])
    }
}

impl From<u64> for M512 {
    fn from(value: u64) -> Self {
        Self::from(value as u128)
    }
}

impl From<u32> for M512 {
    fn from(value: u32) -> Self {
        Self::from(value as u128)
    }
}

impl From<u16> for M512 {
    fn from(value: u16) -> Self {
        Self::from(value as u128)
    }
}

impl From<u8> for M512 {
    fn from(value: u8) -> Self {
        Self::from(value as u128)
    }
}

impl<const N: usize> From<SmallU<N>> for M512 {
    fn from(value: SmallU<N>) -> Self {
        Self::from(value.val() as u128)
    }
}

impl From<M512> for [u128; 4] {
    fn from(value: M512) -> Self {
        let result: [u128; 4] = unsafe { transmute_copy(&value.0) };

        result
    }
}

impl From<M512> for __m512i {
    #[inline(always)]
    fn from(value: M512) -> Self {
        value.0
    }
}
impl<U: NumCast<u128>> NumCast<M512> for U {
    fn num_cast_from(val: M512) -> Self {
        let [low, _, _, _] = val.into();
        Self::num_cast_from(low)
    }
}

impl SerializeBytes for M512 {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

        let raw_values: [u128; 4] = (*self).into();

        for &val in &raw_values {
            write_buf.put_u128_le(val);
        }

        Ok(())
    }
}

impl DeserializeBytes for M512 {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        assert_enough_data_for(&read_buf, size_of::<Self>())?;

        let raw_values = core::array::from_fn(|_| read_buf.get_u128_le());

        Ok(Self::from(raw_values))
    }
}

impl_divisible_bitmask!(M512, 1, 2, 4);

impl Default for M512 {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { _mm512_setzero_si512() })
    }
}

impl BitAnd for M512 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm512_and_si512(self.0, rhs.0) })
    }
}

impl BitAndAssign for M512 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs
    }
}

impl BitOr for M512 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm512_or_si512(self.0, rhs.0) })
    }
}

impl BitOrAssign for M512 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs
    }
}

impl BitXor for M512 {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm512_xor_si512(self.0, rhs.0) })
    }
}

impl BitXorAssign for M512 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl Not for M512 {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        const ONES: __m512i = m512_from_u128s!(u128::MAX, u128::MAX, u128::MAX, u128::MAX,);

        self ^ Self(ONES)
    }
}

impl Shl<usize> for M512 {
    type Output = Self;

    /// TODO: this is not the most efficient implementation
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self::Output {
        match rhs {
            rhs if rhs >= 512 => Self::ZERO,
            0 => self,
            rhs => {
                let [mut val_0, mut val_1, mut val_2, mut val_3]: [u128; 4] = self.into();
                if rhs >= 384 {
                    val_3 = val_0 << (rhs - 384);
                    val_2 = 0;
                    val_1 = 0;
                    val_0 = 0;
                } else if rhs > 256 {
                    val_3 = (val_1 << (rhs - 256)) + (val_0 >> (128usize - (rhs - 256)));
                    val_2 = val_0 << (rhs - 256);
                    val_1 = 0;
                    val_0 = 0;
                } else if rhs == 256 {
                    val_3 = val_1;
                    val_2 = val_0;
                    val_1 = 0;
                    val_0 = 0;
                } else if rhs > 128 {
                    val_3 = (val_2 << (rhs - 128)) + (val_1 >> (128usize - (rhs - 128)));
                    val_2 = (val_1 << (rhs - 128)) + (val_0 >> (128usize - (rhs - 128)));
                    val_1 = val_0 << (rhs - 128);
                    val_0 = 0;
                } else if rhs == 128 {
                    val_3 = val_2;
                    val_2 = val_1;
                    val_1 = val_0;
                    val_0 = 0;
                } else {
                    val_3 = (val_3 << rhs) + (val_2 >> (128usize - rhs));
                    val_2 = (val_2 << rhs) + (val_1 >> (128usize - rhs));
                    val_1 = (val_1 << rhs) + (val_0 >> (128usize - rhs));
                    val_0 <<= rhs;
                }
                [val_0, val_1, val_2, val_3].into()
            }
        }
    }
}

impl Shr<usize> for M512 {
    type Output = Self;

    /// TODO: this is not the most efficient implementation
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self::Output {
        match rhs {
            rhs if rhs >= 512 => Self::ZERO,
            0 => self,
            rhs => {
                let [mut val_0, mut val_1, mut val_2, mut val_3]: [u128; 4] = self.into();
                if rhs >= 384 {
                    val_0 = val_3 >> (rhs - 384);
                    val_1 = 0;
                    val_2 = 0;
                    val_3 = 0;
                } else if rhs > 256 {
                    val_0 = (val_2 >> (rhs - 256)) + (val_3 << (128usize - (rhs - 256)));
                    val_1 = val_3 >> (rhs - 256);
                    val_2 = 0;
                    val_3 = 0;
                } else if rhs == 256 {
                    val_0 = val_2;
                    val_1 = val_3;
                    val_2 = 0;
                    val_3 = 0;
                } else if rhs > 128 {
                    val_0 = (val_1 >> (rhs - 128)) + (val_2 << (128usize - (rhs - 128)));
                    val_1 = (val_2 >> (rhs - 128)) + (val_3 << (128usize - (rhs - 128)));
                    val_2 = val_3 >> (rhs - 128);
                    val_3 = 0;
                } else if rhs == 128 {
                    val_0 = val_1;
                    val_1 = val_2;
                    val_2 = val_3;
                    val_3 = 0;
                } else {
                    val_0 = (val_0 >> rhs) + (val_1 << (128usize - rhs));
                    val_1 = (val_1 >> rhs) + (val_2 << (128usize - rhs));
                    val_2 = (val_2 >> rhs) + (val_3 << (128usize - rhs));
                    val_3 >>= rhs;
                }
                [val_0, val_1, val_2, val_3].into()
            }
        }
    }
}

impl PartialEq for M512 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let pcmp = _mm512_cmpeq_epi32_mask(self.0, other.0);
            pcmp == 0xFFFF
        }
    }
}

impl Eq for M512 {}

impl PartialOrd for M512 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for M512 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        <[u128; 4]>::from(*self).cmp(&<[u128; 4]>::from(*other))
    }
}

impl Distribution<M512> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> M512 {
        let val: [u128; 4] = rng.random();
        val.into()
    }
}

impl std::fmt::Display for M512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data: [u128; 4] = (*self).into();
        write!(f, "{data:02X?}")
    }
}

impl std::fmt::Debug for M512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "M512({self})")
    }
}

#[repr(align(64))]
pub struct AlignedData(pub [u128; 4]);

macro_rules! m512_from_u128s {
    ($($values:expr,)+) => {{
        let aligned_data = $crate::arch::x86_64::m512::AlignedData([$($values,)*]);
        unsafe {* (aligned_data.0.as_ptr() as *const __m512i)}
    }};
}

pub(super) use m512_from_u128s;

impl UnderlierType for M512 {
    const LOG_BITS: usize = 9;
}

impl UnderlierWithBitOps for M512 {
    const ZERO: Self = { Self(m512_from_u128s!(0, 0, 0, 0,)) };
    const ONE: Self = { Self(m512_from_u128s!(1, 0, 0, 0,)) };
    const ONES: Self = { Self(m512_from_u128s!(u128::MAX, u128::MAX, u128::MAX, u128::MAX,)) };

    #[inline(always)]
    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        let (a, b) = unsafe { interleave_bits(self.0, other.0, log_block_len) };
        (Self(a), Self(b))
    }

    #[inline(always)]
    fn transpose(self, other: Self, log_bit_len: usize) -> (Self, Self) {
        let (a, b) = unsafe { transpose_bits(self.0, other.0, log_bit_len) };
        (Self(a), Self(b))
    }

    #[inline(always)]
    unsafe fn spread<T>(self, log_block_len: usize, block_idx: usize) -> Self
    where
        T: UnderlierWithBitOps,
        Self: Divisible<T>,
    {
        match T::LOG_BITS {
            0 => match log_block_len {
                0 => unsafe {
                    let bit = get_block_values::<_, U1, 1>(self, block_idx)[0];
                    Self::fill_with_bit(bit.val())
                },
                1 => unsafe {
                    let bits = get_block_values::<_, U1, 2>(self, block_idx);
                    let values = bits.map(|b| u128::fill_with_bit(b.val()));

                    Self::from_fn::<u128>(|i| values[i / 2])
                },
                2 => unsafe {
                    let bits = get_block_values::<_, U1, 4>(self, block_idx);
                    let values = bits.map(|b| u128::fill_with_bit(b.val()));

                    Self::from_fn::<u128>(|i| values[i])
                },
                3 => unsafe {
                    let bits = get_block_values::<_, U1, 8>(self, block_idx);
                    let values = bits.map(|b| u64::fill_with_bit(b.val()));

                    Self::from_fn::<u64>(|i| values[i])
                },
                4 => unsafe {
                    let bits = get_block_values::<_, U1, 16>(self, block_idx);
                    let values = bits.map(|b| u32::fill_with_bit(b.val()));

                    Self::from_fn::<u32>(|i| values[i])
                },
                5 => unsafe {
                    let bits = get_block_values::<_, U1, 32>(self, block_idx);
                    let values = bits.map(|b| u16::fill_with_bit(b.val()));

                    Self::from_fn::<u16>(|i| values[i])
                },
                6 => unsafe {
                    let bits = get_block_values::<_, U1, 64>(self, block_idx);
                    let values = bits.map(|b| u8::fill_with_bit(b.val()));

                    Self::from_fn::<u8>(|i| values[i])
                },
                _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
            },
            1 => match log_block_len {
                0 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 1>(self, block_idx)[0];

                    _mm512_set1_epi8(bytes as _).into()
                },
                1 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 2>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 32])
                },
                2 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 4>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 16])
                },
                3 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 8>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 8])
                },
                4 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 16>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 4])
                },
                5 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 32>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 2])
                },
                6 => unsafe {
                    let bytes = get_spread_bytes::<_, U2, 64>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i])
                },
                _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
            },
            2 => match log_block_len {
                0 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 1>(self, block_idx)[0];

                    _mm512_set1_epi8(bytes as _).into()
                },
                1 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 2>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 32])
                },
                2 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 4>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 16])
                },
                3 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 8>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 8])
                },
                4 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 16>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 4])
                },
                5 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 32>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i / 2])
                },
                6 => unsafe {
                    let bytes = get_spread_bytes::<_, U4, 64>(self, block_idx);

                    Self::from_fn::<u8>(|i| bytes[i])
                },
                _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
            },
            3 => match log_block_len {
                0 => unsafe { _mm512_permutexvar_epi8(LOG_B8_0[block_idx], self.0).into() },
                1 => unsafe { _mm512_permutexvar_epi8(LOG_B8_1[block_idx], self.0).into() },
                2 => unsafe { _mm512_permutexvar_epi8(LOG_B8_2[block_idx], self.0).into() },
                3 => unsafe { _mm512_permutexvar_epi8(LOG_B8_3[block_idx], self.0).into() },
                4 => unsafe { _mm512_permutexvar_epi8(LOG_B8_4[block_idx], self.0).into() },
                5 => unsafe { _mm512_permutexvar_epi8(LOG_B8_5[block_idx], self.0).into() },
                6 => self,
                _ => panic!("unsupported block length"),
            },
            4 => match log_block_len {
                0 => unsafe { _mm512_permutexvar_epi8(LOG_B16_0[block_idx], self.0).into() },
                1 => unsafe { _mm512_permutexvar_epi8(LOG_B16_1[block_idx], self.0).into() },
                2 => unsafe { _mm512_permutexvar_epi8(LOG_B16_2[block_idx], self.0).into() },
                3 => unsafe { _mm512_permutexvar_epi8(LOG_B16_3[block_idx], self.0).into() },
                4 => unsafe { _mm512_permutexvar_epi8(LOG_B16_4[block_idx], self.0).into() },
                5 => self,
                _ => panic!("unsupported block length"),
            },
            5 => match log_block_len {
                0 => unsafe { _mm512_permutexvar_epi8(LOG_B32_0[block_idx], self.0).into() },
                1 => unsafe { _mm512_permutexvar_epi8(LOG_B32_1[block_idx], self.0).into() },
                2 => unsafe { _mm512_permutexvar_epi8(LOG_B32_2[block_idx], self.0).into() },
                3 => unsafe { _mm512_permutexvar_epi8(LOG_B32_3[block_idx], self.0).into() },
                4 => self,
                _ => panic!("unsupported block length"),
            },
            6 => match log_block_len {
                0 => unsafe { _mm512_permutexvar_epi8(LOG_B64_0[block_idx], self.0).into() },
                1 => unsafe { _mm512_permutexvar_epi8(LOG_B64_1[block_idx], self.0).into() },
                2 => unsafe { _mm512_permutexvar_epi8(LOG_B64_2[block_idx], self.0).into() },
                3 => self,
                _ => panic!("unsupported block length"),
            },
            7 => match log_block_len {
                0 => unsafe { _mm512_permutexvar_epi8(LOG_B128_0[block_idx], self.0).into() },
                1 => unsafe { _mm512_permutexvar_epi8(LOG_B128_1[block_idx], self.0).into() },
                2 => self,
                _ => panic!("unsupported block length"),
            },
            _ => unsafe { spread_fallback(self, log_block_len, block_idx) },
        }
    }
}

unsafe impl Zeroable for M512 {}

unsafe impl Pod for M512 {}

unsafe impl Send for M512 {}

unsafe impl Sync for M512 {}

impl<Scalar: BinaryField> From<__m512i> for PackedPrimitiveType<M512, Scalar> {
    fn from(value: __m512i) -> Self {
        Self::from(M512::from(value))
    }
}

impl<Scalar: BinaryField> From<[u128; 4]> for PackedPrimitiveType<M512, Scalar> {
    fn from(value: [u128; 4]) -> Self {
        Self::from(M512::from(value))
    }
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M512, Scalar>> for __m512i {
    fn from(value: PackedPrimitiveType<M512, Scalar>) -> Self {
        value.to_underlier().into()
    }
}

#[inline]
unsafe fn interleave_bits(a: __m512i, b: __m512i, log_block_len: usize) -> (__m512i, __m512i) {
    match log_block_len {
        0 => unsafe {
            let mask = _mm512_set1_epi8(0x55i8);
            interleave_bits_imm::<1>(a, b, mask)
        },
        1 => unsafe {
            let mask = _mm512_set1_epi8(0x33i8);
            interleave_bits_imm::<2>(a, b, mask)
        },
        2 => unsafe {
            let mask = _mm512_set1_epi8(0x0fi8);
            interleave_bits_imm::<4>(a, b, mask)
        },
        3 => unsafe {
            let shuffle = _mm512_set_epi8(
                15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1,
                14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
                15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
            );
            let a = _mm512_shuffle_epi8(a, shuffle);
            let b = _mm512_shuffle_epi8(b, shuffle);
            let a_prime = _mm512_unpacklo_epi8(a, b);
            let b_prime = _mm512_unpackhi_epi8(a, b);
            (a_prime, b_prime)
        },
        4 => unsafe {
            let shuffle = _mm512_set_epi8(
                15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2,
                13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15,
                14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
            );
            let a = _mm512_shuffle_epi8(a, shuffle);
            let b = _mm512_shuffle_epi8(b, shuffle);
            let a_prime = _mm512_unpacklo_epi16(a, b);
            let b_prime = _mm512_unpackhi_epi16(a, b);
            (a_prime, b_prime)
        },
        5 => unsafe {
            let shuffle = _mm512_set_epi8(
                15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4,
                11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15,
                14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0,
            );
            let a = _mm512_shuffle_epi8(a, shuffle);
            let b = _mm512_shuffle_epi8(b, shuffle);
            let a_prime = _mm512_unpacklo_epi32(a, b);
            let b_prime = _mm512_unpackhi_epi32(a, b);
            (a_prime, b_prime)
        },
        6 => unsafe {
            let a_prime = _mm512_unpacklo_epi64(a, b);
            let b_prime = _mm512_unpackhi_epi64(a, b);
            (a_prime, b_prime)
        },
        7 => unsafe {
            let a_prime = _mm512_permutex2var_epi64(
                a,
                _mm512_set_epi64(
                    0b1101, 0b1100, 0b0101, 0b0100, 0b1001, 0b1000, 0b0001, 0b0000,
                ),
                b,
            );
            let b_prime = _mm512_permutex2var_epi64(
                a,
                _mm512_set_epi64(
                    0b1111, 0b1110, 0b0111, 0b0110, 0b1011, 0b1010, 0b0011, 0b0010,
                ),
                b,
            );
            (a_prime, b_prime)
        },
        8 => unsafe {
            let a_prime = _mm512_permutex2var_epi64(
                a,
                _mm512_set_epi64(
                    0b1011, 0b1010, 0b1001, 0b1000, 0b0011, 0b0010, 0b0001, 0b0000,
                ),
                b,
            );
            let b_prime = _mm512_permutex2var_epi64(
                a,
                _mm512_set_epi64(
                    0b1111, 0b1110, 0b1101, 0b1100, 0b0111, 0b0110, 0b0101, 0b0100,
                ),
                b,
            );
            (a_prime, b_prime)
        },
        _ => panic!("unsupported block length"),
    }
}

#[inline]
unsafe fn interleave_bits_imm<const BLOCK_LEN: u32>(
    a: __m512i,
    b: __m512i,
    mask: __m512i,
) -> (__m512i, __m512i) {
    unsafe {
        let t = _mm512_and_si512(_mm512_xor_si512(_mm512_srli_epi64::<BLOCK_LEN>(a), b), mask);
        let a_prime = _mm512_xor_si512(a, _mm512_slli_epi64::<BLOCK_LEN>(t));
        let b_prime = _mm512_xor_si512(b, t);
        (a_prime, b_prime)
    }
}

static LOG_B8_0: [__m512i; 64] = precompute_spread_mask::<64>(0, 3);
static LOG_B8_1: [__m512i; 32] = precompute_spread_mask::<32>(1, 3);
static LOG_B8_2: [__m512i; 16] = precompute_spread_mask::<16>(2, 3);
static LOG_B8_3: [__m512i; 8] = precompute_spread_mask::<8>(3, 3);
static LOG_B8_4: [__m512i; 4] = precompute_spread_mask::<4>(4, 3);
static LOG_B8_5: [__m512i; 2] = precompute_spread_mask::<2>(5, 3);

static LOG_B16_0: [__m512i; 32] = precompute_spread_mask::<32>(0, 4);
static LOG_B16_1: [__m512i; 16] = precompute_spread_mask::<16>(1, 4);
static LOG_B16_2: [__m512i; 8] = precompute_spread_mask::<8>(2, 4);
static LOG_B16_3: [__m512i; 4] = precompute_spread_mask::<4>(3, 4);
static LOG_B16_4: [__m512i; 2] = precompute_spread_mask::<2>(4, 4);

static LOG_B32_0: [__m512i; 16] = precompute_spread_mask::<16>(0, 5);
static LOG_B32_1: [__m512i; 8] = precompute_spread_mask::<8>(1, 5);
static LOG_B32_2: [__m512i; 4] = precompute_spread_mask::<4>(2, 5);
static LOG_B32_3: [__m512i; 2] = precompute_spread_mask::<2>(3, 5);

static LOG_B64_0: [__m512i; 8] = precompute_spread_mask::<8>(0, 6);
static LOG_B64_1: [__m512i; 4] = precompute_spread_mask::<4>(1, 6);
static LOG_B64_2: [__m512i; 2] = precompute_spread_mask::<2>(2, 6);

static LOG_B128_0: [__m512i; 4] = precompute_spread_mask::<4>(0, 7);
static LOG_B128_1: [__m512i; 2] = precompute_spread_mask::<2>(1, 7);

const fn precompute_spread_mask<const BLOCK_IDX_AMOUNT: usize>(
    log_block_len: usize,
    t_log_bits: usize,
) -> [__m512i; BLOCK_IDX_AMOUNT] {
    let element_log_width = t_log_bits - 3;

    let element_width = 1 << element_log_width;

    let block_size = 1 << (log_block_len + element_log_width);
    let repeat = 1 << (6 - element_log_width - log_block_len);
    let mut masks = [[0u8; 64]; BLOCK_IDX_AMOUNT];

    let mut block_idx = 0;

    while block_idx < BLOCK_IDX_AMOUNT {
        let base = block_idx * block_size;
        let mut j = 0;
        while j < 64 {
            masks[block_idx][j] =
                (base + ((j / element_width) / repeat) * element_width + j % element_width) as u8;
            j += 1;
        }
        block_idx += 1;
    }
    let mut m512_masks = [m512_from_u128s!(0, 0, 0, 0,); BLOCK_IDX_AMOUNT];

    let mut block_idx = 0;

    while block_idx < BLOCK_IDX_AMOUNT {
        let mut u128s = [0; 4];
        let mut i = 0;
        while i < 4 {
            unsafe {
                u128s[i] = u128::from_le_bytes(
                    *(masks[block_idx].as_ptr().add(16 * i) as *const [u8; 16]),
                );
            }
            i += 1;
        }
        m512_masks[block_idx] = m512_from_u128s!(u128s[0], u128s[1], u128s[2], u128s[3],);
        block_idx += 1;
    }

    m512_masks
}

#[inline(always)]
unsafe fn transpose_bits(a: __m512i, b: __m512i, log_block_len: usize) -> (__m512i, __m512i) {
    match log_block_len {
        0..=3 => unsafe {
            let shuffle = _mm512_set_epi8(
                15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1,
                14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
                15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
            );
            let (mut a, mut b) = transpose_with_shuffle(a, b, shuffle);
            for log_block_len in (log_block_len..3).rev() {
                (a, b) = interleave_bits(a, b, log_block_len);
            }

            (a, b)
        },
        4 => unsafe {
            let shuffle = _mm512_set_epi8(
                15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2,
                13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15,
                14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
            );
            transpose_with_shuffle(a, b, shuffle)
        },
        5 => unsafe {
            let shuffle = _mm512_set_epi8(
                15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4,
                11, 10, 9, 8, 3, 2, 1, 0, 15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0, 15,
                14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0,
            );
            transpose_with_shuffle(a, b, shuffle)
        },
        6 => unsafe {
            (
                _mm512_permutex2var_epi64(
                    a,
                    _mm512_set_epi64(
                        0b1110, 0b1100, 0b1010, 0b1000, 0b0110, 0b0100, 0b0010, 0b0000,
                    ),
                    b,
                ),
                _mm512_permutex2var_epi64(
                    a,
                    _mm512_set_epi64(
                        0b1111, 0b1101, 0b1011, 0b1001, 0b0111, 0b0101, 0b0011, 0b0001,
                    ),
                    b,
                ),
            )
        },
        7 => unsafe {
            (
                _mm512_permutex2var_epi64(
                    a,
                    _mm512_set_epi64(
                        0b1101, 0b1100, 0b1001, 0b1000, 0b0101, 0b0100, 0b0001, 0b0000,
                    ),
                    b,
                ),
                _mm512_permutex2var_epi64(
                    a,
                    _mm512_set_epi64(
                        0b1111, 0b1110, 0b1011, 0b1010, 0b0111, 0b0110, 0b0011, 0b0010,
                    ),
                    b,
                ),
            )
        },
        8 => unsafe {
            (
                _mm512_permutex2var_epi64(
                    a,
                    _mm512_set_epi64(
                        0b1011, 0b1010, 0b1001, 0b1000, 0b0011, 0b0010, 0b0001, 0b0000,
                    ),
                    b,
                ),
                _mm512_permutex2var_epi64(
                    a,
                    _mm512_set_epi64(
                        0b1111, 0b1110, 0b1101, 0b1100, 0b0111, 0b0110, 0b0101, 0b0100,
                    ),
                    b,
                ),
            )
        },
        _ => panic!("unsupported block length"),
    }
}

unsafe fn transpose_with_shuffle(a: __m512i, b: __m512i, shuffle: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let (a, b) = (
            _mm512_shuffle_epi8(a, shuffle),
            _mm512_shuffle_epi8(b, shuffle),
        );

        (
            _mm512_permutex2var_epi64(
                a,
                _mm512_set_epi64(
                    0b1110, 0b1100, 0b1010, 0b1000, 0b0110, 0b0100, 0b0010, 0b0000,
                ),
                b,
            ),
            _mm512_permutex2var_epi64(
                a,
                _mm512_set_epi64(
                    0b1111, 0b1101, 0b1011, 0b1001, 0b0111, 0b0101, 0b0011, 0b0001,
                ),
                b,
            ),
        )
    }
}

// Divisible implementations using SIMD extract/insert intrinsics

impl Divisible<M256> for M512 {
    const LOG_N: usize = 1;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = M256> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = M256> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = M256> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> M256 {
        unsafe {
            match index {
                0 => M256(_mm512_extracti64x4_epi64(self.0, 0)),
                1 => M256(_mm512_extracti64x4_epi64(self.0, 1)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: M256) -> Self {
        unsafe {
            match index {
                0 => Self(_mm512_inserti64x4(self.0, val.0, 0)),
                1 => Self(_mm512_inserti64x4(self.0, val.0, 1)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: M256) -> Self {
        unsafe { Self(_mm512_broadcast_i64x4(val.0)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = M256>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [M256; 2] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(2).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<M128> for M512 {
    const LOG_N: usize = 2;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = M128> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = M128> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = M128> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> M128 {
        unsafe {
            match index {
                0 => M128(_mm512_extracti32x4_epi32(self.0, 0)),
                1 => M128(_mm512_extracti32x4_epi32(self.0, 1)),
                2 => M128(_mm512_extracti32x4_epi32(self.0, 2)),
                3 => M128(_mm512_extracti32x4_epi32(self.0, 3)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: M128) -> Self {
        unsafe {
            match index {
                0 => Self(_mm512_inserti32x4(self.0, val.0, 0)),
                1 => Self(_mm512_inserti32x4(self.0, val.0, 1)),
                2 => Self(_mm512_inserti32x4(self.0, val.0, 2)),
                3 => Self(_mm512_inserti32x4(self.0, val.0, 3)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: M128) -> Self {
        unsafe { Self(_mm512_broadcast_i32x4(val.0)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = M128>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [M128; 4] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(4).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u128> for M512 {
    const LOG_N: usize = 2;

    #[inline]
    fn value_iter(value: Self) -> impl ExactSizeIterator<Item = u128> + Send + Clone {
        mapget::value_iter(value)
    }

    #[inline]
    fn ref_iter(value: &Self) -> impl ExactSizeIterator<Item = u128> + Send + Clone + '_ {
        mapget::value_iter(*value)
    }

    #[inline]
    fn slice_iter(slice: &[Self]) -> impl ExactSizeIterator<Item = u128> + Send + Clone + '_ {
        mapget::slice_iter(slice)
    }

    #[inline]
    fn get(self, index: usize) -> u128 {
        u128::from(Divisible::<M128>::get(self, index))
    }

    #[inline]
    fn set(self, index: usize, val: u128) -> Self {
        Divisible::<M128>::set(self, index, M128::from(val))
    }

    #[inline]
    fn broadcast(val: u128) -> Self {
        Divisible::<M128>::broadcast(M128::from(val))
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u128>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u128; 4] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(4).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u64> for M512 {
    const LOG_N: usize = 3;

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
        // Extract M128 lane, then use M128's get
        let lane_idx = index / 2;
        let sub_idx = index % 2;
        let lane = Divisible::<M128>::get(self, lane_idx);
        Divisible::<u64>::get(lane, sub_idx)
    }

    #[inline]
    fn set(self, index: usize, val: u64) -> Self {
        let lane_idx = index / 2;
        let sub_idx = index % 2;
        let lane = Divisible::<M128>::get(self, lane_idx);
        let new_lane = Divisible::<u64>::set(lane, sub_idx, val);
        Divisible::<M128>::set(self, lane_idx, new_lane)
    }

    #[inline]
    fn broadcast(val: u64) -> Self {
        unsafe { Self(_mm512_set1_epi64(val as i64)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u64>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u64; 8] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(8).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u32> for M512 {
    const LOG_N: usize = 4;

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
        // Extract M128 lane, then use M128's get
        let lane_idx = index / 4;
        let sub_idx = index % 4;
        let lane = Divisible::<M128>::get(self, lane_idx);
        Divisible::<u32>::get(lane, sub_idx)
    }

    #[inline]
    fn set(self, index: usize, val: u32) -> Self {
        let lane_idx = index / 4;
        let sub_idx = index % 4;
        let lane = Divisible::<M128>::get(self, lane_idx);
        let new_lane = Divisible::<u32>::set(lane, sub_idx, val);
        Divisible::<M128>::set(self, lane_idx, new_lane)
    }

    #[inline]
    fn broadcast(val: u32) -> Self {
        unsafe { Self(_mm512_set1_epi32(val as i32)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u32>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u32; 16] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(16).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u16> for M512 {
    const LOG_N: usize = 5;

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
        // Extract M128 lane, then use M128's get
        let lane_idx = index / 8;
        let sub_idx = index % 8;
        let lane = Divisible::<M128>::get(self, lane_idx);
        Divisible::<u16>::get(lane, sub_idx)
    }

    #[inline]
    fn set(self, index: usize, val: u16) -> Self {
        let lane_idx = index / 8;
        let sub_idx = index % 8;
        let lane = Divisible::<M128>::get(self, lane_idx);
        let new_lane = Divisible::<u16>::set(lane, sub_idx, val);
        Divisible::<M128>::set(self, lane_idx, new_lane)
    }

    #[inline]
    fn broadcast(val: u16) -> Self {
        unsafe { Self(_mm512_set1_epi16(val as i16)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u16>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u16; 32] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(32).enumerate() {
            arr[i] = val;
        }
        result
    }
}

impl Divisible<u8> for M512 {
    const LOG_N: usize = 6;

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
        // Extract M128 lane, then use M128's get
        let lane_idx = index / 16;
        let sub_idx = index % 16;
        let lane = Divisible::<M128>::get(self, lane_idx);
        Divisible::<u8>::get(lane, sub_idx)
    }

    #[inline]
    fn set(self, index: usize, val: u8) -> Self {
        let lane_idx = index / 16;
        let sub_idx = index % 16;
        let lane = Divisible::<M128>::get(self, lane_idx);
        let new_lane = Divisible::<u8>::set(lane, sub_idx, val);
        Divisible::<M128>::set(self, lane_idx, new_lane)
    }

    #[inline]
    fn broadcast(val: u8) -> Self {
        unsafe { Self(_mm512_set1_epi8(val as i8)) }
    }

    #[inline]
    fn from_iter(iter: impl Iterator<Item = u8>) -> Self {
        let mut result = Self::ZERO;
        let arr: &mut [u8; 64] = bytemuck::cast_mut(&mut result);
        for (i, val) in iter.take(64).enumerate() {
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
    use crate::underlier::single_element_mask_bits;

    fn check_roundtrip<T>(val: M512)
    where
        T: From<M512>,
        M512: From<T>,
    {
        assert_eq!(M512::from(T::from(val)), val);
    }

    #[test]
    fn test_constants() {
        assert_eq!(M512::default(), M512::ZERO);
        assert_eq!(M512::from(0u128), M512::ZERO);
        assert_eq!(M512::from([1u128, 0u128, 0u128, 0u128]), M512::ONE);
    }

    #[derive(Default)]
    struct ByteData([u128; 4]);

    impl ByteData {
        const fn get_bit(&self, i: usize) -> u8 {
            if self.0[i / 128] & (1u128 << (i % 128)) == 0 {
                0
            } else {
                1
            }
        }

        fn set_bit(&mut self, i: usize, val: u8) {
            self.0[i / 128] &= !(1 << (i % 128));
            self.0[i / 128] |= (val as u128) << (i % 128);
        }
    }

    impl From<ByteData> for M512 {
        fn from(value: ByteData) -> Self {
            let vals: [u128; 4] = unsafe { std::mem::transmute(value) };
            vals.into()
        }
    }

    impl From<[u128; 4]> for ByteData {
        fn from(value: [u128; 4]) -> Self {
            unsafe { std::mem::transmute(value) }
        }
    }

    impl Shl<usize> for ByteData {
        type Output = Self;

        fn shl(self, rhs: usize) -> Self::Output {
            let mut result = Self::default();
            for i in 0..512 {
                if i >= rhs {
                    result.set_bit(i, self.get_bit(i - rhs));
                }
            }

            result
        }
    }

    impl Shr<usize> for ByteData {
        type Output = Self;

        fn shr(self, rhs: usize) -> Self::Output {
            let mut result = Self::default();
            for i in 0..512 {
                if i + rhs < 512 {
                    result.set_bit(i, self.get_bit(i + rhs));
                }
            }

            result
        }
    }

    fn get(value: M512, log_block_len: usize, index: usize) -> M512 {
        (value >> (index << log_block_len)) & single_element_mask_bits::<M512>(1 << log_block_len)
    }

    proptest! {
        #[test]
        fn test_conversion(a in any::<[u128; 4]>()) {
            check_roundtrip::<[u128; 4]>(a.into());
            check_roundtrip::<__m512i>(a.into());
        }

        #[test]
        fn test_binary_bit_operations([a, b] in any::<[[u128;4];2]>()) {
            assert_eq!(M512::from([a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]), M512::from(a) & M512::from(b));
            assert_eq!(M512::from([a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]), M512::from(a) | M512::from(b));
            assert_eq!(M512::from([a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]), M512::from(a) ^ M512::from(b));
        }

        #[test]
        fn test_negate(a in any::<[u128; 4]>()) {
            assert_eq!(M512::from([!a[0], !a[1], !a[2], !a[3]]), !M512::from(a))
        }

        #[test]
        fn test_shifts(a in any::<[u128; 4]>(), rhs in 0..255usize) {
            assert_eq!(M512::from(a) << rhs, M512::from(ByteData::from(a) << rhs));
            assert_eq!(M512::from(a) >> rhs, M512::from(ByteData::from(a) >> rhs));
        }

        #[test]
        fn test_interleave_bits(a in any::<[u128; 4]>(), b in any::<[u128; 4]>(), height in 0usize..9) {
            let a = M512::from(a);
            let b = M512::from(b);
            let (c, d) = unsafe {interleave_bits(a.0, b.0, height)};
            let (c, d) = (M512::from(c), M512::from(d));

            let block_len = 1usize << height;
            for i in (0..512/block_len).step_by(2) {
                assert_eq!(get(c, height, i), get(a, height, i));
                assert_eq!(get(c, height, i+1), get(b, height, i));
                assert_eq!(get(d, height, i), get(a, height, i+1));
                assert_eq!(get(d, height, i+1), get(b, height, i+1));
            }
        }
    }

    #[test]
    fn test_fill_with_bit() {
        assert_eq!(
            M512::fill_with_bit(1),
            M512::from([u128::MAX, u128::MAX, u128::MAX, u128::MAX])
        );
        assert_eq!(M512::fill_with_bit(0), M512::from(0u128));
    }

    #[test]
    fn test_eq() {
        let a = M512::from(0u128);
        let b = M512::from(42u128);
        let c = M512::from(u128::MAX);
        let d = M512::from([u128::MAX, u128::MAX, u128::MAX, u128::MAX]);

        assert_eq!(a, a);
        assert_eq!(b, b);
        assert_eq!(c, c);
        assert_eq!(d, d);

        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(b, c);
        assert_ne!(b, d);
        assert_ne!(c, d);
    }

    #[test]
    fn test_serialize_and_deserialize_m512() {
        let mut rng = StdRng::from_seed([0; 32]);

        let original_value = M512::from(core::array::from_fn(|_| rng.random::<u128>()));

        let mut buf = BytesMut::new();
        original_value.serialize(&mut buf).unwrap();

        let deserialized_value = M512::deserialize(buf.freeze()).unwrap();

        assert_eq!(original_value, deserialized_value);
    }
}
