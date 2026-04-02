// Copyright 2024-2025 Irreducible Inc.

use std::{
    arch::aarch64::*,
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

use super::super::portable::{
    packed::PackedPrimitiveType, packed_arithmetic::interleave_mask_even,
};
use crate::{
    BinaryField,
    underlier::{
        NumCast, SmallU, UnderlierType, UnderlierWithBitOps,
        divisible::{Divisible, mapget},
        impl_divisible_bitmask,
    },
};

/// 128-bit value that is used for 128-bit SIMD operations
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct M128(uint64x2_t);

impl M128 {
    pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Self(unsafe { std::mem::transmute::<u128, uint64x2_t>(u128::from_le_bytes(bytes)) })
    }

    pub const fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Self(unsafe { std::mem::transmute::<u128, uint64x2_t>(u128::from_be_bytes(bytes)) })
    }

    pub const fn from_u128(value: u128) -> Self {
        Self(unsafe { std::mem::transmute::<u128, uint64x2_t>(value) })
    }

    #[inline]
    pub fn shuffle_u8(self, src: [u8; 16]) -> Self {
        unsafe { vqtbl1q_u8(self.into(), Self::from_le_bytes(src).into()).into() }
    }
}

impl Default for M128 {
    fn default() -> Self {
        Self(unsafe { vdupq_n_u64(0) })
    }
}

impl PartialEq for M128 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let cmp = vceqq_u64(self.0, other.0);
            vminvq_u32(vreinterpretq_u32_u64(cmp)) == u32::MAX
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

unsafe impl Zeroable for M128 {}
unsafe impl Pod for M128 {}

impl From<M128> for u128 {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_p128_u64(value.0) }
    }
}
impl From<M128> for uint8x16_t {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_u8_u64(value.0) }
    }
}
impl From<M128> for uint16x8_t {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_u16_u64(value.0) }
    }
}
impl From<M128> for uint32x4_t {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_u32_u64(value.0) }
    }
}
impl From<M128> for uint64x2_t {
    fn from(value: M128) -> Self {
        value.0
    }
}
impl From<M128> for poly8x16_t {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_p8_u64(value.0) }
    }
}
impl From<M128> for poly16x8_t {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_p16_u64(value.0) }
    }
}
impl From<M128> for poly64x2_t {
    fn from(value: M128) -> Self {
        unsafe { vreinterpretq_p64_u64(value.0) }
    }
}

impl From<u128> for M128 {
    fn from(value: u128) -> Self {
        Self(unsafe { vreinterpretq_u64_p128(value) })
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

impl From<uint8x16_t> for M128 {
    fn from(value: uint8x16_t) -> Self {
        Self(unsafe { vreinterpretq_u64_u8(value) })
    }
}
impl From<uint16x8_t> for M128 {
    fn from(value: uint16x8_t) -> Self {
        Self(unsafe { vreinterpretq_u64_u16(value) })
    }
}
impl From<uint32x4_t> for M128 {
    fn from(value: uint32x4_t) -> Self {
        Self(unsafe { vreinterpretq_u64_u32(value) })
    }
}
impl From<uint64x2_t> for M128 {
    fn from(value: uint64x2_t) -> Self {
        Self(value)
    }
}
impl From<poly8x16_t> for M128 {
    fn from(value: poly8x16_t) -> Self {
        Self(unsafe { vreinterpretq_u64_p8(value) })
    }
}
impl From<poly16x8_t> for M128 {
    fn from(value: poly16x8_t) -> Self {
        Self(unsafe { vreinterpretq_u64_p16(value) })
    }
}
impl From<poly64x2_t> for M128 {
    fn from(value: poly64x2_t) -> Self {
        Self(unsafe { vreinterpretq_u64_p64(value) })
    }
}

impl SerializeBytes for M128 {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        assert_enough_space_for(&write_buf, std::mem::size_of::<Self>())?;

        write_buf.put_u128_le(u128::from(*self));

        Ok(())
    }
}

impl DeserializeBytes for M128 {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        assert_enough_data_for(&read_buf, std::mem::size_of::<Self>())?;

        Ok(Self::from(read_buf.get_u128_le()))
    }
}

impl_divisible_bitmask!(M128, 1, 2, 4);

// Manual Divisible implementations using NEON intrinsics

impl Divisible<u128> for M128 {
    const LOG_N: usize = 0;

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
        match index {
            0 => self.into(),
            _ => panic!("index out of bounds"),
        }
    }

    #[inline]
    fn set(self, index: usize, val: u128) -> Self {
        match index {
            0 => Self::from(val),
            _ => panic!("index out of bounds"),
        }
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
                0 => vgetq_lane_u64(self.0, 0),
                1 => vgetq_lane_u64(self.0, 1),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u64) -> Self {
        unsafe {
            match index {
                0 => Self(vsetq_lane_u64(val, self.0, 0)),
                1 => Self(vsetq_lane_u64(val, self.0, 1)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u64) -> Self {
        unsafe { Self(vdupq_n_u64(val)) }
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
            let v: uint32x4_t = self.into();
            match index {
                0 => vgetq_lane_u32(v, 0),
                1 => vgetq_lane_u32(v, 1),
                2 => vgetq_lane_u32(v, 2),
                3 => vgetq_lane_u32(v, 3),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u32) -> Self {
        unsafe {
            let v: uint32x4_t = self.into();
            match index {
                0 => Self::from(vsetq_lane_u32(val, v, 0)),
                1 => Self::from(vsetq_lane_u32(val, v, 1)),
                2 => Self::from(vsetq_lane_u32(val, v, 2)),
                3 => Self::from(vsetq_lane_u32(val, v, 3)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u32) -> Self {
        unsafe { Self::from(vdupq_n_u32(val)) }
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
            let v: uint16x8_t = self.into();
            match index {
                0 => vgetq_lane_u16(v, 0),
                1 => vgetq_lane_u16(v, 1),
                2 => vgetq_lane_u16(v, 2),
                3 => vgetq_lane_u16(v, 3),
                4 => vgetq_lane_u16(v, 4),
                5 => vgetq_lane_u16(v, 5),
                6 => vgetq_lane_u16(v, 6),
                7 => vgetq_lane_u16(v, 7),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u16) -> Self {
        unsafe {
            let v: uint16x8_t = self.into();
            match index {
                0 => Self::from(vsetq_lane_u16(val, v, 0)),
                1 => Self::from(vsetq_lane_u16(val, v, 1)),
                2 => Self::from(vsetq_lane_u16(val, v, 2)),
                3 => Self::from(vsetq_lane_u16(val, v, 3)),
                4 => Self::from(vsetq_lane_u16(val, v, 4)),
                5 => Self::from(vsetq_lane_u16(val, v, 5)),
                6 => Self::from(vsetq_lane_u16(val, v, 6)),
                7 => Self::from(vsetq_lane_u16(val, v, 7)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u16) -> Self {
        unsafe { Self::from(vdupq_n_u16(val)) }
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
            let v: uint8x16_t = self.into();
            match index {
                0 => vgetq_lane_u8(v, 0),
                1 => vgetq_lane_u8(v, 1),
                2 => vgetq_lane_u8(v, 2),
                3 => vgetq_lane_u8(v, 3),
                4 => vgetq_lane_u8(v, 4),
                5 => vgetq_lane_u8(v, 5),
                6 => vgetq_lane_u8(v, 6),
                7 => vgetq_lane_u8(v, 7),
                8 => vgetq_lane_u8(v, 8),
                9 => vgetq_lane_u8(v, 9),
                10 => vgetq_lane_u8(v, 10),
                11 => vgetq_lane_u8(v, 11),
                12 => vgetq_lane_u8(v, 12),
                13 => vgetq_lane_u8(v, 13),
                14 => vgetq_lane_u8(v, 14),
                15 => vgetq_lane_u8(v, 15),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn set(self, index: usize, val: u8) -> Self {
        unsafe {
            let v: uint8x16_t = self.into();
            match index {
                0 => Self::from(vsetq_lane_u8(val, v, 0)),
                1 => Self::from(vsetq_lane_u8(val, v, 1)),
                2 => Self::from(vsetq_lane_u8(val, v, 2)),
                3 => Self::from(vsetq_lane_u8(val, v, 3)),
                4 => Self::from(vsetq_lane_u8(val, v, 4)),
                5 => Self::from(vsetq_lane_u8(val, v, 5)),
                6 => Self::from(vsetq_lane_u8(val, v, 6)),
                7 => Self::from(vsetq_lane_u8(val, v, 7)),
                8 => Self::from(vsetq_lane_u8(val, v, 8)),
                9 => Self::from(vsetq_lane_u8(val, v, 9)),
                10 => Self::from(vsetq_lane_u8(val, v, 10)),
                11 => Self::from(vsetq_lane_u8(val, v, 11)),
                12 => Self::from(vsetq_lane_u8(val, v, 12)),
                13 => Self::from(vsetq_lane_u8(val, v, 13)),
                14 => Self::from(vsetq_lane_u8(val, v, 14)),
                15 => Self::from(vsetq_lane_u8(val, v, 15)),
                _ => panic!("index out of bounds"),
            }
        }
    }

    #[inline]
    fn broadcast(val: u8) -> Self {
        unsafe { Self::from(vdupq_n_u8(val)) }
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

impl Not for M128 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        unsafe { vmvnq_u8(self.into()).into() }
    }
}

impl BitAnd for M128 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe { vandq_u64(self.0, rhs.0).into() }
    }
}

impl BitAndAssign for M128 {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitOr for M128 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe { vorrq_u64(self.0, rhs.0).into() }
    }
}

impl BitOrAssign for M128 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitXor for M128 {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe { veorq_u64(self.0, rhs.0).into() }
    }
}

impl BitXorAssign for M128 {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl Shr<usize> for M128 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        Self::from(u128::from(self) >> rhs)
    }
}

impl Shl<usize> for M128 {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        Self::from(u128::from(self) << rhs)
    }
}

impl Distribution<M128> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> M128 {
        M128::from(rng.random::<u128>())
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

impl UnderlierType for M128 {
    const LOG_BITS: usize = 7;
}

impl UnderlierWithBitOps for M128 {
    const ZERO: Self = Self::from_u128(0);
    const ONE: Self = Self::from_u128(1);
    const ONES: Self = Self::from_u128(u128::MAX);

    #[inline]
    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: [M128; 3] = [
            M128::from_u128(interleave_mask_even!(u128, 0)),
            M128::from_u128(interleave_mask_even!(u128, 1)),
            M128::from_u128(interleave_mask_even!(u128, 2)),
        ];

        unsafe {
            seq!(LOG_BLOCK_LEN in 0..=2 {
                if log_block_len == LOG_BLOCK_LEN {
                    let (a, b) = (self.into(), other.into());
                    let mask = MASKS[LOG_BLOCK_LEN].into();
                    let t = vandq_u64(veorq_u64(vshrq_n_u64(a, 1 << LOG_BLOCK_LEN), b), mask);
                    let c = veorq_u64(a, vshlq_n_u64(t, 1 << LOG_BLOCK_LEN));
                    let d = veorq_u64(b, t);
                    return (c.into(), d.into());
                }
            });
            match log_block_len {
                3 => {
                    let (a, b) = (self.into(), other.into());
                    let c = vtrn1q_u8(a, b);
                    let d = vtrn2q_u8(a, b);
                    (c.into(), d.into())
                }
                4 => {
                    let (a, b) = (self.into(), other.into());
                    let c = vtrn1q_u16(a, b);
                    let d = vtrn2q_u16(a, b);
                    (c.into(), d.into())
                }
                5 => {
                    let (a, b) = (self.into(), other.into());
                    let c = vtrn1q_u32(a, b);
                    let d = vtrn2q_u32(a, b);
                    (c.into(), d.into())
                }
                6 => {
                    let (a, b) = (self.into(), other.into());
                    let c = vtrn1q_u64(a, b);
                    let d = vtrn2q_u64(a, b);
                    (c.into(), d.into())
                }
                _ => panic!("Unsupported block length"),
            }
        }
    }

    #[inline]
    fn transpose(self, other: Self, log_block_len: usize) -> (Self, Self) {
        unsafe {
            match log_block_len {
                0..=3 => {
                    let (a, b) = (self.into(), other.into());
                    let (mut a, mut b) = (Self::from(vuzp1q_u8(a, b)), Self::from(vuzp2q_u8(a, b)));

                    for log_block_len in (log_block_len..3).rev() {
                        (a, b) = a.interleave(b, log_block_len);
                    }

                    (a, b)
                }
                4 => {
                    let (a, b) = (self.into(), other.into());
                    (vuzp1q_u16(a, b).into(), vuzp2q_u16(a, b).into())
                }
                5 => {
                    let (a, b) = (self.into(), other.into());
                    (vuzp1q_u32(a, b).into(), vuzp2q_u32(a, b).into())
                }
                6 => {
                    let (a, b) = (self.into(), other.into());
                    (vuzp1q_u64(a, b).into(), vuzp2q_u64(a, b).into())
                }
                _ => panic!("Unsupported block length"),
            }
        }
    }
}

impl<Scalar: BinaryField> From<u128> for PackedPrimitiveType<M128, Scalar> {
    fn from(value: u128) -> Self {
        Self::from(M128::from(value))
    }
}

impl<Scalar: BinaryField> From<PackedPrimitiveType<M128, Scalar>> for u128 {
    fn from(value: PackedPrimitiveType<M128, Scalar>) -> Self {
        value.to_underlier().into()
    }
}

impl<U: NumCast<u128>> NumCast<M128> for U {
    fn num_cast_from(val: M128) -> Self {
        Self::num_cast_from(val.into())
    }
}

#[cfg(test)]
mod tests {
    use binius_utils::bytes::BytesMut;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

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
