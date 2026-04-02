// Copyright 2023-2025 Irreducible Inc.

//! Binary field implementation of GF(2^128) with a modulus of X^128 + X^7 + X^2 + X + 1.
//! This is the GHASH field used in AES-GCM.

use std::{
    any::TypeId,
    fmt::{self, Debug, Display, Formatter},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{
    DeserializeBytes, SerializationError, SerializeBytes,
    bytes::{Buf, BufMut},
    iter::IterExtensions,
};
use bytemuck::{Pod, Zeroable};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use super::{
    arithmetic_traits::InvertOrZero,
    binary_field::{BinaryField, BinaryField1b, TowerField},
    extension::ExtensionField,
    underlier::WithUnderlier,
};
use crate::{
    AESTowerField8b, Field,
    arch::packed_ghash_128::PackedBinaryGhash1x128b,
    arithmetic_traits::Square,
    binary_field_arithmetic::{
        invert_or_zero_using_packed, multiple_using_packed, square_using_packed,
    },
    transpose::square_transforms_extension_field,
    underlier::{Divisible, NumCast, U1, UnderlierWithBitOps},
};

#[derive(
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Zeroable,
    bytemuck::TransparentWrapper,
)]
#[repr(transparent)]
pub struct BinaryField128bGhash(pub(crate) u128);

impl BinaryField128bGhash {
    #[inline]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn val(self) -> u128 {
        self.0
    }

    #[inline]
    pub fn mul_x(self) -> Self {
        let val = self.to_underlier();
        let shifted = val << 1;

        // GHASH irreducible polynomial: x^128 + x^7 + x^2 + x + 1
        // When the high bit is set, we need to XOR with the reduction polynomial 0x87
        // All 1s if the top bit is set, all 0s otherwise
        let mask = (val >> 127).wrapping_neg();
        let result = shifted ^ (0x87 & mask);

        Self::from_underlier(result)
    }

    #[inline]
    pub fn mul_inv_x(self) -> Self {
        let val = self.to_underlier();
        let shifted = val >> 1;

        // If low bit was set, we need to add compensation for the remainder
        // When dividing by x with remainder 1, we add x^(-1) = x^127 to the result
        // Since x^128 ≡ x^7 + x^2 + x + 1, we have x^127 ≡ x^6 + x + 1
        // So 0x43 = x^6 + x + 1 (bits 6, 1, 0) and we set bit 127 for the x^127 term
        // All 1s if the bottom bit is set, all 0s otherwise
        let mask = (val & 1).wrapping_neg();
        let result = shifted ^ (((1u128 << 127) | 0x43) & mask);

        Self::from_underlier(result)
    }
}

unsafe impl WithUnderlier for BinaryField128bGhash {
    type Underlier = u128;
}

impl Neg for BinaryField128bGhash {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self
    }
}

impl Add<Self> for BinaryField128bGhash {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Add<&Self> for BinaryField128bGhash {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: &Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Sub<Self> for BinaryField128bGhash {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Sub<&Self> for BinaryField128bGhash {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Mul<Self> for BinaryField128bGhash {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        multiple_using_packed::<PackedBinaryGhash1x128b>(self, rhs)
    }
}

impl Mul<&Self> for BinaryField128bGhash {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl AddAssign<Self> for BinaryField128bGhash {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<&Self> for BinaryField128bGhash {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + rhs;
    }
}

impl SubAssign<Self> for BinaryField128bGhash {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<&Self> for BinaryField128bGhash {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - rhs;
    }
}

impl MulAssign<Self> for BinaryField128bGhash {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<&Self> for BinaryField128bGhash {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * rhs;
    }
}

impl Sum<Self> for BinaryField128bGhash {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Self> for BinaryField128bGhash {
    #[inline]
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl Product<Self> for BinaryField128bGhash {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a> Product<&'a Self> for BinaryField128bGhash {
    #[inline]
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl Square for BinaryField128bGhash {
    #[inline]
    fn square(self) -> Self {
        square_using_packed::<PackedBinaryGhash1x128b>(self)
    }
}

impl Field for BinaryField128bGhash {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const CHARACTERISTIC: usize = 2;
    const MULTIPLICATIVE_GENERATOR: Self = Self(0x494ef99794d5244f9152df59d87a9186);

    fn double(&self) -> Self {
        Self::ZERO
    }
}

impl Distribution<BinaryField128bGhash> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BinaryField128bGhash {
        BinaryField128bGhash(rng.random())
    }
}

impl InvertOrZero for BinaryField128bGhash {
    #[inline]
    fn invert_or_zero(self) -> Self {
        invert_or_zero_using_packed::<PackedBinaryGhash1x128b>(self)
    }
}

impl From<u128> for BinaryField128bGhash {
    #[inline]
    fn from(value: u128) -> Self {
        Self(value)
    }
}

impl From<BinaryField128bGhash> for u128 {
    #[inline]
    fn from(value: BinaryField128bGhash) -> Self {
        value.0
    }
}

impl Display for BinaryField128bGhash {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "0x{repr:0>32x}", repr = self.0)
    }
}

impl Debug for BinaryField128bGhash {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryField128bGhash({self})")
    }
}

unsafe impl Pod for BinaryField128bGhash {}

impl TryInto<BinaryField1b> for BinaryField128bGhash {
    type Error = ();

    #[inline]
    fn try_into(self) -> Result<BinaryField1b, Self::Error> {
        if self == Self::ZERO {
            Ok(BinaryField1b::ZERO)
        } else if self == Self::ONE {
            Ok(BinaryField1b::ONE)
        } else {
            Err(())
        }
    }
}

impl From<BinaryField1b> for BinaryField128bGhash {
    #[inline]
    fn from(value: BinaryField1b) -> Self {
        debug_assert_eq!(Self::ZERO, Self(0));

        Self(Self::ONE.0 & u128::fill_with_bit(value.val().val()))
    }
}

impl Add<BinaryField1b> for BinaryField128bGhash {
    type Output = Self;

    #[inline]
    fn add(self, rhs: BinaryField1b) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl Sub<BinaryField1b> for BinaryField128bGhash {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: BinaryField1b) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl Mul<BinaryField1b> for BinaryField128bGhash {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: BinaryField1b) -> Self::Output {
        crate::tracing::trace_multiplication!(BinaryField128bGhash, BinaryField1b);

        Self(self.0 & u128::fill_with_bit(u8::from(rhs.0)))
    }
}

impl AddAssign<BinaryField1b> for BinaryField128bGhash {
    #[inline]
    fn add_assign(&mut self, rhs: BinaryField1b) {
        *self = *self + rhs;
    }
}

impl SubAssign<BinaryField1b> for BinaryField128bGhash {
    #[inline]
    fn sub_assign(&mut self, rhs: BinaryField1b) {
        *self = *self - rhs;
    }
}

impl MulAssign<BinaryField1b> for BinaryField128bGhash {
    #[inline]
    fn mul_assign(&mut self, rhs: BinaryField1b) {
        *self = *self * rhs;
    }
}

impl Add<BinaryField128bGhash> for BinaryField1b {
    type Output = BinaryField128bGhash;

    #[inline]
    fn add(self, rhs: BinaryField128bGhash) -> Self::Output {
        rhs + self
    }
}

impl Sub<BinaryField128bGhash> for BinaryField1b {
    type Output = BinaryField128bGhash;

    #[inline]
    fn sub(self, rhs: BinaryField128bGhash) -> Self::Output {
        rhs - self
    }
}

impl Mul<BinaryField128bGhash> for BinaryField1b {
    type Output = BinaryField128bGhash;

    #[inline]
    fn mul(self, rhs: BinaryField128bGhash) -> Self::Output {
        rhs * self
    }
}

impl ExtensionField<BinaryField1b> for BinaryField128bGhash {
    const LOG_DEGREE: usize = 7;

    #[inline]
    fn basis(i: usize) -> Self {
        assert!(i < 128, "index {i} out of range for degree 128");
        Self::new(1 << i)
    }

    #[inline]
    fn from_bases_sparse(
        base_elems: impl IntoIterator<Item = BinaryField1b>,
        log_stride: usize,
    ) -> Self {
        assert!(
            log_stride == 7,
            "log_stride must be 7 for BinaryField128bGhash"
        );
        let value = base_elems
            .into_iter()
            .enumerate()
            .fold(0, |value, (i, elem)| value | (u128::from(elem.0) << i));
        Self::new(value)
    }

    #[inline]
    fn iter_bases(&self) -> impl Iterator<Item = BinaryField1b> {
        Divisible::<U1>::value_iter(self.0).map_skippable(BinaryField1b::from)
    }

    #[inline]
    fn into_iter_bases(self) -> impl Iterator<Item = BinaryField1b> {
        Divisible::<U1>::value_iter(self.0).map_skippable(BinaryField1b::from)
    }

    #[inline]
    unsafe fn get_base_unchecked(&self, i: usize) -> BinaryField1b {
        BinaryField1b(U1::num_cast_from(self.0 >> i))
    }

    #[inline]
    fn square_transpose(values: &mut [Self]) {
        square_transforms_extension_field::<BinaryField1b, Self>(values)
    }
}

impl SerializeBytes for BinaryField128bGhash {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.0.serialize(write_buf)
    }
}

impl DeserializeBytes for BinaryField128bGhash {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Ok(Self(DeserializeBytes::deserialize(read_buf)?))
    }
}

impl BinaryField for BinaryField128bGhash {}

impl TowerField for BinaryField128bGhash {
    fn min_tower_level(self) -> usize {
        match self {
            Self::ZERO | Self::ONE => 0,
            _ => 7,
        }
    }

    fn mul_primitive(self, _iota: usize) -> Self {
        // This method could be implemented by multiplying by isomorphic alpha value
        // But it's not being used as for now
        unimplemented!()
    }
}

impl From<AESTowerField8b> for BinaryField128bGhash {
    fn from(value: AESTowerField8b) -> Self {
        const LOOKUP_TABLE: [BinaryField128bGhash; 256] = [
            BinaryField128bGhash(0x00000000000000000000000000000000),
            BinaryField128bGhash(0x00000000000000000000000000000001),
            BinaryField128bGhash(0x0dcb364640a222fe6b8330483c2e9849),
            BinaryField128bGhash(0x0dcb364640a222fe6b8330483c2e9848),
            BinaryField128bGhash(0x3d5bd35c94646a247573da4a5f7710ed),
            BinaryField128bGhash(0x3d5bd35c94646a247573da4a5f7710ec),
            BinaryField128bGhash(0x3090e51ad4c648da1ef0ea02635988a4),
            BinaryField128bGhash(0x3090e51ad4c648da1ef0ea02635988a5),
            BinaryField128bGhash(0x6d58c4e181f9199f41a12db1f974f3ac),
            BinaryField128bGhash(0x6d58c4e181f9199f41a12db1f974f3ad),
            BinaryField128bGhash(0x6093f2a7c15b3b612a221df9c55a6be5),
            BinaryField128bGhash(0x6093f2a7c15b3b612a221df9c55a6be4),
            BinaryField128bGhash(0x500317bd159d73bb34d2f7fba603e341),
            BinaryField128bGhash(0x500317bd159d73bb34d2f7fba603e340),
            BinaryField128bGhash(0x5dc821fb553f51455f51c7b39a2d7b08),
            BinaryField128bGhash(0x5dc821fb553f51455f51c7b39a2d7b09),
            BinaryField128bGhash(0xa72ec17764d7ced55e2f716f4ede412f),
            BinaryField128bGhash(0xa72ec17764d7ced55e2f716f4ede412e),
            BinaryField128bGhash(0xaae5f7312475ec2b35ac412772f0d966),
            BinaryField128bGhash(0xaae5f7312475ec2b35ac412772f0d967),
            BinaryField128bGhash(0x9a75122bf0b3a4f12b5cab2511a951c2),
            BinaryField128bGhash(0x9a75122bf0b3a4f12b5cab2511a951c3),
            BinaryField128bGhash(0x97be246db011860f40df9b6d2d87c98b),
            BinaryField128bGhash(0x97be246db011860f40df9b6d2d87c98a),
            BinaryField128bGhash(0xca760596e52ed74a1f8e5cdeb7aab283),
            BinaryField128bGhash(0xca760596e52ed74a1f8e5cdeb7aab282),
            BinaryField128bGhash(0xc7bd33d0a58cf5b4740d6c968b842aca),
            BinaryField128bGhash(0xc7bd33d0a58cf5b4740d6c968b842acb),
            BinaryField128bGhash(0xf72dd6ca714abd6e6afd8694e8dda26e),
            BinaryField128bGhash(0xf72dd6ca714abd6e6afd8694e8dda26f),
            BinaryField128bGhash(0xfae6e08c31e89f90017eb6dcd4f33a27),
            BinaryField128bGhash(0xfae6e08c31e89f90017eb6dcd4f33a26),
            BinaryField128bGhash(0x4d52354a3a3d8c865cb10fbabcf00118),
            BinaryField128bGhash(0x4d52354a3a3d8c865cb10fbabcf00119),
            BinaryField128bGhash(0x4099030c7a9fae7837323ff280de9951),
            BinaryField128bGhash(0x4099030c7a9fae7837323ff280de9950),
            BinaryField128bGhash(0x7009e616ae59e6a229c2d5f0e38711f5),
            BinaryField128bGhash(0x7009e616ae59e6a229c2d5f0e38711f4),
            BinaryField128bGhash(0x7dc2d050eefbc45c4241e5b8dfa989bc),
            BinaryField128bGhash(0x7dc2d050eefbc45c4241e5b8dfa989bd),
            BinaryField128bGhash(0x200af1abbbc495191d10220b4584f2b4),
            BinaryField128bGhash(0x200af1abbbc495191d10220b4584f2b5),
            BinaryField128bGhash(0x2dc1c7edfb66b7e77693124379aa6afd),
            BinaryField128bGhash(0x2dc1c7edfb66b7e77693124379aa6afc),
            BinaryField128bGhash(0x1d5122f72fa0ff3d6863f8411af3e259),
            BinaryField128bGhash(0x1d5122f72fa0ff3d6863f8411af3e258),
            BinaryField128bGhash(0x109a14b16f02ddc303e0c80926dd7a10),
            BinaryField128bGhash(0x109a14b16f02ddc303e0c80926dd7a11),
            BinaryField128bGhash(0xea7cf43d5eea4253029e7ed5f22e4037),
            BinaryField128bGhash(0xea7cf43d5eea4253029e7ed5f22e4036),
            BinaryField128bGhash(0xe7b7c27b1e4860ad691d4e9dce00d87e),
            BinaryField128bGhash(0xe7b7c27b1e4860ad691d4e9dce00d87f),
            BinaryField128bGhash(0xd7272761ca8e287777eda49fad5950da),
            BinaryField128bGhash(0xd7272761ca8e287777eda49fad5950db),
            BinaryField128bGhash(0xdaec11278a2c0a891c6e94d79177c893),
            BinaryField128bGhash(0xdaec11278a2c0a891c6e94d79177c892),
            BinaryField128bGhash(0x872430dcdf135bcc433f53640b5ab39b),
            BinaryField128bGhash(0x872430dcdf135bcc433f53640b5ab39a),
            BinaryField128bGhash(0x8aef069a9fb1793228bc632c37742bd2),
            BinaryField128bGhash(0x8aef069a9fb1793228bc632c37742bd3),
            BinaryField128bGhash(0xba7fe3804b7731e8364c892e542da376),
            BinaryField128bGhash(0xba7fe3804b7731e8364c892e542da377),
            BinaryField128bGhash(0xb7b4d5c60bd513165dcfb96668033b3f),
            BinaryField128bGhash(0xb7b4d5c60bd513165dcfb96668033b3e),
            BinaryField128bGhash(0x553e92e8bc0ae9a795ed1f57f3632d4d),
            BinaryField128bGhash(0x553e92e8bc0ae9a795ed1f57f3632d4c),
            BinaryField128bGhash(0x58f5a4aefca8cb59fe6e2f1fcf4db504),
            BinaryField128bGhash(0x58f5a4aefca8cb59fe6e2f1fcf4db505),
            BinaryField128bGhash(0x686541b4286e8383e09ec51dac143da0),
            BinaryField128bGhash(0x686541b4286e8383e09ec51dac143da1),
            BinaryField128bGhash(0x65ae77f268cca17d8b1df555903aa5e9),
            BinaryField128bGhash(0x65ae77f268cca17d8b1df555903aa5e8),
            BinaryField128bGhash(0x386656093df3f038d44c32e60a17dee1),
            BinaryField128bGhash(0x386656093df3f038d44c32e60a17dee0),
            BinaryField128bGhash(0x35ad604f7d51d2c6bfcf02ae363946a8),
            BinaryField128bGhash(0x35ad604f7d51d2c6bfcf02ae363946a9),
            BinaryField128bGhash(0x053d8555a9979a1ca13fe8ac5560ce0c),
            BinaryField128bGhash(0x053d8555a9979a1ca13fe8ac5560ce0d),
            BinaryField128bGhash(0x08f6b313e935b8e2cabcd8e4694e5645),
            BinaryField128bGhash(0x08f6b313e935b8e2cabcd8e4694e5644),
            BinaryField128bGhash(0xf210539fd8dd2772cbc26e38bdbd6c62),
            BinaryField128bGhash(0xf210539fd8dd2772cbc26e38bdbd6c63),
            BinaryField128bGhash(0xffdb65d9987f058ca0415e708193f42b),
            BinaryField128bGhash(0xffdb65d9987f058ca0415e708193f42a),
            BinaryField128bGhash(0xcf4b80c34cb94d56beb1b472e2ca7c8f),
            BinaryField128bGhash(0xcf4b80c34cb94d56beb1b472e2ca7c8e),
            BinaryField128bGhash(0xc280b6850c1b6fa8d532843adee4e4c6),
            BinaryField128bGhash(0xc280b6850c1b6fa8d532843adee4e4c7),
            BinaryField128bGhash(0x9f48977e59243eed8a63438944c99fce),
            BinaryField128bGhash(0x9f48977e59243eed8a63438944c99fcf),
            BinaryField128bGhash(0x9283a13819861c13e1e073c178e70787),
            BinaryField128bGhash(0x9283a13819861c13e1e073c178e70786),
            BinaryField128bGhash(0xa2134422cd4054c9ff1099c31bbe8f23),
            BinaryField128bGhash(0xa2134422cd4054c9ff1099c31bbe8f22),
            BinaryField128bGhash(0xafd872648de276379493a98b2790176a),
            BinaryField128bGhash(0xafd872648de276379493a98b2790176b),
            BinaryField128bGhash(0x186ca7a286376521c95c10ed4f932c55),
            BinaryField128bGhash(0x186ca7a286376521c95c10ed4f932c54),
            BinaryField128bGhash(0x15a791e4c69547dfa2df20a573bdb41c),
            BinaryField128bGhash(0x15a791e4c69547dfa2df20a573bdb41d),
            BinaryField128bGhash(0x253774fe12530f05bc2fcaa710e43cb8),
            BinaryField128bGhash(0x253774fe12530f05bc2fcaa710e43cb9),
            BinaryField128bGhash(0x28fc42b852f12dfbd7acfaef2ccaa4f1),
            BinaryField128bGhash(0x28fc42b852f12dfbd7acfaef2ccaa4f0),
            BinaryField128bGhash(0x7534634307ce7cbe88fd3d5cb6e7dff9),
            BinaryField128bGhash(0x7534634307ce7cbe88fd3d5cb6e7dff8),
            BinaryField128bGhash(0x78ff5505476c5e40e37e0d148ac947b0),
            BinaryField128bGhash(0x78ff5505476c5e40e37e0d148ac947b1),
            BinaryField128bGhash(0x486fb01f93aa169afd8ee716e990cf14),
            BinaryField128bGhash(0x486fb01f93aa169afd8ee716e990cf15),
            BinaryField128bGhash(0x45a48659d3083464960dd75ed5be575d),
            BinaryField128bGhash(0x45a48659d3083464960dd75ed5be575c),
            BinaryField128bGhash(0xbf4266d5e2e0abf497736182014d6d7a),
            BinaryField128bGhash(0xbf4266d5e2e0abf497736182014d6d7b),
            BinaryField128bGhash(0xb2895093a242890afcf051ca3d63f533),
            BinaryField128bGhash(0xb2895093a242890afcf051ca3d63f532),
            BinaryField128bGhash(0x8219b5897684c1d0e200bbc85e3a7d97),
            BinaryField128bGhash(0x8219b5897684c1d0e200bbc85e3a7d96),
            BinaryField128bGhash(0x8fd283cf3626e32e89838b806214e5de),
            BinaryField128bGhash(0x8fd283cf3626e32e89838b806214e5df),
            BinaryField128bGhash(0xd21aa2346319b26bd6d24c33f8399ed6),
            BinaryField128bGhash(0xd21aa2346319b26bd6d24c33f8399ed7),
            BinaryField128bGhash(0xdfd1947223bb9095bd517c7bc417069f),
            BinaryField128bGhash(0xdfd1947223bb9095bd517c7bc417069e),
            BinaryField128bGhash(0xef417168f77dd84fa3a19679a74e8e3b),
            BinaryField128bGhash(0xef417168f77dd84fa3a19679a74e8e3a),
            BinaryField128bGhash(0xe28a472eb7dffab1c822a6319b601672),
            BinaryField128bGhash(0xe28a472eb7dffab1c822a6319b601673),
            BinaryField128bGhash(0x93252331bf042b11512625b1f09fa87e),
            BinaryField128bGhash(0x93252331bf042b11512625b1f09fa87f),
            BinaryField128bGhash(0x9eee1577ffa609ef3aa515f9ccb13037),
            BinaryField128bGhash(0x9eee1577ffa609ef3aa515f9ccb13036),
            BinaryField128bGhash(0xae7ef06d2b6041352455fffbafe8b893),
            BinaryField128bGhash(0xae7ef06d2b6041352455fffbafe8b892),
            BinaryField128bGhash(0xa3b5c62b6bc263cb4fd6cfb393c620da),
            BinaryField128bGhash(0xa3b5c62b6bc263cb4fd6cfb393c620db),
            BinaryField128bGhash(0xfe7de7d03efd328e1087080009eb5bd2),
            BinaryField128bGhash(0xfe7de7d03efd328e1087080009eb5bd3),
            BinaryField128bGhash(0xf3b6d1967e5f10707b04384835c5c39b),
            BinaryField128bGhash(0xf3b6d1967e5f10707b04384835c5c39a),
            BinaryField128bGhash(0xc326348caa9958aa65f4d24a569c4b3f),
            BinaryField128bGhash(0xc326348caa9958aa65f4d24a569c4b3e),
            BinaryField128bGhash(0xceed02caea3b7a540e77e2026ab2d376),
            BinaryField128bGhash(0xceed02caea3b7a540e77e2026ab2d377),
            BinaryField128bGhash(0x340be246dbd3e5c40f0954debe41e951),
            BinaryField128bGhash(0x340be246dbd3e5c40f0954debe41e950),
            BinaryField128bGhash(0x39c0d4009b71c73a648a6496826f7118),
            BinaryField128bGhash(0x39c0d4009b71c73a648a6496826f7119),
            BinaryField128bGhash(0x0950311a4fb78fe07a7a8e94e136f9bc),
            BinaryField128bGhash(0x0950311a4fb78fe07a7a8e94e136f9bd),
            BinaryField128bGhash(0x049b075c0f15ad1e11f9bedcdd1861f5),
            BinaryField128bGhash(0x049b075c0f15ad1e11f9bedcdd1861f4),
            BinaryField128bGhash(0x595326a75a2afc5b4ea8796f47351afd),
            BinaryField128bGhash(0x595326a75a2afc5b4ea8796f47351afc),
            BinaryField128bGhash(0x549810e11a88dea5252b49277b1b82b4),
            BinaryField128bGhash(0x549810e11a88dea5252b49277b1b82b5),
            BinaryField128bGhash(0x6408f5fbce4e967f3bdba32518420a10),
            BinaryField128bGhash(0x6408f5fbce4e967f3bdba32518420a11),
            BinaryField128bGhash(0x69c3c3bd8eecb4815058936d246c9259),
            BinaryField128bGhash(0x69c3c3bd8eecb4815058936d246c9258),
            BinaryField128bGhash(0xde77167b8539a7970d972a0b4c6fa966),
            BinaryField128bGhash(0xde77167b8539a7970d972a0b4c6fa967),
            BinaryField128bGhash(0xd3bc203dc59b856966141a437041312f),
            BinaryField128bGhash(0xd3bc203dc59b856966141a437041312e),
            BinaryField128bGhash(0xe32cc527115dcdb378e4f0411318b98b),
            BinaryField128bGhash(0xe32cc527115dcdb378e4f0411318b98a),
            BinaryField128bGhash(0xeee7f36151ffef4d1367c0092f3621c2),
            BinaryField128bGhash(0xeee7f36151ffef4d1367c0092f3621c3),
            BinaryField128bGhash(0xb32fd29a04c0be084c3607bab51b5aca),
            BinaryField128bGhash(0xb32fd29a04c0be084c3607bab51b5acb),
            BinaryField128bGhash(0xbee4e4dc44629cf627b537f28935c283),
            BinaryField128bGhash(0xbee4e4dc44629cf627b537f28935c282),
            BinaryField128bGhash(0x8e7401c690a4d42c3945ddf0ea6c4a27),
            BinaryField128bGhash(0x8e7401c690a4d42c3945ddf0ea6c4a26),
            BinaryField128bGhash(0x83bf3780d006f6d252c6edb8d642d26e),
            BinaryField128bGhash(0x83bf3780d006f6d252c6edb8d642d26f),
            BinaryField128bGhash(0x7959d70ce1ee694253b85b6402b1e849),
            BinaryField128bGhash(0x7959d70ce1ee694253b85b6402b1e848),
            BinaryField128bGhash(0x7492e14aa14c4bbc383b6b2c3e9f7000),
            BinaryField128bGhash(0x7492e14aa14c4bbc383b6b2c3e9f7001),
            BinaryField128bGhash(0x44020450758a036626cb812e5dc6f8a4),
            BinaryField128bGhash(0x44020450758a036626cb812e5dc6f8a5),
            BinaryField128bGhash(0x49c93216352821984d48b16661e860ed),
            BinaryField128bGhash(0x49c93216352821984d48b16661e860ec),
            BinaryField128bGhash(0x140113ed601770dd121976d5fbc51be5),
            BinaryField128bGhash(0x140113ed601770dd121976d5fbc51be4),
            BinaryField128bGhash(0x19ca25ab20b55223799a469dc7eb83ac),
            BinaryField128bGhash(0x19ca25ab20b55223799a469dc7eb83ad),
            BinaryField128bGhash(0x295ac0b1f4731af9676aac9fa4b20b08),
            BinaryField128bGhash(0x295ac0b1f4731af9676aac9fa4b20b09),
            BinaryField128bGhash(0x2491f6f7b4d138070ce99cd7989c9341),
            BinaryField128bGhash(0x2491f6f7b4d138070ce99cd7989c9340),
            BinaryField128bGhash(0xc61bb1d9030ec2b6c4cb3ae603fc8533),
            BinaryField128bGhash(0xc61bb1d9030ec2b6c4cb3ae603fc8532),
            BinaryField128bGhash(0xcbd0879f43ace048af480aae3fd21d7a),
            BinaryField128bGhash(0xcbd0879f43ace048af480aae3fd21d7b),
            BinaryField128bGhash(0xfb406285976aa892b1b8e0ac5c8b95de),
            BinaryField128bGhash(0xfb406285976aa892b1b8e0ac5c8b95df),
            BinaryField128bGhash(0xf68b54c3d7c88a6cda3bd0e460a50d97),
            BinaryField128bGhash(0xf68b54c3d7c88a6cda3bd0e460a50d96),
            BinaryField128bGhash(0xab43753882f7db29856a1757fa88769f),
            BinaryField128bGhash(0xab43753882f7db29856a1757fa88769e),
            BinaryField128bGhash(0xa688437ec255f9d7eee9271fc6a6eed6),
            BinaryField128bGhash(0xa688437ec255f9d7eee9271fc6a6eed7),
            BinaryField128bGhash(0x9618a6641693b10df019cd1da5ff6672),
            BinaryField128bGhash(0x9618a6641693b10df019cd1da5ff6673),
            BinaryField128bGhash(0x9bd39022563193f39b9afd5599d1fe3b),
            BinaryField128bGhash(0x9bd39022563193f39b9afd5599d1fe3a),
            BinaryField128bGhash(0x613570ae67d90c639ae44b894d22c41c),
            BinaryField128bGhash(0x613570ae67d90c639ae44b894d22c41d),
            BinaryField128bGhash(0x6cfe46e8277b2e9df1677bc1710c5c55),
            BinaryField128bGhash(0x6cfe46e8277b2e9df1677bc1710c5c54),
            BinaryField128bGhash(0x5c6ea3f2f3bd6647ef9791c31255d4f1),
            BinaryField128bGhash(0x5c6ea3f2f3bd6647ef9791c31255d4f0),
            BinaryField128bGhash(0x51a595b4b31f44b98414a18b2e7b4cb8),
            BinaryField128bGhash(0x51a595b4b31f44b98414a18b2e7b4cb9),
            BinaryField128bGhash(0x0c6db44fe62015fcdb456638b45637b0),
            BinaryField128bGhash(0x0c6db44fe62015fcdb456638b45637b1),
            BinaryField128bGhash(0x01a68209a6823702b0c656708878aff9),
            BinaryField128bGhash(0x01a68209a6823702b0c656708878aff8),
            BinaryField128bGhash(0x3136671372447fd8ae36bc72eb21275d),
            BinaryField128bGhash(0x3136671372447fd8ae36bc72eb21275c),
            BinaryField128bGhash(0x3cfd515532e65d26c5b58c3ad70fbf14),
            BinaryField128bGhash(0x3cfd515532e65d26c5b58c3ad70fbf15),
            BinaryField128bGhash(0x8b49849339334e30987a355cbf0c842b),
            BinaryField128bGhash(0x8b49849339334e30987a355cbf0c842a),
            BinaryField128bGhash(0x8682b2d579916ccef3f9051483221c62),
            BinaryField128bGhash(0x8682b2d579916ccef3f9051483221c63),
            BinaryField128bGhash(0xb61257cfad572414ed09ef16e07b94c6),
            BinaryField128bGhash(0xb61257cfad572414ed09ef16e07b94c7),
            BinaryField128bGhash(0xbbd96189edf506ea868adf5edc550c8f),
            BinaryField128bGhash(0xbbd96189edf506ea868adf5edc550c8e),
            BinaryField128bGhash(0xe6114072b8ca57afd9db18ed46787787),
            BinaryField128bGhash(0xe6114072b8ca57afd9db18ed46787786),
            BinaryField128bGhash(0xebda7634f8687551b25828a57a56efce),
            BinaryField128bGhash(0xebda7634f8687551b25828a57a56efcf),
            BinaryField128bGhash(0xdb4a932e2cae3d8baca8c2a7190f676a),
            BinaryField128bGhash(0xdb4a932e2cae3d8baca8c2a7190f676b),
            BinaryField128bGhash(0xd681a5686c0c1f75c72bf2ef2521ff23),
            BinaryField128bGhash(0xd681a5686c0c1f75c72bf2ef2521ff22),
            BinaryField128bGhash(0x2c6745e45de480e5c6554433f1d2c504),
            BinaryField128bGhash(0x2c6745e45de480e5c6554433f1d2c505),
            BinaryField128bGhash(0x21ac73a21d46a21badd6747bcdfc5d4d),
            BinaryField128bGhash(0x21ac73a21d46a21badd6747bcdfc5d4c),
            BinaryField128bGhash(0x113c96b8c980eac1b3269e79aea5d5e9),
            BinaryField128bGhash(0x113c96b8c980eac1b3269e79aea5d5e8),
            BinaryField128bGhash(0x1cf7a0fe8922c83fd8a5ae31928b4da0),
            BinaryField128bGhash(0x1cf7a0fe8922c83fd8a5ae31928b4da1),
            BinaryField128bGhash(0x413f8105dc1d997a87f4698208a636a8),
            BinaryField128bGhash(0x413f8105dc1d997a87f4698208a636a9),
            BinaryField128bGhash(0x4cf4b7439cbfbb84ec7759ca3488aee1),
            BinaryField128bGhash(0x4cf4b7439cbfbb84ec7759ca3488aee0),
            BinaryField128bGhash(0x7c6452594879f35ef287b3c857d12645),
            BinaryField128bGhash(0x7c6452594879f35ef287b3c857d12644),
            BinaryField128bGhash(0x71af641f08dbd1a0990483806bffbe0c),
            BinaryField128bGhash(0x71af641f08dbd1a0990483806bffbe0d),
        ];

        LOOKUP_TABLE[value.0 as usize]
    }
}

#[inline(always)]
pub fn is_ghash_tower<F: TowerField>() -> bool {
    TypeId::of::<F>() == TypeId::of::<BinaryField128bGhash>()
        || TypeId::of::<F>() == TypeId::of::<BinaryField1b>()
}

#[cfg(test)]
mod tests {
    use proptest::{prelude::any, proptest};

    use super::*;
    use crate::binary_field::tests::is_binary_field_valid_generator;

    #[test]
    fn test_ghash_mul() {
        let a = BinaryField128bGhash(1u128);
        let b = BinaryField128bGhash(1u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(1u128));

        let a = BinaryField128bGhash(1u128);
        let b = BinaryField128bGhash(2u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(2u128));

        let a = BinaryField128bGhash(1u128);
        let b = BinaryField128bGhash(1297182698762987u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(1297182698762987u128));

        let a = BinaryField128bGhash(2u128);
        let b = BinaryField128bGhash(2u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(4u128));

        let a = BinaryField128bGhash(2u128);
        let b = BinaryField128bGhash(3u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(6u128));

        let a = BinaryField128bGhash(3u128);
        let b = BinaryField128bGhash(3u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(5u128));

        let a = BinaryField128bGhash(1u128 << 127);
        let b = BinaryField128bGhash(2u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(0b10000111));

        let a = BinaryField128bGhash((1u128 << 127) + 1);
        let b = BinaryField128bGhash(2u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(0b10000101));

        let a = BinaryField128bGhash(3u128 << 126);
        let b = BinaryField128bGhash(2u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(0b10000111 + (1u128 << 127)));

        let a = BinaryField128bGhash(1u128 << 127);
        let b = BinaryField128bGhash(4u128);
        let c = a * b;

        assert_eq!(c, BinaryField128bGhash::from(0b10000111 << 1));

        let a = BinaryField128bGhash(1u128 << 127);
        let b = BinaryField128bGhash(1u128 << 122);
        let c = a * b;

        assert_eq!(
            c,
            BinaryField128bGhash::from((0b00000111 << 121) + 0b10000111)
        );
    }

    #[test]
    fn test_multiplicative_generator() {
        assert!(is_binary_field_valid_generator::<BinaryField128bGhash>());
    }

    #[test]
    fn test_mul_x() {
        let test_cases = [
            0x0,                                    // Zero
            0x1,                                    // One
            0x2,                                    // Two
            0x80000000000000000000000000000000u128, // High bit set
            0x40000000000000000000000000000000u128, // Second highest bit
            0xffffffffffffffffffffffffffffffffu128, // All bits set
            0x87u128,                               // GHASH reduction polynomial
            0x21ac73a21d46a21badd6747bcdfc5d4d,     // Random value
        ];

        for &value in &test_cases {
            let field_val = BinaryField128bGhash::new(value);
            let mul_x_result = field_val.mul_x();
            let regular_mul_result = field_val * BinaryField128bGhash::new(2u128);

            assert_eq!(
                mul_x_result, regular_mul_result,
                "mul_x and regular multiplication by 2 differ for value {:#x}",
                value
            );
        }
    }

    #[test]
    fn test_mul_inv_x() {
        let test_cases = [
            0x0,                                    // Zero
            0x1,                                    // One
            0x2,                                    // Two
            0x1u128,                                // Low bit set
            0x3u128,                                // Two lowest bits set
            0xffffffffffffffffffffffffffffffffu128, // All bits set
            0x87u128,                               // GHASH reduction polynomial
            0x21ac73a21d46a21badd6747bcdfc5d4d,     // Random value
        ];

        for &value in &test_cases {
            let field_val = BinaryField128bGhash::new(value);
            let mul_inv_x_result = field_val.mul_inv_x();
            let regular_mul_result = field_val
                * BinaryField128bGhash::new(2u128)
                    .invert()
                    .expect("2 is invertible");

            assert_eq!(
                mul_inv_x_result, regular_mul_result,
                "mul_inv_x and regular multiplication by 2 differ for value {:#x}",
                value
            );
        }
    }

    proptest! {
        #[test]
        fn test_conversion_from_aes_consistency(a in any::<u8>(), b in any::<u8>()) {
            let a_val = AESTowerField8b::new(a);
            let b_val = AESTowerField8b::new(b);
            let converted_a = BinaryField128bGhash::from(a_val);
            let converted_b = BinaryField128bGhash::from(b_val);
            assert_eq!(BinaryField128bGhash::from(a_val * b_val), converted_a * converted_b);
        }
    }
}
