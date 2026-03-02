//! Definitions of Montgomery field elements.
// The mojarity of the code are taken from Plonky3.
// Ideally we want to simply import or wrap plonky3's field implementation.
// But plonky3 has a feature flag on avx256/avx512 where expander
// decides whether to use avx256 or avx512 at compile time.
//
// So we re-implement the field in our own crate.

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
    io::{Read, Write},
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ethnum::U256;
use serdes::{ExpSerde, SerdeResult};
use utils::{from_monty, monty_reduce, to_monty};

use crate::{FFTField, Field};

mod param;
pub use param::*;

mod utils;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
pub use neon::PackedMontyParameters;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx512::PackedMontyParameters;

// Fallback, use avx2
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
mod avx256;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
pub use avx256::PackedMontyParameters;

#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // Packed field implementations rely on this!
pub struct MontyField31<MP: MontyParameters> {
    /// The MONTY form of the field element, saved as a positive integer less than `P`.
    ///
    /// This is `pub(crate)` for tests and delayed reduction strategies. If you're accessing
    /// `value` outside of those, you're likely doing something fishy.
    pub value: u32,
    _phantom: PhantomData<MP>,
}

impl<MP: FieldParameters> Display for MontyField31<MP> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", from_monty::<MP>(self.value))
    }
}

impl<MP: FieldParameters> Debug for MontyField31<MP> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<MP: FieldParameters> MontyField31<MP> {
    /// The standard way to crate a new element.
    /// Note that new converts the input into MONTY form so should be avoided in performance
    /// critical implementations.
    #[inline(always)]
    pub const fn new(value: u32) -> Self {
        Self {
            value: to_monty::<MP>(value),
            _phantom: PhantomData,
        }
    }

    /// Create a new field element from something already in MONTY form.
    /// This is `pub(crate)` for tests and delayed reduction strategies. If you're using it outside
    /// of those, you're likely doing something fishy.
    #[inline(always)]
    pub(crate) const fn new_monty(value: u32) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    /// Produce a u32 in range [0, P) from a field element corresponding to the true value.
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) const fn to_u32(elem: &Self) -> u32 {
        from_monty::<MP>(elem.value)
    }

    /// Convert a constant u32 array into a constant array of field elements.
    /// Constant version of array.map(MontyField31::new).
    #[inline]
    pub const fn new_array<const N: usize>(input: [u32; N]) -> [Self; N] {
        let mut output = [Self::new_monty(0); N];
        let mut i = 0;
        while i < N {
            output[i] = Self::new(input[i]);
            i += 1;
        }
        output
    }

    /// Convert a constant 2d u32 array into a constant 2d array of field elements.
    /// Constant version of array.map(MontyField31::new_array).
    #[inline]
    pub const fn new_2d_array<const N: usize, const M: usize>(
        input: [[u32; N]; M],
    ) -> [[Self; N]; M] {
        let mut output = [[Self::new_monty(0); N]; M];
        let mut i = 0;
        while i < M {
            output[i] = Self::new_array(input[i]);
            i += 1;
        }
        output
    }
}

impl<MP: FieldParameters> Neg for MontyField31<MP> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::ZERO - self
    }
}

impl<MP: FieldParameters> From<u32> for MontyField31<MP> {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl<MP: FieldParameters> ExpSerde for MontyField31<MP> {
    #[inline(always)]
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        // Note: BabyBear's impl of as_u32_unchecked() converts to canonical form
        writer.write_all(from_monty::<MP>(self.value).to_le_bytes().as_ref())?;
        Ok(())
    }

    /// Note: This function performs modular reduction on inputs and
    /// converts from canonical to Montgomery form.
    #[inline(always)]
    #[allow(const_evaluatable_unchecked)]
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut u = [0u8; 4];
        reader.read_exact(&mut u)?;
        let v = u32::from_le_bytes(u);
        Ok(Self::new(v))
    }
}

impl<MP: FieldParameters> PartialOrd for MontyField31<MP> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<MP: FieldParameters> Ord for MontyField31<MP> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl<MP: FieldParameters> Sub<&MontyField31<MP>> for MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline]
    fn sub(self, rhs: &MontyField31<MP>) -> MontyField31<MP> {
        self.sub(*rhs)
    }
}

impl<MP: FieldParameters> Sub<MontyField31<MP>> for MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline]
    fn sub(self, rhs: MontyField31<MP>) -> MontyField31<MP> {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { MP::PRIME } else { 0 };
        diff = diff.wrapping_add(corr);
        Self::new_monty(diff)
    }
}

impl<MP: FieldParameters> SubAssign for MontyField31<MP> {
    #[inline]
    fn sub_assign(&mut self, rhs: MontyField31<MP>) {
        *self = (*self).sub(rhs)
    }
}

impl<MP: FieldParameters> SubAssign<&MontyField31<MP>> for MontyField31<MP> {
    #[inline]
    fn sub_assign(&mut self, rhs: &MontyField31<MP>) {
        *self = (*self).sub(rhs)
    }
}

// ========================
// additions
// ========================

impl<MP: FieldParameters> Add<&MontyField31<MP>> for MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline]
    fn add(self, rhs: &MontyField31<MP>) -> MontyField31<MP> {
        self.add(*rhs)
    }
}

impl<MP: FieldParameters> Add<MontyField31<MP>> for MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline]
    fn add(self, rhs: MontyField31<MP>) -> MontyField31<MP> {
        let mut sum = self.value + rhs.value;
        let (corr_sum, over) = sum.overflowing_sub(MP::PRIME);
        if !over {
            sum = corr_sum;
        }
        Self::new_monty(sum)
    }
}

impl<MP: FieldParameters> AddAssign for MontyField31<MP> {
    #[inline]
    fn add_assign(&mut self, rhs: MontyField31<MP>) {
        *self = (*self).add(rhs)
    }
}

impl<'b, MP: FieldParameters> AddAssign<&'b MontyField31<MP>> for MontyField31<MP> {
    #[inline]
    fn add_assign(&mut self, rhs: &'b MontyField31<MP>) {
        *self = (*self).add(rhs)
    }
}

impl<T, MP: FieldParameters> Sum<T> for MontyField31<MP>
where
    T: core::borrow::Borrow<Self>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
    }
}

// ========================
// multiplications
// ========================
impl<MP: FieldParameters> Mul<MontyField31<MP>> for MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline]
    fn mul(self, rhs: MontyField31<MP>) -> MontyField31<MP> {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self::new_monty(monty_reduce::<MP>(long_prod))
    }
}

impl<'b, MP: FieldParameters> Mul<&'b MontyField31<MP>> for MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline]
    fn mul(self, rhs: &'b MontyField31<MP>) -> MontyField31<MP> {
        self.mul(*rhs)
    }
}

impl<MP: FieldParameters> Mul<MontyField31<MP>> for &MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline(always)]
    fn mul(self, rhs: MontyField31<MP>) -> MontyField31<MP> {
        *self * rhs
    }
}

impl<MP: FieldParameters> Mul<&MontyField31<MP>> for &MontyField31<MP> {
    type Output = MontyField31<MP>;

    #[inline(always)]
    fn mul(self, rhs: &MontyField31<MP>) -> MontyField31<MP> {
        *self * *rhs
    }
}

impl<MP: FieldParameters> MulAssign for MontyField31<MP> {
    #[inline]
    fn mul_assign(&mut self, rhs: MontyField31<MP>) {
        *self = (*self).mul(&rhs)
    }
}

impl<'b, MP: FieldParameters> MulAssign<&'b MontyField31<MP>> for MontyField31<MP> {
    #[inline]
    fn mul_assign(&mut self, rhs: &'b MontyField31<MP>) {
        *self = (*self).mul(rhs)
    }
}

impl<T, MP: FieldParameters> Product<T> for MontyField31<MP>
where
    T: core::borrow::Borrow<Self>,
{
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, item| acc * item.borrow())
    }
}

impl<MP: FieldParameters> Field for MontyField31<MP> {
    const NAME: &'static str = "Monty Field";

    const SIZE: usize = 32 / 8;

    const FIELD_SIZE: usize = 32;

    const ZERO: Self = Self::new(0);

    const ONE: Self = Self::new(1);

    const MODULUS: U256 = U256([MP::PRIME as u128, 0]);

    // See test below
    const INV_2: Self = Self {
        value: 134217727,
        _phantom: PhantomData,
    };

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }

    /// Uses rejection sampling to avoid bias.
    fn random_unsafe(mut rng: impl rand::RngCore) -> Self {
        Self::new(rng.next_u32())
    }

    fn random_bool(mut rng: impl rand::RngCore) -> Self {
        (rng.next_u32() & 1).into()
    }

    fn inv(&self) -> Option<Self> {
        <MP as FieldParameters>::try_inverse(self)
    }

    /// Converts to canonical form.
    #[inline(always)]
    fn as_u32_unchecked(&self) -> u32 {
        from_monty::<MP>(self.value)
    }

    #[inline(always)]
    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 4);
        // Note: From<u32> performs modular reduction
        u32::from_le_bytes(bytes[..4].try_into().unwrap()).into()
    }

    #[inline(always)]
    fn to_u256(&self) -> U256 {
        U256([self.as_u32_unchecked() as u128, 0])
    }

    #[inline(always)]
    fn from_u256(v: U256) -> Self {
        Self::new(v.as_u32())
    }
}

impl<MP: FieldParameters> FFTField for MontyField31<MP> {
    const TWO_ADICITY: usize = MP::TWO_ADICITY;

    fn root_of_unity() -> Self {
        Self::from(MP::TWO_ADICITY_GENERATOR_IN_U32)
    }
}
