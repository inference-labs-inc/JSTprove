// Copyright 2024-2025 Irreducible Inc.

// This is because derive(bytemuck::TransparentWrapper) adds some type constraints to
// PackedPrimitiveType in addition to the type constraints we define. Even more, annoying, the
// allow attribute has to be added to the module, it doesn't work to add it to the struct
// definition.
#![allow(clippy::multiple_bound_locations)]

use std::{
    fmt::Debug,
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use binius_utils::{checked_arithmetics::checked_int_div, iter::IterExtensions};
use bytemuck::{Pod, TransparentWrapper, Zeroable};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use crate::{
    BinaryField, Divisible, ExtensionField, Field, PackedField,
    arithmetic_traits::{InvertOrZero, MulAlpha, Square},
    field::FieldOps,
    underlier::{NumCast, UnderlierType, UnderlierWithBitOps, WithUnderlier},
};

#[derive(PartialEq, Eq, Clone, Copy, Default, bytemuck::TransparentWrapper)]
#[repr(transparent)]
#[transparent(U)]
pub struct PackedPrimitiveType<U: UnderlierType, Scalar: BinaryField>(
    pub U,
    pub PhantomData<Scalar>,
);

impl<U: UnderlierType, Scalar: BinaryField> PackedPrimitiveType<U, Scalar> {
    pub const WIDTH: usize = {
        assert!(U::BITS.is_multiple_of(Scalar::N_BITS));

        U::BITS / Scalar::N_BITS
    };

    pub const LOG_WIDTH: usize = {
        let result = Self::WIDTH.ilog2();

        assert!(2usize.pow(result) == Self::WIDTH);

        result as usize
    };

    #[inline]
    pub const fn from_underlier(val: U) -> Self {
        Self(val, PhantomData)
    }

    #[inline]
    pub const fn to_underlier(self) -> U {
        self.0
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField>
    PackedPrimitiveType<U, Scalar>
{
    #[inline]
    pub fn broadcast(scalar: Scalar) -> Self {
        U::broadcast_subvalue(scalar.to_underlier()).into()
    }
}

unsafe impl<U: UnderlierType, Scalar: BinaryField> WithUnderlier
    for PackedPrimitiveType<U, Scalar>
{
    type Underlier = U;

    #[inline(always)]
    fn to_underlier(self) -> Self::Underlier {
        TransparentWrapper::peel(self)
    }

    #[inline(always)]
    fn to_underlier_ref(&self) -> &Self::Underlier {
        TransparentWrapper::peel_ref(self)
    }

    #[inline(always)]
    fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
        TransparentWrapper::peel_mut(self)
    }

    #[inline(always)]
    fn to_underliers_ref(val: &[Self]) -> &[Self::Underlier] {
        TransparentWrapper::peel_slice(val)
    }

    #[inline(always)]
    fn to_underliers_ref_mut(val: &mut [Self]) -> &mut [Self::Underlier] {
        TransparentWrapper::peel_slice_mut(val)
    }

    #[inline(always)]
    fn from_underlier(val: Self::Underlier) -> Self {
        TransparentWrapper::wrap(val)
    }

    #[inline(always)]
    fn from_underlier_ref(val: &Self::Underlier) -> &Self {
        TransparentWrapper::wrap_ref(val)
    }

    #[inline(always)]
    fn from_underlier_ref_mut(val: &mut Self::Underlier) -> &mut Self {
        TransparentWrapper::wrap_mut(val)
    }

    #[inline(always)]
    fn from_underliers_ref(val: &[Self::Underlier]) -> &[Self] {
        TransparentWrapper::wrap_slice(val)
    }

    #[inline(always)]
    fn from_underliers_ref_mut(val: &mut [Self::Underlier]) -> &mut [Self] {
        TransparentWrapper::wrap_slice_mut(val)
    }
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Debug for PackedPrimitiveType<U, Scalar>
where
    Self: PackedField,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = checked_int_div(U::BITS, Scalar::N_BITS);
        let values_str = self
            .iter()
            .map(|value| format!("{value}"))
            .collect::<Vec<_>>()
            .join(",");

        write!(f, "Packed{}x{}([{}])", width, Scalar::N_BITS, values_str)
    }
}

impl<U: UnderlierType, Scalar: BinaryField> From<U> for PackedPrimitiveType<U, Scalar> {
    #[inline]
    fn from(val: U) -> Self {
        Self(val, PhantomData)
    }
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Neg for PackedPrimitiveType<U, Scalar> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self
    }
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Add for PackedPrimitiveType<U, Scalar> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        (self.0 ^ rhs.0).into()
    }
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Add<&Self> for PackedPrimitiveType<U, Scalar> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: &Self) -> Self::Output {
        (self.0 ^ rhs.0).into()
    }
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Sub for PackedPrimitiveType<U, Scalar> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 ^ rhs.0).into()
    }
}

impl<U: UnderlierWithBitOps, Scalar: BinaryField> Sub<&Self> for PackedPrimitiveType<U, Scalar> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &Self) -> Self::Output {
        (self.0 ^ rhs.0).into()
    }
}

impl<U: UnderlierType, Scalar: BinaryField> AddAssign for PackedPrimitiveType<U, Scalar>
where
    Self: Add<Output = Self>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<U: UnderlierType, Scalar: BinaryField> AddAssign<&Self> for PackedPrimitiveType<U, Scalar>
where
    Self: for<'a> Add<&'a Self, Output = Self>,
{
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + rhs;
    }
}

impl<U: UnderlierType, Scalar: BinaryField> SubAssign for PackedPrimitiveType<U, Scalar>
where
    Self: Sub<Output = Self>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<U: UnderlierType, Scalar: BinaryField> SubAssign<&Self> for PackedPrimitiveType<U, Scalar>
where
    Self: for<'a> Sub<&'a Self, Output = Self>,
{
    fn sub_assign(&mut self, rhs: &Self) {
        *self = *self - rhs;
    }
}

impl<U: UnderlierType, Scalar: BinaryField> MulAssign for PackedPrimitiveType<U, Scalar>
where
    Self: Mul<Output = Self>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<U: UnderlierType, Scalar: BinaryField> MulAssign<&Self> for PackedPrimitiveType<U, Scalar>
where
    Self: for<'a> Mul<&'a Self, Output = Self>,
{
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * rhs;
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> Add<Scalar>
    for PackedPrimitiveType<U, Scalar>
{
    type Output = Self;

    fn add(self, rhs: Scalar) -> Self::Output {
        self + Self::broadcast(rhs)
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> Sub<Scalar>
    for PackedPrimitiveType<U, Scalar>
{
    type Output = Self;

    fn sub(self, rhs: Scalar) -> Self::Output {
        self - Self::broadcast(rhs)
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> Mul<Scalar>
    for PackedPrimitiveType<U, Scalar>
where
    Self: Mul<Output = Self>,
{
    type Output = Self;

    fn mul(self, rhs: Scalar) -> Self::Output {
        self * Self::broadcast(rhs)
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> AddAssign<Scalar>
    for PackedPrimitiveType<U, Scalar>
{
    fn add_assign(&mut self, rhs: Scalar) {
        *self += Self::broadcast(rhs);
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> SubAssign<Scalar>
    for PackedPrimitiveType<U, Scalar>
{
    fn sub_assign(&mut self, rhs: Scalar) {
        *self -= Self::broadcast(rhs);
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> MulAssign<Scalar>
    for PackedPrimitiveType<U, Scalar>
where
    Self: MulAssign<Self>,
{
    fn mul_assign(&mut self, rhs: Scalar) {
        *self *= Self::broadcast(rhs);
    }
}

impl<U: UnderlierType, Scalar: BinaryField> Sum for PackedPrimitiveType<U, Scalar>
where
    Self: Add<Output = Self>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::from(U::default()), |result, next| result + next)
    }
}

impl<'a, U: UnderlierType, Scalar: BinaryField> Sum<&'a Self> for PackedPrimitiveType<U, Scalar>
where
    Self: Add<Output = Self>,
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::from(U::default()), |result, next| result + *next)
    }
}

impl<U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField> Product
    for PackedPrimitiveType<U, Scalar>
where
    Self: Mul<Output = Self>,
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::broadcast(Scalar::ONE), |result, next| result * next)
    }
}

impl<'a, U: UnderlierWithBitOps + Divisible<Scalar::Underlier>, Scalar: BinaryField>
    Product<&'a Self> for PackedPrimitiveType<U, Scalar>
where
    Self: Mul<Output = Self>,
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::broadcast(Scalar::ONE), |result, next| result * *next)
    }
}

impl<U: UnderlierType, Scalar: BinaryField> Mul<&Self> for PackedPrimitiveType<U, Scalar>
where
    Self: Mul<Output = Self>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

unsafe impl<U: UnderlierType + Zeroable, Scalar: BinaryField> Zeroable
    for PackedPrimitiveType<U, Scalar>
{
}

unsafe impl<U: UnderlierType + Pod, Scalar: BinaryField> Pod for PackedPrimitiveType<U, Scalar> {}

impl<U, Scalar> FieldOps for PackedPrimitiveType<U, Scalar>
where
    Self: Square + InvertOrZero + Mul<Output = Self>,
    U: UnderlierWithBitOps + Divisible<Scalar::Underlier>,
    Scalar: BinaryField,
{
    type Scalar = Scalar;

    #[inline]
    fn zero() -> Self {
        Self::from_underlier(U::ZERO)
    }

    #[inline]
    fn one() -> Self {
        Self::broadcast(Scalar::ONE)
    }

    fn square_transpose<FSub: Field>(elems: &mut [Self])
    where
        Scalar: ExtensionField<FSub>,
    {
        let log_degree = <Scalar as ExtensionField<FSub>>::LOG_DEGREE;
        let degree = <Scalar as ExtensionField<FSub>>::DEGREE;
        assert_eq!(elems.len(), degree);

        let log_sub_bits = Scalar::N_BITS.ilog2() as usize - log_degree;

        // See Hacker's Delight, Section 7-3.
        for i in 0..log_degree {
            for j in 0..1 << (log_degree - i - 1) {
                for k in 0..1 << i {
                    let idx0 = (j << (i + 1)) | k;
                    let idx1 = idx0 | (1 << i);
                    let (u0, u1) = elems[idx0].0.interleave(elems[idx1].0, i + log_sub_bits);
                    elems[idx0] = u0.into();
                    elems[idx1] = u1.into();
                }
            }
        }
    }
}

impl<U, Scalar> PackedField for PackedPrimitiveType<U, Scalar>
where
    Self: Square + InvertOrZero + Mul<Output = Self>,
    U: UnderlierWithBitOps + Divisible<Scalar::Underlier>,
    Scalar: BinaryField,
{
    const LOG_WIDTH: usize = (U::BITS / Scalar::N_BITS).ilog2() as usize;

    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> Self::Scalar {
        Scalar::from_underlier(unsafe { self.0.get_subvalue(i) })
    }

    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, scalar: Scalar) {
        unsafe {
            self.0.set_subvalue(i, scalar.to_underlier());
        }
    }

    #[inline]
    fn iter(&self) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
        Divisible::<Scalar::Underlier>::ref_iter(&self.0)
            .map(|underlier| Scalar::from_underlier(underlier))
    }

    #[inline]
    fn into_iter(self) -> impl Iterator<Item = Self::Scalar> + Send + Clone {
        Divisible::<Scalar::Underlier>::value_iter(self.0)
            .map(|underlier| Scalar::from_underlier(underlier))
    }

    #[inline]
    fn iter_slice(slice: &[Self]) -> impl Iterator<Item = Self::Scalar> + Send + Clone + '_ {
        Divisible::<Scalar::Underlier>::slice_iter(Self::to_underliers_ref(slice))
            .map_skippable(|underlier| Scalar::from_underlier(underlier))
    }

    #[inline]
    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        assert!(log_block_len < Self::LOG_WIDTH);
        let log_bit_len = Self::Scalar::N_BITS.ilog2() as usize;
        let (c, d) = self.0.interleave(other.0, log_block_len + log_bit_len);
        (c.into(), d.into())
    }

    #[inline]
    fn unzip(self, other: Self, log_block_len: usize) -> (Self, Self) {
        assert!(log_block_len < Self::LOG_WIDTH);
        let log_bit_len = Self::Scalar::N_BITS.ilog2() as usize;
        let (c, d) = self.0.transpose(other.0, log_block_len + log_bit_len);
        (c.into(), d.into())
    }

    #[inline]
    unsafe fn spread_unchecked(self, log_block_len: usize, block_idx: usize) -> Self {
        debug_assert!(
            log_block_len <= Self::LOG_WIDTH,
            "{} <= {}",
            log_block_len,
            Self::LOG_WIDTH
        );
        debug_assert!(
            block_idx < 1 << (Self::LOG_WIDTH - log_block_len),
            "{} < {}",
            block_idx,
            1 << (Self::LOG_WIDTH - log_block_len)
        );

        unsafe {
            self.0
                .spread::<<Self::Scalar as WithUnderlier>::Underlier>(log_block_len, block_idx)
                .into()
        }
    }

    #[inline]
    fn broadcast(scalar: Self::Scalar) -> Self {
        Self::broadcast(scalar)
    }

    #[inline]
    fn from_fn(mut f: impl FnMut(usize) -> Self::Scalar) -> Self {
        U::from_fn(move |i| f(i).to_underlier()).into()
    }
}

impl<U: UnderlierType, Scalar: BinaryField> Distribution<PackedPrimitiveType<U, Scalar>>
    for StandardUniform
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedPrimitiveType<U, Scalar> {
        PackedPrimitiveType::from_underlier(U::random(rng))
    }
}

/// Multiply `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD multiplication is faster.
#[allow(dead_code)]
pub fn mul_as_bigger_type<PT1, PT2>(lhs: PT1, rhs: PT1) -> PT1
where
    PT1: PackedField + WithUnderlier,
    PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier,
    PT2::Underlier: From<PT1::Underlier>,
    PT1::Underlier: NumCast<PT2::Underlier>,
{
    let bigger_lhs = PT2::from_underlier(lhs.to_underlier().into());
    let bigger_rhs = PT2::from_underlier(rhs.to_underlier().into());

    let bigger_result = bigger_lhs * bigger_rhs;

    PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
}

/// Square `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD square is faster.
#[allow(dead_code)]
pub fn square_as_bigger_type<PT1, PT2>(val: PT1) -> PT1
where
    PT1: PackedField + WithUnderlier,
    PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier,
    PT2::Underlier: From<PT1::Underlier>,
    PT1::Underlier: NumCast<PT2::Underlier>,
{
    let bigger_val = PT2::from_underlier(val.to_underlier().into());

    let bigger_result = bigger_val.square();

    PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
}

/// Invert `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD invert is faster.
#[allow(dead_code)]
pub fn invert_as_bigger_type<PT1, PT2>(val: PT1) -> PT1
where
    PT1: PackedField + WithUnderlier,
    PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier,
    PT2::Underlier: From<PT1::Underlier>,
    PT1::Underlier: NumCast<PT2::Underlier>,
{
    let bigger_val = PT2::from_underlier(val.to_underlier().into());

    let bigger_result = bigger_val.invert_or_zero();

    PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
}

/// Multiply by alpha `PT1` values by upcasting to wider `PT2` type with the same scalar.
/// This is useful for the cases when SIMD multiply by alpha is faster.
#[allow(dead_code)]
pub fn mul_alpha_as_bigger_type<PT1, PT2>(val: PT1) -> PT1
where
    PT1: PackedField + WithUnderlier,
    PT2: PackedField<Scalar = PT1::Scalar> + WithUnderlier + MulAlpha,
    PT2::Underlier: From<PT1::Underlier>,
    PT1::Underlier: NumCast<PT2::Underlier>,
{
    let bigger_val = PT2::from_underlier(val.to_underlier().into());

    let bigger_result = bigger_val.mul_alpha();

    PT1::from_underlier(PT1::Underlier::num_cast_from(bigger_result.to_underlier()))
}
