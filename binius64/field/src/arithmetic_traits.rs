// Copyright 2024-2025 Irreducible Inc.

/// Value that can be multiplied by itself
pub trait Square {
    /// Returns the value multiplied by itself
    fn square(self) -> Self;
}

/// Value that can be inverted
pub trait InvertOrZero {
    /// Returns the inverted value or zero in case when `self` is zero
    fn invert_or_zero(self) -> Self;
}

/// Value that can be multiplied by alpha
pub trait MulAlpha {
    /// Multiply self by alpha
    fn mul_alpha(self) -> Self;
}

/// Multiplication that is parameterized with some some strategy.
pub trait TaggedMul<Strategy> {
    fn mul(self, rhs: Self) -> Self;
}

macro_rules! impl_mul_with {
    ($name:ident @ $strategy:ty) => {
        impl std::ops::Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                $crate::tracing::trace_multiplication!($name);

                $crate::arithmetic_traits::TaggedMul::<$strategy>::mul(self, rhs)
            }
        }
    };
    ($name:ty => $bigger:ty) => {
        impl std::ops::Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self {
                $crate::arch::portable::packed::mul_as_bigger_type::<_, $bigger>(self, rhs)
            }
        }
    };
}

pub(crate) use impl_mul_with;

/// Square operation that is parameterized with some some strategy.
pub trait TaggedSquare<Strategy> {
    fn square(self) -> Self;
}

macro_rules! impl_square_with {
    ($name:ident @ $strategy:ty) => {
        impl $crate::arithmetic_traits::Square for $name {
            #[inline]
            fn square(self) -> Self {
                $crate::arithmetic_traits::TaggedSquare::<$strategy>::square(self)
            }
        }
    };
    ($name:ty => $bigger:ty) => {
        impl $crate::arithmetic_traits::Square for $name {
            #[inline]
            fn square(self) -> Self {
                $crate::arch::portable::packed::square_as_bigger_type::<_, $bigger>(self)
            }
        }
    };
}

pub(crate) use impl_square_with;

/// Invert or zero operation that is parameterized with some some strategy.
pub trait TaggedInvertOrZero<Strategy> {
    fn invert_or_zero(self) -> Self;
}

macro_rules! impl_invert_with {
    ($name:ident @ $strategy:ty) => {
        impl $crate::arithmetic_traits::InvertOrZero for $name {
            #[inline]
            fn invert_or_zero(self) -> Self {
                $crate::arithmetic_traits::TaggedInvertOrZero::<$strategy>::invert_or_zero(self)
            }
        }
    };
    ($name:ty => $bigger:ty) => {
        impl $crate::arithmetic_traits::InvertOrZero for $name {
            #[inline]
            fn invert_or_zero(self) -> Self {
                $crate::arch::portable::packed::invert_as_bigger_type::<_, $bigger>(self)
            }
        }
    };
}

pub(crate) use impl_invert_with;

/// Multiply by alpha operation that is parameterized with some some strategy.
pub trait TaggedMulAlpha<Strategy> {
    fn mul_alpha(self) -> Self;
}

macro_rules! impl_mul_alpha_with {
    ($name:ident @ $strategy:ty) => {
        impl $crate::arithmetic_traits::MulAlpha for $name {
            #[inline]
            fn mul_alpha(self) -> Self {
                $crate::arithmetic_traits::TaggedMulAlpha::<$strategy>::mul_alpha(self)
            }
        }
    };
    ($name:ty => $bigger:ty) => {
        impl $crate::arithmetic_traits::MulAlpha for $name {
            #[inline]
            fn mul_alpha(self) -> Self {
                $crate::arch::portable::packed::mul_alpha_as_bigger_type::<_, $bigger>(self)
            }
        }
    };
}

pub(crate) use impl_mul_alpha_with;
