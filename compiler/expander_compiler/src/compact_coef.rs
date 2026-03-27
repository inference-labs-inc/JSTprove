use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::Field;

#[derive(Clone, Copy)]
pub enum CompactCoef<F: Field> {
    Small(i64),
    Full(F),
}

impl<F: Field> CompactCoef<F> {
    #[inline(always)]
    pub fn zero() -> Self {
        CompactCoef::Small(0)
    }

    #[inline(always)]
    pub fn one() -> Self {
        CompactCoef::Small(1)
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        match self {
            CompactCoef::Small(v) => *v == 0,
            CompactCoef::Full(f) => f.is_zero(),
        }
    }

    #[inline(always)]
    pub fn is_one(&self) -> bool {
        match self {
            CompactCoef::Small(1) => true,
            CompactCoef::Small(_) => false,
            CompactCoef::Full(f) => *f == F::ONE,
        }
    }

    #[inline]
    pub fn to_field(self) -> F {
        match self {
            CompactCoef::Small(v) => small_to_field::<F>(v),
            CompactCoef::Full(f) => f,
        }
    }

    #[inline(always)]
    pub fn from_field(f: F) -> Self {
        CompactCoef::Full(f)
    }

    #[inline]
    pub fn optimistic_inv(&self) -> Option<Self> {
        match self {
            CompactCoef::Small(0) => None,
            CompactCoef::Small(1) => Some(CompactCoef::Small(1)),
            _ => self.to_field().inv().map(CompactCoef::Full),
        }
    }

    #[inline(always)]
    pub fn double(&self) -> Self {
        match self {
            CompactCoef::Small(v) => match v.checked_mul(2) {
                Some(r) => CompactCoef::Small(r),
                None => CompactCoef::Full(self.to_field().double()),
            },
            CompactCoef::Full(f) => CompactCoef::Full(f.double()),
        }
    }

    pub fn to_u256(&self) -> ethnum::U256 {
        self.to_field().to_u256()
    }
}

#[inline]
fn small_to_field<F: Field>(v: i64) -> F {
    if v >= 0 && v <= u32::MAX as i64 {
        F::from(v as u32)
    } else if v >= 0 {
        F::from_u256(ethnum::U256::from(v as u64))
    } else if v == i64::MIN {
        -F::from_u256(ethnum::U256::from(i64::MAX as u64 + 1))
    } else {
        -small_to_field::<F>(-v)
    }
}

impl<F: Field> From<F> for CompactCoef<F> {
    #[inline(always)]
    fn from(f: F) -> Self {
        CompactCoef::Full(f)
    }
}

impl<F: Field> fmt::Debug for CompactCoef<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CompactCoef::Small(v) => write!(f, "Small({v})"),
            CompactCoef::Full(fld) => write!(f, "Full({fld:?})"),
        }
    }
}

impl<F: Field> Hash for CompactCoef<F> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_field().hash(state);
    }
}

impl<F: Field> PartialEq for CompactCoef<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CompactCoef::Small(a), CompactCoef::Small(b)) => a == b,
            (CompactCoef::Full(a), CompactCoef::Full(b)) => a == b,
            _ => self.to_field() == other.to_field(),
        }
    }
}

impl<F: Field> Eq for CompactCoef<F> {}

impl<F: Field> PartialOrd for CompactCoef<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Field> Ord for CompactCoef<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_field().cmp(&other.to_field())
    }
}

impl<F: Field> Default for CompactCoef<F> {
    #[inline(always)]
    fn default() -> Self {
        CompactCoef::Small(0)
    }
}

impl<F: Field> Neg for CompactCoef<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        match self {
            CompactCoef::Small(0) => CompactCoef::Small(0),
            CompactCoef::Small(v) => match v.checked_neg() {
                Some(r) => CompactCoef::Small(r),
                None => CompactCoef::Full(-self.to_field()),
            },
            CompactCoef::Full(f) => CompactCoef::Full(-f),
        }
    }
}

impl<F: Field> Add for CompactCoef<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (CompactCoef::Small(a), CompactCoef::Small(b)) => match a.checked_add(b) {
                Some(r) => CompactCoef::Small(r),
                None => CompactCoef::Full(self.to_field() + rhs.to_field()),
            },
            _ => CompactCoef::Full(self.to_field() + rhs.to_field()),
        }
    }
}

impl<F: Field> AddAssign for CompactCoef<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field> Sub for CompactCoef<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            (CompactCoef::Small(a), CompactCoef::Small(b)) => match a.checked_sub(b) {
                Some(r) => CompactCoef::Small(r),
                None => CompactCoef::Full(self.to_field() - rhs.to_field()),
            },
            _ => CompactCoef::Full(self.to_field() - rhs.to_field()),
        }
    }
}

impl<F: Field> SubAssign for CompactCoef<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F: Field> Mul for CompactCoef<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (CompactCoef::Small(0), _) | (_, CompactCoef::Small(0)) => CompactCoef::Small(0),
            (CompactCoef::Small(1), other) | (other, CompactCoef::Small(1)) => other,
            (CompactCoef::Small(-1), other) | (other, CompactCoef::Small(-1)) => -other,
            (CompactCoef::Small(a), CompactCoef::Small(b)) => match a.checked_mul(b) {
                Some(r) => CompactCoef::Small(r),
                None => CompactCoef::Full(self.to_field() * rhs.to_field()),
            },
            _ => CompactCoef::Full(self.to_field() * rhs.to_field()),
        }
    }
}

impl<F: Field> MulAssign for CompactCoef<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
