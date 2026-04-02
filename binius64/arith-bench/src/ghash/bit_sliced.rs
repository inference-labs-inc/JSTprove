// Copyright 2025 Irreducible Inc.
use std::array;

use crate::Underlier;

/// Straightforward bit-sliced multiplication for 128-bit elements.
#[inline]
pub fn mul_naive<U>(x: [U; 128], y: [U; 128]) -> [U; 128]
where
    U: Underlier,
{
    let mut result = [U::zero(); 256];
    cmul_naive::<U, Level256>(&x, &y, &mut result);

    reduce_bit_sliced(&mut result);

    *<&[U; 128]>::try_from(&result[0..128]).expect("slice is of the correct size")
}

/// Bit-sliced Karatsuba multiplication for 128-bit elements.
/// Use Katasuba reduction up to degree 4, where the naive algorithm is used.
#[inline]
pub fn mul_katatsuba<U>(x: [U; 128], y: [U; 128]) -> [U; 128]
where
    U: Underlier,
{
    let mut result = [U::zero(); 256];
    cmul_karatsuba_impl::<U, Level256>(&x, &y, &mut result);

    reduce_bit_sliced(&mut result);

    *<&[U; 128]>::try_from(&result[0..128]).expect("slice is of the correct size")
}

/// Reduces the result of the bit-sliced caryless multiplication
#[inline]
fn reduce_bit_sliced<U: Underlier>(value: &mut [U; 256]) {
    for i in (0..128).rev() {
        let current = value[i + 128];

        U::xor_assign(&mut value[i], current);
        U::xor_assign(&mut value[i + 1], current);
        U::xor_assign(&mut value[i + 2], current);
        U::xor_assign(&mut value[i + 7], current);
    }
}

/// Trait representing a level to be able to split the arrays and use the recursion for Karatsuba
/// multiplication.
trait Level {
    const N: usize;

    type Data<U>: AsRef<[U]>;
    type DataMut<U>: AsMut<[U]>;
    type Prev: Level;

    fn split<U>(data: &Self::Data<U>) -> (&PrevLevelData<U, Self>, &PrevLevelData<U, Self>)
    where
        U: Underlier;

    fn split_mut<U>(
        data: &mut Self::DataMut<U>,
    ) -> (
        &mut PrevLevelDataMut<U, Self>,
        &mut PrevLevelDataMut<U, Self>,
    )
    where
        U: Underlier;

    fn default<U>() -> Self::DataMut<U>
    where
        U: Underlier;

    fn xor<U>(x: &Self::Data<U>, y: &Self::Data<U>) -> Self::Data<U>
    where
        U: Underlier;
}

type PrevLevelData<U, L> = <<L as Level>::Prev as Level>::Data<U>;
type PrevLevelDataMut<U, L> = <<L as Level>::Prev as Level>::DataMut<U>;

struct Level1;

impl Level for Level1 {
    const N: usize = 1;

    type Data<U> = [U; Self::N];
    type DataMut<U> = [U; Self::N];
    type Prev = Self;

    #[inline]
    fn split<U>(_data: &Self::Data<U>) -> (&PrevLevelData<U, Self>, &PrevLevelData<U, Self>)
    where
        U: Underlier,
    {
        panic!("Level1 cannot be split");
    }

    #[inline]
    fn split_mut<U>(
        _data: &mut Self::DataMut<U>,
    ) -> (
        &mut PrevLevelDataMut<U, Self>,
        &mut PrevLevelDataMut<U, Self>,
    )
    where
        U: Underlier,
    {
        panic!("Level1 cannot be split");
    }

    #[inline]
    fn default<U>() -> Self::DataMut<U>
    where
        U: Underlier,
    {
        [U::zero(); Self::N]
    }

    #[inline]
    fn xor<U>(x: &Self::Data<U>, y: &Self::Data<U>) -> Self::Data<U>
    where
        U: Underlier,
    {
        array::from_fn(|i| U::xor(x[i], y[i]))
    }
}

macro_rules! define_level {
    ($n:literal, $current:ident, $prev:ident) => {
        struct $current;

        impl Level for $current {
            const N: usize = $n;

            type Data<U> = [U; Self::N];
            type DataMut<U> = [U; Self::N];
            type Prev = $prev;

            #[inline]
            fn split<U>(
                data: &Self::Data<U>,
            ) -> (
                &<Self::Prev as Level>::Data<U>,
                &<Self::Prev as Level>::Data<U>,
            )
            where
                U: Underlier,
            {
                let (lhs, rhs) = data.split_at(Self::N / 2);
                (lhs.try_into().unwrap(), rhs.try_into().unwrap())
            }

            #[inline]
            fn split_mut<U>(
                data: &mut Self::DataMut<U>,
            ) -> (
                &mut <Self::Prev as Level>::DataMut<U>,
                &mut <Self::Prev as Level>::DataMut<U>,
            )
            where
                U: Underlier,
            {
                let (lhs, rhs) = data.split_at_mut(Self::N / 2);
                (lhs.try_into().unwrap(), rhs.try_into().unwrap())
            }

            #[inline]
            fn default<U>() -> Self::DataMut<U>
            where
                U: Underlier,
            {
                [U::zero(); Self::N]
            }

            #[inline]
            fn xor<U>(x: &Self::Data<U>, y: &Self::Data<U>) -> Self::Data<U>
            where
                U: Underlier,
            {
                array::from_fn(|i| U::xor(x[i], y[i]))
            }
        }
    };
}

define_level!(2, Level2, Level1);
define_level!(4, Level4, Level2);
define_level!(8, Level8, Level4);
define_level!(16, Level16, Level8);
define_level!(32, Level32, Level16);
define_level!(64, Level64, Level32);
define_level!(128, Level128, Level64);
define_level!(256, Level256, Level128);

#[inline]
fn cmul_karatsuba_impl<U, L>(
    x: &<<L as Level>::Prev as Level>::Data<U>,
    y: &<<L as Level>::Prev as Level>::Data<U>,
    out: &mut <L as Level>::DataMut<U>,
) where
    U: Underlier,
    L: Level,
{
    if L::N == 8 {
        return cmul_naive::<U, L>(x, y, out);
    }

    let (x0, x1) = L::Prev::split::<U>(x);
    let (y0, y1) = L::Prev::split::<U>(y);
    let (out0, out1) = L::split_mut(out);

    cmul_karatsuba_impl::<U, L::Prev>(x0, y0, out0);
    cmul_karatsuba_impl::<U, L::Prev>(x1, y1, out1);

    let x_0_xor_x_1 = <<L::Prev as Level>::Prev as Level>::xor(x0, x1);
    let y_0_xor_y_1 = <<L::Prev as Level>::Prev as Level>::xor(y0, y1);
    let mut tmp = L::Prev::default::<U>();
    cmul_karatsuba_impl::<U, L::Prev>(&x_0_xor_x_1, &y_0_xor_y_1, &mut tmp);
    for i in 0..L::Prev::N {
        U::xor_assign(&mut tmp.as_mut()[i], out0.as_mut()[i]);
        U::xor_assign(&mut tmp.as_mut()[i], out1.as_mut()[i]);
    }

    for i in 0..L::Prev::N {
        U::xor_assign(&mut out.as_mut()[i + L::Prev::N / 2], tmp.as_mut()[i]);
    }
}

#[inline]
fn cmul_naive<U, L>(
    x: &<<L as Level>::Prev as Level>::Data<U>,
    y: &<<L as Level>::Prev as Level>::Data<U>,
    out: &mut <L as Level>::DataMut<U>,
) where
    U: Underlier,
    L: Level,
{
    for (i, x) in x.as_ref().iter().copied().enumerate() {
        for (j, y) in y.as_ref().iter().copied().enumerate() {
            U::xor_assign(&mut out.as_mut()[i + j], U::and(x, y));
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const ONE_U64: [u64; 128] = {
        let mut result = [0; 128];
        result[0] = u64::MAX;
        result
    };

    proptest! {
        #[test]
        fn test_same_result_karatsuba_naive(
            x in any::<[u64; 128]>(),
            y in any::<[u64; 128]>()
        ) {
            let result_karatsuba = mul_katatsuba(x, y);
            let result_naive = mul_naive(x, y);
            assert_eq!(result_karatsuba, result_naive);
        }

        #[test]
        fn test_mul_commutative(x in any::<[u64; 128]>(), y in any::<[u64; 128]>()) {
            // Test that a * b = b * a
            let xy = mul_naive(x, y);
            let yx = mul_naive(y, x);
            assert_eq!(xy, yx);
        }

        #[test]
        fn test_mul_identity(a in any::<[u64; 128]>()) {
            // Test that a * ONE_U64 = a
            let ab = mul_naive(a, ONE_U64);
            assert_eq!(ab, a);
        }
    }
}
