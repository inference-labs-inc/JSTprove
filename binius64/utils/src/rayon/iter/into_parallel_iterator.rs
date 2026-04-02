// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

use std::collections::HashSet;

use super::{ParallelIterator, ParallelWrapper};

pub trait IntoParallelIterator {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item;

    fn into_par_iter(self) -> Self::Iter;
}

impl<I: ParallelIterator> IntoParallelIterator for I {
    type Iter = I;
    type Item = I::Item;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        self
    }
}

impl<T, const N: usize> IntoParallelIterator for [T; N] {
    type Item = T;
    type Iter = ParallelWrapper<<[T; N] as IntoIterator>::IntoIter>;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        ParallelWrapper::new(self.into_iter())
    }
}

impl<T> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Iter = ParallelWrapper<<Self as IntoIterator>::IntoIter>;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        ParallelWrapper::new(self.into_iter())
    }
}

impl<T> IntoParallelIterator for HashSet<T> {
    type Item = T;
    type Iter = ParallelWrapper<<Self as IntoIterator>::IntoIter>;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        ParallelWrapper::new(self.into_iter())
    }
}

impl<'a, T> IntoParallelIterator for &'a [T] {
    type Item = &'a T;
    type Iter = ParallelWrapper<<std::slice::Iter<'a, T> as IntoIterator>::IntoIter>;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        ParallelWrapper::new(self.iter())
    }
}

impl<'a, T> IntoParallelIterator for &'a mut [T] {
    type Item = &'a mut T;
    type Iter = ParallelWrapper<<std::slice::IterMut<'a, T> as IntoIterator>::IntoIter>;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        ParallelWrapper::new(self.iter_mut())
    }
}

impl<Idx> IntoParallelIterator for std::ops::Range<Idx>
where
    Self: Iterator<Item = Idx>,
{
    type Item = Idx;
    type Iter = ParallelWrapper<Self>;

    #[inline(always)]
    fn into_par_iter(self) -> Self::Iter {
        ParallelWrapper::new(self)
    }
}

pub trait IntoParallelRefIterator<'data> {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item: 'data;

    // Required method
    fn par_iter(&'data self) -> Self::Iter;
}

impl<'a, T> IntoParallelRefIterator<'a> for &'a [T] {
    type Iter = ParallelWrapper<std::slice::Iter<'a, T>>;
    type Item = &'a T;

    #[inline(always)]
    fn par_iter(&'a self) -> Self::Iter {
        ParallelWrapper::new(self.iter())
    }
}

impl<'data, I: 'data + ?Sized> IntoParallelRefIterator<'data> for I
where
    &'data I: IntoParallelIterator,
{
    type Iter = <&'data I as IntoParallelIterator>::Iter;
    type Item = <&'data I as IntoParallelIterator>::Item;

    #[inline(always)]
    fn par_iter(&'data self) -> Self::Iter {
        self.into_par_iter()
    }
}

pub trait IntoParallelRefMutIterator<'data> {
    type Iter: ParallelIterator<Item = Self::Item>;
    type Item: 'data;

    fn par_iter_mut(&'data mut self) -> Self::Iter;
}

impl<'data, I: 'data + ?Sized> IntoParallelRefMutIterator<'data> for I
where
    &'data mut I: IntoParallelIterator,
{
    type Iter = <&'data mut I as IntoParallelIterator>::Iter;
    type Item = <&'data mut I as IntoParallelIterator>::Item;

    #[inline(always)]
    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.into_par_iter()
    }
}

macro_rules! multizip_impls {
    ($(
        $Tuple:ident {
            $(($idx:tt) -> $T:ident)+
        }
    )+) => {
        $(
            impl<$( $T, )+> IntoParallelIterator for ($( $T, )+)
            where
                $(
                    $T: IntoParallelIterator,
                    $T::Iter: crate::rayon::prelude::IndexedParallelIterator,
                )+
            {
                type Item = ($( $T::Item, )+);
                type Iter = ParallelWrapper<itertools::Zip<($( <$T::Iter as crate::rayon::prelude::ParallelIterator>::Inner, )+)>>;

                #[inline(always)]
                fn into_par_iter(self) -> Self::Iter {
                    ParallelWrapper::new(itertools::multizip(( $( crate::rayon::prelude::ParallelIterator::into_inner(self.$idx.into_par_iter()), )+ )))
                }
            }

            impl<'a, $( $T, )+> IntoParallelIterator for &'a ($( $T, )+)
            where
                $(
                    $T: IntoParallelRefIterator<'a>,
                    $T::Iter: crate::rayon::prelude::IndexedParallelIterator,
                )+
            {
                type Item = ($( $T::Item, )+);
                type Iter = ParallelWrapper<itertools::Zip<($( <$T::Iter as crate::rayon::prelude::ParallelIterator>::Inner, )+)>>;

                #[inline(always)]
                fn into_par_iter(self) -> Self::Iter {
                    ParallelWrapper::new(itertools::multizip(( $( crate::rayon::prelude::ParallelIterator::into_inner(self.$idx.par_iter()), )+ )))
                }
            }

            impl<'a, $( $T, )+> IntoParallelIterator for &'a mut ($( $T, )+)
            where
                $(
                    $T: IntoParallelRefMutIterator<'a>,
                    $T::Iter: crate::rayon::prelude::IndexedParallelIterator,
                )+
            {
                type Item = ($( $T::Item, )+);
                type Iter = ParallelWrapper<itertools::Zip<($( <$T::Iter as crate::rayon::prelude::ParallelIterator>::Inner, )+)>>;

                #[inline(always)]
                fn into_par_iter(self) -> Self::Iter {
                    ParallelWrapper::new(itertools::multizip(( $( crate::rayon::prelude::ParallelIterator::into_inner(self.$idx.par_iter_mut()), )+ )))
                }
            }

            impl<$( $T, )+> crate::rayon::prelude::IndexedParallelIteratorInner for itertools::Zip<($( $T, )+)>
            where
                $( $T: crate::rayon::prelude::IndexedParallelIteratorInner, )+
            {
            }
        )+
    }
}

multizip_impls! {
    Tuple1 {
        (0) -> A
    }
    Tuple2 {
        (0) -> A
        (1) -> B
    }
    Tuple3 {
        (0) -> A
        (1) -> B
        (2) -> C
    }
    Tuple4 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
    }
    Tuple5 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
    }
    Tuple6 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
    }
    Tuple7 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
    }
    Tuple8 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
    }
    Tuple9 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
    }
    Tuple10 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
    }
    Tuple11 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
        (10) -> K
    }
    Tuple12 {
        (0) -> A
        (1) -> B
        (2) -> C
        (3) -> D
        (4) -> E
        (5) -> F
        (6) -> G
        (7) -> H
        (8) -> I
        (9) -> J
        (10) -> K
        (11) -> L
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterate_tuple() {
        let a: &[usize] = &[1, 2, 3];
        let b: &[usize] = &[4, 5, 6];
        let c: &[usize] = &[7, 8, 9];

        let result = (a, b, c).into_par_iter().collect::<Vec<_>>();
        assert_eq!(result, vec![(&1, &4, &7), (&2, &5, &8), (&3, &6, &9)]);
    }

    #[test]
    fn iterate_tuple_mut() {
        let a: &mut [usize] = &mut [1, 2, 3];
        let b: &mut [usize] = &mut [4, 5, 6];
        let c: &mut [usize] = &mut [7, 8, 9];

        let result = (a, b, c).into_par_iter().collect::<Vec<_>>();
        assert_eq!(
            result,
            vec![
                (&mut 1, &mut 4, &mut 7),
                (&mut 2, &mut 5, &mut 8),
                (&mut 3, &mut 6, &mut 9)
            ]
        );
    }
}
