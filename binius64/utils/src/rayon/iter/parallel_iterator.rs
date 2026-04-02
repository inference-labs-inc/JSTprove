// Copyright 2025 Irreducible Inc.
// The code is initially based on `maybe-rayon` crate, https://github.com/shssoichiro/maybe-rayon
// Original: Copyright (c) 2021 Joshua Holmer
// Licensed under MIT License

use std::{
    cmp::Ordering,
    collections::LinkedList,
    iter::{Product, Sum},
    ops::{ControlFlow, Deref},
};

use itertools::Either;

use self::private::Try;
use super::{FromParallelIterator, IntoParallelIterator, parallel_wrapper::ParallelWrapper};
use crate::rayon::iter::indexed_parallel_iterator::Chain;

/// `rayon::prelude::ParallelIterator` has at least `fold` method with a signature
/// that is not compatible with the one in `std::iter::Iterator`. That's why we can't use
/// `std::iter::Iterator` as a drop-in replacement for `rayon::prelude::ParallelIterator`
/// (like it is in the original `maybe-rayon` crate) or even derive from it (this would raise
/// ambiguous methods error at the place of usages allowing fully-qualified call syntax only which
/// looks super-ugly).
/// That's why we have this trait and implement `ParallelIterator` for `ParallelWrapper<I:
/// ParallelIteratorInner>`.
pub(crate) trait ParallelIteratorInner: Sized + Iterator {
    #[inline]
    fn for_each<OP>(self, op: OP)
    where
        OP: Fn(Self::Item),
    {
        Iterator::for_each(self, op)
    }

    #[inline]
    fn for_each_with<OP, T>(self, mut init: T, op: OP)
    where
        OP: Fn(&mut T, Self::Item),
    {
        Iterator::for_each(self, |item| op(&mut init, item));
    }

    #[inline]
    fn for_each_init<OP, INIT, T>(self, init: INIT, op: OP)
    where
        OP: Fn(&mut T, Self::Item),
        INIT: Fn() -> T,
    {
        let mut init = init();
        Iterator::for_each(self, |item| op(&mut init, item));
    }

    #[inline]
    fn try_for_each<OP, R>(self, op: OP) -> R
    where
        OP: Fn(Self::Item) -> R,
        R: Try<Output = ()>,
    {
        for v in self {
            let result = op(v);
            if let ControlFlow::Break(r) = result.branch() {
                return R::from_residual(r);
            }
        }

        Try::from_output(())
    }

    #[inline]
    fn try_for_each_with<OP, T, R>(self, mut init: T, op: OP) -> R
    where
        OP: Fn(&mut T, Self::Item) -> R,
        R: Try<Output = ()>,
    {
        for v in self {
            let result = op(&mut init, v);
            if let ControlFlow::Break(r) = result.branch() {
                return R::from_residual(r);
            }
        }

        Try::from_output(())
    }

    #[inline]
    fn try_for_each_init<OP, INIT, T, R>(self, init: INIT, op: OP) -> R
    where
        OP: Fn(&mut T, Self::Item) -> R,
        INIT: Fn() -> T,
        R: Try<Output = ()>,
    {
        self.try_for_each_with(init(), op)
    }

    #[inline]
    fn count(self) -> usize {
        Iterator::count(self)
    }

    #[inline]
    fn map<F, R>(self, map_op: F) -> std::iter::Map<Self, F>
    where
        F: Fn(Self::Item) -> R,
    {
        Iterator::map(self, map_op)
    }

    #[inline]
    fn map_with<F, T, R>(
        self,
        mut init: T,
        map_op: F,
    ) -> std::iter::Map<Self, impl FnMut(Self::Item) -> R>
    where
        F: Fn(&mut T, Self::Item) -> R,
    {
        Iterator::map(self, move |item| map_op(&mut init, item))
    }

    #[inline]
    fn map_init<F, INIT, T, R>(
        self,
        init: INIT,
        map_op: F,
    ) -> std::iter::Map<Self, impl FnMut(Self::Item) -> R>
    where
        F: Fn(&mut T, Self::Item) -> R,
        INIT: Fn() -> T,
    {
        self.map_with(init(), map_op)
    }

    #[inline]
    fn cloned<'a, T>(self) -> impl ParallelIteratorInner<Item = T>
    where
        T: 'a + Clone,
        Self::Item: Deref<Target = T>,
    {
        Iterator::map(self, |x| x.clone())
    }

    #[inline]
    fn copied<'a, T>(self) -> impl ParallelIteratorInner<Item = T>
    where
        T: 'a + Copy,
        Self::Item: Deref<Target = T>,
    {
        Iterator::map(self, |x: <Self as Iterator>::Item| *x)
    }

    #[inline]
    fn inspect<OP>(self, inspect_op: OP) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        OP: Fn(&Self::Item),
    {
        Iterator::inspect(self, inspect_op)
    }

    #[inline]
    fn update<F>(self, update_op: F) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        F: Fn(&mut Self::Item),
    {
        ParallelIteratorInner::map(self, move |mut item| {
            update_op(&mut item);
            item
        })
    }

    #[inline]
    fn filter<P>(self, filter_op: P) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        Iterator::filter(self, filter_op)
    }

    #[inline]
    fn filter_map<P, R>(self, filter_op: P) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        P: Fn(Self::Item) -> Option<Self::Item>,
    {
        Iterator::filter_map(self, filter_op)
    }

    #[inline]
    fn flat_map<F, PI>(self, map_op: F) -> impl ParallelIteratorInner<Item = PI::Item>
    where
        F: Fn(Self::Item) -> PI,
        PI: ParallelIteratorInner,
    {
        Iterator::flat_map(self, map_op)
    }

    #[inline]
    fn flat_map_iter<F, SI>(self, map_op: F) -> impl ParallelIteratorInner<Item = SI::Item>
    where
        F: Fn(Self::Item) -> SI,
        SI: ParallelIteratorInner,
    {
        ParallelIteratorInner::flat_map(self, map_op)
    }

    #[inline]
    fn flatten(
        self,
    ) -> impl ParallelIteratorInner<Item = <<Self::Item as IntoIterator>::IntoIter as Iterator>::Item>
    where
        Self::Item: IntoIterator,
    {
        Iterator::flatten(self)
    }

    #[inline]
    fn flatten_iter(
        self,
    ) -> impl ParallelIteratorInner<Item = <<Self::Item as IntoIterator>::IntoIter as Iterator>::Item>
    where
        Self::Item: IntoIterator,
    {
        ParallelIteratorInner::flatten(self)
    }

    #[inline]
    fn reduce<OP, ID>(self, identity: ID, op: OP) -> Self::Item
    where
        OP: Fn(Self::Item, Self::Item) -> Self::Item,
        ID: Fn() -> Self::Item,
    {
        Iterator::reduce(self, op).unwrap_or_else(identity)
    }

    #[inline]
    fn reduce_with<OP>(self, op: OP) -> Option<Self::Item>
    where
        OP: Fn(Self::Item, Self::Item) -> Self::Item,
    {
        Iterator::reduce(self, op)
    }

    #[inline]
    fn try_reduce<T, OP, ID>(self, identity: ID, op: OP) -> Self::Item
    where
        OP: Fn(T, T) -> Self::Item,
        ID: Fn() -> T,
        Self::Item: Try<Output = T>,
    {
        self.try_reduce_with(op)
            .unwrap_or_else(|| Self::Item::from_output(identity()))
    }

    /// Since `std::iterator::Iterator::try_reduce` is not stable yet, we have to implement it
    /// manually.
    #[inline]
    fn try_reduce_with<T, OP>(self, op: OP) -> Option<Self::Item>
    where
        OP: Fn(T, T) -> Self::Item,
        Self::Item: Try<Output = T>,
    {
        let mut iterator = self.into_iter();
        match iterator.next() {
            None => None,
            Some(first) => {
                let mut accum = first;

                loop {
                    match accum.branch() {
                        ControlFlow::Break(r) => break Some(Self::Item::from_residual(r)),
                        ControlFlow::Continue(acc) => match iterator.next() {
                            None => break Some(Self::Item::from_output(acc)),
                            Some(next) => match next.branch() {
                                ControlFlow::Break(r) => break Some(Self::Item::from_residual(r)),
                                ControlFlow::Continue(c) => {
                                    accum = op(acc, c);
                                }
                            },
                        },
                    }
                }
            }
        }
    }

    #[inline]
    fn fold<T, ID, F>(self, identity: ID, fold_op: F) -> impl ParallelIteratorInner<Item = T>
    where
        F: Fn(T, Self::Item) -> T,
        ID: Fn() -> T,
    {
        self.fold_with(identity(), fold_op)
    }

    #[inline]
    fn fold_with<F, T>(self, init: T, fold_op: F) -> impl ParallelIteratorInner<Item = T>
    where
        F: Fn(T, Self::Item) -> T,
    {
        std::iter::once(Iterator::fold(self, init, fold_op))
    }

    #[inline]
    fn try_fold<T, R, ID, F>(self, identity: ID, fold_op: F) -> impl ParallelIteratorInner<Item = R>
    where
        F: Fn(T, Self::Item) -> R,
        ID: Fn() -> T,
        R: Try<Output = T>,
    {
        self.try_fold_with(identity(), fold_op)
    }

    #[inline]
    fn try_fold_with<F, T, R>(self, init: T, fold_op: F) -> impl ParallelIteratorInner<Item = R>
    where
        F: Fn(T, Self::Item) -> R,
        R: Try<Output = T>,
    {
        let mut accum = init;
        for item in self {
            accum = match fold_op(accum, item).branch() {
                ControlFlow::Break(r) => return std::iter::once(R::from_residual(r)),
                ControlFlow::Continue(c) => c,
            };
        }

        std::iter::once(R::from_output(accum))
    }

    #[inline]
    fn sum<S>(self) -> S
    where
        S: Sum<Self::Item> + Sum<S>,
    {
        Iterator::sum(self)
    }

    #[inline]
    fn product<P>(self) -> P
    where
        P: Product<Self::Item> + Product<P>,
    {
        Iterator::product(self)
    }

    #[inline]
    fn min(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        Iterator::min(self)
    }

    #[inline]
    fn min_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering,
    {
        Iterator::min_by(self, f)
    }

    #[inline]
    fn min_by_key<K, F>(self, f: F) -> Option<Self::Item>
    where
        K: Ord,
        F: Fn(&Self::Item) -> K,
    {
        Iterator::min_by_key(self, f)
    }

    #[inline]
    fn max(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        Iterator::max(self)
    }

    #[inline]
    fn max_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering,
    {
        Iterator::max_by(self, f)
    }

    #[inline]
    fn max_by_key<K, F>(self, f: F) -> Option<Self::Item>
    where
        K: Ord,
        F: Fn(&Self::Item) -> K,
    {
        Iterator::max_by_key(self, f)
    }

    #[inline]
    fn chain<C>(self, chain: C) -> Chain<Self, C>
    where
        C: ParallelIteratorInner<Item = Self::Item>,
    {
        Chain {
            inner: Iterator::chain(self, chain),
        }
    }

    #[inline]
    fn find_any<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        self.find_first(predicate)
    }

    #[inline]
    fn find_first<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        self.into_iter().find(predicate)
    }

    #[inline]
    fn find_last<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        Iterator::filter(self, predicate).last()
    }

    #[inline]
    fn find_map_any<P, R>(self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R>,
    {
        self.find_map_first(predicate)
    }

    #[inline]
    fn find_map_first<P, R>(self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R>,
    {
        self.into_iter().find_map(predicate)
    }

    #[inline]
    fn find_map_last<P, R>(self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R>,
    {
        Iterator::filter_map(self, predicate).last()
    }

    #[inline]
    fn any<P>(mut self, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool,
    {
        Iterator::any(&mut self, predicate)
    }

    #[inline]
    fn all<P>(mut self, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool,
    {
        Iterator::all(&mut self, predicate)
    }

    #[inline]
    fn while_some<T>(self) -> impl ParallelIteratorInner<Item = T>
    where
        Self: ParallelIteratorInner<Item = Option<T>>,
        T: Send,
    {
        Iterator::map_while(self, |x| x)
    }

    #[inline(always)]
    fn panic_fuse(self) -> impl ParallelIteratorInner<Item = Self::Item> {
        self
    }

    #[inline]
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
    where
        Self: ParallelIteratorInner<Item = (A, B)>,
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
        A: Send,
        B: Send,
    {
        let (left, right): (FromA, FromB) = Iterator::unzip(self);
        (left, right)
    }

    #[inline]
    fn partition<A, B, P>(self, predicate: P) -> (A, B)
    where
        A: Default + Extend<Self::Item>,
        B: Default + Extend<Self::Item>,
        P: Fn(&Self::Item) -> bool,
    {
        let mut left = A::default();
        let mut right = B::default();

        for item in self {
            if predicate(&item) {
                left.extend(std::iter::once(item));
            } else {
                right.extend(std::iter::once(item));
            }
        }

        (left, right)
    }

    #[inline]
    fn partition_map<A, B, P, L, R>(self, predicate: P) -> (A, B)
    where
        A: Default + Extend<L>,
        B: Default + Extend<R>,
        P: Fn(Self::Item) -> Either<L, R>,
    {
        let mut left = A::default();
        let mut right = B::default();

        for item in self {
            match predicate(item) {
                Either::Left(l) => left.extend(std::iter::once(l)),
                Either::Right(r) => right.extend(std::iter::once(r)),
            }
        }

        (left, right)
    }

    #[inline]
    fn intersperse(self, element: Self::Item) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        Self::Item: Clone,
    {
        struct Intersperse<I: Iterator> {
            iter: I,
            element: I::Item,
            insert: bool,
        }

        impl<I> Intersperse<I>
        where
            I: Iterator,
            I::Item: Clone,
        {
            const fn new(iter: I, element: I::Item) -> Self {
                Self {
                    iter,
                    element,
                    insert: false,
                }
            }
        }

        impl<I> Iterator for Intersperse<I>
        where
            I: Iterator,
            I::Item: Clone,
        {
            type Item = I::Item;

            fn next(&mut self) -> Option<Self::Item> {
                if self.insert {
                    self.insert = false;
                    Some(self.element.clone())
                } else {
                    self.insert = true;
                    self.iter.next()
                }
            }
        }

        Intersperse::new(self.into_iter(), element)
    }

    #[inline]
    fn take_any(self, n: usize) -> impl ParallelIteratorInner<Item = Self::Item> {
        Iterator::take(self, n)
    }

    #[inline]
    fn skip_any(self, n: usize) -> impl ParallelIteratorInner<Item = Self::Item> {
        Iterator::skip(self, n)
    }

    #[inline]
    fn take_any_while<P>(self, predicate: P) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        Iterator::take_while(self, predicate)
    }

    #[inline]
    fn skip_any_while<P>(self, predicate: P) -> impl ParallelIteratorInner<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        Iterator::skip_while(self, predicate)
    }

    #[inline]
    fn collect_vec_list(self) -> LinkedList<Vec<Self::Item>> {
        std::iter::once(Iterator::collect(self)).collect()
    }

    #[inline(always)]
    fn opt_len(&self) -> Option<usize> {
        None
    }
}

impl<I: Iterator> ParallelIteratorInner for I {}

#[allow(private_bounds)]
pub trait ParallelIterator: Sized {
    type Inner: ParallelIteratorInner<Item = Self::Item>;
    type Item;

    fn into_inner(self) -> Self::Inner;
    fn as_inner(&self) -> &Self::Inner;

    #[inline]
    fn for_each<OP>(self, op: OP)
    where
        OP: Fn(Self::Item),
    {
        ParallelIteratorInner::for_each(self.into_inner(), op)
    }

    #[inline]
    fn for_each_with<OP, T>(self, init: T, op: OP)
    where
        OP: Fn(&mut T, Self::Item),
    {
        ParallelIteratorInner::for_each_with(self.into_inner(), init, op)
    }

    #[inline]
    fn for_each_init<OP, INIT, T>(self, init: INIT, op: OP)
    where
        OP: Fn(&mut T, Self::Item),
        INIT: Fn() -> T,
    {
        ParallelIteratorInner::for_each_init(self.into_inner(), init, op)
    }

    #[inline]
    fn try_for_each<OP, R>(self, op: OP) -> R
    where
        OP: Fn(Self::Item) -> R,
        R: Try<Output = ()>,
    {
        ParallelIteratorInner::try_for_each(self.into_inner(), op)
    }

    #[inline]
    fn try_for_each_with<OP, T, R>(self, init: T, op: OP) -> R
    where
        OP: Fn(&mut T, Self::Item) -> R,
        R: Try<Output = ()>,
    {
        ParallelIteratorInner::try_for_each_with(self.into_inner(), init, op)
    }

    #[inline]
    fn try_for_each_init<OP, INIT, T, R>(self, init: INIT, op: OP) -> R
    where
        OP: Fn(&mut T, Self::Item) -> R,
        INIT: Fn() -> T,
        R: Try<Output = ()>,
    {
        ParallelIteratorInner::try_for_each_init(self.into_inner(), init, op)
    }

    #[inline]
    fn count(self) -> usize {
        ParallelIteratorInner::count(self.into_inner())
    }

    #[inline]
    fn map<F, R>(self, map_op: F) -> ParallelWrapper<std::iter::Map<Self::Inner, F>>
    where
        F: Fn(Self::Item) -> R,
    {
        ParallelWrapper::new(ParallelIteratorInner::map(self.into_inner(), map_op))
    }

    #[inline]
    fn map_with<F, T, R>(
        self,
        init: T,
        map_op: F,
    ) -> ParallelWrapper<std::iter::Map<Self::Inner, impl FnMut(Self::Item) -> R>>
    where
        F: Fn(&mut T, Self::Item) -> R,
    {
        ParallelWrapper::new(ParallelIteratorInner::map_with(
            self.into_inner(),
            init,
            map_op,
        ))
    }

    #[inline]
    fn map_init<F, INIT, T, R>(
        self,
        init: INIT,
        map_op: F,
    ) -> ParallelWrapper<std::iter::Map<Self::Inner, impl FnMut(Self::Item) -> R>>
    where
        F: Fn(&mut T, Self::Item) -> R,
        INIT: Fn() -> T,
    {
        ParallelWrapper::new(ParallelIteratorInner::map_init(
            self.into_inner(),
            init,
            map_op,
        ))
    }

    #[inline]
    fn cloned<'a, T>(self) -> impl ParallelIterator<Item = T>
    where
        T: 'a + Clone,
        Self::Item: Deref<Target = T>,
    {
        ParallelWrapper::new(ParallelIteratorInner::cloned(self.into_inner()))
    }

    #[inline]
    fn copied<'a, T>(self) -> impl ParallelIterator<Item = T>
    where
        T: 'a + Copy,
        Self::Item: Deref<Target = T>,
    {
        ParallelWrapper::new(ParallelIteratorInner::copied(self.into_inner()))
    }

    #[inline]
    fn inspect<OP>(self, inspect_op: OP) -> impl ParallelIterator<Item = Self::Item>
    where
        OP: Fn(&Self::Item),
    {
        ParallelWrapper::new(ParallelIteratorInner::inspect(
            self.into_inner(),
            inspect_op,
        ))
    }

    #[inline]
    fn update<F>(self, update_op: F) -> impl ParallelIterator<Item = Self::Item>
    where
        F: Fn(&mut Self::Item),
    {
        ParallelWrapper::new(ParallelIteratorInner::update(self.into_inner(), update_op))
    }

    #[inline]
    fn filter<P>(self, filter_op: P) -> impl ParallelIterator<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        ParallelWrapper::new(ParallelIteratorInner::filter(self.into_inner(), filter_op))
    }

    #[inline]
    fn filter_map<P, R>(self, filter_op: P) -> impl ParallelIterator<Item = Self::Item>
    where
        P: Fn(Self::Item) -> Option<Self::Item>,
    {
        ParallelWrapper::new(ParallelIteratorInner::filter_map::<P, R>(
            self.into_inner(),
            filter_op,
        ))
    }

    #[inline]
    fn flat_map<F, PI>(self, map_op: F) -> impl ParallelIterator<Item = PI::Item>
    where
        F: Fn(Self::Item) -> PI,
        PI: IntoParallelIterator,
    {
        ParallelWrapper::new(ParallelIteratorInner::flat_map(
            self.into_inner(),
            move |x| map_op(x).into_par_iter().into_inner(),
        ))
    }

    #[inline]
    fn flat_map_iter<F, SI>(self, map_op: F) -> impl ParallelIterator<Item = SI::Item>
    where
        F: Fn(Self::Item) -> SI,
        SI: IntoIterator,
        SI::Item: Send,
    {
        ParallelWrapper::new(ParallelIteratorInner::flat_map_iter(
            self.into_inner(),
            move |x| map_op(x).into_iter(),
        ))
    }

    #[inline]
    fn flatten(
        self,
    ) -> impl ParallelIterator<Item = <<Self::Item as IntoIterator>::IntoIter as Iterator>::Item>
    where
        Self::Item: IntoIterator,
    {
        ParallelWrapper::new(ParallelIteratorInner::flatten(self.into_inner()))
    }

    #[inline]
    fn flatten_iter(
        self,
    ) -> impl ParallelIterator<Item = <<Self::Item as IntoIterator>::IntoIter as Iterator>::Item>
    where
        Self::Item: IntoIterator,
    {
        ParallelWrapper::new(ParallelIteratorInner::flatten_iter(self.into_inner()))
    }

    #[inline]
    fn reduce<OP, ID>(self, identity: ID, op: OP) -> Self::Item
    where
        OP: Fn(Self::Item, Self::Item) -> Self::Item,
        ID: Fn() -> Self::Item,
    {
        ParallelIteratorInner::reduce(self.into_inner(), identity, op)
    }

    #[inline]
    fn reduce_with<OP>(self, op: OP) -> Option<Self::Item>
    where
        OP: Fn(Self::Item, Self::Item) -> Self::Item,
    {
        ParallelIteratorInner::reduce_with(self.into_inner(), op)
    }

    #[inline]
    fn try_reduce<T, OP, ID>(self, identity: ID, op: OP) -> Self::Item
    where
        OP: Fn(T, T) -> Self::Item,
        ID: Fn() -> T,
        Self::Item: Try<Output = T>,
    {
        ParallelIteratorInner::try_reduce(self.into_inner(), identity, op)
    }

    #[inline]
    fn try_reduce_with<T, OP>(self, op: OP) -> Option<Self::Item>
    where
        OP: Fn(T, T) -> Self::Item,
        Self::Item: Try<Output = T>,
    {
        ParallelIteratorInner::try_reduce_with(self.into_inner(), op)
    }

    #[inline]
    fn fold<T, ID, F>(self, identity: ID, fold_op: F) -> impl ParallelIterator<Item = T>
    where
        F: Fn(T, Self::Item) -> T,
        ID: Fn() -> T,
    {
        ParallelWrapper::new(ParallelIteratorInner::fold(
            self.into_inner(),
            identity,
            fold_op,
        ))
    }

    #[inline]
    fn fold_with<F, T>(self, init: T, fold_op: F) -> impl ParallelIterator<Item = T>
    where
        F: Fn(T, Self::Item) -> T,
    {
        ParallelWrapper::new(ParallelIteratorInner::fold_with(
            self.into_inner(),
            init,
            fold_op,
        ))
    }

    #[inline]
    fn try_fold<T, R, ID, F>(self, identity: ID, fold_op: F) -> impl ParallelIterator<Item = R>
    where
        F: Fn(T, Self::Item) -> R,
        ID: Fn() -> T,
        R: Try<Output = T>,
    {
        ParallelWrapper::new(ParallelIteratorInner::try_fold(
            self.into_inner(),
            identity,
            fold_op,
        ))
    }

    #[inline]
    fn try_fold_with<F, T, R>(self, init: T, fold_op: F) -> impl ParallelIterator<Item = R>
    where
        F: Fn(T, Self::Item) -> R,
        R: Try<Output = T>,
    {
        ParallelWrapper::new(ParallelIteratorInner::try_fold_with(
            self.into_inner(),
            init,
            fold_op,
        ))
    }

    #[inline]
    fn sum<S>(self) -> S
    where
        S: Sum<Self::Item> + Sum<S>,
    {
        ParallelIteratorInner::sum(self.into_inner())
    }

    #[inline]
    fn product<P>(self) -> P
    where
        P: Product<Self::Item> + Product<P>,
    {
        ParallelIteratorInner::product(self.into_inner())
    }

    #[inline]
    fn min(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        ParallelIteratorInner::min(self.into_inner())
    }

    #[inline]
    fn min_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering,
    {
        ParallelIteratorInner::min_by(self.into_inner(), f)
    }

    #[inline]
    fn min_by_key<K, F>(self, f: F) -> Option<Self::Item>
    where
        K: Ord,
        F: Fn(&Self::Item) -> K,
    {
        ParallelIteratorInner::min_by_key(self.into_inner(), f)
    }

    #[inline]
    fn max(self) -> Option<Self::Item>
    where
        Self::Item: Ord,
    {
        ParallelIteratorInner::max(self.into_inner())
    }

    #[inline]
    fn max_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering,
    {
        ParallelIteratorInner::max_by(self.into_inner(), f)
    }

    #[inline]
    fn max_by_key<K, F>(self, f: F) -> Option<Self::Item>
    where
        K: Ord,
        F: Fn(&Self::Item) -> K,
    {
        ParallelIteratorInner::max_by_key(self.into_inner(), f)
    }

    #[inline]
    fn chain<C>(
        self,
        chain: C,
    ) -> ParallelWrapper<Chain<Self::Inner, <C::Iter as ParallelIterator>::Inner>>
    where
        C: IntoParallelIterator<Item = Self::Item>,
    {
        ParallelWrapper::new(ParallelIteratorInner::chain(
            self.into_inner(),
            chain.into_par_iter().into_inner(),
        ))
    }

    #[inline]
    fn find_any<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        ParallelIteratorInner::find_any(self.into_inner(), predicate)
    }

    #[inline]
    fn find_first<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        ParallelIteratorInner::find_first(self.into_inner(), predicate)
    }

    #[inline]
    fn find_last<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        ParallelIteratorInner::find_last(self.into_inner(), predicate)
    }

    #[inline]
    fn find_map_any<P, R>(self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R>,
    {
        ParallelIteratorInner::find_map_any(self.into_inner(), predicate)
    }

    #[inline]
    fn find_map_first<P, R>(self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R>,
    {
        ParallelIteratorInner::find_map_first(self.into_inner(), predicate)
    }

    #[inline]
    fn find_map_last<P, R>(self, predicate: P) -> Option<R>
    where
        P: Fn(Self::Item) -> Option<R>,
    {
        ParallelIteratorInner::find_map_last(self.into_inner(), predicate)
    }

    #[inline]
    fn any<P>(self, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool,
    {
        ParallelIteratorInner::any(self.into_inner(), predicate)
    }

    #[inline]
    fn all<P>(self, predicate: P) -> bool
    where
        P: Fn(Self::Item) -> bool,
    {
        ParallelIteratorInner::all(self.into_inner(), predicate)
    }

    #[inline]
    fn while_some<T>(self) -> impl ParallelIterator<Item = T>
    where
        Self: ParallelIterator<Item = Option<T>>,
        T: Send,
    {
        ParallelWrapper::new(ParallelIteratorInner::while_some(self.into_inner()))
    }

    #[inline]
    fn panic_fuse(self) -> impl ParallelIterator<Item = Self::Item> {
        ParallelWrapper::new(ParallelIteratorInner::panic_fuse(self.into_inner()))
    }

    #[inline]
    fn collect<C>(self) -> C
    where
        C: FromParallelIterator<Self::Item>,
    {
        C::from_par_iter(self.into_par_iter())
    }

    #[inline]
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
    where
        Self: ParallelIterator<Item = (A, B)>,
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
        A: Send,
        B: Send,
    {
        ParallelIteratorInner::unzip(self.into_inner())
    }

    #[inline]
    fn partition<A, B, P>(self, predicate: P) -> (A, B)
    where
        A: Default + Extend<Self::Item>,
        B: Default + Extend<Self::Item>,
        P: Fn(&Self::Item) -> bool,
    {
        ParallelIteratorInner::partition(self.into_inner(), predicate)
    }

    #[inline]
    fn partition_map<A, B, P, L, R>(self, predicate: P) -> (A, B)
    where
        A: Default + Extend<L>,
        B: Default + Extend<R>,
        P: Fn(Self::Item) -> Either<L, R>,
    {
        ParallelIteratorInner::partition_map(self.into_inner(), predicate)
    }

    #[inline]
    fn intersperse(self, element: Self::Item) -> impl ParallelIterator<Item = Self::Item>
    where
        Self::Item: Clone,
    {
        ParallelWrapper::new(ParallelIteratorInner::intersperse(
            self.into_inner(),
            element,
        ))
    }

    #[inline]
    fn take_any(self, n: usize) -> impl ParallelIterator<Item = Self::Item> {
        ParallelWrapper::new(ParallelIteratorInner::take_any(self.into_inner(), n))
    }

    #[inline]
    fn skip_any(self, n: usize) -> impl ParallelIterator<Item = Self::Item> {
        ParallelWrapper::new(ParallelIteratorInner::skip_any(self.into_inner(), n))
    }

    #[inline]
    fn take_any_while<P>(self, predicate: P) -> impl ParallelIterator<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        ParallelWrapper::new(ParallelIteratorInner::take_any_while(
            self.into_inner(),
            predicate,
        ))
    }

    #[inline]
    fn skip_any_while<P>(self, predicate: P) -> impl ParallelIterator<Item = Self::Item>
    where
        P: Fn(&Self::Item) -> bool,
    {
        ParallelWrapper::new(ParallelIteratorInner::skip_any_while(
            self.into_inner(),
            predicate,
        ))
    }

    #[inline]
    fn collect_vec_list(self) -> LinkedList<Vec<Self::Item>> {
        ParallelIteratorInner::collect_vec_list(self.into_inner())
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        ParallelIteratorInner::opt_len(self.as_inner())
    }
}

impl<I: Iterator> ParallelIterator for ParallelWrapper<I> {
    type Item = I::Item;
    type Inner = I;

    #[inline(always)]
    fn into_inner(self) -> I {
        self.0
    }

    #[inline(always)]
    fn as_inner(&self) -> &I {
        &self.0
    }
}

impl<I: Iterator> IntoIterator for ParallelWrapper<I> {
    type Item = I::Item;
    type IntoIter = I;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.0
    }
}

impl<L, R> ParallelIterator for Either<L, R>
where
    L: ParallelIterator,
    R: ParallelIterator<Item = L::Item>,
{
    type Item = L::Item;
    type Inner = Either<L::Inner, R::Inner>;

    #[inline(always)]
    fn into_inner(self) -> Self::Inner {
        match self {
            Either::Left(left) => Either::Left(left.into_inner()),
            Either::Right(right) => Either::Right(right.into_inner()),
        }
    }

    #[inline(always)]
    fn as_inner(&self) -> &Self::Inner {
        // This is a limitation - we can't return a reference to Either<L::Inner, R::Inner>
        // from Either<L, R> without storing it. For now, we'll panic.
        // This method is rarely used in practice.
        unimplemented!("as_inner not supported for Either")
    }
}

/// This section is copied from the `rayon` crate.
/// Since `Try` in the standard library is unstable, we need to define our own version.
mod private {
    use std::{
        convert::Infallible,
        ops::ControlFlow::{self, Break, Continue},
        task::Poll,
    };

    /// Clone of `std::ops::Try`.
    ///
    /// Implementing this trait is not permitted outside of `rayon`.
    pub trait Try {
        type Output;
        type Residual;

        fn from_output(output: Self::Output) -> Self;

        fn from_residual(residual: Self::Residual) -> Self;

        fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
    }

    impl<B, C> Try for ControlFlow<B, C> {
        type Output = C;
        type Residual = ControlFlow<B, Infallible>;

        fn from_output(output: Self::Output) -> Self {
            Continue(output)
        }

        fn from_residual(residual: Self::Residual) -> Self {
            match residual {
                Break(b) => Break(b),
                Continue(_) => unreachable!(),
            }
        }

        fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
            match self {
                Continue(c) => Continue(c),
                Break(b) => Break(Break(b)),
            }
        }
    }

    impl<T> Try for Option<T> {
        type Output = T;
        type Residual = Option<Infallible>;

        fn from_output(output: Self::Output) -> Self {
            Some(output)
        }

        fn from_residual(residual: Self::Residual) -> Self {
            match residual {
                None => None,
                Some(_) => unreachable!(),
            }
        }

        fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
            match self {
                Some(c) => Continue(c),
                None => Break(None),
            }
        }
    }

    impl<T, E> Try for Result<T, E> {
        type Output = T;
        type Residual = Result<Infallible, E>;

        fn from_output(output: Self::Output) -> Self {
            Ok(output)
        }

        fn from_residual(residual: Self::Residual) -> Self {
            match residual {
                Err(e) => Err(e),
                Ok(_) => unreachable!(),
            }
        }

        fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
            match self {
                Ok(c) => Continue(c),
                Err(e) => Break(Err(e)),
            }
        }
    }

    impl<T, E> Try for Poll<Result<T, E>> {
        type Output = Poll<T>;
        type Residual = Result<Infallible, E>;

        fn from_output(output: Self::Output) -> Self {
            output.map(Ok)
        }

        fn from_residual(residual: Self::Residual) -> Self {
            match residual {
                Err(e) => Self::Ready(Err(e)),
                Ok(_) => unreachable!(),
            }
        }

        fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
            match self {
                Self::Pending => Continue(Poll::Pending),
                Self::Ready(Ok(c)) => Continue(Poll::Ready(c)),
                Self::Ready(Err(e)) => Break(Err(e)),
            }
        }
    }

    impl<T, E> Try for Poll<Option<Result<T, E>>> {
        type Output = Poll<Option<T>>;
        type Residual = Result<Infallible, E>;

        fn from_output(output: Self::Output) -> Self {
            match output {
                Poll::Ready(o) => Self::Ready(o.map(Ok)),
                Poll::Pending => Self::Pending,
            }
        }

        fn from_residual(residual: Self::Residual) -> Self {
            match residual {
                Err(e) => Self::Ready(Some(Err(e))),
                Ok(_) => unreachable!(),
            }
        }

        fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
            match self {
                Self::Pending => Continue(Poll::Pending),
                Self::Ready(None) => Continue(Poll::Ready(None)),
                Self::Ready(Some(Ok(c))) => Continue(Poll::Ready(Some(c))),
                Self::Ready(Some(Err(e))) => Break(Err(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_try_reduce() {
        fn reduce(data: Vec<Option<i32>>) -> Option<Option<i32>> {
            data.into_par_iter()
                .try_reduce_with(|a, b| if b == 42 { None } else { Some(a + b) })
        }

        let data = vec![];
        assert_eq!(reduce(data), None);

        let data = vec![Some(1), Some(2), Some(3)];
        assert_eq!(reduce(data), Some(Some(6)));

        let data = vec![Some(1), Some(42)];
        assert_eq!(reduce(data), Some(None));
    }

    #[test]
    fn check_try_fold() {
        fn fold(data: Vec<i32>) -> Vec<Option<i32>> {
            data.into_par_iter()
                .try_fold_with(0, |a, b| if b == 42 { None } else { Some(a + b) })
                .collect::<Vec<_>>()
        }

        let data = vec![];
        assert_eq!(fold(data), vec![Some(0)]);

        let data = vec![1, 2, 3];
        assert_eq!(fold(data), vec![Some(6)]);

        let data = vec![1, 42];
        assert_eq!(fold(data), vec![None]);
    }
}
