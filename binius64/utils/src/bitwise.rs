// Copyright 2025 Irreducible Inc.

use std::{
    marker::PhantomData,
    ops::{BitAnd, Shr},
};

use trait_set::trait_set;

use super::random_access_sequence::RandomAccessSequence;

// A trait alias for a type that can act as a bitmask.
//
// It is intended to be a drop in constraint for a primitive integer type.
// Take note that `Shr` implementation sign extends signed types.
trait_set! {
    pub trait Bitwise =
        BitAnd<Output=Self> +
        Shr<usize, Output=Self> +
        From<u8> +
        PartialEq<Self> +
        Sized +
        Sync +
        Copy;
}

/// An adaptor structure that wraps a slice of integers and presents a
/// random access sequence of bits at a given offset.
pub struct BitSelector<B: Bitwise, S: AsRef<[B]>> {
    bit_offset: usize,
    slice: S,
    _b_marker: PhantomData<B>,
}

impl<B: Bitwise, S: AsRef<[B]>> BitSelector<B, S> {
    pub fn new(bit_offset: usize, slice: S) -> Self {
        Self {
            bit_offset,
            slice,
            _b_marker: PhantomData,
        }
    }
}

impl<B: Bitwise, S: AsRef<[B]>> RandomAccessSequence<bool> for BitSelector<B, S> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.slice.as_ref().len()
    }

    #[inline(always)]
    fn get(&self, index: usize) -> bool {
        (self.slice.as_ref()[index] >> self.bit_offset) & B::from(1u8) != B::from(0u8)
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        unsafe {
            (*self.slice.as_ref().get_unchecked(index) >> self.bit_offset) & B::from(1u8)
                != B::from(0u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_selector_on_sequential_integer_range() {
        let log_n = 10;

        let integers = (0u16..1 << log_n).collect::<Vec<_>>();

        for bit_offset in 0..log_n {
            let selector = BitSelector::new(bit_offset, &integers);

            for i in 0..1 << log_n {
                assert_eq!(selector.get(i), (i >> bit_offset) & 1 != 0);
            }
        }
    }
}
