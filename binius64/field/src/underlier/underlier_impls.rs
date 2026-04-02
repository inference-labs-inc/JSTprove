// Copyright 2024-2025 Irreducible Inc.

use super::{
    small_uint::{U1, U2, U4},
    underlier_type::{NumCast, UnderlierType},
    underlier_with_bit_ops::UnderlierWithBitOps,
};
use crate::arch::{interleave_mask_even, interleave_with_mask};

macro_rules! impl_underlier_type {
	($name:ty) => {
		impl UnderlierType for $name {
			const LOG_BITS: usize =
				binius_utils::checked_arithmetics::checked_log_2(Self::BITS as _);
		}
	};
	() => {};
	($name:ty, $($tail:ty),+) => {
		impl_underlier_type!($name);
		impl_underlier_type!($($tail),+);
	}
}

impl_underlier_type!(u8, u16, u32, u64, u128);

impl UnderlierWithBitOps for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const ONES: Self = Self::MAX;

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[u8] = &[
            interleave_mask_even!(u8, 0),
            interleave_mask_even!(u8, 1),
            interleave_mask_even!(u8, 2),
        ];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

impl UnderlierWithBitOps for u16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const ONES: Self = Self::MAX;

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[u16] = &[
            interleave_mask_even!(u16, 0),
            interleave_mask_even!(u16, 1),
            interleave_mask_even!(u16, 2),
            interleave_mask_even!(u16, 3),
        ];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

impl UnderlierWithBitOps for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const ONES: Self = Self::MAX;

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[u32] = &[
            interleave_mask_even!(u32, 0),
            interleave_mask_even!(u32, 1),
            interleave_mask_even!(u32, 2),
            interleave_mask_even!(u32, 3),
            interleave_mask_even!(u32, 4),
        ];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

impl UnderlierWithBitOps for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const ONES: Self = Self::MAX;

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[u64] = &[
            interleave_mask_even!(u64, 0),
            interleave_mask_even!(u64, 1),
            interleave_mask_even!(u64, 2),
            interleave_mask_even!(u64, 3),
            interleave_mask_even!(u64, 4),
            interleave_mask_even!(u64, 5),
        ];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

impl UnderlierWithBitOps for u128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const ONES: Self = Self::MAX;

    fn interleave(self, other: Self, log_block_len: usize) -> (Self, Self) {
        const MASKS: &[u128] = &[
            interleave_mask_even!(u128, 0),
            interleave_mask_even!(u128, 1),
            interleave_mask_even!(u128, 2),
            interleave_mask_even!(u128, 3),
            interleave_mask_even!(u128, 4),
            interleave_mask_even!(u128, 5),
            interleave_mask_even!(u128, 6),
        ];
        interleave_with_mask(self, other, log_block_len, MASKS)
    }
}

macro_rules! impl_num_cast {
	(@pair U1, U2) => {impl_num_cast!(@small_u_from_small_u U1, U2);};
	(@pair U1, U4) => {impl_num_cast!(@small_u_from_small_u U1, U4);};
	(@pair U2, U4) => {impl_num_cast!(@small_u_from_small_u U2, U4);};
	(@pair U1, $bigger:ty) => {impl_num_cast!(@small_u_from_u U1, $bigger);};
	(@pair U2, $bigger:ty) => {impl_num_cast!(@small_u_from_u U2, $bigger);};
	(@pair U4, $bigger:ty) => {impl_num_cast!(@small_u_from_u U4, $bigger);};
	(@pair $smaller:ident, $bigger:ident) => {
		impl NumCast<$bigger> for $smaller {
			#[inline(always)]
			fn num_cast_from(val: $bigger) -> Self {
				val as _
			}
		}

		impl NumCast<$smaller> for $bigger {
			#[inline(always)]
			fn num_cast_from(val: $smaller) -> Self {
				val as _
			}
		}
	};
	(@small_u_from_small_u $smaller:ty, $bigger:ty) => {
		impl NumCast<$bigger> for $smaller {
			#[inline(always)]
			fn num_cast_from(val: $bigger) -> Self {
				Self::new(val.val()) & Self::ONES
			}
		}

		impl NumCast<$smaller> for $bigger {
			#[inline(always)]
			fn num_cast_from(val: $smaller) -> Self {
				Self::new(val.val())
			}
		}
	};
	(@small_u_from_u $smaller:ty, $bigger:ty) => {
		impl NumCast<$bigger> for $smaller {
			#[inline(always)]
			fn num_cast_from(val: $bigger) -> Self {
				Self::new(val as u8) & Self::ONES
			}
		}

		impl NumCast<$smaller> for $bigger {
			#[inline(always)]
			fn num_cast_from(val: $smaller) -> Self {
				val.val() as _
			}
		}
	};
	($_:ty,) => {};
	(,) => {};
	(all_pairs) => {};
	(all_pairs $_:ty) => {};
	(all_pairs $_:ty,) => {};
	(all_pairs $smaller:ident, $head:ident, $($tail:ident,)*) => {
		impl_num_cast!(@pair $smaller, $head);
		impl_num_cast!(all_pairs $smaller, $($tail,)*);
	};
	($smaller:ident, $($tail:ident,)+) => {
		impl_num_cast!(all_pairs $smaller, $($tail,)+);
		impl_num_cast!($($tail,)+);
	};
}

impl_num_cast!(U1, U2, U4, u8, u16, u32, u64, u128,);
