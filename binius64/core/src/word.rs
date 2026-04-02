// Copyright 2025 Irreducible Inc.
//! [`Word`] related definitions.

use std::{
    fmt,
    ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr},
};

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

/// [`Word`] is 64-bit value and is a fundamental unit of data in Binius64. All computation and
/// constraints operate on it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Word(pub u64);

impl Word {
    /// All zero bit pattern, zero, nil, null.
    pub const ZERO: Word = Word(0);
    /// 1.
    pub const ONE: Word = Word(1);
    /// All bits set to one.
    pub const ALL_ONE: Word = Word(u64::MAX);
    /// 32 lower bits are set to one, all other bits are zero.
    pub const MASK_32: Word = Word(0x00000000FFFFFFFF);
    /// Most Significant Bit is set to one, all other bits are zero.
    ///
    /// This is a canonical representation of true.
    pub const MSB_ONE: Word = Word(0x8000000000000000);
}

impl fmt::Debug for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Word({:#018x})", self.0)
    }
}

impl BitAnd for Word {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Word(self.0 & rhs.0)
    }
}

impl BitOr for Word {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Word(self.0 | rhs.0)
    }
}

impl BitXor for Word {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Word(self.0 ^ rhs.0)
    }
}

impl Shl<u32> for Word {
    type Output = Self;

    fn shl(self, rhs: u32) -> Self::Output {
        Word(self.0 << rhs)
    }
}

impl Shr<u32> for Word {
    type Output = Self;

    fn shr(self, rhs: u32) -> Self::Output {
        Word(self.0 >> rhs)
    }
}

impl Not for Word {
    type Output = Self;

    fn not(self) -> Self::Output {
        Word(!self.0)
    }
}

impl Word {
    /// Creates a new `Word` from a 64-bit unsigned integer.
    pub fn from_u64(value: u64) -> Word {
        Word(value)
    }

    /// Performs 32-bit addition.
    ///
    /// Returns (sum, carry_out) where ith carry_out bit is set to one if there is a carry out at
    /// that bit position.
    pub fn iadd_cout_32(self, rhs: Word) -> (Word, Word) {
        let Word(lhs) = self;
        let Word(rhs) = rhs;
        let full_sum = lhs.wrapping_add(rhs);
        let sum = full_sum & 0x00000000_FFFFFFFF;
        let cout = (lhs & rhs) | ((lhs ^ rhs) & !full_sum);
        (Word(sum), Word(cout))
    }

    /// Performs 64-bit addition with carry input bit.
    ///
    /// cin is a carry-in from the previous addition. Since it can only affect the LSB only, the cin
    /// could be 1 if there is carry over, or 0 otherwise.
    ///
    /// Returns (sum, carry_out) where ith carry_out bit is set to one if there is a carry out at
    /// that bit position.
    pub fn iadd_cin_cout(self, rhs: Word, cin: Word) -> (Word, Word) {
        debug_assert!(cin == Word::ZERO || cin == Word::ONE, "cin must be 0 or 1");
        let Word(lhs) = self;
        let Word(rhs) = rhs;
        let Word(cin) = cin;
        let sum = lhs.wrapping_add(rhs).wrapping_add(cin);
        let cout = (lhs & rhs) | ((lhs ^ rhs) & !sum);
        (Word(sum), Word(cout))
    }

    /// Performs 64-bit subtraction with borrow input bit.
    ///
    /// bin is a borrow-in from the previous subtraction. Since it can only affect the LSB only, the
    /// bin could be 1 if there is borrow over, or 0 otherwise.
    ///
    /// Returns (diff, borrow_out) where ith borrow_out bit is set to one if there is a borrow out
    /// at that bit position.
    pub fn isub_bin_bout(self, rhs: Word, bin: Word) -> (Word, Word) {
        debug_assert!(bin == Word::ZERO || bin == Word::ONE, "bin must be 0 or 1");
        let Word(lhs) = self;
        let Word(rhs) = rhs;
        let Word(bin) = bin;
        let diff = lhs.wrapping_sub(rhs).wrapping_sub(bin);
        let bout = (!lhs & rhs) | (!(lhs ^ rhs) & diff);
        (Word(diff), Word(bout))
    }

    /// Performs shift right by a given number of bits followed by masking with a 32-bit mask.
    pub fn shr_32(self, n: u32) -> Word {
        let Word(value) = self;
        // Shift right logically by n bits and mask with 32-bit mask
        let result = (value >> n) & 0x00000000_FFFFFFFF;
        Word(result)
    }

    /// Shift Arithmetic Right by a given number of bits.
    ///
    /// This is similar to a logical shift right, but it shifts the sign bit to the right.
    pub fn sar(&self, n: u32) -> Word {
        let Word(value) = self;
        let value = *value as i64;
        let result = value >> n;
        Word(result as u64)
    }

    /// Rotate Right by a given number of bits followed by masking with a 32-bit mask.
    pub fn rotr_32(self, n: u32) -> Word {
        let Word(value) = self;
        let value_32 = (value as u32).rotate_right(n);
        Word(value_32 as u64)
    }

    /// Rotate Right by a given number of bits.
    pub fn rotr(self, n: u32) -> Word {
        let Word(value) = self;
        Word(value.rotate_right(n))
    }

    /// Shift Left Logical on 32-bit halves.
    ///
    /// Performs independent logical left shifts on the upper and lower 32-bit halves.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    pub fn sll32(self, n: u32) -> Word {
        let Word(value) = self;
        let n = n & 0x1F; // Only use lower 5 bits

        // Extract 32-bit halves
        let lo = value as u32;
        let hi = (value >> 32) as u32;

        // Shift each half independently
        let lo_shifted = (lo << n) as u64;
        let hi_shifted = ((hi << n) as u64) << 32;

        Word(lo_shifted | hi_shifted)
    }

    /// Shift Right Logical on 32-bit halves.
    ///
    /// Performs independent logical right shifts on the upper and lower 32-bit halves.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    pub fn srl32(self, n: u32) -> Word {
        let Word(value) = self;
        let n = n & 0x1F; // Only use lower 5 bits

        // Extract 32-bit halves
        let lo = value as u32;
        let hi = (value >> 32) as u32;

        // Shift each half independently
        let lo_shifted = (lo >> n) as u64;
        let hi_shifted = ((hi >> n) as u64) << 32;

        Word(lo_shifted | hi_shifted)
    }

    /// Shift Right Arithmetic on 32-bit halves.
    ///
    /// Performs independent arithmetic right shifts on the upper and lower 32-bit halves.
    /// Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift amount
    /// (0-31).
    pub fn sra32(self, n: u32) -> Word {
        let Word(value) = self;
        let n = n & 0x1F; // Only use lower 5 bits

        // Extract 32-bit halves as signed integers
        let lo = value as u32 as i32;
        let hi = (value >> 32) as u32 as i32;

        // Arithmetic shift each half independently
        let lo_shifted = ((lo >> n) as u32) as u64;
        let hi_shifted = (((hi >> n) as u32) as u64) << 32;

        Word(lo_shifted | hi_shifted)
    }

    /// Rotate Right on 32-bit halves.
    ///
    /// Performs independent rotate right operations on the upper and lower 32-bit halves.
    /// Bits shifted off the right end wrap around to the left within each 32-bit half.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    pub fn rotr32(self, n: u32) -> Word {
        let Word(value) = self;
        let n = n & 0x1F; // Only use lower 5 bits

        // Extract 32-bit halves
        let lo = value as u32;
        let hi = (value >> 32) as u32;

        // Rotate each half independently
        let lo_rotated = lo.rotate_right(n) as u64;
        let hi_rotated = (hi.rotate_right(n) as u64) << 32;

        Word(lo_rotated | hi_rotated)
    }

    /// Unsigned integer multiplication.
    ///
    /// Multiplies two 64-bit unsigned integers and returns the 128-bit result split into high and
    /// low 64-bit words, respectively.
    pub fn imul(self, rhs: Word) -> (Word, Word) {
        let Word(lhs) = self;
        let Word(rhs) = rhs;
        let result = (lhs as u128) * (rhs as u128);

        let hi = (result >> 64) as u64;
        let lo = result as u64;
        (Word(hi), Word(lo))
    }

    /// Signed integer multiplication.
    ///
    /// Multiplies two 64-bit signed integers and returns the 128-bit result split into high and
    /// low 64-bit words, respectively.
    pub fn smul(self, rhs: Word) -> (Word, Word) {
        let Word(lhs) = self;
        let Word(rhs) = rhs;
        // Interpret as signed 64-bit integers
        let a = lhs as i64;
        let b = rhs as i64;
        // Perform signed multiplication as 128-bit
        let result = (a as i128) * (b as i128);
        // Extract high and low 64-bit words
        let hi = (result >> 64) as u64;
        let lo = result as u64;
        (Word(hi), Word(lo))
    }

    /// Integer addition.
    ///
    /// Wraps around on overflow.
    pub fn wrapping_add(self, rhs: Word) -> Word {
        Word(self.0.wrapping_add(rhs.0))
    }

    /// Integer subtraction.
    ///
    /// Wraps around on overflow.
    pub fn wrapping_sub(self, rhs: Word) -> Word {
        Word(self.0.wrapping_sub(rhs.0))
    }

    /// Returns the integer value as a 64-bit unsigned integer.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Tests if this Word represents true as an MSB-bool.
    ///
    /// In MSB-bool representation, a value is true if its Most Significant Bit (bit 63) is set to
    /// 1. All other bits are ignored for the boolean value.
    ///
    /// Returns true if the MSB is 1, false otherwise.
    pub fn is_msb_true(self) -> bool {
        (self.0 & 0x8000000000000000) != 0
    }

    /// Tests if this Word represents false as an MSB-bool.
    ///
    /// In MSB-bool representation, a value is false if its Most Significant Bit (bit 63) is 0.
    /// All other bits are ignored for the boolean value.
    ///
    /// Returns true if the MSB is 0, false otherwise.
    pub fn is_msb_false(self) -> bool {
        (self.0 & 0x8000000000000000) == 0
    }
}

impl SerializeBytes for Word {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.0.serialize(write_buf)
    }
}

impl DeserializeBytes for Word {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Ok(Word(u64::deserialize(read_buf)?))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(Word::ZERO, Word(0));
        assert_eq!(Word::ONE, Word(1));
        assert_eq!(Word::ALL_ONE, Word(0xFFFFFFFFFFFFFFFF));
        assert_eq!(Word::MASK_32, Word(0x00000000FFFFFFFF));
        assert_eq!(Word::MSB_ONE, Word(0x8000000000000000));
    }

    #[test]
    fn test_msb_bool() {
        // Test MSB_ONE is true
        assert!(Word::MSB_ONE.is_msb_true());
        assert!(!Word::MSB_ONE.is_msb_false());

        // Test ZERO is false
        assert!(!Word::ZERO.is_msb_true());
        assert!(Word::ZERO.is_msb_false());

        // Test various values with MSB set
        assert!(Word(0x8000000000000000).is_msb_true());
        assert!(Word(0x8000000000000001).is_msb_true());
        assert!(Word(0x80000000FFFFFFFF).is_msb_true());
        assert!(Word(0xFFFFFFFFFFFFFFFF).is_msb_true());

        // Test various values with MSB clear
        assert!(Word(0x7FFFFFFFFFFFFFFF).is_msb_false());
        assert!(Word(0x0000000000000001).is_msb_false());
        assert!(Word(0x00000000FFFFFFFF).is_msb_false());
        assert!(Word(0x7000000000000000).is_msb_false());

        // Verify complementary behavior
        let test_word = Word(0x8123456789ABCDEF);
        assert!(test_word.is_msb_true());
        assert!(!test_word.is_msb_false());

        let test_word2 = Word(0x7123456789ABCDEF);
        assert!(!test_word2.is_msb_true());
        assert!(test_word2.is_msb_false());
    }

    proptest! {
        #[test]
        fn prop_msb_bool(val in any::<u64>()) {
            let word = Word(val);

            // is_msb_true and is_msb_false should be complementary
            assert_eq!(word.is_msb_true(), !word.is_msb_false());
            assert_eq!(word.is_msb_false(), !word.is_msb_true());

            // Check against direct bit manipulation
            let msb_set = (val & 0x8000000000000000) != 0;
            assert_eq!(word.is_msb_true(), msb_set);
            assert_eq!(word.is_msb_false(), !msb_set);

            // MSB operations should ignore lower bits
            let word_with_msb = Word(val | 0x8000000000000000);
            let word_without_msb = Word(val & 0x7FFFFFFFFFFFFFFF);
            assert!(word_with_msb.is_msb_true());
            assert!(word_without_msb.is_msb_false());
        }

        #[test]
        fn prop_bitwise_and(a in any::<u64>(), b in any::<u64>()) {
            let wa = Word(a);
            let wb = Word(b);

            // Basic AND properties
            assert_eq!((wa & wb).0, a & b);
            assert_eq!(wa & Word::ALL_ONE, wa);
            assert_eq!(wa & Word::ZERO, Word::ZERO);
            assert_eq!(wa & wa, wa); // Idempotent

            // Commutative
            assert_eq!(wa & wb, wb & wa);
        }

        #[test]
        fn prop_bitwise_or(a in any::<u64>(), b in any::<u64>()) {
            let wa = Word(a);
            let wb = Word(b);

            // Basic OR properties
            assert_eq!((wa | wb).0, a | b);
            assert_eq!(wa | Word::ZERO, wa);
            assert_eq!(wa | Word::ALL_ONE, Word::ALL_ONE);
            assert_eq!(wa | wa, wa); // Idempotent

            // Commutative
            assert_eq!(wa | wb, wb | wa);
        }

        #[test]
        fn prop_bitwise_xor(a in any::<u64>(), b in any::<u64>()) {
            let wa = Word(a);
            let wb = Word(b);

            // Basic XOR properties
            assert_eq!((wa ^ wb).0, a ^ b);
            assert_eq!(wa ^ Word::ZERO, wa);
            assert_eq!(wa ^ wa, Word::ZERO);
            assert_eq!(wa ^ Word::ALL_ONE, !wa);

            // Commutative
            assert_eq!(wa ^ wb, wb ^ wa);

            // Double XOR cancels
            assert_eq!(wa ^ wb ^ wb, wa);
        }

        #[test]
        fn prop_bitwise_not(a in any::<u64>()) {
            let wa = Word(a);

            // Basic NOT properties
            assert_eq!((!wa).0, !a);
            assert_eq!(!(!wa), wa); // Double negation
            assert_eq!(!Word::ZERO, Word::ALL_ONE);
            assert_eq!(!Word::ALL_ONE, Word::ZERO);

            // De Morgan's laws
            let wb = Word(a.wrapping_add(1));
            assert_eq!(!(wa & wb), !wa | !wb);
            assert_eq!(!(wa | wb), !wa & !wb);
        }

        #[test]
        fn prop_shift_left(val in any::<u64>(), shift in 0u32..64) {
            let w = Word(val);
            assert_eq!((w << shift).0, val << shift);

            // Shifting by 0 is identity
            assert_eq!(w << 0, w);

            // Shifting by 64 or more gives 0
            if shift >= 64 {
                assert_eq!((w << shift).0, 0);
            }
        }

        #[test]
        fn prop_shift_right(val in any::<u64>(), shift in 0u32..64) {
            let w = Word(val);
            assert_eq!((w >> shift).0, val >> shift);

            // Shifting by 0 is identity
            assert_eq!(w >> 0, w);

            // Shifting by 64 or more gives 0
            if shift >= 64 {
                assert_eq!((w >> shift).0, 0);
            }
        }

        #[test]
        fn prop_shift_inverse(val in any::<u64>(), shift in 1u32..64) {
            let w = Word(val);
            // Left then right shift loses high bits
            let mask = (1u64 << (64 - shift)) - 1;
            assert_eq!(((w << shift) >> shift).0, val & mask);

            // Right then left shift loses low bits
            let high_mask = !((1u64 << shift) - 1);
            assert_eq!(((w >> shift) << shift).0, val & high_mask);
        }

        #[test]
        fn prop_sar(val in any::<u64>(), shift in 0u32..64) {
            let w = Word(val);
            let expected = ((val as i64) >> shift) as u64;
            assert_eq!(w.sar(shift).0, expected);

            // SAR by 0 is identity
            assert_eq!(w.sar(0), w);

            // SAR by 63 gives all 0s or all 1s depending on sign
            let sign_extended = if (val as i64) < 0 {
                Word(0xFFFFFFFFFFFFFFFF)
            } else {
                Word(0)
            };
            assert_eq!(w.sar(63), sign_extended);
        }

        #[test]
        fn prop_sar_sign_extension(val in any::<u64>(), shift in 1u32..64) {
            let w = Word(val);
            let result = w.sar(shift);

            // Check sign bit is extended
            let is_negative = (val as i64) < 0;
            if is_negative {
                // High bits should all be 1
                let mask = !((1u64 << (64 - shift)) - 1);
                assert_eq!(result.0 & mask, mask);
            } else {
                // High bits should all be 0
                let mask = !((1u64 << (64 - shift)) - 1);
                assert_eq!(result.0 & mask, 0);
            }
        }

        #[test]
        fn prop_iadd_cout_32(a in any::<u32>(), b in any::<u32>()) {
            let wa = Word(a as u64);
            let wb = Word(b as u64);
            let (sum, cout) = wa.iadd_cout_32(wb);

            // Sum should be masked to 32 bits
            assert_eq!(sum.0, (a as u64 + b as u64) & 0xFFFFFFFF);

            // Carry computation: cout = (a & b) | ((a ^ b) & !sum)
            let expected_cout = (a as u64 & b as u64) | ((a as u64 ^ b as u64) & !sum.0);
            assert_eq!(cout.0, expected_cout);

            // Identity: adding 0 produces no carries
            let (sum0, cout0) = wa.iadd_cout_32(Word::ZERO);
            assert_eq!(sum0.0, a as u64);
            assert_eq!(cout0, Word::ZERO);
        }

        #[test]
        fn prop_iadd_cin_cout(a in any::<u64>(), b in any::<u64>(), cin in 0u64..=1) {
            let wa = Word(a);
            let wb = Word(b);
            let wcin = Word(cin);
            let (sum, cout) = wa.iadd_cin_cout(wb, wcin);

            // Basic addition with carry
            let expected_sum = a.wrapping_add(b).wrapping_add(cin);
            assert_eq!(sum.0, expected_sum);

            // Carry computation: cout at each bit position
            let expected_cout = (a & b) | ((a ^ b) & !expected_sum);
            assert_eq!(cout.0, expected_cout);

            // Without carry in, same as regular addition
            let (sum0, cout0) = wa.iadd_cin_cout(wb, Word::ZERO);
            let full_sum = a.wrapping_add(b);
            assert_eq!(sum0.0, full_sum);
            assert_eq!(cout0.0, (a & b) | ((a ^ b) & !full_sum));
        }

        #[test]
        fn prop_isub_bin_bout(a in any::<u64>(), b in any::<u64>(), bin in 0u64..=1) {
            let wa = Word(a);
            let wb = Word(b);
            let wbin = Word(bin);
            let (diff, bout) = wa.isub_bin_bout(wb, wbin);

            // Basic subtraction with borrow
            let expected_diff = a.wrapping_sub(b).wrapping_sub(bin);
            assert_eq!(diff.0, expected_diff);

            // Borrow computation: bout = (!a & b) | (!(a ^ b) & diff)
            let expected_bout = (!a & b) | (!(a ^ b) & expected_diff);
            assert_eq!(bout.0, expected_bout);

            // Without borrow in
            let (diff0, bout0) = wa.isub_bin_bout(wb, Word::ZERO);
            let expected = a.wrapping_sub(b);
            assert_eq!(diff0.0, expected);
            assert_eq!(bout0.0, (!a & b) | (!(a ^ b) & expected));
        }

        #[test]
        fn prop_shr_32(val in any::<u64>(), shift in 0u32..64) {
            let w = Word(val);
            let result = w.shr_32(shift);

            // Result should be the full value shifted right, then masked to 32 bits
            let expected = (val >> shift) & 0xFFFFFFFF;
            assert_eq!(result.0, expected);

            // Shifting by 0 gives lower 32 bits
            assert_eq!(w.shr_32(0).0, val & 0xFFFFFFFF);

            // Shifting by 32 or more gives upper bits or zeros
            if shift >= 32 {
                assert_eq!(result.0, (val >> shift) & 0xFFFFFFFF);
            }
        }

        #[test]
        fn prop_rotr_32(val in any::<u32>(), rotate in 0u32..64) {
            let w = Word(val as u64);
            let result = w.rotr_32(rotate);

            // Only lower 32 bits are rotated
            let rotate_mod = rotate % 32;
            let val32 = val as u64;
            let expected = if rotate_mod == 0 {
                val32
            } else {
                ((val32 >> rotate_mod) | (val32 << (32 - rotate_mod))) & 0xFFFFFFFF
            };
            assert_eq!(result.0, expected);

            // Rotation by 0 or 32 is identity
            assert_eq!(w.rotr_32(0).0, val32);
            assert_eq!(w.rotr_32(32).0, val32);
        }

        #[test]
        fn prop_rotr(val in any::<u64>(), rotate in 0u32..128) {
            let w = Word(val);
            let result = w.rotr(rotate);

            // Rotation is modulo 64
            let rotate_mod = rotate % 64;
            let expected = if rotate_mod == 0 {
                val
            } else {
                (val >> rotate_mod) | (val << (64 - rotate_mod))
            };
            assert_eq!(result.0, expected);

            // Rotation by 0 or 64 is identity
            assert_eq!(w.rotr(0), w);
            assert_eq!(w.rotr(64), w);

            // Double rotation
            let r1 = rotate % 64;
            let r2 = (64 - r1) % 64;
            if r1 != 0 {
                assert_eq!(w.rotr(r1).rotr(r2), w);
            }
        }

        #[test]
        fn prop_imul(a in any::<u64>(), b in any::<u64>()) {
            let wa = Word(a);
            let wb = Word(b);
            let (hi, lo) = wa.imul(wb);

            // Check against native 128-bit multiplication
            let result = (a as u128) * (b as u128);
            assert_eq!(hi.0, (result >> 64) as u64);
            assert_eq!(lo.0, result as u64);

            // Multiplication by 0 gives 0
            let (hi0, lo0) = wa.imul(Word::ZERO);
            assert_eq!(hi0, Word::ZERO);
            assert_eq!(lo0, Word::ZERO);

            // Multiplication by 1 is identity
            let (hi1, lo1) = wa.imul(Word::ONE);
            assert_eq!(hi1, Word::ZERO);
            assert_eq!(lo1, wa);

            // Commutative
            let (hi_ab, lo_ab) = wa.imul(wb);
            let (hi_reversed, lo_reversed) = wb.imul(wa);
            assert_eq!(hi_ab, hi_reversed);
            assert_eq!(lo_ab, lo_reversed);
        }

        #[test]
        fn prop_sll32(val in any::<u64>(), shift in 0u32..32) {
            let w = Word(val);
            let result = w.sll32(shift);

            // Extract 32-bit halves
            let lo = val as u32;
            let hi = (val >> 32) as u32;

            // Expected result: each half shifted independently
            let expected_lo = ((lo << shift) as u64) & 0xFFFFFFFF;
            let expected_hi = ((hi << shift) as u64) << 32;
            let expected = expected_lo | expected_hi;

            assert_eq!(result.0, expected);

            // Shifting by 0 is identity
            assert_eq!(w.sll32(0), w);

            // Shifting by 31 should move MSB of each half to sign bit
            let w_test = Word(0x40000001_40000001);
            let result_31 = w_test.sll32(31);
            assert_eq!(result_31.0, 0x80000000_80000000);

            // Test that shift amount is masked to 5 bits
            assert_eq!(w.sll32(shift), w.sll32(shift | 0x20));
        }

        #[test]
        fn prop_srl32(val in any::<u64>(), shift in 0u32..32) {
            let w = Word(val);
            let result = w.srl32(shift);

            // Extract 32-bit halves
            let lo = val as u32;
            let hi = (val >> 32) as u32;

            // Expected result: each half shifted independently
            let expected_lo = (lo >> shift) as u64;
            let expected_hi = ((hi >> shift) as u64) << 32;
            let expected = expected_lo | expected_hi;

            assert_eq!(result.0, expected);

            // Shifting by 0 is identity
            assert_eq!(w.srl32(0), w);

            // Shifting by 31 should move LSB to bit 0, clearing upper bits
            let w_test = Word(0x80000000_80000000);
            let result_31 = w_test.srl32(31);
            assert_eq!(result_31.0, 0x00000001_00000001);

            // Test that shift amount is masked to 5 bits
            assert_eq!(w.srl32(shift), w.srl32(shift | 0x20));
        }

        #[test]
        fn prop_sra32(val in any::<u64>(), shift in 0u32..32) {
            let w = Word(val);
            let result = w.sra32(shift);

            // Extract 32-bit halves as signed
            let lo = val as u32 as i32;
            let hi = (val >> 32) as u32 as i32;

            // Expected result: each half arithmetic shifted independently
            let expected_lo = ((lo >> shift) as u32) as u64;
            let expected_hi = (((hi >> shift) as u32) as u64) << 32;
            let expected = expected_lo | expected_hi;

            assert_eq!(result.0, expected);

            // Shifting by 0 is identity
            assert_eq!(w.sra32(0), w);

            // Sign extension test: negative values extend sign bit
            let w_neg = Word(0x80000000_80000000);
            let result_1 = w_neg.sra32(1);
            assert_eq!(result_1.0, 0xC0000000_C0000000);

            // Sign extension test: positive values extend 0
            let w_pos = Word(0x40000000_40000000);
            let result_1_pos = w_pos.sra32(1);
            assert_eq!(result_1_pos.0, 0x20000000_20000000);

            // Shifting by 31 gives all 0s or all 1s in each half
            let result_31 = w.sra32(31);
            let expected_lo_31 = if lo < 0 { 0xFFFFFFFF } else { 0 };
            let expected_hi_31 = if hi < 0 { 0xFFFFFFFF00000000 } else { 0 };
            assert_eq!(result_31.0, expected_lo_31 | expected_hi_31);

            // Test that shift amount is masked to 5 bits
            assert_eq!(w.sra32(shift), w.sra32(shift | 0x20));
        }

        #[test]
        fn prop_rotr32(val in any::<u64>(), rotate in 0u32..32) {
            let w = Word(val);
            let result = w.rotr32(rotate);

            // Extract 32-bit halves
            let lo = val as u32;
            let hi = (val >> 32) as u32;

            // Expected result: each half rotated independently
            let expected_lo = lo.rotate_right(rotate) as u64;
            let expected_hi = ((hi.rotate_right(rotate)) as u64) << 32;
            let expected = expected_lo | expected_hi;

            assert_eq!(result.0, expected);

            // Rotating by 0 is identity
            assert_eq!(w.rotr32(0), w);

            // Rotating by 32 is identity (due to masking to 5 bits)
            assert_eq!(w.rotr32(32), w.rotr32(0));

            // Test that rotate amount is masked to 5 bits
            assert_eq!(w.rotr32(rotate), w.rotr32(rotate | 0x20));

            // Rotation is circular - rotating by n then 32-n gives identity
            if rotate > 0 && rotate < 32 {
                let w_test = Word(0x12345678_9ABCDEF0);
                let rotated = w_test.rotr32(rotate);
                let back = rotated.rotr32(32 - rotate);
                assert_eq!(back, w_test);
            }
        }

        #[test]
        fn prop_smul(a in any::<u64>(), b in any::<u64>()) {
            let wa = Word(a);
            let wb = Word(b);
            let (hi, lo) = wa.smul(wb);

            // Check against native 128-bit signed multiplication
            let result = (a as i64 as i128) * (b as i64 as i128);
            assert_eq!(hi.0, (result >> 64) as u64);
            assert_eq!(lo.0, result as u64);

            // Multiplication by 0 gives 0
            let (hi0, lo0) = wa.smul(Word::ZERO);
            assert_eq!(hi0, Word::ZERO);
            assert_eq!(lo0, Word::ZERO);

            // Multiplication by 1 is identity
            let (hi1, lo1) = wa.smul(Word::ONE);
            let expected_hi = if (a as i64) < 0 { Word(0xFFFFFFFFFFFFFFFF) } else { Word::ZERO };
            assert_eq!(hi1, expected_hi);
            assert_eq!(lo1, wa);

            // Multiplication by -1 negates
            let (hi_neg, lo_neg) = wa.smul(Word(0xFFFFFFFFFFFFFFFF));
            let neg_result = -(a as i64 as i128);
            assert_eq!(hi_neg.0, (neg_result >> 64) as u64);
            assert_eq!(lo_neg.0, neg_result as u64);

            // Commutative
            let (hi_ab, lo_ab) = wa.smul(wb);
            let (hi_reversed, lo_reversed) = wb.smul(wa);
            assert_eq!(hi_ab, hi_reversed);
            assert_eq!(lo_ab, lo_reversed);
        }

        #[test]
        fn prop_wrapping_sub(a in any::<u64>(), b in any::<u64>()) {
            let wa = Word(a);
            let wb = Word(b);
            let result = wa.wrapping_sub(wb);

            assert_eq!(result.0, a.wrapping_sub(b));

            // Subtracting 0 is identity
            assert_eq!(wa.wrapping_sub(Word::ZERO), wa);

            // Subtracting itself gives 0
            assert_eq!(wa.wrapping_sub(wa), Word::ZERO);

            // Adding then subtracting cancels
            let sum = Word(a.wrapping_add(b));
            assert_eq!(sum.wrapping_sub(wb), wa);
        }

        #[test]
        fn prop_conversions(val in any::<u64>()) {
            let word = Word::from_u64(val);
            assert_eq!(word.as_u64(), val);
            assert_eq!(word, Word(val));

            // Round trip
            assert_eq!(Word::from_u64(word.as_u64()), word);
        }

        #[test]
        fn prop_debug_format(val in any::<u64>()) {
            let word = Word(val);
            let debug_str = format!("{:?}", word);
            assert!(debug_str.starts_with("Word(0x"));
            assert!(debug_str.ends_with(")"));
            // Check the hex value is correct (lowercase)
            let expected = format!("Word({:#018x})", val);
            assert_eq!(debug_str, expected);
        }
    }

    #[test]
    fn test_32bit_shift_edge_cases() {
        // Test sll32 edge cases
        let w1 = Word(0x12345678_9ABCDEF0);
        assert_eq!(w1.sll32(4).0, 0x23456780_ABCDEF00);
        assert_eq!(w1.sll32(16).0, 0x56780000_DEF00000);

        // Test that upper bits don't affect lower half and vice versa
        let w2 = Word(0xFFFFFFFF_00000000);
        assert_eq!(w2.sll32(1).0, 0xFFFFFFFE_00000000);
        let w3 = Word(0x00000000_FFFFFFFF);
        assert_eq!(w3.sll32(1).0, 0x00000000_FFFFFFFE);

        // Test srl32 edge cases
        assert_eq!(w1.srl32(4).0, 0x01234567_09ABCDEF);
        assert_eq!(w1.srl32(16).0, 0x00001234_00009ABC);

        // Test sra32 with mixed sign bits
        let w4 = Word(0x80000000_7FFFFFFF); // Negative upper, positive lower
        assert_eq!(w4.sra32(1).0, 0xC0000000_3FFFFFFF);
        assert_eq!(w4.sra32(31).0, 0xFFFFFFFF_00000000);

        let w5 = Word(0x7FFFFFFF_80000000); // Positive upper, negative lower
        assert_eq!(w5.sra32(1).0, 0x3FFFFFFF_C0000000);
        assert_eq!(w5.sra32(31).0, 0x00000000_FFFFFFFF);

        // Test boundary values
        let all_ones = Word(0xFFFFFFFF_FFFFFFFF);
        assert_eq!(all_ones.sll32(1).0, 0xFFFFFFFE_FFFFFFFE);
        assert_eq!(all_ones.srl32(1).0, 0x7FFFFFFF_7FFFFFFF);
        assert_eq!(all_ones.sra32(1).0, 0xFFFFFFFF_FFFFFFFF);

        let alternating = Word(0xAAAAAAAA_55555555);
        assert_eq!(alternating.sll32(1).0, 0x55555554_AAAAAAAA);
        assert_eq!(alternating.srl32(1).0, 0x55555555_2AAAAAAA);
        assert_eq!(alternating.sra32(1).0, 0xD5555555_2AAAAAAA);

        // Test zero shifts
        assert_eq!(w1.sll32(0), w1);
        assert_eq!(w1.srl32(0), w1);
        assert_eq!(w1.sra32(0), w1);

        // Test that shifts are independent between halves
        let w6 = Word(0x00000001_00000000);
        assert_eq!(w6.sll32(31).0, 0x80000000_00000000);
        assert_eq!(w6.srl32(1).0, 0x00000000_00000000);

        // Test rotr32 edge cases
        let w7 = Word(0x80000001_80000001);
        assert_eq!(w7.rotr32(1).0, 0xC0000000_C0000000);
        assert_eq!(w7.rotr32(31).0, 0x00000003_00000003);

        // Test rotr32 rotation wrapping
        let w8 = Word(0x12345678_9ABCDEF0);
        assert_eq!(w8.rotr32(4).0, 0x81234567_09ABCDEF);
        assert_eq!(w8.rotr32(16).0, 0x56781234_DEF09ABC);

        // Test rotr32 with different values in each half
        let w9 = Word(0xFFFF0000_0000FFFF);
        assert_eq!(w9.rotr32(16).0, 0x0000FFFF_FFFF0000);

        // Test rotr32 zero rotation
        assert_eq!(w8.rotr32(0), w8);
    }
}
