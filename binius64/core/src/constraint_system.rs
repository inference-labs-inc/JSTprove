// Copyright 2025 Irreducible Inc.
//! Constraint system and related definitions.

use std::{
    borrow::Cow,
    cmp,
    ops::{Index, IndexMut},
};

use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use crate::{consts, error::ConstraintSystemError, word::Word};

/// A type safe wrapper over an index into the [`ValueVec`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ValueIndex(pub u32);

impl ValueIndex {
    /// The value index that is not considered to be valid.
    pub const INVALID: ValueIndex = ValueIndex(u32::MAX);
}

/// The most sensible default for a value index is invalid.
impl Default for ValueIndex {
    fn default() -> Self {
        Self::INVALID
    }
}

impl SerializeBytes for ValueIndex {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.0.serialize(write_buf)
    }
}

impl DeserializeBytes for ValueIndex {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Ok(ValueIndex(u32::deserialize(read_buf)?))
    }
}

/// A different variants of shifting a value.
///
/// Note that there is no shift left arithmetic because it is redundant.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShiftVariant {
    /// Shift logical left.
    Sll,
    /// Shift logical right.
    Slr,
    /// Shift arithmetic right.
    ///
    /// This is similar to the logical shift right but instead of shifting in 0 bits it will
    /// replicate the sign bit.
    Sar,
    /// Rotate right.
    ///
    /// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
    Rotr,
    /// Shift logical left on 32-bit halves.
    ///
    /// Performs independent logical left shifts on the upper and lower 32-bit halves of the word.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    Sll32,
    /// Shift logical right on 32-bit halves.
    ///
    /// Performs independent logical right shifts on the upper and lower 32-bit halves of the word.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    Srl32,
    /// Shift arithmetic right on 32-bit halves.
    ///
    /// Performs independent arithmetic right shifts on the upper and lower 32-bit halves of the
    /// word. Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift
    /// amount (0-31).
    Sra32,
    /// Rotate right on 32-bit halves.
    ///
    /// Performs independent rotate right operations on the upper and lower 32-bit halves of the
    /// word. Bits shifted off the right end wrap around to the left within each 32-bit half.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    Rotr32,
}

impl SerializeBytes for ShiftVariant {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        let index = match self {
            ShiftVariant::Sll => 0u8,
            ShiftVariant::Slr => 1u8,
            ShiftVariant::Sar => 2u8,
            ShiftVariant::Rotr => 3u8,
            ShiftVariant::Sll32 => 4u8,
            ShiftVariant::Srl32 => 5u8,
            ShiftVariant::Sra32 => 6u8,
            ShiftVariant::Rotr32 => 7u8,
        };
        index.serialize(write_buf)
    }
}

impl DeserializeBytes for ShiftVariant {
    fn deserialize(read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let index = u8::deserialize(read_buf)?;
        match index {
            0 => Ok(ShiftVariant::Sll),
            1 => Ok(ShiftVariant::Slr),
            2 => Ok(ShiftVariant::Sar),
            3 => Ok(ShiftVariant::Rotr),
            4 => Ok(ShiftVariant::Sll32),
            5 => Ok(ShiftVariant::Srl32),
            6 => Ok(ShiftVariant::Sra32),
            7 => Ok(ShiftVariant::Rotr32),
            _ => Err(SerializationError::UnknownEnumVariant {
                name: "ShiftVariant",
                index,
            }),
        }
    }
}

/// Similar to [`ValueIndex`], but represents a value that has been shifted by a certain amount.
///
/// This is used in the operands to constraints like [`AndConstraint`].
///
/// The canonical formto represent a value without any shifting is [`ShiftVariant::Sll`] with
/// amount equals 0.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShiftedValueIndex {
    /// The index of this value in the input values vector.
    pub value_index: ValueIndex,
    /// The flavour of the shift that the value must be shifted by.
    pub shift_variant: ShiftVariant,
    /// The number of bits by which the value must be shifted by.
    ///
    /// Must be less than 64.
    pub amount: usize,
}

impl ShiftedValueIndex {
    /// Create a value index that just uses the specified value. Equivalent to [`Self::sll`] with
    /// amount equals 0.
    pub fn plain(value_index: ValueIndex) -> Self {
        Self {
            value_index,
            shift_variant: ShiftVariant::Sll,
            amount: 0,
        }
    }

    /// Shift Left Logical by the given number of bits.
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 64.
    pub fn sll(value_index: ValueIndex, amount: usize) -> Self {
        assert!(amount < 64, "shift amount n={amount} out of range");
        Self {
            value_index,
            shift_variant: ShiftVariant::Sll,
            amount,
        }
    }

    /// Shift Right Logical by the given number of bits.
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 64.
    pub fn srl(value_index: ValueIndex, amount: usize) -> Self {
        assert!(amount < 64, "shift amount n={amount} out of range");
        Self {
            value_index,
            shift_variant: ShiftVariant::Slr,
            amount,
        }
    }

    /// Shift Right Arithmetic by the given number of bits.
    ///
    /// This is similar to the Shift Right Logical but instead of shifting in 0 bits it will
    /// replicate the sign bit.
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 64.
    pub fn sar(value_index: ValueIndex, amount: usize) -> Self {
        assert!(amount < 64, "shift amount n={amount} out of range");
        Self {
            value_index,
            shift_variant: ShiftVariant::Sar,
            amount,
        }
    }

    /// Rotate Right by the given number of bits.
    ///
    /// Rotates bits to the right, with bits shifted off the right end wrapping around to the left.
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 64.
    pub fn rotr(value_index: ValueIndex, amount: usize) -> Self {
        assert!(amount < 64, "shift amount n={amount} out of range");
        Self {
            value_index,
            shift_variant: ShiftVariant::Rotr,
            amount,
        }
    }

    /// Shift Left Logical on 32-bit halves by the given number of bits.
    ///
    /// Performs independent logical left shifts on the upper and lower 32-bit halves.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 32.
    pub fn sll32(value_index: ValueIndex, amount: usize) -> Self {
        assert!(
            amount < 32,
            "shift amount n={amount} out of range for 32-bit shift"
        );
        Self {
            value_index,
            shift_variant: ShiftVariant::Sll32,
            amount,
        }
    }

    /// Shift Right Logical on 32-bit halves by the given number of bits.
    ///
    /// Performs independent logical right shifts on the upper and lower 32-bit halves.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 32.
    pub fn srl32(value_index: ValueIndex, amount: usize) -> Self {
        assert!(
            amount < 32,
            "shift amount n={amount} out of range for 32-bit shift"
        );
        Self {
            value_index,
            shift_variant: ShiftVariant::Srl32,
            amount,
        }
    }

    /// Shift Right Arithmetic on 32-bit halves by the given number of bits.
    ///
    /// Performs independent arithmetic right shifts on the upper and lower 32-bit halves.
    /// Sign extends each 32-bit half independently. Only uses the lower 5 bits of the shift amount
    /// (0-31).
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 32.
    pub fn sra32(value_index: ValueIndex, amount: usize) -> Self {
        assert!(
            amount < 32,
            "shift amount n={amount} out of range for 32-bit shift"
        );
        Self {
            value_index,
            shift_variant: ShiftVariant::Sra32,
            amount,
        }
    }

    /// Rotate Right on 32-bit halves by the given number of bits.
    ///
    /// Performs independent rotate right operations on the upper and lower 32-bit halves.
    /// Bits shifted off the right end wrap around to the left within each 32-bit half.
    /// Only uses the lower 5 bits of the shift amount (0-31).
    ///
    /// # Panics
    /// Panics if the shift amount is greater than or equal to 32.
    pub fn rotr32(value_index: ValueIndex, amount: usize) -> Self {
        assert!(
            amount < 32,
            "shift amount n={amount} out of range for 32-bit rotate"
        );
        Self {
            value_index,
            shift_variant: ShiftVariant::Rotr32,
            amount,
        }
    }
}

impl SerializeBytes for ShiftedValueIndex {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.value_index.serialize(&mut write_buf)?;
        self.shift_variant.serialize(&mut write_buf)?;
        self.amount.serialize(write_buf)
    }
}

impl DeserializeBytes for ShiftedValueIndex {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let value_index = ValueIndex::deserialize(&mut read_buf)?;
        let shift_variant = ShiftVariant::deserialize(&mut read_buf)?;
        let amount = usize::deserialize(read_buf)?;

        // Validate that amount is within valid range
        if amount >= 64 {
            return Err(SerializationError::InvalidConstruction {
                name: "ShiftedValueIndex::amount",
            });
        }

        Ok(ShiftedValueIndex {
            value_index,
            shift_variant,
            amount,
        })
    }
}

/// Operand type.
///
/// An operand in Binius64 is a vector of shifted values. Each item in the vector represents a
/// term in a XOR combination of shifted values.
///
/// To give a couple examples:
///
/// ```ignore
/// vec![] == 0
/// vec![1] == 1
/// vec![1, 1] == 1 ^ 1
/// vec![x >> 5, y << 5] = (x >> 5) ^ (y << 5)
/// ```
pub type Operand = Vec<ShiftedValueIndex>;

/// AND constraint: `A & B = C`.
///
/// This constraint verifies that the bitwise AND of operands A and B equals operand C.
/// Each operand is computed as the XOR of multiple shifted values from the value vector.
#[derive(Debug, Clone, Default)]
pub struct AndConstraint {
    /// Operand A.
    pub a: Operand,
    /// Operand B.
    pub b: Operand,
    /// Operand C.
    pub c: Operand,
}

impl AndConstraint {
    /// Creates a new AND constraint from XOR combinations of the given unshifted values.
    pub fn plain_abc(
        a: impl IntoIterator<Item = ValueIndex>,
        b: impl IntoIterator<Item = ValueIndex>,
        c: impl IntoIterator<Item = ValueIndex>,
    ) -> AndConstraint {
        AndConstraint {
            a: a.into_iter().map(ShiftedValueIndex::plain).collect(),
            b: b.into_iter().map(ShiftedValueIndex::plain).collect(),
            c: c.into_iter().map(ShiftedValueIndex::plain).collect(),
        }
    }

    /// Creates a new AND constraint from XOR combinations of the given shifted values.
    pub fn abc(
        a: impl IntoIterator<Item = ShiftedValueIndex>,
        b: impl IntoIterator<Item = ShiftedValueIndex>,
        c: impl IntoIterator<Item = ShiftedValueIndex>,
    ) -> AndConstraint {
        AndConstraint {
            a: a.into_iter().collect(),
            b: b.into_iter().collect(),
            c: c.into_iter().collect(),
        }
    }
}

impl SerializeBytes for AndConstraint {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.a.serialize(&mut write_buf)?;
        self.b.serialize(&mut write_buf)?;
        self.c.serialize(write_buf)
    }
}

impl DeserializeBytes for AndConstraint {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let a = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
        let b = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
        let c = Vec::<ShiftedValueIndex>::deserialize(read_buf)?;

        Ok(AndConstraint { a, b, c })
    }
}

/// MUL constraint: `A * B = (HI << 64) | LO`.
///
/// 64-bit unsigned integer multiplication producing 128-bit result split into high and low 64-bit
/// words.
#[derive(Debug, Clone, Default)]
pub struct MulConstraint {
    /// A operand.
    pub a: Operand,
    /// B operand.
    pub b: Operand,
    /// HI operand.
    ///
    /// The high 64 bits of the result of the multiplication.
    pub hi: Operand,
    /// LO operand.
    ///
    /// The low 64 bits of the result of the multiplication.
    pub lo: Operand,
}

impl SerializeBytes for MulConstraint {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.a.serialize(&mut write_buf)?;
        self.b.serialize(&mut write_buf)?;
        self.hi.serialize(&mut write_buf)?;
        self.lo.serialize(write_buf)
    }
}

impl DeserializeBytes for MulConstraint {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let a = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
        let b = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
        let hi = Vec::<ShiftedValueIndex>::deserialize(&mut read_buf)?;
        let lo = Vec::<ShiftedValueIndex>::deserialize(read_buf)?;

        Ok(MulConstraint { a, b, hi, lo })
    }
}

/// The ConstraintSystem is the core data structure in Binius64 that defines the computational
/// constraints to be proven in zero-knowledge. It represents a system of equations over 64-bit
/// words that must be satisfied by a valid values vector [`ValueVec`].
///
/// # Clone
///
/// While this type is cloneable it may be expensive to do so since the constraint systems often
/// can have millions of constraints.
#[derive(Debug, Clone)]
pub struct ConstraintSystem {
    /// Description of the value vector layout expected by this constraint system.
    pub value_vec_layout: ValueVecLayout,
    /// The constants that this constraint system defines.
    ///
    /// Those constants will be going to be available for constraints in the value vector. Those
    /// are known to both prover and verifier.
    pub constants: Vec<Word>,
    /// List of AND constraints that must be satisfied by the values vector.
    pub and_constraints: Vec<AndConstraint>,
    /// List of MUL constraints that must be satisfied by the values vector.
    pub mul_constraints: Vec<MulConstraint>,
}

impl ConstraintSystem {
    /// Serialization format version for compatibility checking
    pub const SERIALIZATION_VERSION: u32 = 2;
}

impl ConstraintSystem {
    /// Creates a new constraint system.
    pub fn new(
        constants: Vec<Word>,
        value_vec_layout: ValueVecLayout,
        and_constraints: Vec<AndConstraint>,
        mul_constraints: Vec<MulConstraint>,
    ) -> Self {
        assert_eq!(constants.len(), value_vec_layout.n_const);
        ConstraintSystem {
            constants,
            value_vec_layout,
            and_constraints,
            mul_constraints,
        }
    }

    /// Ensures that this constraint system is well-formed and ready for proving.
    ///
    /// Specifically checks that:
    ///
    /// - the value vec layout is [valid][`ValueVecLayout::validate`].
    /// - every [shifted value index][`ShiftedValueIndex`] is canonical.
    /// - referenced values indices are in the range.
    /// - constraints do not reference values in the padding area.
    /// - shifts amounts are valid.
    pub fn validate(&self) -> Result<(), ConstraintSystemError> {
        // Validate the value vector layout
        self.value_vec_layout.validate()?;

        for i in 0..self.and_constraints.len() {
            validate_operand(
                &self.and_constraints[i].a,
                &self.value_vec_layout,
                "and",
                i,
                "a",
            )?;
            validate_operand(
                &self.and_constraints[i].b,
                &self.value_vec_layout,
                "and",
                i,
                "b",
            )?;
            validate_operand(
                &self.and_constraints[i].c,
                &self.value_vec_layout,
                "and",
                i,
                "c",
            )?;
        }
        for i in 0..self.mul_constraints.len() {
            validate_operand(
                &self.mul_constraints[i].a,
                &self.value_vec_layout,
                "mul",
                i,
                "a",
            )?;
            validate_operand(
                &self.mul_constraints[i].b,
                &self.value_vec_layout,
                "mul",
                i,
                "b",
            )?;
            validate_operand(
                &self.mul_constraints[i].lo,
                &self.value_vec_layout,
                "mul",
                i,
                "lo",
            )?;
            validate_operand(
                &self.mul_constraints[i].hi,
                &self.value_vec_layout,
                "mul",
                i,
                "hi",
            )?;
        }

        return Ok(());

        fn validate_operand(
            operand: &Operand,
            value_vec_layout: &ValueVecLayout,
            constraint_type: &'static str,
            constraint_index: usize,
            operand_name: &'static str,
        ) -> Result<(), ConstraintSystemError> {
            for term in operand {
                // check canonicity. SLL is the canonical form of the operand.
                if term.amount == 0 && term.shift_variant != ShiftVariant::Sll {
                    return Err(ConstraintSystemError::NonCanonicalShift {
                        constraint_type,
                        constraint_index,
                        operand_name,
                    });
                }
                if term.amount >= 64 {
                    return Err(ConstraintSystemError::ShiftAmountTooLarge {
                        constraint_type,
                        constraint_index,
                        operand_name,
                        shift_amount: term.amount,
                    });
                }
                // Check if the value index is out of bounds.
                if value_vec_layout.is_committed_oob(term.value_index) {
                    return Err(ConstraintSystemError::OutOfRangeValueIndex {
                        constraint_type,
                        constraint_index,
                        operand_name,
                        value_index: term.value_index.0,
                        total_len: value_vec_layout.committed_total_len,
                    });
                }
                // No value should refer to padding.
                if value_vec_layout.is_padding(term.value_index) {
                    return Err(ConstraintSystemError::PaddingValueIndex {
                        constraint_type,
                        constraint_index,
                        operand_name,
                    });
                }
            }
            Ok(())
        }
    }

    /// [Validates][`Self::validate`] and prepares this constraint system for proving/verifying.
    ///
    /// This function performs the following:
    /// 1. Validates the value vector layout (including public input checks)
    /// 2. Validates the constraints.
    /// 3. Pads the AND and MUL constraints to the next po2 size
    pub fn validate_and_prepare(&mut self) -> Result<(), ConstraintSystemError> {
        self.validate()?;

        // Both AND and MUL constraint list have requirements wrt their sizes.
        let and_target_size =
            cmp::max(consts::MIN_AND_CONSTRAINTS, self.and_constraints.len()).next_power_of_two();
        let mul_target_size =
            cmp::max(consts::MIN_MUL_CONSTRAINTS, self.mul_constraints.len()).next_power_of_two();

        self.and_constraints
            .resize_with(and_target_size, AndConstraint::default);
        self.mul_constraints
            .resize_with(mul_target_size, MulConstraint::default);

        Ok(())
    }

    #[cfg(test)]
    fn add_and_constraint(&mut self, and_constraint: AndConstraint) {
        self.and_constraints.push(and_constraint);
    }

    #[cfg(test)]
    fn add_mul_constraint(&mut self, mul_constraint: MulConstraint) {
        self.mul_constraints.push(mul_constraint);
    }

    /// Returns the number of AND constraints in the system.
    pub fn n_and_constraints(&self) -> usize {
        self.and_constraints.len()
    }

    /// Returns the number of MUL  constraints in the system.
    pub fn n_mul_constraints(&self) -> usize {
        self.mul_constraints.len()
    }

    /// The total length of the [`ValueVec`] expected by this constraint system.
    pub fn value_vec_len(&self) -> usize {
        self.value_vec_layout.committed_total_len
    }

    /// Create a new [`ValueVec`] with the size expected by this constraint system.
    pub fn new_value_vec(&self) -> ValueVec {
        ValueVec::new(self.value_vec_layout.clone())
    }
}

impl SerializeBytes for ConstraintSystem {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

        self.value_vec_layout.serialize(&mut write_buf)?;
        self.constants.serialize(&mut write_buf)?;
        self.and_constraints.serialize(&mut write_buf)?;
        self.mul_constraints.serialize(write_buf)
    }
}

impl DeserializeBytes for ConstraintSystem {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let version = u32::deserialize(&mut read_buf)?;
        if version != Self::SERIALIZATION_VERSION {
            return Err(SerializationError::InvalidConstruction {
                name: "ConstraintSystem::version",
            });
        }

        let value_vec_layout = ValueVecLayout::deserialize(&mut read_buf)?;
        let constants = Vec::<Word>::deserialize(&mut read_buf)?;
        let and_constraints = Vec::<AndConstraint>::deserialize(&mut read_buf)?;
        let mul_constraints = Vec::<MulConstraint>::deserialize(read_buf)?;

        if constants.len() != value_vec_layout.n_const {
            return Err(SerializationError::InvalidConstruction {
                name: "ConstraintSystem::constants",
            });
        }

        Ok(ConstraintSystem {
            value_vec_layout,
            constants,
            and_constraints,
            mul_constraints,
        })
    }
}

/// Description of a layout of the value vector for a particular circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueVecLayout {
    /// The number of the constants declared by the circuit.
    pub n_const: usize,
    /// The number of the input output parameters declared by the circuit.
    pub n_inout: usize,
    /// The number of the witness parameters declared by the circuit.
    pub n_witness: usize,
    /// The number of the internal values declared by the circuit.
    ///
    /// Those are outputs and intermediaries created by the gates.
    pub n_internal: usize,

    /// The offset at which `inout` parameters start.
    pub offset_inout: usize,
    /// The offset at which `witness` parameters start.
    ///
    /// The public section of the value vec has the power-of-two size and is greater than the
    /// minimum number of words. By public section we mean the constants and the inout values.
    pub offset_witness: usize,
    /// The total number of committed values in the values vector. This does not include any
    /// scratch values.
    ///
    /// This must be a power-of-two.
    pub committed_total_len: usize,
    /// The number of scratch values at the end of the value vec.
    pub n_scratch: usize,
}

impl ValueVecLayout {
    /// Validates that the value vec layout has a correct shape.
    ///
    /// Specifically checks that:
    ///
    /// - the total committed length is a power of two.
    /// - the public segment (constants and inout values) is padded to the power of two.
    /// - the public segment is not less than the minimum size.
    pub fn validate(&self) -> Result<(), ConstraintSystemError> {
        if !self.committed_total_len.is_power_of_two() {
            return Err(ConstraintSystemError::ValueVecLenNotPowerOfTwo);
        }

        if !self.offset_witness.is_power_of_two() {
            return Err(ConstraintSystemError::PublicInputPowerOfTwo);
        }

        let pub_input_size = self.offset_witness;
        if pub_input_size < consts::MIN_WORDS_PER_SEGMENT {
            return Err(ConstraintSystemError::PublicInputTooShort { pub_input_size });
        }

        Ok(())
    }

    /// Returns true if the given index points to an area that is considered to be padding.
    fn is_padding(&self, index: ValueIndex) -> bool {
        let idx = index.0 as usize;

        // padding 1: between constants and inout section
        if idx >= self.n_const && idx < self.offset_inout {
            return true;
        }

        // padding 2: between the end of inout section and the start of witness section
        let end_of_inout = self.offset_inout + self.n_inout;
        if idx >= end_of_inout && idx < self.offset_witness {
            return true;
        }

        // padding 3: between the last internal value and the total len
        let end_of_internal = self.offset_witness + self.n_witness + self.n_internal;
        if idx >= end_of_internal && idx < self.committed_total_len {
            return true;
        }

        false
    }

    /// Returns true if the given index is out-of-bounds for the committed part of this layout.
    fn is_committed_oob(&self, index: ValueIndex) -> bool {
        index.0 as usize >= self.committed_total_len
    }
}

impl SerializeBytes for ValueVecLayout {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.n_const.serialize(&mut write_buf)?;
        self.n_inout.serialize(&mut write_buf)?;
        self.n_witness.serialize(&mut write_buf)?;
        self.n_internal.serialize(&mut write_buf)?;
        self.offset_inout.serialize(&mut write_buf)?;
        self.offset_witness.serialize(&mut write_buf)?;
        self.committed_total_len.serialize(&mut write_buf)?;
        self.n_scratch.serialize(write_buf)
    }
}

impl DeserializeBytes for ValueVecLayout {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let n_const = usize::deserialize(&mut read_buf)?;
        let n_inout = usize::deserialize(&mut read_buf)?;
        let n_witness = usize::deserialize(&mut read_buf)?;
        let n_internal = usize::deserialize(&mut read_buf)?;
        let offset_inout = usize::deserialize(&mut read_buf)?;
        let offset_witness = usize::deserialize(&mut read_buf)?;
        let committed_total_len = usize::deserialize(&mut read_buf)?;
        let n_scratch = usize::deserialize(read_buf)?;

        Ok(ValueVecLayout {
            n_const,
            n_inout,
            n_witness,
            n_internal,
            offset_inout,
            offset_witness,
            committed_total_len,
            n_scratch,
        })
    }
}

/// The vector of values used in constraint evaluation and proof generation.
///
/// `ValueVec` is the concrete instantiation of values that satisfy (or should satisfy) a
/// [`ConstraintSystem`]. It follows the layout defined by [`ValueVecLayout`] and serves
/// as the primary data structure for both constraint evaluation and polynomial commitment.
///
/// Between these sections, there may be padding regions to satisfy alignment requirements.
/// The total size is always a power of two as required for technical reasons.
#[derive(Clone, Debug)]
pub struct ValueVec {
    layout: ValueVecLayout,
    data: Vec<Word>,
}

impl ValueVec {
    /// Creates a new value vector with the given layout.
    ///
    /// The values are filled with zeros.
    pub fn new(layout: ValueVecLayout) -> ValueVec {
        let size = layout.committed_total_len + layout.n_scratch;
        ValueVec {
            layout,
            data: vec![Word::ZERO; size],
        }
    }

    /// Creates a new value vector with the given layout and data.
    ///
    /// The data is checked to have the correct length.
    pub fn new_from_data(
        layout: ValueVecLayout,
        public: Vec<Word>,
        private: Vec<Word>,
    ) -> Result<ValueVec, ConstraintSystemError> {
        let committed_len = public.len() + private.len();
        if committed_len != layout.committed_total_len {
            return Err(ConstraintSystemError::ValueVecLenMismatch {
                expected: layout.committed_total_len,
                actual: committed_len,
            });
        }

        let full_len = layout.committed_total_len + layout.n_scratch;
        let mut data = public;
        data.reserve(full_len);
        data.extend_from_slice(&private);
        data.resize(full_len, Word::ZERO);

        Ok(ValueVec { layout, data })
    }

    /// The total size of the committed portion of the vector (excluding scratch).
    pub fn size(&self) -> usize {
        self.layout.committed_total_len
    }

    /// Returns the value stored at the given index.
    ///
    /// Panics if the index is out of bounds. Will happily return a value from the padding section.
    pub fn get(&self, index: usize) -> Word {
        self.data[index]
    }

    /// Sets the value at the given index.
    ///
    /// Panics if the index is out of bounds. Will gladly assign a value to the padding section.
    pub fn set(&mut self, index: usize, value: Word) {
        self.data[index] = value;
    }

    /// Returns the public portion of the values vector.
    pub fn public(&self) -> &[Word] {
        &self.data[..self.layout.offset_witness]
    }

    /// Return all non-public values (witness + internal) without scratch space.
    pub fn non_public(&self) -> &[Word] {
        &self.data[self.layout.offset_witness..self.layout.committed_total_len]
    }

    /// Returns the witness portion of the values vector.
    pub fn witness(&self) -> &[Word] {
        let start = self.layout.offset_witness;
        let end = start + self.layout.n_witness;
        &self.data[start..end]
    }

    /// Returns the combined values vector.
    pub fn combined_witness(&self) -> &[Word] {
        let start = 0;
        let end = self.layout.committed_total_len;
        &self.data[start..end]
    }
}

impl Index<ValueIndex> for ValueVec {
    type Output = Word;

    fn index(&self, index: ValueIndex) -> &Self::Output {
        &self.data[index.0 as usize]
    }
}

impl IndexMut<ValueIndex> for ValueVec {
    fn index_mut(&mut self, index: ValueIndex) -> &mut Self::Output {
        &mut self.data[index.0 as usize]
    }
}

/// Values data for zero-knowledge proofs (either public witness or non-public part - private inputs
/// and internal values).
///
/// It uses `Cow<[Word]>` to avoid unnecessary clones while supporting
/// both borrowed and owned data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValuesData<'a> {
    data: Cow<'a, [Word]>,
}

impl<'a> ValuesData<'a> {
    /// Serialization format version for compatibility checking
    pub const SERIALIZATION_VERSION: u32 = 1;

    /// Create a new ValuesData from borrowed data
    pub fn borrowed(data: &'a [Word]) -> Self {
        Self {
            data: Cow::Borrowed(data),
        }
    }

    /// Create a new ValuesData from owned data
    pub fn owned(data: Vec<Word>) -> Self {
        Self {
            data: Cow::Owned(data),
        }
    }

    /// Get the values data as a slice
    pub fn as_slice(&self) -> &[Word] {
        &self.data
    }

    /// Get the number of words in the values data
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the witness is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to owned data, consuming self
    pub fn into_owned(self) -> Vec<Word> {
        self.data.into_owned()
    }

    /// Convert to owned version of ValuesData
    pub fn to_owned(&self) -> ValuesData<'static> {
        ValuesData {
            data: Cow::Owned(self.data.to_vec()),
        }
    }
}

impl<'a> SerializeBytes for ValuesData<'a> {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

        self.data.as_ref().serialize(write_buf)
    }
}

impl DeserializeBytes for ValuesData<'static> {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let version = u32::deserialize(&mut read_buf)?;
        if version != Self::SERIALIZATION_VERSION {
            return Err(SerializationError::InvalidConstruction {
                name: "Witness::version",
            });
        }

        let data = Vec::<Word>::deserialize(read_buf)?;

        Ok(ValuesData::owned(data))
    }
}

impl<'a> From<&'a [Word]> for ValuesData<'a> {
    fn from(data: &'a [Word]) -> Self {
        ValuesData::borrowed(data)
    }
}

impl From<Vec<Word>> for ValuesData<'static> {
    fn from(data: Vec<Word>) -> Self {
        ValuesData::owned(data)
    }
}

impl<'a> AsRef<[Word]> for ValuesData<'a> {
    fn as_ref(&self) -> &[Word] {
        self.as_slice()
    }
}

impl<'a> std::ops::Deref for ValuesData<'a> {
    type Target = [Word];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> From<ValuesData<'a>> for Vec<Word> {
    fn from(value: ValuesData<'a>) -> Self {
        value.into_owned()
    }
}

/// A zero-knowledge proof that can be serialized for cross-host verification.
///
/// This structure contains the complete proof transcript generated by the prover,
/// along with information about the challenger type needed for verification.
/// The proof data represents the Fiat-Shamir transcript that can be deserialized
/// by the verifier to recreate the interactive protocol.
///
/// # Design
///
/// The proof contains:
/// - `data`: The actual proof transcript as bytes (zero-copy with Cow)
/// - `challenger_type`: String identifying the challenger used (e.g., `"HasherChallenger<Sha256>"`)
///
/// This enables complete cross-host verification where a proof generated on one
/// machine can be serialized, transmitted, and verified on another machine with
/// the correct challenger configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<'a> {
    data: Cow<'a, [u8]>,
    challenger_type: String,
}

impl<'a> Proof<'a> {
    /// Serialization format version for compatibility checking
    pub const SERIALIZATION_VERSION: u32 = 1;

    /// Create a new Proof from borrowed transcript data
    pub fn borrowed(data: &'a [u8], challenger_type: String) -> Self {
        Self {
            data: Cow::Borrowed(data),
            challenger_type,
        }
    }

    /// Create a new Proof from owned transcript data
    pub fn owned(data: Vec<u8>, challenger_type: String) -> Self {
        Self {
            data: Cow::Owned(data),
            challenger_type,
        }
    }

    /// Get the proof transcript data as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the challenger type identifier
    pub fn challenger_type(&self) -> &str {
        &self.challenger_type
    }

    /// Get the number of bytes in the proof transcript
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the proof transcript is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to owned data, consuming self
    pub fn into_owned(self) -> (Vec<u8>, String) {
        (self.data.into_owned(), self.challenger_type)
    }

    /// Convert to owned version of Proof
    pub fn to_owned(&self) -> Proof<'static> {
        Proof {
            data: Cow::Owned(self.data.to_vec()),
            challenger_type: self.challenger_type.clone(),
        }
    }
}

impl<'a> SerializeBytes for Proof<'a> {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        Self::SERIALIZATION_VERSION.serialize(&mut write_buf)?;

        self.challenger_type.serialize(&mut write_buf)?;

        self.data.as_ref().serialize(write_buf)
    }
}

impl DeserializeBytes for Proof<'static> {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        let version = u32::deserialize(&mut read_buf)?;
        if version != Self::SERIALIZATION_VERSION {
            return Err(SerializationError::InvalidConstruction {
                name: "Proof::version",
            });
        }

        let challenger_type = String::deserialize(&mut read_buf)?;
        let data = Vec::<u8>::deserialize(read_buf)?;

        Ok(Proof::owned(data, challenger_type))
    }
}

impl<'a> From<(&'a [u8], String)> for Proof<'a> {
    fn from((data, challenger_type): (&'a [u8], String)) -> Self {
        Proof::borrowed(data, challenger_type)
    }
}

impl From<(Vec<u8>, String)> for Proof<'static> {
    fn from((data, challenger_type): (Vec<u8>, String)) -> Self {
        Proof::owned(data, challenger_type)
    }
}

impl<'a> AsRef<[u8]> for Proof<'a> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<'a> std::ops::Deref for Proof<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[cfg(test)]
mod serialization_tests {
    use rand::{RngCore, SeedableRng, rngs::StdRng};

    use super::*;

    pub(crate) fn create_test_constraint_system() -> ConstraintSystem {
        let constants = vec![
            Word::from_u64(1),
            Word::from_u64(42),
            Word::from_u64(0xDEADBEEF),
        ];

        let value_vec_layout = ValueVecLayout {
            n_const: 3,
            n_inout: 2,
            n_witness: 10,
            n_internal: 3,
            offset_inout: 4,         // Must be power of 2 and >= n_const
            offset_witness: 8,       // Must be power of 2 and >= offset_inout + n_inout
            committed_total_len: 16, // Must be power of 2 and >= offset_witness + n_witness
            n_scratch: 0,
        };

        let and_constraints = vec![
            AndConstraint::plain_abc(
                vec![ValueIndex(0), ValueIndex(1)],
                vec![ValueIndex(2)],
                vec![ValueIndex(3), ValueIndex(4)],
            ),
            AndConstraint::abc(
                vec![ShiftedValueIndex::sll(ValueIndex(0), 5)],
                vec![ShiftedValueIndex::srl(ValueIndex(1), 10)],
                vec![ShiftedValueIndex::sar(ValueIndex(2), 15)],
            ),
        ];

        let mul_constraints = vec![MulConstraint {
            a: vec![ShiftedValueIndex::plain(ValueIndex(0))],
            b: vec![ShiftedValueIndex::plain(ValueIndex(1))],
            hi: vec![ShiftedValueIndex::plain(ValueIndex(2))],
            lo: vec![ShiftedValueIndex::plain(ValueIndex(3))],
        }];

        ConstraintSystem::new(
            constants,
            value_vec_layout,
            and_constraints,
            mul_constraints,
        )
    }

    #[test]
    fn test_word_serialization_round_trip() {
        let mut rng = StdRng::seed_from_u64(0);
        let word = Word::from_u64(rng.next_u64());

        let mut buf = Vec::new();
        word.serialize(&mut buf).unwrap();

        let deserialized = Word::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(word, deserialized);
    }

    #[test]
    fn test_shift_variant_serialization_round_trip() {
        let variants = [
            ShiftVariant::Sll,
            ShiftVariant::Slr,
            ShiftVariant::Sar,
            ShiftVariant::Rotr,
        ];

        for variant in variants {
            let mut buf = Vec::new();
            variant.serialize(&mut buf).unwrap();

            let deserialized = ShiftVariant::deserialize(&mut buf.as_slice()).unwrap();
            match (variant, deserialized) {
                (ShiftVariant::Sll, ShiftVariant::Sll)
                | (ShiftVariant::Slr, ShiftVariant::Slr)
                | (ShiftVariant::Sar, ShiftVariant::Sar)
                | (ShiftVariant::Rotr, ShiftVariant::Rotr) => {}
                _ => panic!(
                    "ShiftVariant round trip failed: {:?} != {:?}",
                    variant, deserialized
                ),
            }
        }
    }

    #[test]
    fn test_shift_variant_unknown_variant() {
        // Create invalid variant index
        let mut buf = Vec::new();
        255u8.serialize(&mut buf).unwrap();

        let result = ShiftVariant::deserialize(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializationError::UnknownEnumVariant { name, index } => {
                assert_eq!(name, "ShiftVariant");
                assert_eq!(index, 255);
            }
            _ => panic!("Expected UnknownEnumVariant error"),
        }
    }

    #[test]
    fn test_value_index_serialization_round_trip() {
        let value_index = ValueIndex(12345);

        let mut buf = Vec::new();
        value_index.serialize(&mut buf).unwrap();

        let deserialized = ValueIndex::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(value_index, deserialized);
    }

    #[test]
    fn test_shifted_value_index_serialization_round_trip() {
        let shifted_value_index = ShiftedValueIndex::srl(ValueIndex(42), 23);

        let mut buf = Vec::new();
        shifted_value_index.serialize(&mut buf).unwrap();

        let deserialized = ShiftedValueIndex::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(shifted_value_index.value_index, deserialized.value_index);
        assert_eq!(shifted_value_index.amount, deserialized.amount);
        match (
            shifted_value_index.shift_variant,
            deserialized.shift_variant,
        ) {
            (ShiftVariant::Slr, ShiftVariant::Slr) => {}
            _ => panic!("ShiftVariant mismatch"),
        }
    }

    #[test]
    fn test_shifted_value_index_invalid_amount() {
        // Create a buffer with invalid shift amount (>= 64)
        let mut buf = Vec::new();
        ValueIndex(0).serialize(&mut buf).unwrap();
        ShiftVariant::Sll.serialize(&mut buf).unwrap();
        64usize.serialize(&mut buf).unwrap(); // Invalid amount

        let result = ShiftedValueIndex::deserialize(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializationError::InvalidConstruction { name } => {
                assert_eq!(name, "ShiftedValueIndex::amount");
            }
            _ => panic!("Expected InvalidConstruction error"),
        }
    }

    #[test]
    fn test_and_constraint_serialization_round_trip() {
        let constraint = AndConstraint::abc(
            vec![ShiftedValueIndex::sll(ValueIndex(1), 5)],
            vec![ShiftedValueIndex::srl(ValueIndex(2), 10)],
            vec![
                ShiftedValueIndex::sar(ValueIndex(3), 15),
                ShiftedValueIndex::plain(ValueIndex(4)),
            ],
        );

        let mut buf = Vec::new();
        constraint.serialize(&mut buf).unwrap();

        let deserialized = AndConstraint::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(constraint.a.len(), deserialized.a.len());
        assert_eq!(constraint.b.len(), deserialized.b.len());
        assert_eq!(constraint.c.len(), deserialized.c.len());

        for (orig, deser) in constraint.a.iter().zip(deserialized.a.iter()) {
            assert_eq!(orig.value_index, deser.value_index);
            assert_eq!(orig.amount, deser.amount);
        }
    }

    #[test]
    fn test_mul_constraint_serialization_round_trip() {
        let constraint = MulConstraint {
            a: vec![ShiftedValueIndex::plain(ValueIndex(0))],
            b: vec![ShiftedValueIndex::srl(ValueIndex(1), 32)],
            hi: vec![ShiftedValueIndex::plain(ValueIndex(2))],
            lo: vec![ShiftedValueIndex::plain(ValueIndex(3))],
        };

        let mut buf = Vec::new();
        constraint.serialize(&mut buf).unwrap();

        let deserialized = MulConstraint::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(constraint.a.len(), deserialized.a.len());
        assert_eq!(constraint.b.len(), deserialized.b.len());
        assert_eq!(constraint.hi.len(), deserialized.hi.len());
        assert_eq!(constraint.lo.len(), deserialized.lo.len());
    }

    #[test]
    fn test_value_vec_layout_serialization_round_trip() {
        let layout = ValueVecLayout {
            n_const: 5,
            n_inout: 3,
            n_witness: 12,
            n_internal: 7,
            offset_inout: 8,
            offset_witness: 16,
            committed_total_len: 32,
            n_scratch: 0,
        };

        let mut buf = Vec::new();
        layout.serialize(&mut buf).unwrap();

        let deserialized = ValueVecLayout::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(layout, deserialized);
    }

    #[test]
    fn test_constraint_system_serialization_round_trip() {
        let original = create_test_constraint_system();

        let mut buf = Vec::new();
        original.serialize(&mut buf).unwrap();

        let deserialized = ConstraintSystem::deserialize(&mut buf.as_slice()).unwrap();

        // Check version
        assert_eq!(ConstraintSystem::SERIALIZATION_VERSION, 2);

        // Check value_vec_layout
        assert_eq!(original.value_vec_layout, deserialized.value_vec_layout);

        // Check constants
        assert_eq!(original.constants.len(), deserialized.constants.len());
        for (orig, deser) in original.constants.iter().zip(deserialized.constants.iter()) {
            assert_eq!(orig, deser);
        }

        // Check and_constraints
        assert_eq!(
            original.and_constraints.len(),
            deserialized.and_constraints.len()
        );

        // Check mul_constraints
        assert_eq!(
            original.mul_constraints.len(),
            deserialized.mul_constraints.len()
        );
    }

    #[test]
    fn test_constraint_system_version_mismatch() {
        // Create a buffer with wrong version
        let mut buf = Vec::new();
        999u32.serialize(&mut buf).unwrap(); // Wrong version

        let result = ConstraintSystem::deserialize(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializationError::InvalidConstruction { name } => {
                assert_eq!(name, "ConstraintSystem::version");
            }
            _ => panic!("Expected InvalidConstruction error"),
        }
    }

    #[test]
    fn test_constraint_system_constants_length_mismatch() {
        // Create valid components but with mismatched constants length
        let value_vec_layout = ValueVecLayout {
            n_const: 5, // Expect 5 constants
            n_inout: 2,
            n_witness: 10,
            n_internal: 3,
            offset_inout: 8,
            offset_witness: 16,
            committed_total_len: 32,
            n_scratch: 0,
        };

        let constants = vec![Word::from_u64(1), Word::from_u64(2)]; // Only 2 constants
        let and_constraints: Vec<AndConstraint> = vec![];
        let mul_constraints: Vec<MulConstraint> = vec![];

        // Serialize components manually
        let mut buf = Vec::new();
        ConstraintSystem::SERIALIZATION_VERSION
            .serialize(&mut buf)
            .unwrap();
        value_vec_layout.serialize(&mut buf).unwrap();
        constants.serialize(&mut buf).unwrap();
        and_constraints.serialize(&mut buf).unwrap();
        mul_constraints.serialize(&mut buf).unwrap();

        let result = ConstraintSystem::deserialize(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializationError::InvalidConstruction { name } => {
                assert_eq!(name, "ConstraintSystem::constants");
            }
            _ => panic!("Expected InvalidConstruction error"),
        }
    }

    #[test]
    fn test_serialization_with_different_sources() {
        let original = create_test_constraint_system();

        // Test with Vec<u8> (memory buffer)
        let mut vec_buf = Vec::new();
        original.serialize(&mut vec_buf).unwrap();
        let deserialized1 = ConstraintSystem::deserialize(&mut vec_buf.as_slice()).unwrap();
        assert_eq!(original.constants.len(), deserialized1.constants.len());

        // Test with bytes::BytesMut (another common buffer type)
        let mut bytes_buf = bytes::BytesMut::new();
        original.serialize(&mut bytes_buf).unwrap();
        let deserialized2 = ConstraintSystem::deserialize(bytes_buf.freeze()).unwrap();
        assert_eq!(original.constants.len(), deserialized2.constants.len());
    }

    /// Helper function to create or update the reference binary file for version compatibility
    /// testing. This is not run automatically but can be used to regenerate the reference file
    /// when needed.
    #[test]
    #[ignore] // Use `cargo test -- --ignored create_reference_binary` to run this
    fn create_reference_binary_file() {
        let constraint_system = create_test_constraint_system();

        // Serialize to binary data
        let mut buf = Vec::new();
        constraint_system.serialize(&mut buf).unwrap();

        // Write to reference file.
        let test_data_path = std::path::Path::new("test_data/constraint_system_v2.bin");

        // Create directory if it doesn't exist
        if let Some(parent) = test_data_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        std::fs::write(test_data_path, &buf).unwrap();

        println!("Created reference binary file at: {:?}", test_data_path);
        println!("Binary data length: {} bytes", buf.len());
    }

    /// Test deserialization from a reference binary file to ensure version compatibility.
    /// This test will fail if breaking changes are made without incrementing the version.
    #[test]
    fn test_deserialize_from_reference_binary_file() {
        // We now have v2 format with n_scratch field
        // The v1 file is no longer compatible, so we test with v2
        let binary_data = include_bytes!("../test_data/constraint_system_v2.bin");

        let deserialized = ConstraintSystem::deserialize(&mut binary_data.as_slice()).unwrap();

        assert_eq!(deserialized.value_vec_layout.n_const, 3);
        assert_eq!(deserialized.value_vec_layout.n_inout, 2);
        assert_eq!(deserialized.value_vec_layout.n_witness, 10);
        assert_eq!(deserialized.value_vec_layout.n_internal, 3);
        assert_eq!(deserialized.value_vec_layout.offset_inout, 4);
        assert_eq!(deserialized.value_vec_layout.offset_witness, 8);
        assert_eq!(deserialized.value_vec_layout.committed_total_len, 16);
        assert_eq!(deserialized.value_vec_layout.n_scratch, 0);

        assert_eq!(deserialized.constants.len(), 3);
        assert_eq!(deserialized.constants[0].as_u64(), 1);
        assert_eq!(deserialized.constants[1].as_u64(), 42);
        assert_eq!(deserialized.constants[2].as_u64(), 0xDEADBEEF);

        assert_eq!(deserialized.and_constraints.len(), 2);
        assert_eq!(deserialized.mul_constraints.len(), 1);

        // Verify that the version is what we expect
        // This is implicitly checked during deserialization, but we can also verify
        // the file starts with the correct version bytes
        let version_bytes = &binary_data[0..4]; // First 4 bytes should be version
        let expected_version_bytes = 2u32.to_le_bytes(); // Version 2 in little-endian
        assert_eq!(
            version_bytes, expected_version_bytes,
            "Binary file version mismatch. If you made breaking changes, increment ConstraintSystem::SERIALIZATION_VERSION"
        );
    }

    #[test]
    fn test_witness_serialization_round_trip_owned() {
        let data = vec![
            Word::from_u64(1),
            Word::from_u64(42),
            Word::from_u64(0xDEADBEEF),
            Word::from_u64(0x1234567890ABCDEF),
        ];
        let witness = ValuesData::owned(data.clone());

        let mut buf = Vec::new();
        witness.serialize(&mut buf).unwrap();

        let deserialized = ValuesData::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(witness, deserialized);
        assert_eq!(deserialized.as_slice(), data.as_slice());
    }

    #[test]
    fn test_witness_serialization_round_trip_borrowed() {
        let data = vec![Word::from_u64(123), Word::from_u64(456)];
        let witness = ValuesData::borrowed(&data);

        let mut buf = Vec::new();
        witness.serialize(&mut buf).unwrap();

        let deserialized = ValuesData::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(witness, deserialized);
        assert_eq!(deserialized.as_slice(), data.as_slice());
    }

    #[test]
    fn test_witness_version_mismatch() {
        let mut buf = Vec::new();
        999u32.serialize(&mut buf).unwrap(); // Wrong version
        vec![Word::from_u64(1)].serialize(&mut buf).unwrap(); // Some data

        let result = ValuesData::deserialize(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializationError::InvalidConstruction { name } => {
                assert_eq!(name, "Witness::version");
            }
            _ => panic!("Expected version mismatch error"),
        }
    }

    /// Helper function to create or update the reference binary file for Witness version
    /// compatibility testing.
    #[test]
    #[ignore] // Use `cargo test -- --ignored create_witness_reference_binary` to run this
    fn create_witness_reference_binary_file() {
        let data = vec![
            Word::from_u64(1),
            Word::from_u64(42),
            Word::from_u64(0xDEADBEEF),
            Word::from_u64(0x1234567890ABCDEF),
        ];
        let witness = ValuesData::owned(data);

        let mut buf = Vec::new();
        witness.serialize(&mut buf).unwrap();

        let test_data_path = std::path::Path::new("verifier/core/test_data/witness_v1.bin");

        if let Some(parent) = test_data_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        std::fs::write(test_data_path, &buf).unwrap();

        println!(
            "Created Witness reference binary file at: {:?}",
            test_data_path
        );
        println!("Binary data length: {} bytes", buf.len());
    }

    /// Test deserialization from a reference binary file to ensure Witness version
    /// compatibility. This test will fail if breaking changes are made without incrementing the
    /// version.
    #[test]
    fn test_witness_deserialize_from_reference_binary_file() {
        let binary_data = include_bytes!("../test_data/witness_v1.bin");

        let deserialized = ValuesData::deserialize(&mut binary_data.as_slice()).unwrap();

        assert_eq!(deserialized.len(), 4);
        assert_eq!(deserialized.as_slice()[0].as_u64(), 1);
        assert_eq!(deserialized.as_slice()[1].as_u64(), 42);
        assert_eq!(deserialized.as_slice()[2].as_u64(), 0xDEADBEEF);
        assert_eq!(deserialized.as_slice()[3].as_u64(), 0x1234567890ABCDEF);

        // Verify that the version is what we expect
        // This is implicitly checked during deserialization, but we can also verify
        // the file starts with the correct version bytes
        let version_bytes = &binary_data[0..4]; // First 4 bytes should be version
        let expected_version_bytes = 1u32.to_le_bytes(); // Version 1 in little-endian
        assert_eq!(
            version_bytes, expected_version_bytes,
            "WitnessData binary file version mismatch. If you made breaking changes, increment WitnessData::SERIALIZATION_VERSION"
        );
    }

    #[test]
    fn test_proof_serialization_round_trip_owned() {
        let transcript_data = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        let challenger_type = "HasherChallenger<Sha256>".to_string();
        let proof = Proof::owned(transcript_data.clone(), challenger_type.clone());

        let mut buf = Vec::new();
        proof.serialize(&mut buf).unwrap();

        let deserialized = Proof::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(proof, deserialized);
        assert_eq!(deserialized.as_slice(), transcript_data.as_slice());
        assert_eq!(deserialized.challenger_type(), &challenger_type);
    }

    #[test]
    fn test_proof_serialization_round_trip_borrowed() {
        let transcript_data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let challenger_type = "TestChallenger".to_string();
        let proof = Proof::borrowed(&transcript_data, challenger_type.clone());

        let mut buf = Vec::new();
        proof.serialize(&mut buf).unwrap();

        let deserialized = Proof::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(proof, deserialized);
        assert_eq!(deserialized.as_slice(), transcript_data.as_slice());
        assert_eq!(deserialized.challenger_type(), &challenger_type);
    }

    #[test]
    fn test_proof_empty_transcript() {
        let proof = Proof::owned(vec![], "EmptyProof".to_string());
        assert!(proof.is_empty());
        assert_eq!(proof.len(), 0);

        let mut buf = Vec::new();
        proof.serialize(&mut buf).unwrap();

        let deserialized = Proof::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(proof, deserialized);
        assert!(deserialized.is_empty());
    }

    #[test]
    fn test_proof_large_transcript() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut large_data = vec![0u8; 10000];
        rng.fill_bytes(&mut large_data);

        let challenger_type = "LargeProofChallenger".to_string();
        let proof = Proof::owned(large_data.clone(), challenger_type.clone());

        let mut buf = Vec::new();
        proof.serialize(&mut buf).unwrap();

        let deserialized = Proof::deserialize(&mut buf.as_slice()).unwrap();
        assert_eq!(proof, deserialized);
        assert_eq!(deserialized.len(), 10000);
        assert_eq!(deserialized.challenger_type(), &challenger_type);
    }

    #[test]
    fn test_proof_version_mismatch() {
        let mut buf = Vec::new();
        999u32.serialize(&mut buf).unwrap(); // Wrong version
        "TestChallenger".serialize(&mut buf).unwrap(); // Some challenger type
        vec![0xAAu8].serialize(&mut buf).unwrap(); // Some data

        let result = Proof::deserialize(&mut buf.as_slice());
        assert!(result.is_err());
        match result.unwrap_err() {
            SerializationError::InvalidConstruction { name } => {
                assert_eq!(name, "Proof::version");
            }
            _ => panic!("Expected version mismatch error"),
        }
    }

    #[test]
    fn test_proof_into_owned() {
        let original_data = vec![1, 2, 3, 4, 5];
        let original_challenger = "TestChallenger".to_string();
        let proof = Proof::owned(original_data.clone(), original_challenger.clone());

        let (data, challenger_type) = proof.into_owned();
        assert_eq!(data, original_data);
        assert_eq!(challenger_type, original_challenger);
    }

    #[test]
    fn test_proof_to_owned() {
        let data = vec![0xFF, 0xEE, 0xDD];
        let challenger_type = "BorrowedChallenger".to_string();
        let borrowed_proof = Proof::borrowed(&data, challenger_type.clone());

        let owned_proof = borrowed_proof.to_owned();
        assert_eq!(owned_proof.as_slice(), data);
        assert_eq!(owned_proof.challenger_type(), &challenger_type);
        // Verify it's truly owned (not just borrowed)
        drop(data); // This would fail if owned_proof was still borrowing
        assert_eq!(owned_proof.len(), 3);
    }

    #[test]
    fn test_proof_different_challenger_types() {
        let data = vec![0x42];
        let challengers = vec![
            "HasherChallenger<Sha256>".to_string(),
            "HasherChallenger<Blake2b>".to_string(),
            "CustomChallenger".to_string(),
            "".to_string(), // Empty string should also work
        ];

        for challenger_type in challengers {
            let proof = Proof::owned(data.clone(), challenger_type.clone());
            let mut buf = Vec::new();
            proof.serialize(&mut buf).unwrap();

            let deserialized = Proof::deserialize(&mut buf.as_slice()).unwrap();
            assert_eq!(deserialized.challenger_type(), &challenger_type);
        }
    }

    #[test]
    fn test_proof_serialization_with_different_sources() {
        let transcript_data = vec![0x11, 0x22, 0x33, 0x44];
        let challenger_type = "MultiSourceChallenger".to_string();
        let original = Proof::owned(transcript_data, challenger_type);

        // Test with Vec<u8> (memory buffer)
        let mut vec_buf = Vec::new();
        original.serialize(&mut vec_buf).unwrap();
        let deserialized1 = Proof::deserialize(&mut vec_buf.as_slice()).unwrap();
        assert_eq!(original, deserialized1);

        // Test with bytes::BytesMut (another common buffer type)
        let mut bytes_buf = bytes::BytesMut::new();
        original.serialize(&mut bytes_buf).unwrap();
        let deserialized2 = Proof::deserialize(bytes_buf.freeze()).unwrap();
        assert_eq!(original, deserialized2);
    }

    /// Helper function to create or update the reference binary file for Proof version
    /// compatibility testing.
    #[test]
    #[ignore] // Use `cargo test -- --ignored create_proof_reference_binary` to run this
    fn create_proof_reference_binary_file() {
        let transcript_data = vec![
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54,
            0x32, 0x10, 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
        ];
        let challenger_type = "HasherChallenger<Sha256>".to_string();
        let proof = Proof::owned(transcript_data, challenger_type);

        let mut buf = Vec::new();
        proof.serialize(&mut buf).unwrap();

        let test_data_path = std::path::Path::new("verifier/core/test_data/proof_v1.bin");

        if let Some(parent) = test_data_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }

        std::fs::write(test_data_path, &buf).unwrap();

        println!(
            "Created Proof reference binary file at: {:?}",
            test_data_path
        );
        println!("Binary data length: {} bytes", buf.len());
    }

    /// Test deserialization from a reference binary file to ensure Proof version
    /// compatibility. This test will fail if breaking changes are made without incrementing the
    /// version.
    #[test]
    fn test_proof_deserialize_from_reference_binary_file() {
        let binary_data = include_bytes!("../test_data/proof_v1.bin");

        let deserialized = Proof::deserialize(&mut binary_data.as_slice()).unwrap();

        assert_eq!(deserialized.len(), 24); // 24 bytes of transcript data
        assert_eq!(deserialized.challenger_type(), "HasherChallenger<Sha256>");

        let expected_data = vec![
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54,
            0x32, 0x10, 0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE,
        ];
        assert_eq!(deserialized.as_slice(), expected_data);

        // Verify that the version is what we expect
        // This is implicitly checked during deserialization, but we can also verify
        // the file starts with the correct version bytes
        let version_bytes = &binary_data[0..4]; // First 4 bytes should be version
        let expected_version_bytes = 1u32.to_le_bytes(); // Version 1 in little-endian
        assert_eq!(
            version_bytes, expected_version_bytes,
            "Proof binary file version mismatch. If you made breaking changes, increment Proof::SERIALIZATION_VERSION"
        );
    }

    #[test]
    fn split_values_vec_and_combine() {
        let values = ValueVec::new(ValueVecLayout {
            n_const: 2,
            n_inout: 2,
            n_witness: 2,
            n_internal: 2,
            offset_inout: 2,
            offset_witness: 4,
            committed_total_len: 8,
            n_scratch: 0,
        });

        let public = values.public();
        let non_public = values.non_public();
        let combined =
            ValueVec::new_from_data(values.layout.clone(), public.to_vec(), non_public.to_vec())
                .unwrap();
        assert_eq!(combined.combined_witness(), values.combined_witness());
    }

    #[test]
    fn test_roundtrip_cs_and_witnesses_reconstruct_valuevec_with_scratch() {
        // Layout with non-zero scratch. Public = 8, total committed = 16, scratch = 5
        let layout = ValueVecLayout {
            n_const: 2,
            n_inout: 3,
            n_witness: 4,
            n_internal: 3,
            offset_inout: 4,   // >= n_const and power of two
            offset_witness: 8, // >= offset_inout + n_inout and power of two
            committed_total_len: 16,
            n_scratch: 5, // non-zero scratch
        };

        let constants = vec![Word::from_u64(11), Word::from_u64(22)];
        let cs = ConstraintSystem::new(constants, layout.clone(), vec![], vec![]);

        // Build a ValueVec and fill both committed and scratch with non-zero data
        let mut values = cs.new_value_vec();
        let full_len = layout.committed_total_len + layout.n_scratch;
        for i in 0..full_len {
            // Deterministic pattern
            let val = Word::from_u64(0xA5A5_5A5A ^ (i as u64 * 0x9E37_79B9));
            values.set(i, val);
        }

        // Split into public and non-public witnesses and serialize all artifacts
        let public_data = ValuesData::from(values.public());
        let non_public_data = ValuesData::from(values.non_public());

        let mut buf_cs = Vec::new();
        cs.serialize(&mut buf_cs).unwrap();

        let mut buf_pub = Vec::new();
        public_data.serialize(&mut buf_pub).unwrap();

        let mut buf_non_pub = Vec::new();
        non_public_data.serialize(&mut buf_non_pub).unwrap();

        // Deserialize everything back
        let cs2 = ConstraintSystem::deserialize(&mut buf_cs.as_slice()).unwrap();
        let pub2 = ValuesData::deserialize(&mut buf_pub.as_slice()).unwrap();
        let non_pub2 = ValuesData::deserialize(&mut buf_non_pub.as_slice()).unwrap();

        // Reconstruct ValueVec from deserialized pieces
        let reconstructed = ValueVec::new_from_data(
            cs2.value_vec_layout.clone(),
            pub2.into_owned(),
            non_pub2.into_owned(),
        )
        .unwrap();

        // Ensure committed part matches exactly
        assert_eq!(reconstructed.combined_witness(), values.combined_witness());

        // Scratch is not serialized; reconstructed scratch should be zero-filled
        let scratch_start = layout.committed_total_len;
        let scratch_end = scratch_start + layout.n_scratch;
        for i in scratch_start..scratch_end {
            assert_eq!(
                reconstructed.get(i),
                Word::ZERO,
                "scratch index {i} should be zero"
            );
        }
    }

    #[test]
    fn test_is_padding_comprehensive() {
        // Test layout with all types of padding
        let layout = ValueVecLayout {
            n_const: 2,              // constants at indices 0-1
            n_inout: 3,              // inout at indices 4-6
            n_witness: 5,            // witness at indices 16-20
            n_internal: 10,          // internal at indices 21-30
            offset_inout: 4,         // gap between constants and inout (indices 2-3 are padding)
            offset_witness: 16,      // public section is 16 (power of 2), gap 7-15 is padding
            committed_total_len: 64, // total must be power of 2, gap 31-63 is padding
            n_scratch: 0,
        };

        // Test constants (indices 0-1): NOT padding
        assert!(
            !layout.is_padding(ValueIndex(0)),
            "index 0 should be constant"
        );
        assert!(
            !layout.is_padding(ValueIndex(1)),
            "index 1 should be constant"
        );

        // Test padding between constants and inout (indices 2-3): PADDING
        assert!(
            layout.is_padding(ValueIndex(2)),
            "index 2 should be padding between const and inout"
        );
        assert!(
            layout.is_padding(ValueIndex(3)),
            "index 3 should be padding between const and inout"
        );

        // Test inout values (indices 4-6): NOT padding
        assert!(!layout.is_padding(ValueIndex(4)), "index 4 should be inout");
        assert!(!layout.is_padding(ValueIndex(5)), "index 5 should be inout");
        assert!(!layout.is_padding(ValueIndex(6)), "index 6 should be inout");

        // Test padding between inout and witness (indices 7-15): PADDING
        for i in 7..16 {
            assert!(
                layout.is_padding(ValueIndex(i)),
                "index {} should be padding between inout and witness",
                i
            );
        }

        // Test witness values (indices 16-20): NOT padding
        for i in 16..21 {
            assert!(
                !layout.is_padding(ValueIndex(i)),
                "index {} should be witness",
                i
            );
        }

        // Test internal values (indices 21-30): NOT padding
        for i in 21..31 {
            assert!(
                !layout.is_padding(ValueIndex(i)),
                "index {} should be internal",
                i
            );
        }

        // Test padding after internal values (indices 31-63): PADDING
        for i in 31..64 {
            assert!(
                layout.is_padding(ValueIndex(i)),
                "index {} should be padding after internal",
                i
            );
        }
    }

    #[test]
    fn test_is_padding_minimal_layout() {
        // Test a minimal layout with no gaps except required end padding
        let layout = ValueVecLayout {
            n_const: 4,              // constants at indices 0-3
            n_inout: 4,              // inout at indices 4-7
            n_witness: 4,            // witness at indices 8-11
            n_internal: 4,           // internal at indices 12-15
            offset_inout: 4,         // no gap between constants and inout
            offset_witness: 8,       // no gap between inout and witness
            committed_total_len: 16, // exactly fits all values
            n_scratch: 0,
        };

        // No padding anywhere in this layout
        for i in 0..16 {
            assert!(
                !layout.is_padding(ValueIndex(i)),
                "index {} should not be padding in minimal layout",
                i
            );
        }
    }

    #[test]
    fn test_is_padding_public_section_min_size() {
        // Test layout where public section must be padded to meet MIN_WORDS_PER_SEGMENT
        let layout = ValueVecLayout {
            n_const: 1,              // only 1 constant
            n_inout: 1,              // only 1 inout
            n_witness: 2,            // 2 witness values
            n_internal: 2,           // 2 internal values
            offset_inout: 4,         // padding between const and inout to reach min size
            offset_witness: 8,       // public section padded to 8 (MIN_WORDS_PER_SEGMENT)
            committed_total_len: 16, // power of 2
            n_scratch: 0,
        };

        // Test the single constant
        assert!(
            !layout.is_padding(ValueIndex(0)),
            "index 0 should be constant"
        );

        // Test padding between constant and inout (indices 1-3)
        assert!(
            layout.is_padding(ValueIndex(1)),
            "index 1 should be padding"
        );
        assert!(
            layout.is_padding(ValueIndex(2)),
            "index 2 should be padding"
        );
        assert!(
            layout.is_padding(ValueIndex(3)),
            "index 3 should be padding"
        );

        // Test the single inout value
        assert!(!layout.is_padding(ValueIndex(4)), "index 4 should be inout");

        // Test padding between inout and witness (indices 5-7)
        assert!(
            layout.is_padding(ValueIndex(5)),
            "index 5 should be padding"
        );
        assert!(
            layout.is_padding(ValueIndex(6)),
            "index 6 should be padding"
        );
        assert!(
            layout.is_padding(ValueIndex(7)),
            "index 7 should be padding"
        );

        // Test witness values (indices 8-9)
        assert!(
            !layout.is_padding(ValueIndex(8)),
            "index 8 should be witness"
        );
        assert!(
            !layout.is_padding(ValueIndex(9)),
            "index 9 should be witness"
        );

        // Test internal values (indices 10-11)
        assert!(
            !layout.is_padding(ValueIndex(10)),
            "index 10 should be internal"
        );
        assert!(
            !layout.is_padding(ValueIndex(11)),
            "index 11 should be internal"
        );

        // Test padding at the end (indices 12-15)
        for i in 12..16 {
            assert!(
                layout.is_padding(ValueIndex(i)),
                "index {} should be end padding",
                i
            );
        }
    }

    #[test]
    fn test_is_padding_boundary_conditions() {
        let layout = ValueVecLayout {
            n_const: 2,
            n_inout: 2,
            n_witness: 4,
            n_internal: 4,
            offset_inout: 4,
            offset_witness: 8,
            committed_total_len: 16,
            n_scratch: 0,
        };

        // Test exact boundaries
        assert!(
            !layout.is_padding(ValueIndex(1)),
            "last constant should not be padding"
        );
        assert!(
            layout.is_padding(ValueIndex(2)),
            "first padding after const should be padding"
        );

        assert!(
            layout.is_padding(ValueIndex(3)),
            "last padding before inout should be padding"
        );
        assert!(
            !layout.is_padding(ValueIndex(4)),
            "first inout should not be padding"
        );

        assert!(
            !layout.is_padding(ValueIndex(5)),
            "last inout should not be padding"
        );
        assert!(
            layout.is_padding(ValueIndex(6)),
            "first padding after inout should be padding"
        );

        assert!(
            layout.is_padding(ValueIndex(7)),
            "last padding before witness should be padding"
        );
        assert!(
            !layout.is_padding(ValueIndex(8)),
            "first witness should not be padding"
        );

        assert!(
            !layout.is_padding(ValueIndex(11)),
            "last witness should not be padding"
        );
        assert!(
            !layout.is_padding(ValueIndex(12)),
            "first internal should not be padding"
        );

        assert!(
            !layout.is_padding(ValueIndex(15)),
            "last internal should not be padding"
        );
        // Note: index 16 would be out of bounds, not tested here
    }

    #[test]
    fn test_validate_rejects_padding_references() {
        let mut cs = ConstraintSystem::new(
            vec![Word::from_u64(1)],
            ValueVecLayout {
                n_const: 1,
                n_inout: 1,
                n_witness: 2,
                n_internal: 2,
                offset_inout: 4,
                offset_witness: 8,
                committed_total_len: 16,
                n_scratch: 0,
            },
            vec![],
            vec![],
        );

        // Add constraint that references padding (index 2 is padding between const and inout)
        cs.add_and_constraint(AndConstraint::plain_abc(
            vec![ValueIndex(0)], // valid constant
            vec![ValueIndex(2)], // PADDING!
            vec![ValueIndex(8)], // valid witness
        ));

        let result = cs.validate_and_prepare();
        assert!(
            result.is_err(),
            "Should reject constraint referencing padding"
        );

        match result.unwrap_err() {
            ConstraintSystemError::PaddingValueIndex {
                constraint_type, ..
            } => {
                assert_eq!(constraint_type, "and");
            }
            other => panic!("Expected PaddingValueIndex error, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_accepts_non_padding_references() {
        let mut cs = ConstraintSystem::new(
            vec![Word::from_u64(1), Word::from_u64(2)],
            ValueVecLayout {
                n_const: 2,
                n_inout: 2,
                n_witness: 4,
                n_internal: 4,
                offset_inout: 2,
                offset_witness: 4,
                committed_total_len: 16,
                n_scratch: 0,
            },
            vec![],
            vec![],
        );

        // Add constraint that only references valid non-padding indices
        cs.add_and_constraint(AndConstraint::plain_abc(
            vec![ValueIndex(0), ValueIndex(1)], // constants
            vec![ValueIndex(2), ValueIndex(3)], // inout
            vec![ValueIndex(4), ValueIndex(5)], // witness
        ));

        cs.add_mul_constraint(MulConstraint {
            a: vec![ShiftedValueIndex::plain(ValueIndex(6))], // witness
            b: vec![ShiftedValueIndex::plain(ValueIndex(7))], // witness
            hi: vec![ShiftedValueIndex::plain(ValueIndex(8))], // internal
            lo: vec![ShiftedValueIndex::plain(ValueIndex(9))], // internal
        });

        let result = cs.validate_and_prepare();
        assert!(
            result.is_ok(),
            "Should accept constraints with only valid references: {:?}",
            result
        );
    }

    #[test]
    fn test_is_padding_matches_compiler_requirements() {
        // Test that is_padding correctly handles the MIN_WORDS_PER_SEGMENT requirement
        // as seen in the compiler mod.rs:
        // cur_index = cur_index.max(MIN_WORDS_PER_SEGMENT as u32);
        // cur_index = cur_index.next_power_of_two();

        // Case 1: Very small public section (1 const + 1 inout = 2 total)
        // Should be padded to MIN_WORDS_PER_SEGMENT (8)
        let layout1 = ValueVecLayout {
            n_const: 1,
            n_inout: 1,
            n_witness: 4,
            n_internal: 4,
            offset_inout: 1,   // right after constants
            offset_witness: 8, // padded to MIN_WORDS_PER_SEGMENT
            committed_total_len: 16,
            n_scratch: 0,
        };

        // Verify padding between end of inout (index 2) and offset_witness (8)
        assert!(
            !layout1.is_padding(ValueIndex(0)),
            "const should not be padding"
        );
        assert!(
            !layout1.is_padding(ValueIndex(1)),
            "inout should not be padding"
        );
        for i in 2..8 {
            assert!(
                layout1.is_padding(ValueIndex(i)),
                "index {} should be padding to meet MIN_WORDS_PER_SEGMENT",
                i
            );
        }

        // Case 2: Public section exactly MIN_WORDS_PER_SEGMENT (no extra padding needed)
        let layout2 = ValueVecLayout {
            n_const: 4,
            n_inout: 4,
            n_witness: 8,
            n_internal: 0,
            offset_inout: 4,
            offset_witness: 8, // exactly MIN_WORDS_PER_SEGMENT, already power of 2
            committed_total_len: 16,
            n_scratch: 0,
        };

        // No padding in public section
        for i in 0..8 {
            assert!(
                !layout2.is_padding(ValueIndex(i)),
                "index {} should not be padding",
                i
            );
        }

        // Case 3: Public section between MIN_WORDS_PER_SEGMENT and next power of 2
        // e.g., 10 total needs to round up to 16
        let layout3 = ValueVecLayout {
            n_const: 5,
            n_inout: 5,
            n_witness: 16,
            n_internal: 0,
            offset_inout: 5,
            offset_witness: 16, // rounded up from 10 to 16 (next power of 2)
            committed_total_len: 32,
            n_scratch: 0,
        };

        // Check padding from end of inout (10) to offset_witness (16)
        for i in 0..5 {
            assert!(
                !layout3.is_padding(ValueIndex(i)),
                "const {} should not be padding",
                i
            );
        }
        for i in 5..10 {
            assert!(
                !layout3.is_padding(ValueIndex(i)),
                "inout {} should not be padding",
                i
            );
        }
        for i in 10..16 {
            assert!(
                layout3.is_padding(ValueIndex(i)),
                "index {} should be padding for power-of-2 alignment",
                i
            );
        }

        // Case 4: Test with offsets that show all three padding types
        let layout4 = ValueVecLayout {
            n_const: 2,              // indices 0-1
            n_inout: 2,              // indices 8-9
            n_witness: 4,            // indices 16-19
            n_internal: 4,           // indices 20-23
            offset_inout: 8,         // padding after constants to align
            offset_witness: 16,      // padding after inout to reach power of 2
            committed_total_len: 32, // padding after internal to reach total
            n_scratch: 0,
        };

        // Constants
        assert!(!layout4.is_padding(ValueIndex(0)));
        assert!(!layout4.is_padding(ValueIndex(1)));

        // Padding between constants and inout (indices 2-7)
        for i in 2..8 {
            assert!(
                layout4.is_padding(ValueIndex(i)),
                "padding between const and inout at {}",
                i
            );
        }

        // Inout values
        assert!(!layout4.is_padding(ValueIndex(8)));
        assert!(!layout4.is_padding(ValueIndex(9)));

        // Padding between inout and witness (indices 10-15)
        for i in 10..16 {
            assert!(
                layout4.is_padding(ValueIndex(i)),
                "padding between inout and witness at {}",
                i
            );
        }

        // Witness values
        for i in 16..20 {
            assert!(!layout4.is_padding(ValueIndex(i)), "witness at {}", i);
        }

        // Internal values
        for i in 20..24 {
            assert!(!layout4.is_padding(ValueIndex(i)), "internal at {}", i);
        }

        // Padding after internal to total_len (indices 24-31)
        for i in 24..32 {
            assert!(
                layout4.is_padding(ValueIndex(i)),
                "padding after internal at {}",
                i
            );
        }
    }

    #[test]
    fn test_validate_rejects_out_of_range_indices() {
        let mut cs = ConstraintSystem::new(
            vec![Word::from_u64(1)],
            ValueVecLayout {
                n_const: 1,
                n_inout: 1,
                n_witness: 2,
                n_internal: 2,
                offset_inout: 4,
                offset_witness: 8,
                committed_total_len: 16,
                n_scratch: 0,
            },
            vec![],
            vec![],
        );

        // Add AND constraint that references an out-of-range index
        cs.add_and_constraint(AndConstraint::plain_abc(
            vec![ValueIndex(0)],  // valid constant
            vec![ValueIndex(16)], // OUT OF RANGE! (total_len is 16, so max valid index is 15)
            vec![ValueIndex(8)],  // valid witness
        ));

        let result = cs.validate_and_prepare();
        assert!(
            result.is_err(),
            "Should reject constraint with out-of-range index"
        );

        match result.unwrap_err() {
            ConstraintSystemError::OutOfRangeValueIndex {
                constraint_type,
                operand_name,
                value_index,
                total_len,
                ..
            } => {
                assert_eq!(constraint_type, "and");
                assert_eq!(operand_name, "b");
                assert_eq!(value_index, 16);
                assert_eq!(total_len, 16);
            }
            other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_rejects_out_of_range_in_mul_constraint() {
        let mut cs = ConstraintSystem::new(
            vec![Word::from_u64(1), Word::from_u64(2)],
            ValueVecLayout {
                n_const: 2,
                n_inout: 2,
                n_witness: 4,
                n_internal: 4,
                offset_inout: 2,
                offset_witness: 4,
                committed_total_len: 16,
                n_scratch: 0,
            },
            vec![],
            vec![],
        );

        // Add MUL constraint with out-of-range index in 'hi' operand
        cs.add_mul_constraint(MulConstraint {
            a: vec![ShiftedValueIndex::plain(ValueIndex(0))], // valid
            b: vec![ShiftedValueIndex::plain(ValueIndex(1))], // valid
            hi: vec![ShiftedValueIndex::plain(ValueIndex(100))], // WAY out of range!
            lo: vec![ShiftedValueIndex::plain(ValueIndex(3))], // valid
        });

        let result = cs.validate_and_prepare();
        assert!(
            result.is_err(),
            "Should reject MUL constraint with out-of-range index"
        );

        match result.unwrap_err() {
            ConstraintSystemError::OutOfRangeValueIndex {
                constraint_type,
                operand_name,
                value_index,
                total_len,
                ..
            } => {
                assert_eq!(constraint_type, "mul");
                assert_eq!(operand_name, "hi");
                assert_eq!(value_index, 100);
                assert_eq!(total_len, 16);
            }
            other => panic!("Expected OutOfRangeValueIndex error, got: {:?}", other),
        }
    }

    #[test]
    fn test_validate_checks_out_of_range_before_padding() {
        // This test verifies that out-of-range checking happens before padding checking
        // by using an index that is both out-of-range AND would be in a padding area if it were
        // valid
        let mut cs = ConstraintSystem::new(
            vec![Word::from_u64(1)],
            ValueVecLayout {
                n_const: 1,
                n_inout: 1,
                n_witness: 2,
                n_internal: 2,
                offset_inout: 4,
                offset_witness: 8,
                committed_total_len: 16,
                n_scratch: 0,
            },
            vec![],
            vec![],
        );

        // Index 20 is out of range (>= 16)
        // If it were in range, indices 2-3 and 6-7 would be padding
        cs.add_and_constraint(AndConstraint::plain_abc(
            vec![ValueIndex(0)],
            vec![ValueIndex(20)], // out of range
            vec![ValueIndex(8)],
        ));

        let result = cs.validate_and_prepare();
        assert!(result.is_err());

        // Should get OutOfRangeValueIndex, not PaddingValueIndex
        match result.unwrap_err() {
            ConstraintSystemError::OutOfRangeValueIndex { .. } => {
                // Good, out-of-range was detected first
            }
            other => panic!(
                "Expected OutOfRangeValueIndex to be detected before padding check, got: {:?}",
                other
            ),
        }
    }
}
