// Copyright 2025 Irreducible Inc.

use std::{iter, mem, ops::Range};

use binius_core::{
    ShiftVariant,
    constraint_system::{
        AndConstraint, ConstraintSystem, MulConstraint, Operand, ShiftedValueIndex,
    },
    consts::LOG_WORD_SIZE_BITS,
};
use binius_field::Field;
use binius_utils::serialization::{DeserializeBytes, SerializationError, SerializeBytes};
use bytes::{Buf, BufMut};

use super::{BITAND_ARITY, INTMUL_ARITY, PreparedOperatorData};

/// Represents the type of operations handled by the shift protocol.
///
/// The shift protocol supports two fundamental operation types that correspond
/// to the constraint types in Binius64:
///
/// # Operation Types
///
/// - **BitwiseAnd**: Corresponds to AND constraints of the form `A & B ^ C = 0`
/// - **IntegerMul**: Corresponds to MUL constraints of the form `A * B = (HI << 64) | LO`
///
/// These operations work with shifted value indices to efficiently encode
/// computations on 64-bit words without requiring separate shift constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u16)]
pub enum Operation {
    BitwiseAnd,
    IntegerMul,
}

/// A `Key` specifies an operation, an identifier for a 2D matrix, and a range of constraint
/// indices.
///
/// The matrix encodes constraint information with respect exactly one operand of that operation,
/// one shift variant, and one shift amount. Every `Key` corresponds to a unique word (not
/// referenced) in the `Key`. The `range` specifies a range within a list of constraint indices,
/// those constraint indices in which the word participates with respect to the key. If constraint
/// index `i` is among the values within the range, that means the word participates in constraint
/// `i` of operation `operation` as part of the operand encoded in the `id`, with the word shifted
/// with the shift variant and amount also encoded in the `id`.
///
/// # Relationship to Formal Specification
///
/// The paper defines one `M` multilinear polynomial for each (operation, operand, shift variant)
/// tuple. Each `M` multilinear forms a 3D matrix that decomposes into `WORD_SIZE_BITS`
/// 2D matrices. Each `Key` corresponds to one such 2D matrix. We operate at 2D granularity
/// because the prover performs field operations on 2D matrices during both protocol phases.
///
/// # Structure
///
/// - **Operation**: Constraint type (AND or MUL)
/// - **ID**: Packed encoding of operand index, shift variant, and shift amount
/// - **Range**: Constraint indices where this shifted word appears
///
/// # ID Encoding
///
/// The `id` packs three values:
/// - Operand index (which operand in the constraint)
/// - Shift variant (logical left, logical right, arithmetic right)
/// - Shift amount (0 to `WORD_SIZE_BITS-1` bits)
///
/// This ordering places shift information (fundamental to Binius64) in lower bits,
/// with operation and operand data in higher bits. Future operations can simply extend
/// the `id` range with higher bits without breaking the semantic meaning of lower bits.
///
/// # Performance Considerations
///
/// The operation remains separate from `id` for cleaner code organization with no
/// performance cost. During proving, only the operation needs extraction while
/// the packed operand index, shift variant, and shift amount remain undifferentiated.
#[derive(Debug, Clone)]
pub struct Key {
    pub operation: Operation,
    pub id: u16,
    pub range: Range<u32>,
}

impl Key {
    /// Accumulates the partial evaluations of an operation matrix for the key, partitioned by
    /// operand index.
    ///
    /// A [`Key`] references the operation constraints where one witness word is an operand. This
    /// accumulates the partial evaluation of the operation matrix for this key.
    ///
    /// ## Returns
    /// An iterator of tuples, where the first is the operand ID in the operation and the second is
    /// the accumulated value of the partial evaluation tensor.
    #[inline]
    pub fn accumulate_by_operand<'a, F: Field>(
        &'a self,
        constraint_indices: &'a [ConstraintIndex],
        operator_data: &'a PreparedOperatorData<F>,
    ) -> impl Iterator<Item = (usize, F)> + 'a {
        let Range { start, end } = self.range;

        let mut iter = constraint_indices[start as usize..end as usize].iter();
        let mut acc = F::ZERO;
        let mut maybe_current = iter.next();
        iter::from_fn(move || {
            let current = maybe_current?;

            acc += operator_data.r_x_prime_tensor.as_ref()[current.constraint_index as usize];
            for next in &mut iter {
                maybe_current = Some(next);
                if next.operand_index != current.operand_index {
                    let ret = mem::take(&mut acc);
                    return Some((current.operand_index as usize, ret));
                }
                acc += operator_data.r_x_prime_tensor.as_ref()[next.constraint_index as usize];
            }

            maybe_current = None;
            Some((current.operand_index as usize, mem::take(&mut acc)))
        })
    }

    /// Accumulates the partial evaluation of an operation matrix for the key.
    ///
    /// A [`Key`] references the operation constraints where one witness word is an operand. This
    /// accumulates the partial evaluation of the operation matrix for this key.
    #[inline]
    pub fn accumulate<F: Field>(
        &self,
        constraint_indices: &[ConstraintIndex],
        operator_data: &PreparedOperatorData<F>,
    ) -> F {
        self.accumulate_by_operand(constraint_indices, operator_data)
            .map(|(operand_index, acc)| acc * operator_data.lambda_powers[operand_index])
            .sum()
    }
}

/// A collection of keys that organizes the prover's view of the constraint system.
///
/// The prover operates in both phases by iterating through `key_ranges` (one range per witness
/// word), then accessing the corresponding keys in the `keys` vector. Each key contains a range
/// that indexes into `constraint_indices` to identify which constraints involve that
/// particular shifted operand.
///
/// # Structure
///
/// - **keys**: All keys flattened into a single vector
/// - **key_ranges**: For every word there is a range of keys within the `keys` vector
/// - **constraint_indices**: Flattened list of constraint indices referenced by the keys
///
/// # Organization
///
/// Keys are organized by word index for efficient batch processing. For word `w`,
/// `key_ranges[w]` gives the range of keys in the `keys` vector that correspond to that word.
/// Each key's range field then points into `constraint_indices` to specify which constraints
/// involve that particular shifted operand.
#[derive(Debug, Clone)]
pub struct KeyCollection {
    pub keys: Vec<Key>,
    pub key_ranges: Vec<Range<u32>>,
    pub constraint_indices: Vec<ConstraintIndex>,
}

/// A `BuilderKey` is a key that is being built up during `KeyCollection`
/// construction. It is a temporary structure that is later transformed
/// into a `Key`.
///
/// It differs from a `Key` by storing a vector of constraint indices directly,
/// rather than a range that indexes into the flattened `constraint_indices` vector.
/// During construction, these indices are later flattened to create the final `Key`.
struct BuilderKey {
    pub id: u16,
    pub operation: Operation,
    pub constraint_indices: Vec<ConstraintIndex>,
}

/// Indexes a reference to a shifted value index, appearing in a constraint operand.
#[derive(Debug, Clone)]
pub struct ConstraintIndex {
    operand_index: u8,
    constraint_index: u32,
}

/// Updates the list of `BuilderKey` objects with respect to an operand of an operation during
/// `KeyCollection` construction.
fn update_with_operand(
    operation: Operation,
    operand_index: usize,
    operand_values: impl Iterator<Item = impl AsRef<Operand>>,
    builder_key_lists: &mut [Vec<BuilderKey>],
) {
    for (constraint_idx, operand_value) in operand_values.enumerate() {
        // Each operand value is a Vec<ShiftedValueIndex> - multiple shifted word references
        for ShiftedValueIndex {
            value_index,
            shift_variant,
            amount,
        } in operand_value.as_ref()
        {
            // Access and update the builder keys corresponding to the word index (`value_index.0`)
            let builder_keys = &mut builder_key_lists[value_index.0 as usize];
            // Encode (shift_variant, shift_amount) into a single ID
            let shift_variant_val: u16 = match shift_variant {
                ShiftVariant::Sll => 0,
                ShiftVariant::Slr => 1,
                ShiftVariant::Sar => 2,
                ShiftVariant::Rotr => 3,
                ShiftVariant::Sll32 => 4,
                ShiftVariant::Srl32 => 5,
                ShiftVariant::Sra32 => 6,
                ShiftVariant::Rotr32 => 7,
            };
            let id = (shift_variant_val << LOG_WORD_SIZE_BITS) + *amount as u16;

            // Find existing builder key or create a new one for this (operation, id) pair
            let constraint_index = ConstraintIndex {
                operand_index: operand_index as u8,
                constraint_index: constraint_idx as u32,
            };
            if let Some(builder_key) = builder_keys
                .iter_mut()
                .find(|key| key.id == id && key.operation == operation)
            {
                builder_key.constraint_indices.push(constraint_index);
            } else {
                builder_keys.push(BuilderKey {
                    id,
                    operation,
                    constraint_indices: vec![constraint_index],
                });
            }
        }
    }
}

/// Constructs a `KeyCollection` from a constraint system.
pub fn build_key_collection(cs: &ConstraintSystem) -> KeyCollection {
    // Initialize a temporary list of builder keys lists, one for each word
    let mut builder_key_lists: Vec<Vec<BuilderKey>> = (0..cs.value_vec_layout.committed_total_len)
        .map(|_| Vec::new())
        .collect();

    // Update the builder keys lists with respect to each operand of each operation
    let bitand_operand_getters: [fn(&AndConstraint) -> &Operand; BITAND_ARITY] =
        [|c| &c.a, |c| &c.b, |c| &c.c];
    let intmul_operand_getters: [fn(&MulConstraint) -> &Operand; INTMUL_ARITY] =
        [|c| &c.a, |c| &c.b, |c| &c.lo, |c| &c.hi];

    bitand_operand_getters
        .iter()
        .enumerate()
        .for_each(|(operand_idx, get_operand)| {
            update_with_operand(
                Operation::BitwiseAnd,
                operand_idx,
                cs.and_constraints.iter().map(get_operand),
                &mut builder_key_lists,
            );
        });

    intmul_operand_getters
        .iter()
        .enumerate()
        .for_each(|(operand_idx, get_operand)| {
            update_with_operand(
                Operation::IntegerMul,
                operand_idx,
                cs.mul_constraints.iter().map(get_operand),
                &mut builder_key_lists,
            );
        });

    // Compute all three fields of the key collection from the builder keys lists
    let key_ranges = builder_key_lists
        .iter()
        .scan(0u32, |offset, builder_keys| {
            let start = *offset;
            *offset += builder_keys.len() as u32;
            Some(start..*offset)
        })
        .collect();

    let mut keys = Vec::new();
    let mut constraint_indices = Vec::new();

    for builder_key in builder_key_lists.into_iter().flatten() {
        let BuilderKey {
            id,
            operation,
            constraint_indices: mut builder_constraint_indices,
        } = builder_key;

        // Sort constraint indices by operand index so we can save work in [`Key::accumulate`].
        builder_constraint_indices.sort_by_key(|constraint_index| constraint_index.operand_index);

        let start = constraint_indices.len() as u32;
        constraint_indices.extend(builder_constraint_indices);
        let end = constraint_indices.len() as u32;
        keys.push(Key {
            id,
            operation,
            range: start..end,
        });
    }

    KeyCollection {
        keys,
        key_ranges,
        constraint_indices,
    }
}

// Serialization implementations

impl SerializeBytes for Operation {
    fn serialize(&self, write_buf: impl BufMut) -> Result<(), SerializationError> {
        let val = match self {
            Operation::BitwiseAnd => 0u8,
            Operation::IntegerMul => 1u8,
        };
        val.serialize(write_buf)
    }
}

impl DeserializeBytes for Operation {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
        let val = u8::deserialize(&mut read_buf)?;
        match val {
            0 => Ok(Operation::BitwiseAnd),
            1 => Ok(Operation::IntegerMul),
            _ => Err(SerializationError::UnknownEnumVariant {
                name: "Operation",
                index: val,
            }),
        }
    }
}

impl SerializeBytes for Key {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.operation.serialize(&mut write_buf)?;
        self.id.serialize(&mut write_buf)?;
        self.range.start.serialize(&mut write_buf)?;
        self.range.end.serialize(write_buf)
    }
}

impl DeserializeBytes for Key {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
        let operation = Operation::deserialize(&mut read_buf)?;
        let id = u16::deserialize(&mut read_buf)?;
        let start = u32::deserialize(&mut read_buf)?;
        let end = u32::deserialize(&mut read_buf)?;
        Ok(Key {
            operation,
            id,
            range: start..end,
        })
    }
}

impl SerializeBytes for ConstraintIndex {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        self.operand_index.serialize(&mut write_buf)?;
        self.constraint_index.serialize(write_buf)
    }
}

impl DeserializeBytes for ConstraintIndex {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
        let operand_index = u8::deserialize(&mut read_buf)?;
        let constraint_index = u32::deserialize(&mut read_buf)?;
        Ok(ConstraintIndex {
            operand_index,
            constraint_index,
        })
    }
}

impl SerializeBytes for KeyCollection {
    fn serialize(&self, mut write_buf: impl BufMut) -> Result<(), SerializationError> {
        // Version for forward compatibility
        const VERSION: u32 = 1;
        VERSION.serialize(&mut write_buf)?;

        self.keys.serialize(&mut write_buf)?;

        // Serialize key_ranges as pairs of start/end
        (self.key_ranges.len() as u32).serialize(&mut write_buf)?;
        for range in &self.key_ranges {
            range.start.serialize(&mut write_buf)?;
            range.end.serialize(&mut write_buf)?;
        }

        self.constraint_indices.serialize(write_buf)
    }
}

impl DeserializeBytes for KeyCollection {
    fn deserialize(mut read_buf: impl Buf) -> Result<Self, SerializationError> {
        const VERSION: u32 = 1;
        let version = u32::deserialize(&mut read_buf)?;
        if version != VERSION {
            return Err(SerializationError::InvalidConstruction {
                name: "KeyCollection::version",
            });
        }

        let keys = Vec::<Key>::deserialize(&mut read_buf)?;

        // Deserialize key_ranges
        let len = u32::deserialize(&mut read_buf)? as usize;
        let mut key_ranges = Vec::with_capacity(len);
        for _ in 0..len {
            let start = u32::deserialize(&mut read_buf)?;
            let end = u32::deserialize(&mut read_buf)?;
            key_ranges.push(start..end);
        }

        let constraint_indices = Vec::<ConstraintIndex>::deserialize(&mut read_buf)?;

        Ok(KeyCollection {
            keys,
            key_ranges,
            constraint_indices,
        })
    }
}
