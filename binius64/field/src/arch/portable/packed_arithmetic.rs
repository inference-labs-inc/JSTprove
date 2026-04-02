// Copyright 2024-2025 Irreducible Inc.

use crate::{
    PackedField,
    binary_field::BinaryField,
    linear_transformation::{FieldLinearTransformation, Transformation},
    packed::PackedBinaryField,
    underlier::{UnderlierWithBitOps, WithUnderlier},
};

/// Interleave using the provided even mask slice.
///
/// See [Hacker's Delight](https://dl.acm.org/doi/10.5555/2462741), Section 7-3.
pub fn interleave_with_mask<U: UnderlierWithBitOps>(
    a: U,
    b: U,
    log_block_len: usize,
    even_mask: &[U],
) -> (U, U) {
    assert!(log_block_len < even_mask.len());

    let block_len = 1 << log_block_len;
    let t = ((a >> block_len) ^ b) & even_mask[log_block_len];
    let c = a ^ t << block_len;
    let d = b ^ t;

    (c, d)
}

/// Generate the mask with ones in the odd packed element positions and zeros in even
macro_rules! interleave_mask_even {
    ($underlier:ty, $tower_level:literal) => {{
        let scalar_bits = 1 << $tower_level;

        let mut mask: $underlier = (1 << scalar_bits) - 1;
        let log_width = <$underlier>::LOG_BITS - $tower_level;
        let mut i = 1;
        while i < log_width {
            mask |= mask << (scalar_bits << i);
            i += 1;
        }

        mask
    }};
}

pub(crate) use interleave_mask_even;

/// Packed transformation implementation.
/// Stores bases in a form of:
/// [
///     [<base vec 1> ... <base vec 1>],
///     ...
///     [<base vec N> ... <base vec N>]
/// ]
/// Transformation complexity is `N*log(N)` where `N` is `OP::Scalar::DEGREE`.
pub struct PackedTransformation<OP> {
    bases: Vec<OP>,
}

impl<OP> PackedTransformation<OP>
where
    OP: PackedBinaryField,
{
    pub fn new<Data: AsRef<[OP::Scalar]> + Sync>(
        transformation: &FieldLinearTransformation<OP::Scalar, Data>,
    ) -> Self {
        Self {
            bases: transformation
                .bases()
                .iter()
                .map(|base| OP::broadcast(*base))
                .collect(),
        }
    }
}

/// Broadcast lowest field for each element, e.g. `[<0001><0000>] -> [<1111><0000>]`
fn broadcast_lowest_bit<U: UnderlierWithBitOps>(mut data: U, log_packed_bits: usize) -> U {
    for i in 0..log_packed_bits {
        data |= data << (1 << i)
    }

    data
}

impl<U, IP, OP, IF, OF> Transformation<IP, OP> for PackedTransformation<OP>
where
    IP: PackedField<Scalar = IF> + WithUnderlier<Underlier = U>,
    OP: PackedField<Scalar = OF> + WithUnderlier<Underlier = U>,
    IF: BinaryField,
    OF: BinaryField,
    U: UnderlierWithBitOps,
{
    fn transform(&self, input: &IP) -> OP {
        let mut result = OP::zero();
        let ones = OP::one().to_underlier();
        let mut input = input.to_underlier();

        for base in &self.bases {
            let base_component = input & ones;
            // contains ones at positions which correspond to non-zero components
            let mask = broadcast_lowest_bit(base_component, OF::LOG_DEGREE);
            result += OP::from_underlier(mask & base.to_underlier());
            input = input >> 1;
        }

        result
    }
}
