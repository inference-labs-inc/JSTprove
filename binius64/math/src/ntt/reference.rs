// Copyright 2024-2025 Irreducible Inc.

use binius_field::PackedField;

use crate::{
    field_buffer::FieldSliceMut,
    ntt::{AdditiveNTT, DomainContext},
};

/// Reference implementation of [`AdditiveNTT`].
///
/// This is slow. Do not use in production.
pub struct NeighborsLastReference<DC> {
    pub domain_context: DC,
}

impl<DC: DomainContext> AdditiveNTT for NeighborsLastReference<DC> {
    type Field = DC::Field;

    fn forward_transform<P: PackedField<Scalar = Self::Field>>(
        &self,
        mut data: FieldSliceMut<P>,
        skip_early: usize,
        skip_late: usize,
    ) {
        let log_d = data.log_len();
        input_check(&self.domain_context, log_d, skip_early, skip_late);

        for layer in skip_early..(log_d - skip_late) {
            let num_blocks = 1 << layer;
            let block_size_half = 1 << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.domain_context.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = block_size_half | idx0;
                    // perform butterfly
                    let mut u = data.get(idx0);
                    let mut v = data.get(idx1);
                    u += v * twiddle;
                    v += u;
                    data.set(idx0, u);
                    data.set(idx1, v);
                }
            }
        }
    }

    fn inverse_transform<P: PackedField<Scalar = Self::Field>>(
        &self,
        mut data: FieldSliceMut<P>,
        skip_early: usize,
        skip_late: usize,
    ) {
        let log_d = data.log_len();
        input_check(&self.domain_context, log_d, skip_early, skip_late);

        for layer in (skip_early..(log_d - skip_late)).rev() {
            let num_blocks = 1 << layer;
            let block_size_half = 1 << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.domain_context.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = block_size_half | idx0;
                    // perform butterfly
                    let mut u = data.get(idx0);
                    let mut v = data.get(idx1);
                    v += u;
                    u += v * twiddle;
                    data.set(idx0, u);
                    data.set(idx1, v);
                }
            }
        }
    }

    fn domain_context(&self) -> &impl DomainContext<Field = DC::Field> {
        &self.domain_context
    }
}

/// Checks for the preconditions of the [`AdditiveNTT`] transforms.
///
/// ## Preconditions
///
/// - `skip_early + skip_late <= log_d`
/// - `log_d - skip_late <= domain_context.log_domain_size()`
pub fn input_check(
    domain_context: &impl DomainContext,
    log_d: usize,
    skip_early: usize,
    skip_late: usize,
) {
    // we can't "double-skip" layers
    assert!(skip_early + skip_late <= log_d);

    // we need enough twiddles in `domain_context`
    assert!(log_d - skip_late <= domain_context.log_domain_size());
}
