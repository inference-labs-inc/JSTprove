// Copyright 2025 Irreducible Inc.

use std::ops::{Deref, DerefMut};

use binius_field::{Field, PackedField};
use binius_math::{
    FieldBuffer, FieldSlice,
    multilinear::{
        eq::tensor_prod_eq_ind_prepend,
        fold::{binary_fold_high, fold_highest_var_inplace},
    },
};
use binius_utils::{
    bitwise::{BitSelector, Bitwise},
    checked_arithmetics::checked_log_2,
    random_access_sequence::{MatrixVertSliceSubrange, RandomAccessSequence},
    rayon::prelude::*,
};

/// A helper struct to maintain switchover-related invariants related to a set of 1-bit
/// multilinears represented by a slice of bitmasks.
///
/// # Invariants
/// 1. `i`-th transparent multilinear is equal to a slice of `i`-th bits of all bitmasks
/// 2. For the first `switchover` rounds, the said multilinears are transparent, and their partial
///    evaluations are obtained in linear time by doing binary folds
/// 3. Partial evaluations are stored after `switchover` rounds and folded as usual afterwards.
/// 4. Switchover is never performed before the first round.
///
/// Choosing switchover round value is a balancing act between peak memory consumption and
/// performance.
pub struct BinarySwitchover<'b, P: PackedField, B: Bitwise> {
    n_multilinears: usize,
    bitmasks: &'b [B],
    tensor: FieldBuffer<P>,
    folded: Option<Vec<FieldBuffer<P>>>,
    switchover: usize,
}

impl<'b, F, P, B> BinarySwitchover<'b, P, B>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
{
    /// Construct a new switchover helper.
    ///
    /// # Params
    /// * `bitmasks`       - bitmask representation of 1-bit multilinears
    /// * `n_multilinears` - number of  lower bitmask bits that become multilinears
    /// * `switchover`     - number of rounds after which to do the folding
    pub fn new(n_multilinears: usize, switchover: usize, bitmasks: &'b [B]) -> Self {
        assert!(
            bitmasks.len().is_power_of_two(),
            "Bitmasks represent a collection of multilinears and thus should be of power of two length"
        );

        let n_vars = checked_log_2(bitmasks.len());
        let switchover = switchover.min(n_vars).max(1);
        let mut tensor = FieldBuffer::zeros_truncated(0, switchover);
        tensor.set(0, F::ONE);

        Self {
            n_multilinears,
            bitmasks,
            tensor,
            folded: None,
            // Store this separately as `log_cap` is rounded up to the packing width
            switchover,
        }
    }

    // Number of variables in all 1-bit multilinears at start of sumcheck.
    fn n_vars_transparent(&self) -> usize {
        checked_log_2(self.bitmasks.len())
    }

    /// Get a power-of-two sized aligned chunk of the multilinear at `bit_offset` in the current
    /// round. This method abstracts transparent/folded state handling. Pre-switchover logic
    /// requires a chunk sized scratchpad to hold the result.
    pub fn get_chunk<'switchover, 'scratchpad, Data: DerefMut<Target = [P]>>(
        &'switchover self,
        scratchpad: &'scratchpad mut FieldBuffer<P, Data>,
        bit_offset: usize,
        chunk_vars: usize,
        chunk_index: usize,
    ) -> FieldSlice<'scratchpad, P>
    where
        'switchover: 'scratchpad,
    {
        assert!(bit_offset < self.n_multilinears);
        assert_eq!(scratchpad.len(), 1 << chunk_vars);

        if let Some(folded) = &self.folded {
            folded[bit_offset].chunk(chunk_vars, chunk_index)
        } else {
            get_binary_chunk(
                scratchpad,
                &self.tensor,
                &BitSelector::new(bit_offset, self.bitmasks),
                chunk_vars,
                chunk_index,
            );
            scratchpad.to_ref()
        }
    }

    pub fn fold(&mut self, challenge: F) {
        if let Some(folded) = &mut self.folded {
            // Post-switchover: fold high as usual
            folded
                .par_iter_mut()
                .for_each(|multilinear| fold_highest_var_inplace(multilinear, challenge));
        } else {
            // Pre-switchover: update the folding tensor
            assert!(self.tensor.log_len() < self.switchover);
            tensor_prod_eq_ind_prepend(&mut self.tensor, &[challenge]);

            if self.tensor.log_len() == self.switchover {
                self.perform();
            }
        }
    }

    // Perform the switchover process. This operation is idempotent.
    fn perform(&mut self) {
        if self.folded.is_some() {
            return;
        }

        let folded_n_vars = self.n_vars_transparent() - self.tensor.log_len();

        let all_folded = (0..self.n_multilinears)
            .into_par_iter()
            .map(|bit_offset| {
                let mut folded = FieldBuffer::<P>::zeros(folded_n_vars);
                get_binary_chunk(
                    &mut folded,
                    &self.tensor,
                    &BitSelector::new(bit_offset, self.bitmasks),
                    folded_n_vars,
                    0,
                );

                folded
            })
            .collect::<Vec<_>>();

        self.folded = Some(all_folded);
    }

    pub fn finalize(mut self) -> Vec<FieldBuffer<P>> {
        self.perform();
        self.folded.expect("explicit call to perform()")
    }
}

// Compute a power-of-two sized aligned chunk of the partial evaluation of a binary sequence
// with a tensor using fold high. Conceptually, this method:
//  * takes in a boolean sequence `binary_sequence`, splits it into tensor-sized chunks, then does a
//    tensor product with each of them
//  * the `2^chunk_vars` aligned chunk with index `chunk_index` is put into `dest`
// The folded column is not fully materialized though.
fn get_binary_chunk<P, DataOut, DataIn>(
    dest: &mut FieldBuffer<P, DataOut>,
    tensor: &FieldBuffer<P, DataIn>,
    binary_sequence: &(impl RandomAccessSequence<bool> + Sync),
    chunk_vars: usize,
    chunk_index: usize,
) where
    P: PackedField,
    DataOut: DerefMut<Target = [P]>,
    DataIn: Deref<Target = [P]> + Sync,
{
    assert!(binary_sequence.len().is_power_of_two());
    let sequence_log_len = checked_log_2(binary_sequence.len());

    // We can view fold high as a following matrix operation:
    //  1) Rearrange the sequence into a row-major matrix with `tensor.len()` rows
    //  2) Obtain a view into a chunk-sized vertical slice of said matrix
    //  3) Fold this matrix along columns to get chunk-sized partial evaluations
    let matrix_vert_slice = MatrixVertSliceSubrange::new(
        binary_sequence,
        tensor.log_len(),
        sequence_log_len - tensor.log_len(),
        chunk_vars,
        chunk_index,
    );
    binary_fold_high(dest, tensor, matrix_vert_slice);
}
