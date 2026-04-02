// Copyright 2023-2025 Irreducible Inc.

//! [Reed–Solomon] codes over binary fields.
//!
//! See [`ReedSolomonCode`] for details.

use std::ptr;

use binius_field::{BinaryField, PackedField};
use binius_utils::rayon::prelude::*;
use getset::{CopyGetters, Getters};

use super::{
    FieldBuffer, FieldSlice, FieldSliceMut, binary_subspace::BinarySubspace, ntt::AdditiveNTT,
};
use crate::{
    bit_reverse::{bit_reverse_indices, bit_reverse_packed},
    ntt::DomainContext,
};

/// [Reed–Solomon] codes over binary fields.
///
/// The Reed–Solomon code admits an efficient encoding algorithm over binary fields due to [LCH14].
/// The additive NTT encoding algorithm encodes messages interpreted as the coefficients of a
/// polynomial in a non-standard, novel polynomial basis and the codewords are the polynomial
/// evaluations over a linear subspace of the field. See the [binius-math] crate for more details.
///
/// [Reed–Solomon]: <https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction>
/// [LCH14]: <https://arxiv.org/abs/1404.3458>
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct ReedSolomonCode<F> {
    #[get = "pub"]
    subspace: BinarySubspace<F>,
    log_dimension: usize,
    #[get_copy = "pub"]
    log_inv_rate: usize,
}

impl<F: BinaryField> ReedSolomonCode<F> {
    pub fn new(log_dimension: usize, log_inv_rate: usize) -> Self {
        let subspace = BinarySubspace::with_dim(log_dimension + log_inv_rate);
        Self::with_subspace(subspace, log_dimension, log_inv_rate)
    }

    pub fn with_ntt_subspace(
        ntt: &impl AdditiveNTT<Field = F>,
        log_dimension: usize,
        log_inv_rate: usize,
    ) -> Self {
        Self::with_domain_context_subspace(ntt.domain_context(), log_dimension, log_inv_rate)
    }

    pub fn with_domain_context_subspace(
        domain_context: &impl DomainContext<Field = F>,
        log_dimension: usize,
        log_inv_rate: usize,
    ) -> Self {
        let subspace_dim = log_dimension + log_inv_rate;
        assert!(
            subspace_dim <= domain_context.log_domain_size(),
            "precondition: subspace dimension must not exceed domain context log size"
        );
        let subspace = domain_context.subspace(subspace_dim);
        Self::with_subspace(subspace, log_dimension, log_inv_rate)
    }

    pub fn with_subspace(
        subspace: BinarySubspace<F>,
        log_dimension: usize,
        log_inv_rate: usize,
    ) -> Self {
        assert_eq!(
            subspace.dim(),
            log_dimension + log_inv_rate,
            "precondition: subspace dimension must equal log_dimension + log_inv_rate"
        );
        Self {
            subspace,
            log_dimension,
            log_inv_rate,
        }
    }

    /// The dimension.
    pub const fn dim(&self) -> usize {
        1 << self.dim_bits()
    }

    pub const fn log_dim(&self) -> usize {
        self.log_dimension
    }

    pub const fn log_len(&self) -> usize {
        self.log_dimension + self.log_inv_rate
    }

    /// The block length.
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> usize {
        1 << (self.log_dimension + self.log_inv_rate)
    }

    /// The base-2 log of the dimension.
    const fn dim_bits(&self) -> usize {
        self.log_dimension
    }

    /// The reciprocal of the rate, ie. `self.len() / self.dim()`.
    pub const fn inv_rate(&self) -> usize {
        1 << self.log_inv_rate
    }

    /// Encodes a message with an interleaved Reed–Solomon code.
    ///
    /// This function interprets the message as a batch of independent vectors and applies an
    /// interleaved Reed–Solomon.
    ///
    /// ## Preconditions
    ///
    /// * `data.log_len()` must equal `log_dim() + log_batch_size`.
    /// * The NTT subspace must match the code's subspace.
    ///
    /// ## Postconditions
    ///
    /// * All elements in the output buffer are initialized with the encoded codeword.
    pub fn encode_batch<P, NTT>(
        &self,
        ntt: &NTT,
        data: FieldSlice<P>,
        log_batch_size: usize,
    ) -> FieldBuffer<P>
    where
        P: PackedField<Scalar = F>,
        NTT: AdditiveNTT<Field = F> + Sync,
    {
        assert_eq!(
            ntt.subspace(self.log_len()),
            self.subspace,
            "precondition: NTT subspace must match code subspace"
        );
        assert_eq!(
            data.log_len(),
            self.log_dim() + log_batch_size,
            "precondition: data.log_len() must equal log_dim() + log_batch_size"
        );

        let _scope = tracing::trace_span!(
            "Reed-Solomon encode",
            log_len = self.log_len(),
            log_batch_size = log_batch_size,
            symbol_bits = F::N_BITS,
        )
        .entered();

        // Repeat the message to fill the entire buffer.
        let log_output_len = self.log_dim() + log_batch_size + self.log_inv_rate;
        let output_data = if data.log_len() < P::LOG_WIDTH {
            let mut scalars = data.iter_scalars().collect::<Vec<_>>();
            bit_reverse_indices(&mut scalars);
            let elem_0 = P::from_scalars(scalars.into_iter().cycle());
            vec![elem_0; 1 << log_output_len.saturating_sub(P::LOG_WIDTH)]
        } else {
            let mut output_data = Vec::with_capacity(1 << (log_output_len - P::LOG_WIDTH));

            output_data.extend_from_slice(data.as_ref());

            // Bit-reverse permute the message.
            bit_reverse_packed(FieldSliceMut::from_slice(
                data.log_len(),
                output_data.as_mut_slice(),
            ));

            let log_msg_len_packed = data.log_len() - P::LOG_WIDTH;
            output_data
                .spare_capacity_mut()
                .par_chunks_exact_mut(1 << log_msg_len_packed)
                .enumerate()
                .for_each(|(i, output_chunk)| unsafe {
                    let dst_ptr = output_chunk.as_mut_ptr();

                    // TODO(https://github.com/rust-lang/rust/issues/81944):
                    // Improve unsafe code with Vec::split_at_spare_mut when stable

                    // Safety:
                    // - log_output_len == log_msg_len_packed + self.log_inv_rate
                    // - i + 1 is in the range 1..1 << self.log_inv_rate
                    // - dst_ptr is disjoint from src_ptr and within the Vec capacity
                    let src_ptr = dst_ptr.sub((i + 1) << log_msg_len_packed);
                    ptr::copy_nonoverlapping(src_ptr, dst_ptr, 1 << log_msg_len_packed);
                });

            unsafe {
                // Safety: the vec's spare capacity is fully initialized above.
                output_data.set_len(1 << (log_output_len - P::LOG_WIDTH));
            }

            output_data
        };
        let mut output = FieldBuffer::new(log_output_len, output_data.into_boxed_slice());

        ntt.forward_transform(output.to_mut(), self.log_inv_rate, log_batch_size);
        output
    }
}

#[cfg(test)]
mod tests {
    use binius_field::{
        BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash4x128b, PackedField,
    };
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;
    use crate::{
        FieldBuffer,
        bit_reverse::reverse_bits,
        ntt::{NeighborsLastReference, domain_context::GenericPreExpanded},
        test_utils::random_field_buffer,
    };

    fn test_encode_batch_helper<P: PackedField>(
        log_dim: usize,
        log_inv_rate: usize,
        log_batch_size: usize,
    ) where
        P::Scalar: BinaryField,
    {
        let mut rng = StdRng::seed_from_u64(0);

        let rs_code = ReedSolomonCode::<P::Scalar>::new(log_dim, log_inv_rate);

        // Create NTT with matching subspace
        let subspace = rs_code.subspace().clone();
        let domain_context = GenericPreExpanded::<P::Scalar>::generate_from_subspace(&subspace);
        let ntt = NeighborsLastReference {
            domain_context: &domain_context,
        };

        // Generate random message buffer
        let message = random_field_buffer::<P>(&mut rng, log_dim + log_batch_size);

        // Test the new encode_batch interface
        let encoded_buffer = rs_code.encode_batch(&ntt, message.to_ref(), log_batch_size);

        // Method 2: Reference implementation - apply NTT with zero-padded coefficients to the
        // bit-reversal permuted message.
        let mut reference_buffer = FieldBuffer::zeros(rs_code.log_len() + log_batch_size);
        for (i, val) in message.iter_scalars().enumerate() {
            let bits = (rs_code.log_dim() + log_batch_size) as u32;
            reference_buffer.set(reverse_bits(i, bits), val);
        }

        // Perform large NTT with zero-padded coefficients.
        ntt.forward_transform(reference_buffer.to_mut(), 0, log_batch_size);

        // Compare results
        assert_eq!(
            encoded_buffer.as_ref(),
            reference_buffer.as_ref(),
            "encode_batch_inplace result differs from reference NTT implementation"
        );
    }

    #[test]
    fn test_encode_batch_above_packing_width() {
        // Test with PackedBinaryGhash1x128b
        test_encode_batch_helper::<PackedBinaryGhash1x128b>(4, 2, 0);
        test_encode_batch_helper::<PackedBinaryGhash1x128b>(6, 2, 1);
        test_encode_batch_helper::<PackedBinaryGhash1x128b>(8, 3, 2);

        // Test with PackedBinaryGhash4x128b
        test_encode_batch_helper::<PackedBinaryGhash4x128b>(4, 2, 0);
        test_encode_batch_helper::<PackedBinaryGhash4x128b>(6, 2, 1);
        test_encode_batch_helper::<PackedBinaryGhash4x128b>(8, 3, 2);
    }

    #[test]
    fn test_encode_batch_below_packing_width() {
        // Test where message length is less than the packing width and codeword length is greater.
        test_encode_batch_helper::<PackedBinaryGhash4x128b>(1, 2, 0);
    }
}
