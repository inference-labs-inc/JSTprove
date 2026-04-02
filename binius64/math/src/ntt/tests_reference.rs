// Copyright 2025 Irreducible Inc.
//! This module tests that the NTT implementations are equivalent to a simple reference
//! implementation.

use binius_field::{
    BinaryField, PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b,
    PackedField,
};
use rand::prelude::*;

use super::{AdditiveNTT, DomainContext, NeighborsLastBreadthFirst};
use crate::{
    binary_subspace::BinarySubspace,
    field_buffer::FieldSliceMut,
    ntt::{
        NeighborsLastMultiThread, NeighborsLastReference, NeighborsLastSingleThread,
        domain_context::{
            GaoMateerPreExpanded, GenericOnTheFly, GenericPreExpanded, TraceOneElement,
        },
    },
    test_utils::{B128, Packed128b, random_field_buffer},
};

fn test_transform_equivalence<P: PackedField>(
    mut rng: impl Rng,
    reference: impl Fn(FieldSliceMut<P>, usize, usize),
    transform: impl Fn(FieldSliceMut<P>, usize, usize),
    log_n: usize,
) {
    let half_rounds = log_n / 2;
    for skip_early in [0, 1, half_rounds] {
        for skip_late in [0, 1, log_n - half_rounds] {
            if skip_early + skip_late > log_n {
                continue;
            }

            let mut data_a = random_field_buffer::<P>(&mut rng, log_n);
            let mut data_b = data_a.clone();

            reference(data_a.to_mut(), skip_early, skip_late);
            transform(data_b.to_mut(), skip_early, skip_late);
            assert_eq!(data_a, data_b);
        }
    }
}

fn test_equivalence<P: PackedField>(
    ntt_a: &impl AdditiveNTT<Field = P::Scalar>,
    ntt_b: &impl AdditiveNTT<Field = P::Scalar>,
) where
    P::Scalar: BinaryField,
{
    let mut rng = StdRng::seed_from_u64(0);

    test_transform_equivalence::<P>(
        &mut rng,
        |data, skip_early, skip_late| ntt_a.forward_transform(data, skip_early, skip_late),
        |data, skip_early, skip_late| ntt_b.forward_transform(data, skip_early, skip_late),
        1,
    );
    test_transform_equivalence::<P>(
        &mut rng,
        |data, skip_early, skip_late| ntt_a.forward_transform(data, skip_early, skip_late),
        |data, skip_early, skip_late| ntt_b.forward_transform(data, skip_early, skip_late),
        8,
    );
}

fn test_equivalence_ntts<P: PackedField>(
    domain_context: &(impl DomainContext<Field = P::Scalar> + Sync),
) where
    P::Scalar: BinaryField,
{
    let ntt_ref = NeighborsLastReference { domain_context };
    let ntt_breadth_first = NeighborsLastBreadthFirst { domain_context };
    let ntt_single_2 = NeighborsLastSingleThread {
        domain_context,
        log_base_len: 2,
    };
    let ntt_single_6 = NeighborsLastSingleThread {
        domain_context,
        log_base_len: 6,
    };
    let ntt_multi_3_0 = NeighborsLastMultiThread {
        domain_context,
        log_base_len: 3,
        log_num_shares: 0,
    };
    let ntt_multi_3_1 = NeighborsLastMultiThread {
        domain_context,
        log_base_len: 3,
        log_num_shares: 1,
    };
    let ntt_multi_3_2 = NeighborsLastMultiThread {
        domain_context,
        log_base_len: 3,
        log_num_shares: 2,
    };
    let ntt_multi_3_1000 = NeighborsLastMultiThread {
        domain_context,
        log_base_len: 3,
        log_num_shares: 1000,
    };

    test_equivalence::<P>(&ntt_ref, &ntt_breadth_first);
    test_equivalence::<P>(&ntt_ref, &ntt_single_2);
    test_equivalence::<P>(&ntt_ref, &ntt_single_6);
    test_equivalence::<P>(&ntt_ref, &ntt_multi_3_0);
    test_equivalence::<P>(&ntt_ref, &ntt_multi_3_1);
    test_equivalence::<P>(&ntt_ref, &ntt_multi_3_2);
    test_equivalence::<P>(&ntt_ref, &ntt_multi_3_1000);
}

trait NTTFactory<F: BinaryField> {
    fn create<DC: DomainContext<Field = F> + Sync>(
        &self,
        domain_context: DC,
    ) -> impl AdditiveNTT<Field = F>;
}

struct NeighborsLastReferenceFactory;

impl<F: BinaryField> NTTFactory<F> for NeighborsLastReferenceFactory {
    fn create<DC: DomainContext<Field = F> + Sync>(
        &self,
        domain_context: DC,
    ) -> impl AdditiveNTT<Field = F> {
        NeighborsLastReference { domain_context }
    }
}

struct NeighborsLastBreadthFirstFactory;

impl<F: BinaryField> NTTFactory<F> for NeighborsLastBreadthFirstFactory {
    fn create<DC: DomainContext<Field = F> + Sync>(
        &self,
        domain_context: DC,
    ) -> impl AdditiveNTT<Field = F> {
        NeighborsLastBreadthFirst { domain_context }
    }
}

struct NeighborsLastSingleThreadFactory {
    log_base_len: usize,
}

impl<F: BinaryField> NTTFactory<F> for NeighborsLastSingleThreadFactory {
    fn create<DC: DomainContext<Field = F> + Sync>(
        &self,
        domain_context: DC,
    ) -> impl AdditiveNTT<Field = F> {
        NeighborsLastSingleThread {
            domain_context,
            log_base_len: self.log_base_len,
        }
    }
}

struct NeighborsLastMultiThreadFactory {
    log_base_len: usize,
    log_num_shares: usize,
}

impl<F: BinaryField> NTTFactory<F> for NeighborsLastMultiThreadFactory {
    fn create<DC: DomainContext<Field = F> + Sync>(
        &self,
        domain_context: DC,
    ) -> impl AdditiveNTT<Field = F> {
        NeighborsLastMultiThread {
            domain_context,
            log_base_len: self.log_base_len,
            log_num_shares: self.log_num_shares,
        }
    }
}

#[rstest::rstest]
#[case::neighbors_last_reference(NeighborsLastReferenceFactory)]
#[case::neighbors_last_breadth_first(NeighborsLastBreadthFirstFactory)]
#[case::neighbors_last_single_thread(NeighborsLastSingleThreadFactory { log_base_len: 6 })]
#[case::neighbors_last_multi_thread_1_share(
	NeighborsLastMultiThreadFactory { log_base_len: 3, log_num_shares: 1 }
)]
#[case::neighbors_last_multi_thread_1000_share(
	NeighborsLastMultiThreadFactory { log_base_len: 3, log_num_shares: 1000 }
)]
fn test_forward_transform_is_identity(#[case] ntt_factory: impl NTTFactory<B128>) {
    fn test_forward_transform_is_identity_helper<F, P>(ntt_factory: impl NTTFactory<F>)
    where
        F: BinaryField,
        P: PackedField<Scalar = F>,
    {
        let subspace = BinarySubspace::<F>::with_dim(1);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);

        let ntt = ntt_factory.create(domain_context);

        let mut rng = StdRng::seed_from_u64(0);
        let data = random_field_buffer::<P>(&mut rng, 0);
        let mut data_clone = data.clone();

        ntt.forward_transform(data_clone.to_mut(), 0, 0);

        assert_eq!(data, data_clone);
    }

    test_forward_transform_is_identity_helper::<_, Packed128b>(ntt_factory);
}

fn test_equivalence_ntts_domain_contexts<P: PackedField>()
where
    P::Scalar: BinaryField + TraceOneElement,
{
    let dc_1 = GaoMateerPreExpanded::<P::Scalar>::generate(10);
    test_equivalence_ntts::<P>(&dc_1);

    let subspace = BinarySubspace::with_dim(10);
    let dc_2 = GenericPreExpanded::<P::Scalar>::generate_from_subspace(&subspace);
    test_equivalence_ntts::<P>(&dc_2);
}

#[test]
fn test_equivalence_ntts_domain_contexts_packings() {
    test_equivalence_ntts_domain_contexts::<PackedBinaryGhash1x128b>();
    test_equivalence_ntts_domain_contexts::<PackedBinaryGhash2x128b>();
    test_equivalence_ntts_domain_contexts::<PackedBinaryGhash4x128b>();
}

fn test_composition<P: PackedField>()
where
    P::Scalar: BinaryField,
{
    let log_d = 7;
    let mut rng = StdRng::seed_from_u64(0);
    let data_orig = random_field_buffer::<P>(&mut rng, log_d);
    let mut data = data_orig.clone();

    let subspace = BinarySubspace::<P::Scalar>::with_dim(10);
    let domain_context = GenericPreExpanded::generate_from_subspace(&subspace);
    let ntt = NeighborsLastReference { domain_context };
    for skip_early in [0, 2, 5] {
        for skip_late in [0, 2, 5] {
            if skip_early + skip_late > log_d {
                continue;
            }

            ntt.forward_transform(data.to_mut(), skip_early, skip_late);
            ntt.inverse_transform(data.to_mut(), skip_early, skip_late);
            assert_eq!(data, data_orig);
        }
    }
}

#[test]
fn test_composition_packings() {
    test_composition::<B128>();
    test_composition::<Packed128b>();
}
