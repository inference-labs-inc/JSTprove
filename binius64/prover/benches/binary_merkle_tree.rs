// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

use std::iter::repeat_with;

use binius_field::Random;
use binius_hash::{
    ParallelCompressionAdaptor, ParallelDigest, ParallelMultidigestImpl, PseudoCompressionFunction,
    StdCompression, StdDigest,
    vision_4::parallel_compression::VisionParallelCompression as VisionParallelCompression_4,
    vision_6::parallel_digest::VisionHasherMultiDigest as VisionHasherMultiDigest_6,
};
use binius_prover::merkle_tree::{MerkleTreeProver, prover::BinaryMerkleTreeProver};
use binius_verifier::config::B128;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use digest::{FixedOutputReset, Output, core_api::BlockSizeUser};

const LOG_ELEMS: usize = 17;
const LOG_ELEMS_IN_LEAF: usize = 4;

type F = B128;

fn bench_binary_merkle_tree<H, C>(c: &mut Criterion, compression: C, hash_name: &str)
where
    H: ParallelDigest<Digest: BlockSizeUser + FixedOutputReset>,
    C: PseudoCompressionFunction<Output<H::Digest>, 2> + Sync,
{
    let parallel_compression = ParallelCompressionAdaptor::new(compression);
    let merkle_prover = BinaryMerkleTreeProver::<_, H, _>::new(parallel_compression);
    let mut rng = rand::rng();
    let data = repeat_with(|| F::random(&mut rng))
        .take(1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF))
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group(format!("slow/merkle_tree/{hash_name}"));
    group.throughput(Throughput::Bytes(
        ((1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF)) * std::mem::size_of::<F>()) as u64,
    ));
    group.sample_size(10);
    group.bench_function(
        format!(
            "{} log elems size {}xB64 leaf",
            LOG_ELEMS,
            1 << LOG_ELEMS_IN_LEAF
        ),
        |b| {
            b.iter(|| merkle_prover.commit(&data, 1 << LOG_ELEMS_IN_LEAF));
        },
    );
    group.finish()
}

fn bench_sha256_merkle_tree(c: &mut Criterion) {
    bench_binary_merkle_tree::<StdDigest, _>(c, StdCompression::default(), "SHA-256");
}

const N: usize = 128;
type LeafHasher = ParallelMultidigestImpl<VisionHasherMultiDigest_6<N, { N * 6 }>, N>;

// Use Vision6 for leaves and Vision4 for compression
fn bench_vision_merkle_tree(c: &mut Criterion) {
    let vision_compression = VisionParallelCompression_4::new();
    let merkle_prover = BinaryMerkleTreeProver::<_, LeafHasher, _>::new(vision_compression);
    let mut rng = rand::rng();
    let data = repeat_with(|| F::random(&mut rng))
        .take(1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("slow/merkle_tree/Vision");
    group.throughput(Throughput::Bytes(
        ((1 << (LOG_ELEMS + LOG_ELEMS_IN_LEAF)) * std::mem::size_of::<F>()) as u64,
    ));
    group.sample_size(10);
    group.bench_function(
        format!(
            "{LOG_ELEMS} log elems size {}xB64 leaf",
            1 << LOG_ELEMS_IN_LEAF
        ),
        |b| {
            b.iter(|| merkle_prover.commit(&data, 1 << LOG_ELEMS_IN_LEAF));
        },
    );
    group.finish()
}

criterion_group!(
    binary_merkle_tree,
    bench_sha256_merkle_tree,
    bench_vision_merkle_tree
);
criterion_main!(binary_merkle_tree);
