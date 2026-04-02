// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_field::arch::OptimalPackedB128;
use binius_math::test_utils::random_scalars;
use binius_prover::fold_word::fold_words;
use binius_verifier::config::{B128, WORD_SIZE_BITS};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::Rng;

fn bench_fold_words(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold_words");

    for log_n_words in [12, 16, 20] {
        let n_words = 1 << log_n_words;

        // Set throughput to measure elements per second
        group.throughput(Throughput::Elements(n_words as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(n_words),
            &n_words,
            |b, &n_words| {
                let mut rng = rand::rng();
                let words = (0..n_words)
                    .map(|_| Word::from_u64(rng.random::<u64>()))
                    .collect::<Vec<_>>();
                let vec = random_scalars::<B128>(&mut rng, WORD_SIZE_BITS);

                b.iter(|| fold_words::<_, OptimalPackedB128>(&words, &vec));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fold_words);
criterion_main!(benches);
