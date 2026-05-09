use arith::Field;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use goldilocks::Goldilocks;
use jst_gkr::prover::prove;
use jst_gkr::transcript::Sha256Transcript;
use jst_gkr_engine::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type F = Goldilocks;

fn build_dense_circuit(depth: usize, width_bits: usize) -> (LayeredCircuit<F>, Vec<F>) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let input_size = 1 << width_bits;

    let witness: Vec<F> = (0..input_size)
        .map(|_| F::random_unsafe(&mut rng))
        .collect();

    let mut layers = Vec::with_capacity(depth);
    for _ in 0..depth {
        let mut mul_gates = Vec::new();
        let mut add_gates = Vec::new();
        let out_size = 1 << width_bits;

        for o in 0..out_size {
            let i0 = (2 * o) % input_size;
            let i1 = (2 * o + 1) % input_size;
            mul_gates.push(MulGate {
                o_id: o,
                i_ids: [i0, i1],
                coef: F::ONE,
            });
            add_gates.push(AddGate {
                o_id: o,
                i_id: i0,
                coef: F::ONE,
            });
        }

        layers.push(CircuitLayer {
            input_var_num: width_bits,
            output_var_num: width_bits,
            mul_gates,
            add_gates,
            const_gates: vec![],
        });
    }

    (LayeredCircuit { layers }, witness)
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("jst_gkr_prove");
    group.sample_size(20);

    for &width_bits in &[8, 10, 12, 14, 16] {
        let depth = 4;
        let (circuit, witness) = build_dense_circuit(depth, width_bits);
        let label = format!("d{depth}_w{width_bits}");

        group.bench_with_input(
            BenchmarkId::new("prove", &label),
            &(&circuit, &witness),
            |b, &(circuit, witness)| {
                b.iter(|| {
                    let mut transcript = Sha256Transcript::default();
                    prove::<F, Sha256Transcript>(circuit, witness, &mut transcript)
                });
            },
        );
    }

    group.finish();
}

fn bench_sumcheck_standalone(c: &mut Criterion) {
    let mut group = c.benchmark_group("jst_gkr_sumcheck");
    group.sample_size(30);

    for &n in &[10, 14, 18, 20] {
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let size = 1 << n;
        let f_vals: Vec<F> = (0..size).map(|_| F::random_unsafe(&mut rng)).collect();
        let hg_vals: Vec<F> = (0..size).map(|_| F::random_unsafe(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::new("sumcheck", n), &n, |b, _| {
            b.iter(|| {
                let mut bk_f = f_vals.clone();
                let mut bk_hg = hg_vals.clone();
                let mut transcript = Sha256Transcript::default();
                jst_gkr::sumcheck::prove_sumcheck(&mut bk_f, &mut bk_hg, n, &mut transcript)
            });
        });
    }

    group.finish();
}

fn bench_depth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("jst_gkr_depth");
    group.sample_size(20);

    let width_bits = 12;
    for &depth in &[2, 4, 8, 16] {
        let (circuit, witness) = build_dense_circuit(depth, width_bits);
        let label = format!("d{depth}_w{width_bits}");

        group.bench_with_input(
            BenchmarkId::new("prove", &label),
            &(&circuit, &witness),
            |b, &(circuit, witness)| {
                b.iter(|| {
                    let mut transcript = Sha256Transcript::default();
                    prove::<F, Sha256Transcript>(circuit, witness, &mut transcript)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_prove,
    bench_sumcheck_standalone,
    bench_depth_scaling
);
criterion_main!(benches);
