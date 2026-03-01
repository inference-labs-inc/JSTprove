// this module benchmarks the performance of different field operations

use std::ops::Range;

use arith::Field;
use ark_std::test_rng;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use halo2curves::bn256::Fr;
use polynomials::{EqPolynomial, MultiLinearPoly};

const RANGE: Range<usize> = 9..10;

fn bench_mle_eval<F: Field>(c: &mut Criterion) {
    let mut rng = test_rng();
    for nv in RANGE {
        let mle = MultiLinearPoly::<F>::random(nv, &mut rng);
        let point = (0..nv)
            .map(|_| F::random_unsafe(&mut rng))
            .collect::<Vec<_>>();

        // jolt method
        let label = format!("jolt's mle eval, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| black_box(mle.evaluate_jolt(&point)))
        });

        // hyperplonk method
        let label = format!("hyperplonk's mle eval, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| {
                {
                    let mut mle_eval = mle.clone();
                    mle_eval.fix_variables(point.as_ref())
                };
                black_box(())
            })
        });

        // expander method
        let mut buf = vec![F::zero(); 1 << nv];
        let label = format!("expander's mle eval, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| {
                black_box(MultiLinearPoly::<F>::evaluate_with_buffer(
                    mle.coeffs.as_ref(),
                    point.as_ref(),
                    buf.as_mut(),
                ))
            })
        });
    }
}

fn bench_eq_xr<F: Field>(c: &mut Criterion) {
    let mut rng = test_rng();
    for nv in RANGE {
        let point = (0..nv)
            .map(|_| F::random_unsafe(&mut rng))
            .collect::<Vec<_>>();

        // first method
        let label = format!("jolt's eq_xr, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| black_box(EqPolynomial::<F>::evals_jolt(point.as_ref())))
        });

        // second method
        let label = format!("hyperplonk's eq_xr, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| black_box(EqPolynomial::<F>::build_eq_x_r(point.as_ref())))
        });

        // third method
        let label = format!("expander's eq_xr, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| {
                black_box({
                    let mut eq_x_r = vec![F::zero(); 1 << nv];
                    EqPolynomial::<F>::build_eq_x_r_with_buf(point.as_ref(), &F::ONE, &mut eq_x_r);
                })
            })
        });
    }
}

fn bench_scaled_eq_xr<F: Field>(c: &mut Criterion) {
    let mut rng = test_rng();
    for nv in RANGE {
        let point = (0..nv)
            .map(|_| F::random_unsafe(&mut rng))
            .collect::<Vec<_>>();
        let scalar = F::random_unsafe(&mut rng);

        // first method
        let label = format!("jolt's scaled eq_xr, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| {
                black_box(EqPolynomial::<F>::scaled_evals_jolt(
                    point.as_ref(),
                    &scalar,
                ))
            })
        });

        // second method
        let label = format!("expander's scaled eq_xr, dim = {}", nv);
        c.bench_function(label.as_str(), |b| {
            b.iter(|| {
                black_box({
                    let mut eq_x_r = vec![F::zero(); 1 << nv];
                    EqPolynomial::<F>::build_eq_x_r_with_buf(point.as_ref(), &scalar, &mut eq_x_r);
                })
            })
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_mle_eval::<Fr>(c);
    bench_eq_xr::<Fr>(c);
    bench_scaled_eq_xr::<Fr>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
