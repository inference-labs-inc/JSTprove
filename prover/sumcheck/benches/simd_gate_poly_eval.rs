use arith::{Field, Fr};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mersenne31::M31Ext3;
use rand::thread_rng;

fn gen_random_m31ext3(n: usize) -> Vec<M31Ext3> {
    let mut rng = thread_rng();
    (0..n).map(|_| M31Ext3::random_unsafe(&mut rng)).collect()
}

fn gen_random_fr(n: usize) -> Vec<Fr> {
    let mut rng = thread_rng();
    (0..n).map(|_| Fr::random_unsafe(&mut rng)).collect()
}

#[inline(never)]
fn baseline_eval_m31(
    bk_eq: &[M31Ext3],
    bk_f: &[M31Ext3],
    bk_hg: &[M31Ext3],
    eval_size: usize,
) -> [M31Ext3; 4] {
    let mut p0 = M31Ext3::zero();
    let mut p1 = M31Ext3::zero();
    let mut p2 = M31Ext3::zero();
    let mut p3 = M31Ext3::zero();

    for i in 0..eval_size {
        let eq_v_0 = bk_eq[i * 2];
        let eq_v_1 = bk_eq[i * 2 + 1];
        let f_v_0 = bk_f[i * 2];
        let f_v_1 = bk_f[i * 2 + 1];
        let hg_v_0 = bk_hg[i * 2];
        let hg_v_1 = bk_hg[i * 2 + 1];

        p0 += eq_v_0 * f_v_0 * hg_v_0;
        p1 += eq_v_1 * f_v_1 * hg_v_1;

        let tmp0 = eq_v_1 - eq_v_0;
        let tmp1 = f_v_1 - f_v_0;
        let tmp2 = hg_v_1 - hg_v_0;
        let tmp3 = eq_v_1 + tmp0;
        let tmp4 = f_v_1 + tmp1;
        let tmp5 = hg_v_1 + tmp2;

        p2 += tmp3 * tmp4 * tmp5;
        p3 += (tmp3 + tmp0) * (tmp4 + tmp1) * (tmp5 + tmp2);
    }

    [p0, p1, p2, p3]
}

#[inline(never)]
fn baseline_eval_fr(bk_eq: &[Fr], bk_f: &[Fr], bk_hg: &[Fr], eval_size: usize) -> [Fr; 4] {
    let mut p0 = Fr::zero();
    let mut p1 = Fr::zero();
    let mut p2 = Fr::zero();
    let mut p3 = Fr::zero();

    for i in 0..eval_size {
        let eq_v_0 = bk_eq[i * 2];
        let eq_v_1 = bk_eq[i * 2 + 1];
        let f_v_0 = bk_f[i * 2];
        let f_v_1 = bk_f[i * 2 + 1];
        let hg_v_0 = bk_hg[i * 2];
        let hg_v_1 = bk_hg[i * 2 + 1];

        p0 += eq_v_0 * f_v_0 * hg_v_0;
        p1 += eq_v_1 * f_v_1 * hg_v_1;

        let tmp0 = eq_v_1 - eq_v_0;
        let tmp1 = f_v_1 - f_v_0;
        let tmp2 = hg_v_1 - hg_v_0;
        let tmp3 = eq_v_1 + tmp0;
        let tmp4 = f_v_1 + tmp1;
        let tmp5 = hg_v_1 + tmp2;

        p2 += tmp3 * tmp4 * tmp5;
        p3 += (tmp3 + tmp0) * (tmp4 + tmp1) * (tmp5 + tmp2);
    }

    [p0, p1, p2, p3]
}

#[inline(never)]
fn toom_cook_eval_m31(
    bk_eq: &[M31Ext3],
    bk_f: &[M31Ext3],
    bk_hg: &[M31Ext3],
    eval_size: usize,
) -> [M31Ext3; 4] {
    let mut p0 = M31Ext3::zero();
    let mut p1 = M31Ext3::zero();
    let mut p_neg1 = M31Ext3::zero();
    let mut p_inf = M31Ext3::zero();

    for i in 0..eval_size {
        let eq_v_0 = bk_eq[i * 2];
        let eq_v_1 = bk_eq[i * 2 + 1];
        let f_v_0 = bk_f[i * 2];
        let f_v_1 = bk_f[i * 2 + 1];
        let hg_v_0 = bk_hg[i * 2];
        let hg_v_1 = bk_hg[i * 2 + 1];

        let delta_eq = eq_v_1 - eq_v_0;
        let delta_f = f_v_1 - f_v_0;
        let delta_hg = hg_v_1 - hg_v_0;

        p0 += eq_v_0 * f_v_0 * hg_v_0;
        p1 += eq_v_1 * f_v_1 * hg_v_1;
        p_neg1 += (eq_v_0 - delta_eq) * (f_v_0 - delta_f) * (hg_v_0 - delta_hg);
        p_inf += delta_eq * delta_f * delta_hg;
    }

    let c0 = p0;
    let c3 = p_inf;
    let c1_plus_c2 = p1 - c0 - c3;
    let neg_c1_minus_c2 = p_neg1 - c0 + c3;
    let half = M31Ext3::INV_2;
    let c1 = (c1_plus_c2 - neg_c1_minus_c2) * half;
    let c2 = (c1_plus_c2 + neg_c1_minus_c2) * half;

    let r0 = c0;
    let r1 = c0 + c1 + c2 + c3;
    let r2 = c0 + c1.double() + c2.double().double() + c3.double().double().double();
    let r3 = c0 + c1.mul_by_3() + c2.mul_by_3().mul_by_3() + c3.mul_by_3().mul_by_3().mul_by_3();

    [r0, r1, r2, r3]
}

#[inline(never)]
fn fused_coeff_eval_m31(
    bk_eq: &[M31Ext3],
    bk_f: &[M31Ext3],
    bk_hg: &[M31Ext3],
    eval_size: usize,
) -> [M31Ext3; 4] {
    let mut c0 = M31Ext3::zero();
    let mut c1 = M31Ext3::zero();
    let mut c2 = M31Ext3::zero();
    let mut c3 = M31Ext3::zero();

    for i in 0..eval_size {
        let eq_v_0 = bk_eq[i * 2];
        let eq_v_1 = bk_eq[i * 2 + 1];
        let f_v_0 = bk_f[i * 2];
        let f_v_1 = bk_f[i * 2 + 1];
        let hg_v_0 = bk_hg[i * 2];
        let hg_v_1 = bk_hg[i * 2 + 1];

        let d_eq = eq_v_1 - eq_v_0;
        let d_f = f_v_1 - f_v_0;
        let d_hg = hg_v_1 - hg_v_0;

        let abc = eq_v_0 * f_v_0 * hg_v_0;

        let ab_dc = eq_v_0 * f_v_0 * d_hg;
        let a_db_c = eq_v_0 * d_f * hg_v_0;
        let da_bc = d_eq * f_v_0 * hg_v_0;

        let a_db_dc = eq_v_0 * d_f * d_hg;
        let da_b_dc = d_eq * f_v_0 * d_hg;
        let da_db_c = d_eq * d_f * hg_v_0;

        let da_db_dc = d_eq * d_f * d_hg;

        c0 += abc;
        c1 += ab_dc + a_db_c + da_bc;
        c2 += a_db_dc + da_b_dc + da_db_c;
        c3 += da_db_dc;
    }

    let r0 = c0;
    let r1 = c0 + c1 + c2 + c3;
    let r2 = c0 + c1.double() + c2.double().double() + c3.double().double().double();
    let r3 = c0 + c1.mul_by_3() + c2.mul_by_3().mul_by_3() + c3.mul_by_3().mul_by_3().mul_by_3();

    [r0, r1, r2, r3]
}

fn criterion_simd_gate_m31ext3(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_gate_poly_eval_m31ext3");

    for var_num in [4, 8, 12, 16] {
        let total = 1 << var_num;
        let eval_size = total / 2;
        let bk_eq = gen_random_m31ext3(total);
        let bk_f = gen_random_m31ext3(total);
        let bk_hg = gen_random_m31ext3(total);

        group.bench_function(BenchmarkId::new("baseline", var_num), |b| {
            b.iter(|| black_box(baseline_eval_m31(&bk_eq, &bk_f, &bk_hg, eval_size)))
        });

        group.bench_function(BenchmarkId::new("toom_cook", var_num), |b| {
            b.iter(|| black_box(toom_cook_eval_m31(&bk_eq, &bk_f, &bk_hg, eval_size)))
        });

        group.bench_function(BenchmarkId::new("fused_coeff", var_num), |b| {
            b.iter(|| black_box(fused_coeff_eval_m31(&bk_eq, &bk_f, &bk_hg, eval_size)))
        });
    }

    group.finish();
}

fn criterion_simd_gate_bn254(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_gate_poly_eval_bn254");

    for var_num in [4, 8, 12] {
        let total = 1 << var_num;
        let eval_size = total / 2;
        let bk_eq = gen_random_fr(total);
        let bk_f = gen_random_fr(total);
        let bk_hg = gen_random_fr(total);

        group.bench_function(BenchmarkId::new("baseline", var_num), |b| {
            b.iter(|| black_box(baseline_eval_fr(&bk_eq, &bk_f, &bk_hg, eval_size)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    criterion_simd_gate_m31ext3,
    criterion_simd_gate_bn254
);
criterion_main!(benches);
