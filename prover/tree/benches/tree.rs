use arith::{Field, SimdField};
use ark_std::{rand::RngCore, test_rng};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gf2::{GF2x128, GF2};
use tree::{Leaf, Tree, LEAF_BYTES};
use tynm::type_name;

const FINAL_MT_LEAVES_LOG2: usize = 15;

fn tree_building_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Keccak merkle tree");

    let mut rng = test_rng();
    let mut data_buffer = [0u8; LEAF_BYTES];
    let leaves: Vec<_> = (0..(1 << FINAL_MT_LEAVES_LOG2))
        .map(|_| {
            Leaf::new({
                rng.fill_bytes(&mut data_buffer);
                data_buffer
            })
        })
        .collect();

    for i in 10..=FINAL_MT_LEAVES_LOG2 {
        group
            .bench_function(BenchmarkId::new(format!("2^{i} leaves"), i), |b| {
                let leaves_benchmark = leaves[..(1 << i)].to_vec();

                b.iter(|| {
                    Tree::new_with_leaves(leaves_benchmark.clone());
                })
            })
            .sample_size(10);
    }
}

fn compact_field_elem_tree_building_benchmark_generic<F, PackF>(c: &mut Criterion)
where
    F: Field,
    PackF: SimdField<Scalar = F>,
{
    let mut group = c.benchmark_group(format!(
        "Keccak merkle tree with field element {} packed by SIMD field element {}",
        type_name::<F>(),
        type_name::<PackF>()
    ));
    let num_of_elems_in_leaf = LEAF_BYTES * 8 / F::FIELD_SIZE;

    let mut rng = test_rng();
    let field_elems: Vec<_> = (0..(1 << FINAL_MT_LEAVES_LOG2) * num_of_elems_in_leaf)
        .map(|_| F::random_unsafe(&mut rng))
        .collect();

    for i in 10..=FINAL_MT_LEAVES_LOG2 {
        group
            .bench_function(BenchmarkId::new(format!("2^{i} leaves"), i), |b| {
                let field_elems_benchmark = field_elems[..(1 << i) * num_of_elems_in_leaf].to_vec();

                b.iter(|| {
                    Tree::compact_new_with_field_elems::<F, PackF>(field_elems_benchmark.clone());
                })
            })
            .sample_size(10);
    }
}

fn compact_field_elem_tree_building_benchmark(c: &mut Criterion) {
    compact_field_elem_tree_building_benchmark_generic::<GF2, GF2x128>(c);
}

fn compact_packed_field_elem_tree_building_benchmark_generic<F, PackF>(c: &mut Criterion)
where
    F: Field,
    PackF: SimdField<Scalar = F>,
{
    let mut group = c.benchmark_group(format!(
        "Keccak merkle tree with SIMD field element {}",
        type_name::<PackF>()
    ));
    let num_of_elems_in_leaf = LEAF_BYTES / PackF::SIZE;

    let mut rng = test_rng();
    let field_elems: Vec<_> = (0..(1 << FINAL_MT_LEAVES_LOG2) * num_of_elems_in_leaf)
        .map(|_| PackF::random_unsafe(&mut rng))
        .collect();

    for i in 10..=FINAL_MT_LEAVES_LOG2 {
        group
            .bench_function(BenchmarkId::new(format!("2^{i} leaves"), i), |b| {
                let field_elems_benchmark = field_elems[..(1 << i) * num_of_elems_in_leaf].to_vec();

                b.iter(|| {
                    Tree::compact_new_with_packed_field_elems::<F, PackF>(
                        field_elems_benchmark.clone(),
                    );
                })
            })
            .sample_size(10);
    }
}

fn compact_packed_field_elem_tree_building_benchmark(c: &mut Criterion) {
    compact_packed_field_elem_tree_building_benchmark_generic::<GF2, GF2x128>(c);
}

criterion_group!(
    bench,
    tree_building_benchmark,
    compact_field_elem_tree_building_benchmark,
    compact_packed_field_elem_tree_building_benchmark
);
criterion_main!(bench);
