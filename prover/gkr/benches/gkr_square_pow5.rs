#![allow(clippy::all)]
use circuit::{Circuit, CircuitLayer, CoefType, GateUni};
use config_macros::declare_gkr_config;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gkr::Prover;
use gkr_engine::{
    ExpanderPCS, FieldEngine, GKREngine, GKRScheme, M31x16Config, MPIConfig,
    StructuredReferenceString,
};
use gkr_hashers::SHA256hasher;
use poly_commit::{expander_pcs_init_testing_only, raw::RawExpanderGKR};
use std::hint::black_box;
use transcript::BytesHashTranscript;

fn build_pow5_circuit<C: FieldEngine>(input_var_num: usize) -> Circuit<C> {
    let mut circuit = Circuit::default();
    let n = 1 << input_var_num;

    let mut layer = CircuitLayer {
        input_var_num,
        output_var_num: input_var_num,
        ..Default::default()
    };
    for i in 0..n {
        layer.uni.push(GateUni {
            i_ids: [i],
            o_id: i,
            coef: C::CircuitField::from(1),
            coef_type: CoefType::Constant,
            gate_type: 12345,
        });
    }
    circuit.layers.push(layer);

    let mut layer2 = CircuitLayer {
        input_var_num,
        output_var_num: input_var_num,
        ..Default::default()
    };
    for i in 0..n {
        layer2.uni.push(GateUni {
            i_ids: [i],
            o_id: i,
            coef: C::CircuitField::from(1),
            coef_type: CoefType::Constant,
            gate_type: 12345,
        });
    }
    circuit.layers.push(layer2);

    circuit.identify_rnd_coefs();
    circuit
}

fn prover_run<Cfg: GKREngine>(
    mpi_config: &MPIConfig,
    circuit: &mut Circuit<Cfg::FieldConfig>,
    pcs_params: &<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
    pcs_proving_key: &<<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::PKey,
    pcs_scratch: &mut <Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::ScratchPad,
) where
    Cfg::FieldConfig: FieldEngine,
{
    let mut prover = Prover::<Cfg>::new(mpi_config.clone());
    prover.prepare_mem(circuit);
    prover.prove(circuit, pcs_params, pcs_proving_key, pcs_scratch);
}

fn criterion_pow5_bench(c: &mut Criterion) {
    declare_gkr_config!(
        M31x16Gkr2,
        FieldType::M31x16,
        FiatShamirHashType::SHA256,
        PCSCommitmentType::Raw,
        GKRScheme::GkrSquare
    );

    let input_var_num: usize = 16;
    let mpi_config = MPIConfig::prover_new();
    let mut circuit = build_pow5_circuit::<<M31x16Gkr2 as GKREngine>::FieldConfig>(input_var_num);
    circuit.set_random_input_for_test();

    let (pcs_params, pcs_proving_key, _pcs_vk, mut pcs_scratch) =
        expander_pcs_init_testing_only::<
            <M31x16Gkr2 as GKREngine>::FieldConfig,
            <M31x16Gkr2 as GKREngine>::PCSConfig,
        >(circuit.log_input_size(), &mpi_config);

    let mut group = c.benchmark_group("gkr_square_pow5");
    group.bench_function(
        BenchmarkId::new("m31x16_pow5_2layers", input_var_num),
        |b| {
            b.iter(|| {
                prover_run::<M31x16Gkr2>(
                    &mpi_config,
                    &mut circuit,
                    &pcs_params,
                    &pcs_proving_key,
                    &mut pcs_scratch,
                );
                black_box(())
            })
        },
    );
    group.finish();
}

criterion_group!(benches, criterion_pow5_bench);
criterion_main!(benches);
