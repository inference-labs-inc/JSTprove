use circuit::Circuit;
use config_macros::declare_gkr_config;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gkr::{
    utils::{
        KECCAK_BN254_CIRCUIT, KECCAK_BN254_WITNESS, KECCAK_M31_CIRCUIT, KECCAK_M31_WITNESS,
        POSEIDON_M31_CIRCUIT,
    },
    Prover,
};
use gkr_engine::{
    BN254Config, ExpanderPCS, FieldEngine, GKREngine, GKRScheme, M31x16Config, MPIConfig,
    StructuredReferenceString,
};
use gkr_hashers::SHA256hasher;
use poly_commit::{expander_pcs_init_testing_only, raw::RawExpanderGKR};
use std::hint::black_box;
use transcript::BytesHashTranscript;

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

fn benchmark_setup<Cfg: GKREngine>(
    circuit_file: &str,
    witness_file: Option<&str>,
) -> (
    MPIConfig,
    Circuit<Cfg::FieldConfig>,
    <Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::Params,
    <<Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::SRS as StructuredReferenceString>::PKey,
    <Cfg::PCSConfig as ExpanderPCS<Cfg::FieldConfig>>::ScratchPad,
) {
    let mpi_config = MPIConfig::prover_new();
    let mut circuit = Circuit::<Cfg::FieldConfig>::load_circuit(circuit_file);

    if let Some(witness_file) = witness_file {
        circuit.prover_load_witness_file(witness_file, &mpi_config);
    } else {
        circuit.set_random_input_for_test();
    }

    let (pcs_params, pcs_proving_key, _pcs_verification_key, pcs_scratch) =
        expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
            circuit.log_input_size(),
            &mpi_config,
        );

    (
        mpi_config,
        circuit,
        pcs_params,
        pcs_proving_key,
        pcs_scratch,
    )
}

fn criterion_gkr_keccak(c: &mut Criterion) {
    declare_gkr_config!(
        M31x16ConfigSha2,
        FieldType::M31x16,
        FiatShamirHashType::SHA256,
        PCSCommitmentType::Raw,
        GKRScheme::Vanilla
    );
    declare_gkr_config!(
        BN254ConfigSha2,
        FieldType::BN254,
        FiatShamirHashType::SHA256,
        PCSCommitmentType::Raw,
        GKRScheme::Vanilla
    );

    let (m31_config, mut m31_circuit, m31_pcs_params, m31_pcs_proving_key, mut m31_pcs_scratch) =
        benchmark_setup::<M31x16ConfigSha2>(KECCAK_M31_CIRCUIT, Some(KECCAK_M31_WITNESS));
    let (
        bn254_config,
        mut bn254_circuit,
        bn254_pcs_params,
        bn254_pcs_proving_key,
        mut bn254_pcs_scratch,
    ) = benchmark_setup::<BN254ConfigSha2>(KECCAK_BN254_CIRCUIT, Some(KECCAK_BN254_WITNESS));

    let num_keccak_m31 = 2 * <M31x16ConfigSha2 as GKREngine>::FieldConfig::get_field_pack_size();
    let num_keccak_bn254 = 2 * <BN254ConfigSha2 as GKREngine>::FieldConfig::get_field_pack_size();

    let mut group = c.benchmark_group("single thread proving keccak by GKR vanilla");
    group.bench_function(
        BenchmarkId::new(
            format!(
                "Over M31, with {} keccak instances per proof",
                num_keccak_m31
            ),
            0,
        ),
        |b| {
            b.iter(|| {
                {
                    prover_run::<M31x16ConfigSha2>(
                        &m31_config,
                        &mut m31_circuit,
                        &m31_pcs_params,
                        &m31_pcs_proving_key,
                        &mut m31_pcs_scratch,
                    );
                    black_box(())
                };
            })
        },
    );

    group.bench_function(
        BenchmarkId::new(
            format!(
                "Over BN254, with {} keccak instances per proof",
                num_keccak_bn254
            ),
            0,
        ),
        |b| {
            b.iter(|| {
                {
                    prover_run::<BN254ConfigSha2>(
                        &bn254_config,
                        &mut bn254_circuit,
                        &bn254_pcs_params,
                        &bn254_pcs_proving_key,
                        &mut bn254_pcs_scratch,
                    );
                    black_box(())
                };
            })
        },
    );
}

fn criterion_gkr_poseidon(c: &mut Criterion) {
    declare_gkr_config!(
        M31x16ConfigSha2,
        FieldType::M31x16,
        FiatShamirHashType::SHA256,
        PCSCommitmentType::Raw,
        GKRScheme::GkrSquare
    );

    let (m31_config, mut m31_circuit, pcs_params, pcs_proving_key, mut pcs_scratch) =
        benchmark_setup::<M31x16ConfigSha2>(POSEIDON_M31_CIRCUIT, None);

    let mut group = c.benchmark_group("single thread proving poseidon by GKR^2");
    let num_poseidon_m31 =
        120 * <M31x16ConfigSha2 as GKREngine>::FieldConfig::get_field_pack_size();

    group.bench_function(
        BenchmarkId::new(
            format!(
                "Over M31, with {} poseidon instances per proof",
                num_poseidon_m31
            ),
            0,
        ),
        |b| {
            b.iter(|| {
                {
                    prover_run::<M31x16ConfigSha2>(
                        &m31_config,
                        &mut m31_circuit,
                        &pcs_params,
                        &pcs_proving_key,
                        &mut pcs_scratch,
                    );
                    black_box(())
                };
            })
        },
    );
}

criterion_group!(benches, criterion_gkr_keccak, criterion_gkr_poseidon);
criterion_main!(benches);
