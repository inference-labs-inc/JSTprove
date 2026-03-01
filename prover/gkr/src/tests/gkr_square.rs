use arith::SimdField;
use circuit::{Circuit, CircuitLayer, CoefType, GateConst, GateUni};
use config_macros::declare_gkr_config;
use gkr_engine::{FieldEngine, GKREngine, GKRScheme, M31x16Config, MPIConfig, MPIEngine};
use gkr_hashers::SHA256hasher;
use poly_commit::{expander_pcs_init_testing_only, RawExpanderGKR};
use transcript::BytesHashTranscript;

use crate::{Prover, Verifier};

/// A simple GKR2 test circuit:
/// ```text
///         N_0_0     N_0_1             Layer 0 (Output)
///    x11 /   \    /    |  \
///  N_1_0     N_1_1  N_1_2  N_1_3      Layer 1
///     |       |    /  |      |   \
/// Pow5|       |  /    |      |    PI[0]
///  N_2_0     N_2_1   N_2_2  N_2_3     Layer 2 (Input)
/// ```
/// (Unmarked lines are `+` gates with coeff 1)
pub fn gkr_square_test_circuit<C: FieldEngine>() -> Circuit<C> {
    let mut circuit = Circuit::default();

    // Layer 1
    let mut l1 = CircuitLayer {
        input_var_num: 2,
        output_var_num: 2,
        ..Default::default()
    };
    // N_1_3 += PI[0] (public input)
    l1.const_.push(GateConst {
        i_ids: [],
        o_id: 3,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::PublicInput(0),
        gate_type: 0,
    });
    // N_1_0 += (N_2_0)^5
    l1.uni.push(GateUni {
        i_ids: [0],
        o_id: 0,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12345,
    });

    // N_1_1 += N_2_1
    l1.uni.push(GateUni {
        i_ids: [1],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_1_2 += N_2_1
    l1.uni.push(GateUni {
        i_ids: [1],
        o_id: 2,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_1_2 += N_2_2
    l1.uni.push(GateUni {
        i_ids: [2],
        o_id: 2,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_1_3 += N_2_3
    l1.uni.push(GateUni {
        i_ids: [3],
        o_id: 3,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    circuit.layers.push(l1);

    // Output layer
    let mut output_layer = CircuitLayer {
        input_var_num: 2,
        output_var_num: 1,
        ..Default::default()
    };
    // N_0_0 += 11 * N_1_0
    output_layer.uni.push(GateUni {
        i_ids: [0],
        o_id: 0,
        coef: C::CircuitField::from(11),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_0_0 += N_1_1
    output_layer.uni.push(GateUni {
        i_ids: [1],
        o_id: 0,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_0_1 += N_1_1
    output_layer.uni.push(GateUni {
        i_ids: [1],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_0_1 += N_1_2
    output_layer.uni.push(GateUni {
        i_ids: [2],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    // N_0_1 += N_1_3
    output_layer.uni.push(GateUni {
        i_ids: [3],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    circuit.layers.push(output_layer);

    circuit.identify_rnd_coefs();
    circuit
}

#[test]
fn gkr_square_correctness_test() {
    declare_gkr_config!(
        GkrConfigType,
        FieldType::M31,
        FiatShamirHashType::SHA256,
        PolynomialCommitmentType::Raw,
        GKRScheme::GkrSquare
    );
    type GkrFieldConfigType = <GkrConfigType as GKREngine>::FieldConfig;
    let mpi_config = MPIConfig::prover_new();
    let mut circuit = gkr_square_test_circuit::<GkrFieldConfigType>();
    // Set input layers with N_2_0 = 3, N_2_1 = 5, N_2_2 = 7,
    // and N_2_3 varying from 0 to 15
    let mut final_vals = (0..16).map(|x| x.into()).collect::<Vec<_>>(); // Add variety for MPI participants
    final_vals[0] +=
        <GkrFieldConfigType as FieldEngine>::CircuitField::from(mpi_config.world_rank as u32);
    let final_vals = <GkrFieldConfigType as FieldEngine>::SimdCircuitField::pack(&final_vals);
    circuit.layers[0].input_vals = vec![2.into(), 3.into(), 5.into(), final_vals];
    // Set public input PI[0] = 13
    circuit.public_input = vec![13.into()];

    do_prove_verify::<GkrConfigType>(&mpi_config, &mut circuit);

}

fn do_prove_verify<Cfg: GKREngine>(
    mpi_config: &MPIConfig,
    circuit: &mut Circuit<Cfg::FieldConfig>,
) {
    circuit.evaluate();

    let (pcs_params, pcs_proving_key, pcs_verification_key, mut pcs_scratch) =
        expander_pcs_init_testing_only::<Cfg::FieldConfig, Cfg::PCSConfig>(
            circuit.log_input_size(),
            mpi_config,
        );

    // Prove
    let mut prover = Prover::<Cfg>::new(mpi_config.clone());
    prover.prepare_mem(circuit);
    let (claimed_v, proof) = prover.prove(circuit, &pcs_params, &pcs_proving_key, &mut pcs_scratch);

    // Verify if root process
    if mpi_config.is_root() {
        let verifier = Verifier::<Cfg>::new(mpi_config.clone());
        let public_input = circuit.public_input.clone();
        assert!(verifier.verify(
            circuit,
            &public_input,
            &claimed_v,
            &pcs_params,
            &pcs_verification_key,
            &proof
        ))
    }
}
