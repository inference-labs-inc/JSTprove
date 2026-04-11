use arith::{Field, SimdField};
use circuit::{Circuit, CircuitLayer, CoefType, GateConst, GateUni};
use gkr_engine::{FieldEngine, GKREngine, MPIConfig, Proof};
use poly_commit::expander_pcs_init_testing_only;

use crate::{M31x16ConfigSha2RawSquare, Prover, Verifier};

type Cfg = M31x16ConfigSha2RawSquare;
type FC = <Cfg as GKREngine>::FieldConfig;

fn gkr_square_test_circuit<C: FieldEngine>() -> Circuit<C> {
    let mut circuit = Circuit::default();
    let mut l1 = CircuitLayer {
        input_var_num: 2,
        output_var_num: 2,
        ..Default::default()
    };
    l1.const_.push(GateConst {
        i_ids: [],
        o_id: 3,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::PublicInput(0),
        gate_type: 0,
    });
    l1.uni.push(GateUni {
        i_ids: [0],
        o_id: 0,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12345,
    });
    l1.uni.push(GateUni {
        i_ids: [1],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    l1.uni.push(GateUni {
        i_ids: [1],
        o_id: 2,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    l1.uni.push(GateUni {
        i_ids: [2],
        o_id: 2,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    l1.uni.push(GateUni {
        i_ids: [3],
        o_id: 3,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    circuit.layers.push(l1);
    let mut output_layer = CircuitLayer {
        input_var_num: 2,
        output_var_num: 1,
        ..Default::default()
    };
    output_layer.uni.push(GateUni {
        i_ids: [0],
        o_id: 0,
        coef: C::CircuitField::from(11),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    output_layer.uni.push(GateUni {
        i_ids: [1],
        o_id: 0,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    output_layer.uni.push(GateUni {
        i_ids: [1],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
    output_layer.uni.push(GateUni {
        i_ids: [2],
        o_id: 1,
        coef: C::CircuitField::from(1),
        coef_type: CoefType::Constant,
        gate_type: 12346,
    });
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

fn setup_circuit() -> Circuit<FC> {
    let mpi_config = MPIConfig::prover_new();
    let mut circuit = gkr_square_test_circuit::<FC>();
    let mut final_vals = (0..16)
        .map(|x| <FC as FieldEngine>::CircuitField::from(x))
        .collect::<Vec<_>>();
    final_vals[0] += <FC as FieldEngine>::CircuitField::from(mpi_config.world_rank as u32);
    let final_vals = <FC as FieldEngine>::SimdCircuitField::pack(&final_vals);
    circuit.layers[0].input_vals = vec![2.into(), 3.into(), 5.into(), final_vals];
    circuit.public_input = vec![13.into()];
    circuit
}

fn do_prove(circuit: &mut Circuit<FC>) -> (<FC as FieldEngine>::ChallengeField, Proof) {
    let mpi_config = MPIConfig::prover_new();
    circuit.evaluate();
    let (pcs_params, pcs_proving_key, _, mut pcs_scratch) =
        expander_pcs_init_testing_only::<FC, <Cfg as GKREngine>::PCSConfig>(
            circuit.log_input_size(),
            &mpi_config,
        );
    let mut prover = Prover::<Cfg>::new(mpi_config);
    prover.prepare_mem(circuit);
    prover.prove(circuit, &pcs_params, &pcs_proving_key, &mut pcs_scratch)
}

fn do_verify(
    circuit: &mut Circuit<FC>,
    public_input: &[<FC as FieldEngine>::SimdCircuitField],
    claimed_v: &<FC as FieldEngine>::ChallengeField,
    proof: &Proof,
) -> bool {
    let mpi_config = MPIConfig::prover_new();
    let (pcs_params, _, pcs_verification_key, _) = expander_pcs_init_testing_only::<
        FC,
        <Cfg as GKREngine>::PCSConfig,
    >(circuit.log_input_size(), &mpi_config);
    let verifier = Verifier::<Cfg>::new(mpi_config);
    verifier.verify(
        circuit,
        public_input,
        claimed_v,
        &pcs_params,
        &pcs_verification_key,
        proof,
    )
}

#[test]
fn adversarial_forged_claimed_v() {
    let mut circuit = setup_circuit();
    let (claimed_v, proof) = do_prove(&mut circuit);
    let public_input = circuit.public_input.clone();

    let forged_v = claimed_v + <FC as FieldEngine>::ChallengeField::from(1u32);
    assert!(
        !do_verify(&mut circuit, &public_input, &forged_v, &proof),
        "verifier must reject forged claimed_v"
    );
}

#[test]
fn adversarial_wrong_public_input() {
    let mut circuit = setup_circuit();
    let (claimed_v, proof) = do_prove(&mut circuit);
    let mut wrong_pi = circuit.public_input.clone();
    wrong_pi[0] = 999.into();
    assert!(
        !do_verify(&mut circuit, &wrong_pi, &claimed_v, &proof),
        "verifier must reject proof with altered public input"
    );
}

#[test]
fn adversarial_zero_claimed_v() {
    let mut circuit = setup_circuit();
    let (_claimed_v, proof) = do_prove(&mut circuit);
    let public_input = circuit.public_input.clone();
    let zero_v = <FC as FieldEngine>::ChallengeField::ZERO;
    assert!(
        !do_verify(&mut circuit, &public_input, &zero_v, &proof),
        "verifier must reject zero claimed_v"
    );
}

#[test]
fn adversarial_truncated_proof_bytes() {
    let mut circuit = setup_circuit();
    let (claimed_v, proof) = do_prove(&mut circuit);
    let public_input = circuit.public_input.clone();

    for fraction in &[2, 4, 8, 16] {
        let cutoff = proof.bytes.len() / fraction;
        if cutoff == 0 {
            continue;
        }
        let truncated = Proof {
            bytes: proof.bytes[..cutoff].to_vec(),
        };
        assert!(
            !do_verify(&mut circuit, &public_input, &claimed_v, &truncated),
            "verifier must reject proof truncated to 1/{fraction}"
        );
    }
}

#[test]
fn adversarial_bitflip_proof_stream() {
    use rand::{Rng, SeedableRng};

    let mut circuit = setup_circuit();
    let (claimed_v, proof) = do_prove(&mut circuit);
    let public_input = circuit.public_input.clone();

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345678);
    for trial in 0..50 {
        let byte_idx = rng.gen_range(0..proof.bytes.len());
        let bit_idx: u8 = rng.gen_range(0..8);
        let mut corrupted_bytes = proof.bytes.clone();
        corrupted_bytes[byte_idx] ^= 1 << bit_idx;
        let corrupted = Proof {
            bytes: corrupted_bytes,
        };
        assert!(
            !do_verify(&mut circuit, &public_input, &claimed_v, &corrupted),
            "trial {trial}: bitflip at byte {byte_idx} bit {bit_idx} was accepted"
        );
    }
}

#[test]
fn adversarial_replayed_proof_different_inputs() {
    let mut circuit_a = setup_circuit();
    let (claimed_v, proof) = do_prove(&mut circuit_a);

    let mut circuit_b = gkr_square_test_circuit::<FC>();
    let different_vals = (0..16)
        .map(|x| <FC as FieldEngine>::CircuitField::from(x + 100))
        .collect::<Vec<_>>();
    let different_vals = <FC as FieldEngine>::SimdCircuitField::pack(&different_vals);
    circuit_b.layers[0].input_vals = vec![99.into(), 88.into(), 77.into(), different_vals];
    circuit_b.public_input = vec![13.into()];
    circuit_b.evaluate();

    let public_input = circuit_b.public_input.clone();
    assert!(
        !do_verify(&mut circuit_b, &public_input, &claimed_v, &proof),
        "verifier must reject proof replayed against different circuit inputs"
    );
}

#[test]
fn adversarial_random_proof_bytes() {
    use rand::{Rng, SeedableRng};

    let mut circuit = setup_circuit();
    let (claimed_v, _proof) = do_prove(&mut circuit);
    let public_input = circuit.public_input.clone();

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xf01e);
    for trial in 0..20 {
        let len: usize = rng.gen_range(1..2048);
        let random_bytes: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
        let random_proof = Proof {
            bytes: random_bytes,
        };
        assert!(
            !do_verify(&mut circuit, &public_input, &claimed_v, &random_proof),
            "trial {trial}: random proof bytes were accepted"
        );
    }
}

#[test]
fn adversarial_empty_proof() {
    let mut circuit = setup_circuit();
    let (claimed_v, _proof) = do_prove(&mut circuit);
    let public_input = circuit.public_input.clone();
    let empty_proof = Proof { bytes: vec![] };
    assert!(
        !do_verify(&mut circuit, &public_input, &claimed_v, &empty_proof),
        "verifier must reject empty proof"
    );
}
