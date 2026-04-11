use super::prove::{prove, HolographicProof, LayerEvalPoint, LayerHolographicOpening};
use super::setup::setup;
use super::verify::{verify, VerifyError};
use circuit::{Circuit, CircuitLayer, CoefType, GateAdd, GateMul, GateUni, StructureInfo};
use gkr_engine::{Goldilocksx1Config, Transcript};
type C = Goldilocksx1Config;
use arith::Field;
use goldilocks::{Goldilocks, GoldilocksExt4};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use transcript::BytesHashTranscript;
type Sha2T = BytesHashTranscript<gkr_hashers::SHA256hasher>;

fn rng_for(label: &str) -> ChaCha20Rng {
    let mut seed = [0u8; 32];
    let bytes = label.as_bytes();
    let n = bytes.len().min(32);
    seed[..n].copy_from_slice(&bytes[..n]);
    ChaCha20Rng::from_seed(seed)
}

fn make_layer(input_var_num: usize, output_var_num: usize) -> CircuitLayer<C> {
    CircuitLayer {
        input_var_num,
        output_var_num,
        input_vals: Vec::new(),
        output_vals: Vec::new(),
        mul: Vec::new(),
        add: Vec::new(),
        const_: Vec::new(),
        uni: Vec::new(),
        structure_info: StructureInfo::default(),
    }
}

fn mul_gate(o: usize, x: usize, y: usize, coef: u64) -> GateMul<C> {
    GateMul {
        i_ids: [x, y],
        o_id: o,
        coef_type: CoefType::Constant,
        coef: Goldilocks::from(coef),
        gate_type: 0,
    }
}

fn add_gate(o: usize, x: usize, coef: u64) -> GateAdd<C> {
    GateAdd {
        i_ids: [x],
        o_id: o,
        coef_type: CoefType::Constant,
        coef: Goldilocks::from(coef),
        gate_type: 0,
    }
}

fn uni_gate(o: usize, x: usize, coef: u64) -> GateUni<C> {
    GateUni {
        i_ids: [x],
        o_id: o,
        coef: Goldilocks::from(coef),
        coef_type: CoefType::Constant,
        gate_type: 12345,
    }
}

fn build_two_layer_circuit() -> Circuit<C> {
    let mut layer0 = make_layer(2, 2);
    layer0.mul.push(mul_gate(0, 1, 2, 5));
    layer0.mul.push(mul_gate(1, 2, 3, 7));
    layer0.add.push(add_gate(2, 0, 13));
    layer0.add.push(add_gate(3, 1, 17));
    let mut layer1 = make_layer(2, 2);
    layer1.mul.push(mul_gate(3, 0, 1, 19));
    layer1.add.push(add_gate(1, 2, 23));
    Circuit {
        layers: vec![layer0, layer1],
        public_input: Vec::new(),
        expected_num_output_zeros: 0,
        rnd_coefs_identified: false,
        rnd_coefs: Vec::new(),
    }
}

fn build_circuit_with_uni() -> Circuit<C> {
    let mut layer0 = make_layer(2, 2);
    layer0.uni.push(uni_gate(0, 0, 1));
    layer0.add.push(add_gate(1, 1, 3));
    let mut layer1 = make_layer(2, 2);
    layer1.mul.push(mul_gate(0, 0, 1, 11));
    layer1.add.push(add_gate(1, 2, 7));
    Circuit {
        layers: vec![layer0, layer1],
        public_input: Vec::new(),
        expected_num_output_zeros: 0,
        rnd_coefs_identified: false,
        rnd_coefs: Vec::new(),
    }
}

fn random_eval_points(
    rng: &mut ChaCha20Rng,
    pk: &super::setup::HolographicProvingKey<C>,
) -> Vec<LayerEvalPoint<GoldilocksExt4>> {
    pk.layers
        .iter()
        .map(|layer| {
            let n_z = layer.n_z;
            let n_x = layer.n_x;
            let mul_z: Vec<GoldilocksExt4> = (0..n_z)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let mul_x: Vec<GoldilocksExt4> = (0..n_x)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let mul_y: Vec<GoldilocksExt4> = (0..n_x)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let add_z: Vec<GoldilocksExt4> = (0..n_z)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let add_x: Vec<GoldilocksExt4> = (0..n_x)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let mul_claim = layer
                .mul
                .as_ref()
                .map(|w| w.poly.evaluate::<GoldilocksExt4>(&mul_z, &mul_x, &mul_y))
                .unwrap_or(GoldilocksExt4::ZERO);
            let add_claim = layer
                .add
                .as_ref()
                .map(|w| w.poly.evaluate::<GoldilocksExt4>(&add_z, &add_x, &[]))
                .unwrap_or(GoldilocksExt4::ZERO);
            let uni_z: Vec<GoldilocksExt4> = (0..n_z)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let uni_x: Vec<GoldilocksExt4> = (0..n_x)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let uni_claim = layer
                .uni
                .as_ref()
                .map(|w| w.poly.evaluate::<GoldilocksExt4>(&uni_z, &uni_x, &[]))
                .unwrap_or(GoldilocksExt4::ZERO);
            let cst_z: Vec<GoldilocksExt4> = (0..n_z)
                .map(|_| GoldilocksExt4::random_unsafe(&mut *rng))
                .collect();
            let cst_claim = layer
                .cst
                .as_ref()
                .map(|w| w.poly.evaluate::<GoldilocksExt4>(&cst_z, &[], &[]))
                .unwrap_or(GoldilocksExt4::ZERO);
            LayerEvalPoint {
                layer_index: layer.layer_index,
                mul_z,
                mul_x,
                mul_y,
                add_z,
                add_x,
                uni_z,
                uni_x,
                cst_z,
                mul_claim,
                add_claim,
                uni_claim,
                cst_claim,
            }
        })
        .collect()
}

fn honest_prove_verify(
    circuit: Circuit<C>,
    seed: &str,
) -> (
    super::setup::HolographicProvingKey<C>,
    super::setup::HolographicVerifyingKey,
    Vec<LayerEvalPoint<GoldilocksExt4>>,
    HolographicProof<GoldilocksExt4>,
) {
    let mut rng = rng_for(seed);
    let (pk, vk) = setup::<C>(circuit).unwrap();
    let eval_points = random_eval_points(&mut rng, &pk);
    let mut prover_t = Sha2T::new();
    let proof = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut prover_t).unwrap();
    let mut verifier_t = Sha2T::new();
    verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t)
        .expect("honest proof must verify");
    (pk, vk, eval_points, proof)
}

#[test]
fn adversarial_forged_mul_claim_detected_by_verifier() {
    let (_pk, vk, mut eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "forge_mul");
    eval_points[0].mul_claim += GoldilocksExt4::ONE;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        matches!(
            result,
            Err(VerifyError::ClaimMismatch {
                layer: 0,
                which: "mul"
            })
        ),
        "verifier must reject when mul_claim is forged post-prove; got {result:?}"
    );
}

#[test]
fn adversarial_forged_add_claim_detected_by_verifier() {
    let (_pk, vk, mut eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "forge_add");
    eval_points[1].add_claim += GoldilocksExt4::ONE;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        matches!(
            result,
            Err(VerifyError::ClaimMismatch {
                layer: 1,
                which: "add"
            })
        ),
        "verifier must reject when add_claim is forged post-prove; got {result:?}"
    );
}

#[test]
fn adversarial_forged_uni_claim_detected_by_verifier() {
    let (_pk, vk, mut eval_points, proof) =
        honest_prove_verify(build_circuit_with_uni(), "forge_uni");
    eval_points[0].uni_claim += GoldilocksExt4::ONE;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        matches!(
            result,
            Err(VerifyError::ClaimMismatch {
                layer: 0,
                which: "uni"
            })
        ),
        "verifier must reject when uni_claim is forged post-prove; got {result:?}"
    );
}

#[test]
fn adversarial_cross_layer_proof_replay() {
    let (_pk, vk, eval_points, mut proof) =
        honest_prove_verify(build_two_layer_circuit(), "cross_layer");

    let layer0_opening = proof.layers[0].clone();
    proof.layers[1] = LayerHolographicOpening {
        layer_index: 1,
        mul: layer0_opening.mul,
        add: layer0_opening.add,
        uni: layer0_opening.uni,
        cst: layer0_opening.cst,
    };

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject cross-layer replayed openings; got {result:?}"
    );
}

#[test]
fn adversarial_swapped_layer_order() {
    let (_pk, vk, eval_points, mut proof) =
        honest_prove_verify(build_two_layer_circuit(), "swap_layers");

    proof.layers.swap(0, 1);

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject swapped layer order; got {result:?}"
    );
}

#[test]
fn adversarial_schema_injection_mul_on_add_only_layer() {
    let mut layer_mul_only = make_layer(2, 2);
    layer_mul_only.mul.push(mul_gate(0, 1, 2, 5));
    let mut layer_add_only = make_layer(2, 2);
    layer_add_only.add.push(add_gate(1, 0, 7));
    let circuit: Circuit<C> = Circuit {
        layers: vec![layer_mul_only, layer_add_only],
        public_input: Vec::new(),
        expected_num_output_zeros: 0,
        rnd_coefs_identified: false,
        rnd_coefs: Vec::new(),
    };

    let (_pk, vk, eval_points, mut proof) = honest_prove_verify(circuit, "schema_inject");

    let stolen_mul = proof.layers[0].mul.clone();
    proof.layers[1].mul = stolen_mul;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject mul opening injected into add-only layer; got {result:?}"
    );
}

#[test]
fn adversarial_schema_removal_mul_stripped() {
    let (_pk, vk, eval_points, mut proof) =
        honest_prove_verify(build_two_layer_circuit(), "schema_strip");

    proof.layers[0].mul = None;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject when mul opening is stripped; got {result:?}"
    );
}

#[test]
fn adversarial_vk_circuit_mismatch() {
    let mut circuit_a = build_two_layer_circuit();
    circuit_a.layers[0].mul[0].coef = Goldilocks::from(999u64);

    let mut rng_a = rng_for("vk_mismatch_a");
    let (pk_a, _vk_a) = setup::<C>(circuit_a).unwrap();
    let eval_points_a = random_eval_points(&mut rng_a, &pk_a);
    let mut prover_t_a = Sha2T::new();
    let proof_a =
        prove::<C, GoldilocksExt4, Sha2T>(&pk_a, &eval_points_a, &mut prover_t_a).unwrap();

    let circuit_b = build_two_layer_circuit();
    let (_pk_b, vk_b) = setup::<C>(circuit_b).unwrap();

    let mut verifier_t = Sha2T::new();
    let result =
        verify::<C, GoldilocksExt4, Sha2T>(&vk_b, &eval_points_a, &proof_a, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject proof from circuit A verified against VK from circuit B; got {result:?}"
    );
}

#[test]
fn adversarial_eval_point_perturbation_after_honest_prove() {
    let (_pk, vk, mut eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "eval_perturb");

    eval_points[0].mul_z[0] += GoldilocksExt4::ONE;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject when eval point is perturbed post-prove; got {result:?}"
    );
}

#[test]
fn adversarial_zero_proof_forgery() {
    let (_pk, vk, eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "zero_forgery");

    let zero_proof = HolographicProof {
        layers: proof
            .layers
            .iter()
            .map(|l| LayerHolographicOpening {
                layer_index: l.layer_index,
                mul: None,
                add: None,
                uni: None,
                cst: None,
            })
            .collect(),
    };

    let mut verifier_t = Sha2T::new();
    let result =
        verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &zero_proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject proof with all openings stripped; got {result:?}"
    );
}

#[test]
fn adversarial_duplicate_layer_proof() {
    let (_pk, vk, eval_points, proof) = honest_prove_verify(build_two_layer_circuit(), "dup_layer");

    let mut dup_proof = proof.clone();
    dup_proof.layers[1] = dup_proof.layers[0].clone();
    dup_proof.layers[1].layer_index = 1;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &dup_proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject proof with duplicated layer-0 opening at layer-1; got {result:?}"
    );
}

#[test]
fn adversarial_tampered_opening_evalclaim_all_layers() {
    let (_pk, vk, eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "tamper_all");

    for target_layer in 0..proof.layers.len() {
        for which in &["mul", "add"] {
            let mut tampered = proof.clone();
            let layer = &mut tampered.layers[target_layer];
            let opening = match *which {
                "mul" => &mut layer.mul,
                "add" => &mut layer.add,
                _ => unreachable!(),
            };
            if let Some(ref mut o) = opening {
                o.skeleton.evalclaim.claimed_eval += GoldilocksExt4::ONE;
            } else {
                continue;
            }

            let mut verifier_t = Sha2T::new();
            let result =
                verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &tampered, &mut verifier_t);
            assert!(
                result.is_err(),
                "verifier must reject tampered {which} evalclaim at layer {target_layer}; got {result:?}"
            );
        }
    }
}

#[test]
fn adversarial_empty_proof() {
    let (_pk, vk, eval_points, _proof) =
        honest_prove_verify(build_two_layer_circuit(), "empty_proof");

    let empty = HolographicProof { layers: vec![] };
    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &empty, &mut verifier_t);
    assert!(
        matches!(result, Err(VerifyError::LayerCountMismatch { .. })),
        "verifier must reject empty proof; got {result:?}"
    );
}

#[test]
fn adversarial_extra_layer_appended() {
    let (_pk, vk, eval_points, mut proof) =
        honest_prove_verify(build_two_layer_circuit(), "extra_layer");

    proof.layers.push(proof.layers[0].clone());

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject proof with extra layer appended; got {result:?}"
    );
}

#[test]
fn fuzz_serialized_proof_bitflip_all_rejected() {
    use serdes::ExpSerde;

    let (_pk, vk, eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "bitflip_fuzz");

    let mut proof_bytes = Vec::new();
    proof.serialize_into(&mut proof_bytes).unwrap();

    let mut rng = rng_for("bitflip_positions");
    let num_trials = 200;
    let mut rejected = 0;
    let mut accepted_positions = Vec::new();

    for _trial in 0..num_trials {
        use rand::Rng;
        let byte_idx = rng.gen_range(0..proof_bytes.len());
        let bit_idx = rng.gen_range(0..8u8);

        let mut corrupted = proof_bytes.clone();
        corrupted[byte_idx] ^= 1 << bit_idx;

        let deserialized = HolographicProof::<GoldilocksExt4>::deserialize_from(&corrupted[..]);
        match deserialized {
            Ok(bad_proof) => {
                let mut verifier_t = Sha2T::new();
                let result = verify::<C, GoldilocksExt4, Sha2T>(
                    &vk,
                    &eval_points,
                    &bad_proof,
                    &mut verifier_t,
                );
                if result.is_err() {
                    rejected += 1;
                } else {
                    accepted_positions.push((byte_idx, bit_idx));
                }
            }
            Err(_) => {
                rejected += 1;
            }
        }
    }

    let acceptance_rate = accepted_positions.len() as f64 / num_trials as f64;
    assert!(
        acceptance_rate < 0.10,
        "bitflip acceptance rate {:.1}% ({} of {num_trials}) exceeds 10% threshold \
         — positions: {accepted_positions:?}",
        acceptance_rate * 100.0,
        accepted_positions.len()
    );
}

#[test]
fn fuzz_random_proof_bytes_all_rejected() {
    use rand::Rng;
    use serdes::ExpSerde;

    let (_pk, vk, eval_points, _proof) =
        honest_prove_verify(build_two_layer_circuit(), "random_bytes");

    let mut rng = rng_for("random_proof_bytes");

    for trial in 0..50 {
        let len = rng.gen_range(1..4096);
        let random_bytes: Vec<u8> = (0..len).map(|_| rng.gen()).collect();

        let deserialized = HolographicProof::<GoldilocksExt4>::deserialize_from(&random_bytes[..]);
        match deserialized {
            Ok(bad_proof) => {
                let mut verifier_t = Sha2T::new();
                let result = verify::<C, GoldilocksExt4, Sha2T>(
                    &vk,
                    &eval_points,
                    &bad_proof,
                    &mut verifier_t,
                );
                assert!(
                    result.is_err(),
                    "trial {trial}: random bytes deserialized into an accepted proof — soundness breach"
                );
            }
            Err(_) => {}
        }
    }
}

#[test]
fn adversarial_wrong_transcript_state() {
    let (_pk, vk, eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "wrong_transcript");

    let mut poisoned_t = Sha2T::new();
    poisoned_t.append_field_element(&GoldilocksExt4::from(Goldilocks::from(0xdead_beef_u64)));

    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut poisoned_t);
    assert!(
        result.is_err(),
        "verifier must reject proof verified with poisoned transcript; got {result:?}"
    );
}

#[test]
fn adversarial_claim_mismatch_without_proof_tamper() {
    let (_pk, vk, mut eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "claim_mismatch");

    eval_points[0].mul_claim += GoldilocksExt4::ONE;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        matches!(result, Err(VerifyError::ClaimMismatch { layer: 0, which: "mul" })),
        "verifier must detect claim mismatch when eval_point claim differs from proof; got {result:?}"
    );
}

#[test]
fn adversarial_all_claims_zeroed_post_prove() {
    let (_pk, vk, mut eval_points, proof) =
        honest_prove_verify(build_two_layer_circuit(), "all_zero_claims");

    for ep in &mut eval_points {
        ep.mul_claim = GoldilocksExt4::ZERO;
        ep.add_claim = GoldilocksExt4::ZERO;
        ep.uni_claim = GoldilocksExt4::ZERO;
        ep.cst_claim = GoldilocksExt4::ZERO;
    }

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject when all claims are zeroed post-prove; got {result:?}"
    );
}

#[test]
fn adversarial_proof_from_different_eval_points() {
    let mut rng_a = rng_for("diff_eval_a");
    let mut rng_b = rng_for("diff_eval_b");
    let circuit = build_two_layer_circuit();
    let (pk, vk) = setup::<C>(circuit).unwrap();

    let eval_points_a = random_eval_points(&mut rng_a, &pk);
    let eval_points_b = random_eval_points(&mut rng_b, &pk);

    let mut prover_t = Sha2T::new();
    let proof_a = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points_a, &mut prover_t).unwrap();

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points_b, &proof_a, &mut verifier_t);
    assert!(
        result.is_err(),
        "verifier must reject proof verified at different eval points than it was proved at; got {result:?}"
    );
}

#[test]
fn adversarial_layer_index_out_of_range() {
    let (_pk, vk, eval_points, mut proof) =
        honest_prove_verify(build_two_layer_circuit(), "idx_out_range");

    proof.layers[0].layer_index = 999;

    let mut verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(&vk, &eval_points, &proof, &mut verifier_t);
    assert!(
        matches!(result, Err(VerifyError::LayerIndexMismatch { .. })),
        "verifier must reject out-of-range layer index; got {result:?}"
    );
}
