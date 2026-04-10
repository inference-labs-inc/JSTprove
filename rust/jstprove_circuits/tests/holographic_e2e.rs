//! Phase 4 end-to-end test: full holographic-GKR pipeline against
//! a hand-built tiny circuit, exercised through the
//! `gkr::holographic::{setup, prove, verify}` cryptographic core
//! and the `jstprove_io::bundle::{write_bundle_with_vk,
//! read_vk_only}` distribution layer.
//!
//! This test does NOT route through the existing
//! `compile_goldilocks_whir_pq` path because the expander_compiler
//! emits gate kinds (Random coefficients, `const_` gates) that the
//! Phase 2a wiring extractor currently rejects. A synthetic
//! hand-built `expander_circuit::Circuit<C::FieldConfig>` with only
//! `mul` and `add` gates lets the keystone integration test land
//! today; the broader compiler-generated coverage will follow when
//! the wiring extractor grows handlers for const_ / uni / Random
//! coefs.

use arith::Field;
use circuit::{Circuit, CircuitLayer, CoefType, GateAdd, GateMul, StructureInfo};
use gkr::holographic::{HolographicVerifyingKey, LayerEvalPoint, prove, setup, verify};
use gkr_engine::{GoldilocksExt4x1Config, Transcript};
use goldilocks::{Goldilocks, GoldilocksExt4};
use jstprove_io::bundle::{
    BundleBlobs, bundle_has_vk, read_bundle, read_vk_only, write_bundle_with_vk,
};
use serdes::ExpSerde;
use tempfile::TempDir;
use transcript::BytesHashTranscript;
type C = GoldilocksExt4x1Config;
type Sha2T = BytesHashTranscript<gkr_hashers::SHA256hasher>;

#[derive(serde::Serialize, serde::Deserialize)]
struct StubMeta {
    name: String,
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

fn build_tiny_two_layer_circuit() -> Circuit<C> {
    // Layer 0:
    //   out[0] = 5  · in[1] · in[2]
    //   out[1] = 7  · in[2] · in[3]
    //   out[2] = 13 · in[0]            (add)
    //   out[3] = 17 · in[1]            (add)
    let mut layer0 = make_layer(2, 2);
    layer0.mul.push(mul_gate(0, 1, 2, 5));
    layer0.mul.push(mul_gate(1, 2, 3, 7));
    layer0.add.push(add_gate(2, 0, 13));
    layer0.add.push(add_gate(3, 1, 17));

    // Layer 1:
    //   out[3] = 19 · prev[0] · prev[1]
    //   out[1] = 23 · prev[2]           (add)
    //   out[2] = 29 · prev[3]           (add)
    let mut layer1 = make_layer(2, 2);
    layer1.mul.push(mul_gate(3, 0, 1, 19));
    layer1.add.push(add_gate(1, 2, 23));
    layer1.add.push(add_gate(2, 3, 29));

    Circuit {
        layers: vec![layer0, layer1],
        public_input: Vec::new(),
        expected_num_output_zeros: 0,
        rnd_coefs_identified: false,
        rnd_coefs: Vec::new(),
    }
}

fn random_eval_points(
    pk: &gkr::holographic::HolographicProvingKey<C>,
    seed: u64,
) -> Vec<LayerEvalPoint<GoldilocksExt4>> {
    use ark_std::test_rng;
    use rand::{RngCore, SeedableRng};
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let _ = test_rng();
    let mut next = move || -> GoldilocksExt4 {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        GoldilocksExt4::from_uniform_bytes(&bytes)
    };
    pk.layers
        .iter()
        .map(|layer| {
            let n_z = layer.n_z;
            let n_x = layer.n_x;
            let mul_z: Vec<GoldilocksExt4> = (0..n_z).map(|_| next()).collect();
            let mul_x: Vec<GoldilocksExt4> = (0..n_x).map(|_| next()).collect();
            let mul_y: Vec<GoldilocksExt4> = (0..n_x).map(|_| next()).collect();
            let add_z: Vec<GoldilocksExt4> = (0..n_z).map(|_| next()).collect();
            let add_x: Vec<GoldilocksExt4> = (0..n_x).map(|_| next()).collect();
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
            let uni_z: Vec<GoldilocksExt4> = (0..n_z).map(|_| next()).collect();
            let uni_x: Vec<GoldilocksExt4> = (0..n_x).map(|_| next()).collect();
            let uni_claim = layer
                .uni
                .as_ref()
                .map(|w| w.poly.evaluate::<GoldilocksExt4>(&uni_z, &uni_x, &[]))
                .unwrap_or(GoldilocksExt4::ZERO);
            let cst_z: Vec<GoldilocksExt4> = (0..n_z).map(|_| next()).collect();
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

#[test]
fn end_to_end_holographic_pipeline_with_bundle_distribution() {
    // ----- Phase 2b: setup ------------------------------------------
    let circuit = build_tiny_two_layer_circuit();
    let (pk, vk) = setup::<C>(circuit).expect("setup must succeed on a tiny mul/add circuit");
    assert_eq!(vk.n_layers, 2);
    assert_eq!(vk.layers.len(), 2);
    assert!(vk.layers[0].mul.is_some());
    assert!(vk.layers[0].add.is_some());
    assert!(vk.layers[1].mul.is_some());
    assert!(vk.layers[1].add.is_some());

    // ----- Phase 3a: serialize VK and ship as a bundle --------------
    let mut vk_bytes = Vec::new();
    vk.serialize_into(&mut vk_bytes).unwrap();
    assert!(
        vk_bytes.len() < 1024,
        "VK is supposed to be lightweight; got {} bytes",
        vk_bytes.len()
    );

    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("model_bundle");
    let circuit_blob = vec![0xCCu8; 16]; // stub circuit blob
    let ws_blob = vec![0xDDu8; 8]; // stub witness solver blob
    write_bundle_with_vk::<StubMeta>(
        &dir,
        &circuit_blob,
        &ws_blob,
        Some(&vk_bytes),
        Some(StubMeta {
            name: "tiny".into(),
        }),
        None,
        true,
    )
    .unwrap();
    assert!(bundle_has_vk(&dir));

    // Validator path: read just the VK without touching the circuit
    let vk_bytes_validator = read_vk_only(&dir).expect("read_vk_only must succeed");
    assert_eq!(vk_bytes_validator, vk_bytes);

    // Round-trip the VK back through the holographic deserializer
    let vk_decoded = HolographicVerifyingKey::deserialize_from(&vk_bytes_validator[..]).unwrap();
    assert_eq!(vk_decoded.n_layers, vk.n_layers);
    assert_eq!(vk_decoded.version, vk.version);

    // Sanity: full read_bundle still returns the circuit + ws + vk
    let blobs: BundleBlobs<StubMeta> = read_bundle(&dir).unwrap();
    assert_eq!(blobs.circuit, circuit_blob);
    assert_eq!(blobs.witness_solver, ws_blob);
    assert_eq!(blobs.vk, Some(vk_bytes.clone()));

    // ----- Phase 2c: prove ------------------------------------------
    let eval_points = random_eval_points(&pk, 0xc0_ffee_d00d_dead);
    let mut prover_t = Sha2T::new();
    let proof = prove::<C, GoldilocksExt4, Sha2T>(&pk, &eval_points, &mut prover_t)
        .expect("prove must succeed");
    assert_eq!(proof.layers.len(), 2);

    // ----- Phase 2d: verify against the bundle-distributed VK ------
    let mut verifier_t = Sha2T::new();
    verify::<C, GoldilocksExt4, Sha2T>(&vk_decoded, &eval_points, &proof, &mut verifier_t)
        .expect("verifier must accept honest proof against the bundle-distributed VK");

    // ----- Soundness anchor: tampered proof must be rejected --------
    let mut tampered_proof = proof.clone();
    if let Some(mul) = tampered_proof.layers[0].mul.as_mut() {
        let original = mul.skeleton.evalclaim.claimed_eval;
        mul.skeleton.evalclaim.claimed_eval = original + GoldilocksExt4::ONE;
    }
    let mut tamper_verifier_t = Sha2T::new();
    let result = verify::<C, GoldilocksExt4, Sha2T>(
        &vk_decoded,
        &eval_points,
        &tampered_proof,
        &mut tamper_verifier_t,
    );
    assert!(
        result.is_err(),
        "verifier must reject a tampered proof; got {result:?}"
    );
}
