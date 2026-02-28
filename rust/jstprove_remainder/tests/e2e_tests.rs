#![allow(clippy::pedantic)]
#![allow(
    clippy::identity_op,
    clippy::manual_memcpy,
    clippy::needless_range_loop,
    clippy::useless_vec
)]

use frontend::abstract_expr::AbstractExpression;
use frontend::layouter::builder::{CircuitBuilder, LayerVisibility};
use jstprove_remainder::runner::witness::compute_multiplicities;
use jstprove_remainder::util::i64_to_fr;
use remainder::mle::evals::MultilinearExtension;
use remainder::prover::helpers::{
    prove_circuit_with_runtime_optimized_config, test_circuit_with_runtime_optimized_config,
    verify_circuit_with_proof_config,
};
use shared_types::transcript::poseidon_sponge::PoseidonSponge;
use shared_types::Fr;

fn matmul_native(a: &[i64], a_rows: usize, a_cols: usize, b: &[i64], b_cols: usize) -> Vec<i64> {
    let mut result = vec![0i64; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
        }
    }
    result
}

#[test]
fn test_matmul_prove_verify() {
    let a_rows = 4;
    let a_cols = 2;
    let b_cols = 2;

    let a_rows_vars = 2;
    let a_cols_vars = 1;
    let b_cols_vars = 1;

    let a_data: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let b_data: Vec<i64> = vec![10, 20, 30, 40];
    let c_data = matmul_native(&a_data, a_rows, a_cols, &b_data, b_cols);

    let a_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(a_data.iter().map(|&v| i64_to_fr(v)).collect());
    let b_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(b_data.iter().map(|&v| i64_to_fr(v)).collect());
    let c_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(c_data.iter().map(|&v| i64_to_fr(v)).collect());

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let a_node = builder.add_input_shred("A", a_rows_vars + a_cols_vars, &public);
    let b_node = builder.add_input_shred("B", a_cols_vars + b_cols_vars, &public);
    let expected_c = builder.add_input_shred("Expected C", a_rows_vars + b_cols_vars, &public);

    let computed_c = builder.add_matmult_node(
        &a_node,
        (a_rows_vars, a_cols_vars),
        &b_node,
        (a_cols_vars, b_cols_vars),
    );

    let diff = builder.add_sector(computed_c - expected_c);
    builder.set_output(&diff);

    let mut circuit = builder.build_without_layer_combination().unwrap();

    circuit.set_input("A", a_mle);
    circuit.set_input("B", b_mle);
    circuit.set_input("Expected C", c_mle);

    let provable = circuit.gen_provable_circuit().unwrap();
    test_circuit_with_runtime_optimized_config(&provable);
}

#[test]
fn test_gemm_with_bias_and_rescale() {
    let alpha: i64 = 1 << 18;

    let m = 4usize;
    let k = 2usize;
    let n = 2usize;

    let m_vars = 2usize;
    let k_vars = 1usize;
    let n_vars = 1usize;

    let input: Vec<i64> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let weights: Vec<i64> = vec![1 * alpha, 2 * alpha, 3 * alpha, 4 * alpha];
    let bias: Vec<i64> = vec![0, 0];

    let raw_result = matmul_native(&input, m, k, &weights, n);
    let mut with_bias = raw_result.clone();
    for i in 0..m {
        for j in 0..n {
            with_bias[i * n + j] += bias[j];
        }
    }

    let rescaled: Vec<i64> = with_bias.iter().map(|&v| v / alpha).collect();
    let remainders: Vec<i64> = with_bias.iter().map(|&v| v % alpha).collect();

    assert!(
        remainders.iter().all(|&r| r == 0),
        "sanity: all remainders should be 0 for this test case"
    );

    let input_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(input.iter().map(|&v| i64_to_fr(v)).collect());
    let weights_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(weights.iter().map(|&v| i64_to_fr(v)).collect());
    let expected_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(rescaled.iter().map(|&v| i64_to_fr(v)).collect());
    let quotient_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(rescaled.iter().map(|&v| i64_to_fr(v)).collect());
    let remainder_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(remainders.iter().map(|&v| i64_to_fr(v)).collect());

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let input_node = builder.add_input_shred("Input", m_vars + k_vars, &public);
    let weight_node = builder.add_input_shred("Weights", k_vars + n_vars, &public);
    let expected_node = builder.add_input_shred("Expected", m_vars + n_vars, &public);

    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);
    let quotient_node = builder.add_input_shred("Quotient", m_vars + n_vars, &committed);
    let remainder_node = builder.add_input_shred("Remainder", m_vars + n_vars, &committed);

    let matmul_result = builder.add_matmult_node(
        &input_node,
        (m_vars, k_vars),
        &weight_node,
        (k_vars, n_vars),
    );

    let alpha_fr = i64_to_fr(alpha);
    let rescale_check = builder.add_sector(
        matmul_result.expr()
            - AbstractExpression::scaled(quotient_node.expr(), alpha_fr)
            - remainder_node.expr(),
    );
    builder.set_output(&rescale_check);

    let output_check = builder.add_sector(quotient_node.expr() - expected_node.expr());
    builder.set_output(&output_check);

    let mut circuit = builder.build_without_layer_combination().unwrap();

    circuit.set_input("Input", input_mle);
    circuit.set_input("Weights", weights_mle);
    circuit.set_input("Expected", expected_mle);
    circuit.set_input("Quotient", quotient_mle);
    circuit.set_input("Remainder", remainder_mle);

    let provable = circuit.gen_provable_circuit().unwrap();
    test_circuit_with_runtime_optimized_config(&provable);
}

#[test]
fn test_elementwise_add_prove_verify() {
    let data_a: Vec<i64> = vec![10, 20, 30, 40];
    let data_b: Vec<i64> = vec![1, 2, 3, 4];
    let expected: Vec<i64> = vec![11, 22, 33, 44];

    let a_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(data_a.iter().map(|&v| i64_to_fr(v)).collect());
    let b_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(data_b.iter().map(|&v| i64_to_fr(v)).collect());
    let expected_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(expected.iter().map(|&v| i64_to_fr(v)).collect());

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let a_node = builder.add_input_shred("A", 2, &public);
    let b_node = builder.add_input_shred("B", 2, &public);
    let expected_node = builder.add_input_shred("Expected", 2, &public);

    let sum = builder.add_sector(a_node.expr() + b_node.expr());
    let diff = builder.add_sector(sum - expected_node);
    builder.set_output(&diff);

    let mut circuit = builder.build_without_layer_combination().unwrap();

    circuit.set_input("A", a_mle);
    circuit.set_input("B", b_mle);
    circuit.set_input("Expected", expected_mle);

    let provable = circuit.gen_provable_circuit().unwrap();
    test_circuit_with_runtime_optimized_config(&provable);
}

#[test]
fn test_cast_passthrough_prove_verify() {
    // Cast is a no-op in the ZK circuit: the quantiser has already resolved
    // types, so field elements pass through unchanged.  Prove that the
    // identity constraint  output - input = 0  is satisfied under the GKR
    // prover, using a mix of positive and negative quantised values.
    let data: Vec<i64> = vec![1, -2, 3, -4, 5, -6, 7, -8];
    let num_vars = 3; // 2^3 = 8 elements

    let input_mle = MultilinearExtension::new(data.iter().map(|&v| i64_to_fr(v)).collect());
    let expected_mle = MultilinearExtension::new(data.iter().map(|&v| i64_to_fr(v)).collect());

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let input_node = builder.add_input_shred("Input", num_vars, &public);
    let expected_node = builder.add_input_shred("Expected", num_vars, &public);

    // Cast is identity: output == input, so this sector must be zero.
    let diff = builder.add_sector(input_node.expr() - expected_node.expr());
    builder.set_output(&diff);

    let mut circuit = builder.build_with_layer_combination().unwrap();
    circuit.set_input("Input", input_mle);
    circuit.set_input("Expected", expected_mle);

    let provable = circuit.gen_provable_circuit().unwrap();
    test_circuit_with_runtime_optimized_config(&provable);
}

#[test]
fn test_logup_range_check() {
    let table_num_vars: usize = 4;
    let witness_num_vars: usize = 3;
    let table_size: u64 = 1 << table_num_vars;

    let table_mle = MultilinearExtension::new((0..table_size).map(Fr::from).collect());

    let witness_values: Vec<u64> = vec![0, 3, 7, 15, 3, 7, 0, 1];
    let mut multiplicities: Vec<u32> = vec![0; table_size as usize];
    for &v in &witness_values {
        multiplicities[v as usize] += 1;
    }
    let witness_mle: MultilinearExtension<Fr> = witness_values.into();
    let multiplicities_mle: MultilinearExtension<Fr> = multiplicities.into();

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let table = builder.add_input_shred("Table", table_num_vars, &public);

    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);
    let witness = builder.add_input_shred("Witness", witness_num_vars, &committed);
    let mults = builder.add_input_shred("Multiplicities", table_num_vars, &committed);

    let fs_node = builder.add_fiat_shamir_challenge_node(1);
    let lookup_table = builder.add_lookup_table(&table, &fs_node);
    let _constraint = builder.add_lookup_constraint(&lookup_table, &witness, &mults);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("Table", table_mle.clone());
    prover_circuit.set_input("Witness", witness_mle);
    prover_circuit.set_input("Multiplicities", multiplicities_mle);

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("Table", table_mle);
    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_matmul_full_prove_verify_roundtrip() {
    let a_data: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let b_data: Vec<i64> = vec![10, 20, 30, 40];
    let c_data = matmul_native(&a_data, 4, 2, &b_data, 2);

    let a_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(a_data.iter().map(|&v| i64_to_fr(v)).collect());
    let b_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(b_data.iter().map(|&v| i64_to_fr(v)).collect());
    let c_mle: MultilinearExtension<Fr> =
        MultilinearExtension::new(c_data.iter().map(|&v| i64_to_fr(v)).collect());

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let a_node = builder.add_input_shred("A", 3, &public);
    let b_node = builder.add_input_shred("B", 2, &public);
    let expected_c = builder.add_input_shred("Expected C", 3, &public);

    let computed_c = builder.add_matmult_node(&a_node, (2, 1), &b_node, (1, 1));
    let diff = builder.add_sector(computed_c - expected_c);
    builder.set_output(&diff);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("A", a_mle.clone());
    prover_circuit.set_input("B", b_mle.clone());
    prover_circuit.set_input("Expected C", c_mle.clone());

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("A", a_mle);
    verifier_circuit.set_input("B", b_mle);
    verifier_circuit.set_input("Expected C", c_mle);
    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

fn to_fr_vec(data: &[i64]) -> Vec<Fr> {
    data.iter().map(|&v| i64_to_fr(v)).collect()
}

fn rescale_array(values: &[i64], alpha: i64, offset: i64) -> (Vec<i64>, Vec<i64>) {
    let mut quotients = Vec::with_capacity(values.len());
    let mut remainders = Vec::with_capacity(values.len());
    for &v in values {
        let shifted = alpha * offset + v;
        let q_shifted = shifted / alpha;
        let r = shifted - alpha * q_shifted;
        quotients.push(q_shifted - offset);
        remainders.push(r);
    }
    (quotients, remainders)
}

#[test]
fn test_multi_layer_gemm_relu_prove_verify() {
    let alpha: i64 = 1 << 18;
    let offset: i64 = 1 << 30;

    let input: Vec<i64> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let fc1_weight: Vec<i64> = vec![alpha, 0, 0, alpha, alpha, 0, 0, alpha];
    let fc1_bias: Vec<i64> = vec![0; 8];

    let fc1_mm = matmul_native(&input, 4, 4, &fc1_weight, 2);
    let fc1_after_bias: Vec<i64> = fc1_mm
        .iter()
        .zip(fc1_bias.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (fc1_q, fc1_r) = rescale_array(&fc1_after_bias, alpha, offset);

    let relu1_out: Vec<i64> = fc1_q.iter().map(|&x| x.max(0)).collect();
    let relu1_di: Vec<i64> = relu1_out
        .iter()
        .zip(fc1_q.iter())
        .map(|(o, x)| o - x)
        .collect();
    let relu1_dz: Vec<i64> = relu1_out.clone();

    let fc2_weight: Vec<i64> = vec![alpha, 2 * alpha, 3 * alpha, alpha];
    let fc2_bias: Vec<i64> = vec![0; 8];

    let fc2_mm = matmul_native(&relu1_out, 4, 2, &fc2_weight, 2);
    let fc2_after_bias: Vec<i64> = fc2_mm
        .iter()
        .zip(fc2_bias.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (fc2_q, fc2_r) = rescale_array(&fc2_after_bias, alpha, offset);

    let expected = fc2_q.clone();

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", 4, &public);
    let fc1_w_node = builder.add_input_shred("fc1_weight", 3, &public);
    let fc1_b_node = builder.add_input_shred("fc1_bias", 3, &public);
    let fc2_w_node = builder.add_input_shred("fc2_weight", 2, &public);
    let fc2_b_node = builder.add_input_shred("fc2_bias", 3, &public);
    let zero_node = builder.add_input_shred("zero", 3, &public);
    let expected_node = builder.add_input_shred("expected", 3, &public);

    let fc1_q_node = builder.add_input_shred("fc1_q", 3, &committed);
    let fc1_r_node = builder.add_input_shred("fc1_r", 3, &committed);
    let relu1_max_node = builder.add_input_shred("relu1_max", 3, &committed);
    let relu1_di_node = builder.add_input_shred("relu1_di", 3, &committed);
    let relu1_dz_node = builder.add_input_shred("relu1_dz", 3, &committed);
    let fc2_q_node = builder.add_input_shred("fc2_q", 3, &committed);
    let fc2_r_node = builder.add_input_shred("fc2_r", 3, &committed);

    let alpha_fr = i64_to_fr(alpha);

    let fc1_mm_node = builder.add_matmult_node(&input_node, (2, 2), &fc1_w_node, (2, 1));
    let fc1_check = builder.add_sector(
        fc1_mm_node.expr() + fc1_b_node.expr()
            - AbstractExpression::scaled(fc1_q_node.expr(), alpha_fr)
            - fc1_r_node.expr(),
    );
    builder.set_output(&fc1_check);

    let r1_c1 =
        builder.add_sector(relu1_max_node.expr() - fc1_q_node.expr() - relu1_di_node.expr());
    builder.set_output(&r1_c1);
    let r1_c2 = builder.add_sector(relu1_max_node.expr() - zero_node.expr() - relu1_dz_node.expr());
    builder.set_output(&r1_c2);
    let r1_prod = builder.add_sector(AbstractExpression::products(vec![
        relu1_di_node.id(),
        relu1_dz_node.id(),
    ]));
    builder.set_output(&r1_prod);

    let fc2_mm_node = builder.add_matmult_node(&relu1_max_node, (2, 1), &fc2_w_node, (1, 1));
    let fc2_check = builder.add_sector(
        fc2_mm_node.expr() + fc2_b_node.expr()
            - AbstractExpression::scaled(fc2_q_node.expr(), alpha_fr)
            - fc2_r_node.expr(),
    );
    builder.set_output(&fc2_check);

    let out_check = builder.add_sector(fc2_q_node.expr() - expected_node.expr());
    builder.set_output(&out_check);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    prover_circuit.set_input(
        "fc1_weight",
        MultilinearExtension::new(to_fr_vec(&fc1_weight)),
    );
    prover_circuit.set_input("fc1_bias", MultilinearExtension::new(to_fr_vec(&fc1_bias)));
    prover_circuit.set_input(
        "fc2_weight",
        MultilinearExtension::new(to_fr_vec(&fc2_weight)),
    );
    prover_circuit.set_input("fc2_bias", MultilinearExtension::new(to_fr_vec(&fc2_bias)));
    prover_circuit.set_input("zero", MultilinearExtension::new(vec![Fr::from(0u64); 8]));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&expected)));
    prover_circuit.set_input("fc1_q", MultilinearExtension::new(to_fr_vec(&fc1_q)));
    prover_circuit.set_input("fc1_r", MultilinearExtension::new(to_fr_vec(&fc1_r)));
    prover_circuit.set_input(
        "relu1_max",
        MultilinearExtension::new(to_fr_vec(&relu1_out)),
    );
    prover_circuit.set_input("relu1_di", MultilinearExtension::new(to_fr_vec(&relu1_di)));
    prover_circuit.set_input("relu1_dz", MultilinearExtension::new(to_fr_vec(&relu1_dz)));
    prover_circuit.set_input("fc2_q", MultilinearExtension::new(to_fr_vec(&fc2_q)));
    prover_circuit.set_input("fc2_r", MultilinearExtension::new(to_fr_vec(&fc2_r)));

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    verifier_circuit.set_input(
        "fc1_weight",
        MultilinearExtension::new(to_fr_vec(&fc1_weight)),
    );
    verifier_circuit.set_input("fc1_bias", MultilinearExtension::new(to_fr_vec(&fc1_bias)));
    verifier_circuit.set_input(
        "fc2_weight",
        MultilinearExtension::new(to_fr_vec(&fc2_weight)),
    );
    verifier_circuit.set_input("fc2_bias", MultilinearExtension::new(to_fr_vec(&fc2_bias)));
    verifier_circuit.set_input("zero", MultilinearExtension::new(vec![Fr::from(0u64); 8]));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&expected)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

fn transpose_matrix(data: &[i64], rows: usize, cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

fn pad_to_size(data: &[i64], target: usize) -> Vec<i64> {
    let mut padded = data.to_vec();
    padded.resize(target, 0);
    padded
}

fn broadcast_bias(bias: &[i64], bias_len: usize, total_padded: usize) -> Vec<i64> {
    let mut out = vec![0i64; total_padded];
    for j in 0..bias_len {
        out[j] = bias[j];
    }
    out
}

fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}
fn log2(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

fn pad_matrix(
    data: &[i64],
    orig_rows: usize,
    orig_cols: usize,
    pad_rows: usize,
    pad_cols: usize,
) -> Vec<i64> {
    let mut out = vec![0i64; pad_rows * pad_cols];
    for r in 0..orig_rows {
        for c in 0..orig_cols {
            out[r * pad_cols + c] = data[r * orig_cols + c];
        }
    }
    out
}

fn padded_matmul(a: &[i64], a_rows: usize, a_cols: usize, b: &[i64], b_cols: usize) -> Vec<i64> {
    let mut out = vec![0i64; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0i64;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            out[i * b_cols + j] = sum;
        }
    }
    out
}

#[test]
fn test_lenet_fc_prove_verify() {
    use jstprove_remainder::onnx::parser;
    use std::path::Path;

    let alpha: i64 = 1 << 18;
    let offset: i64 = 1 << 30;

    let parsed = parser::parse_onnx(Path::new("models/lenet.onnx")).unwrap();

    let fc1_w_raw: Vec<i64> = parsed.initializers["fc1.weight"]
        .float_data
        .iter()
        .map(|&f| (f * alpha as f64).round() as i64)
        .collect();
    let fc1_b_raw: Vec<i64> = parsed.initializers["fc1.bias"]
        .float_data
        .iter()
        .map(|&f| (f * (alpha as f64).powi(2)).round() as i64)
        .collect();
    let fc2_w_raw: Vec<i64> = parsed.initializers["fc2.weight"]
        .float_data
        .iter()
        .map(|&f| (f * alpha as f64).round() as i64)
        .collect();
    let fc2_b_raw: Vec<i64> = parsed.initializers["fc2.bias"]
        .float_data
        .iter()
        .map(|&f| (f * (alpha as f64).powi(2)).round() as i64)
        .collect();
    let fc3_w_raw: Vec<i64> = parsed.initializers["fc3.weight"]
        .float_data
        .iter()
        .map(|&f| (f * alpha as f64).round() as i64)
        .collect();
    let fc3_b_raw: Vec<i64> = parsed.initializers["fc3.bias"]
        .float_data
        .iter()
        .map(|&f| (f * (alpha as f64).powi(2)).round() as i64)
        .collect();

    let fc1_w_t = transpose_matrix(&fc1_w_raw, 120, 400);
    let fc2_w_t = transpose_matrix(&fc2_w_raw, 84, 120);
    let fc3_w_t = transpose_matrix(&fc3_w_raw, 10, 84);

    let k1 = next_pow2(400);
    let n1 = next_pow2(120);
    let k2 = next_pow2(120);
    let n2 = next_pow2(84);
    let k3 = next_pow2(84);
    let n3 = next_pow2(10);

    let k1v = log2(k1);
    let n1v = log2(n1);
    let k2v = log2(k2);
    let n2v = log2(n2);
    let k3v = log2(k3);
    let n3v = log2(n3);

    let input_raw: Vec<i64> = (0..400).map(|i| ((i as i64 * 7 + 3) % 201) - 100).collect();
    let input_padded = pad_to_size(&input_raw, k1);

    let fc1_w_padded = pad_matrix(&fc1_w_t, 400, 120, k1, n1);
    let fc1_b_padded = broadcast_bias(&fc1_b_raw, 120, n1);

    let fc1_mm = padded_matmul(&input_padded, 1, k1, &fc1_w_padded, n1);
    let fc1_ab: Vec<i64> = fc1_mm
        .iter()
        .zip(fc1_b_padded.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (fc1_q, fc1_r) = rescale_array(&fc1_ab, alpha, offset);

    let relu1_out: Vec<i64> = fc1_q.iter().map(|&x| x.max(0)).collect();
    let relu1_di: Vec<i64> = relu1_out
        .iter()
        .zip(fc1_q.iter())
        .map(|(o, x)| o - x)
        .collect();
    let relu1_dz: Vec<i64> = relu1_out.clone();

    let fc2_w_padded = pad_matrix(&fc2_w_t, 120, 84, k2, n2);
    let fc2_b_padded = broadcast_bias(&fc2_b_raw, 84, n2);

    let fc2_mm = padded_matmul(&relu1_out, 1, k2, &fc2_w_padded, n2);
    let fc2_ab: Vec<i64> = fc2_mm
        .iter()
        .zip(fc2_b_padded.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (fc2_q, fc2_r) = rescale_array(&fc2_ab, alpha, offset);

    let relu2_out: Vec<i64> = fc2_q.iter().map(|&x| x.max(0)).collect();
    let relu2_di: Vec<i64> = relu2_out
        .iter()
        .zip(fc2_q.iter())
        .map(|(o, x)| o - x)
        .collect();
    let relu2_dz: Vec<i64> = relu2_out.clone();

    let fc3_w_padded = pad_matrix(&fc3_w_t, 84, 10, k3, n3);
    let fc3_b_padded = broadcast_bias(&fc3_b_raw, 10, n3);

    let fc3_mm = padded_matmul(&relu2_out, 1, k3, &fc3_w_padded, n3);
    let fc3_ab: Vec<i64> = fc3_mm
        .iter()
        .zip(fc3_b_padded.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (fc3_q, fc3_r) = rescale_array(&fc3_ab, alpha, offset);

    let expected = fc3_q.clone();

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", k1v, &public);
    let fc1_w_node = builder.add_input_shred("fc1_w", k1v + n1v, &public);
    let fc1_b_node = builder.add_input_shred("fc1_b", n1v, &public);
    let fc2_w_node = builder.add_input_shred("fc2_w", k2v + n2v, &public);
    let fc2_b_node = builder.add_input_shred("fc2_b", n2v, &public);
    let fc3_w_node = builder.add_input_shred("fc3_w", k3v + n3v, &public);
    let fc3_b_node = builder.add_input_shred("fc3_b", n3v, &public);
    let zero7_node = builder.add_input_shred("zero7", n1v, &public);
    let expected_node = builder.add_input_shred("expected", n3v, &public);

    let fc1_q_node = builder.add_input_shred("fc1_q", n1v, &committed);
    let fc1_r_node = builder.add_input_shred("fc1_r", n1v, &committed);
    let r1_max_node = builder.add_input_shred("r1_max", n1v, &committed);
    let r1_di_node = builder.add_input_shred("r1_di", n1v, &committed);
    let r1_dz_node = builder.add_input_shred("r1_dz", n1v, &committed);
    let fc2_q_node = builder.add_input_shred("fc2_q", n2v, &committed);
    let fc2_r_node = builder.add_input_shred("fc2_r", n2v, &committed);
    let r2_max_node = builder.add_input_shred("r2_max", n2v, &committed);
    let r2_di_node = builder.add_input_shred("r2_di", n2v, &committed);
    let r2_dz_node = builder.add_input_shred("r2_dz", n2v, &committed);
    let fc3_q_node = builder.add_input_shred("fc3_q", n3v, &committed);
    let fc3_r_node = builder.add_input_shred("fc3_r", n3v, &committed);

    let alpha_fr = i64_to_fr(alpha);

    let fc1_mm_node = builder.add_matmult_node(&input_node, (0, k1v), &fc1_w_node, (k1v, n1v));
    let fc1_chk = builder.add_sector(
        fc1_mm_node.expr() + fc1_b_node.expr()
            - AbstractExpression::scaled(fc1_q_node.expr(), alpha_fr)
            - fc1_r_node.expr(),
    );
    builder.set_output(&fc1_chk);

    let r1_c1 = builder.add_sector(r1_max_node.expr() - fc1_q_node.expr() - r1_di_node.expr());
    builder.set_output(&r1_c1);
    let r1_c2 = builder.add_sector(r1_max_node.expr() - zero7_node.expr() - r1_dz_node.expr());
    builder.set_output(&r1_c2);
    let r1_prod = builder.add_sector(AbstractExpression::products(vec![
        r1_di_node.id(),
        r1_dz_node.id(),
    ]));
    builder.set_output(&r1_prod);

    let fc2_mm_node = builder.add_matmult_node(&r1_max_node, (0, k2v), &fc2_w_node, (k2v, n2v));
    let fc2_chk = builder.add_sector(
        fc2_mm_node.expr() + fc2_b_node.expr()
            - AbstractExpression::scaled(fc2_q_node.expr(), alpha_fr)
            - fc2_r_node.expr(),
    );
    builder.set_output(&fc2_chk);

    let r2_c1 = builder.add_sector(r2_max_node.expr() - fc2_q_node.expr() - r2_di_node.expr());
    builder.set_output(&r2_c1);
    let r2_c2 = builder.add_sector(r2_max_node.expr() - zero7_node.expr() - r2_dz_node.expr());
    builder.set_output(&r2_c2);
    let r2_prod = builder.add_sector(AbstractExpression::products(vec![
        r2_di_node.id(),
        r2_dz_node.id(),
    ]));
    builder.set_output(&r2_prod);

    let fc3_mm_node = builder.add_matmult_node(&r2_max_node, (0, k3v), &fc3_w_node, (k3v, n3v));
    let fc3_chk = builder.add_sector(
        fc3_mm_node.expr() + fc3_b_node.expr()
            - AbstractExpression::scaled(fc3_q_node.expr(), alpha_fr)
            - fc3_r_node.expr(),
    );
    builder.set_output(&fc3_chk);

    let out_chk = builder.add_sector(fc3_q_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    macro_rules! set_input {
        ($circuit:expr, $name:expr, $data:expr) => {
            $circuit.set_input($name, MultilinearExtension::new(to_fr_vec(&$data)));
        };
    }
    set_input!(prover_circuit, "input", input_padded);
    set_input!(prover_circuit, "fc1_w", fc1_w_padded);
    set_input!(prover_circuit, "fc1_b", fc1_b_padded);
    set_input!(prover_circuit, "fc2_w", fc2_w_padded);
    set_input!(prover_circuit, "fc2_b", fc2_b_padded);
    set_input!(prover_circuit, "fc3_w", fc3_w_padded);
    set_input!(prover_circuit, "fc3_b", fc3_b_padded);
    prover_circuit.set_input("zero7", MultilinearExtension::new(vec![Fr::from(0u64); n1]));
    set_input!(prover_circuit, "expected", expected);
    set_input!(prover_circuit, "fc1_q", fc1_q);
    set_input!(prover_circuit, "fc1_r", fc1_r);
    set_input!(prover_circuit, "r1_max", relu1_out);
    set_input!(prover_circuit, "r1_di", relu1_di);
    set_input!(prover_circuit, "r1_dz", relu1_dz);
    set_input!(prover_circuit, "fc2_q", fc2_q);
    set_input!(prover_circuit, "fc2_r", fc2_r);
    set_input!(prover_circuit, "r2_max", relu2_out);
    set_input!(prover_circuit, "r2_di", relu2_di);
    set_input!(prover_circuit, "r2_dz", relu2_dz);
    set_input!(prover_circuit, "fc3_q", fc3_q);
    set_input!(prover_circuit, "fc3_r", fc3_r);

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    set_input!(verifier_circuit, "input", input_padded);
    set_input!(verifier_circuit, "fc1_w", fc1_w_padded);
    set_input!(verifier_circuit, "fc1_b", fc1_b_padded);
    set_input!(verifier_circuit, "fc2_w", fc2_w_padded);
    set_input!(verifier_circuit, "fc2_b", fc2_b_padded);
    set_input!(verifier_circuit, "fc3_w", fc3_w_padded);
    set_input!(verifier_circuit, "fc3_b", fc3_b_padded);
    verifier_circuit.set_input("zero7", MultilinearExtension::new(vec![Fr::from(0u64); n1]));
    set_input!(verifier_circuit, "expected", expected);

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_conv2d_prove_verify() {
    let alpha: i64 = 1 << 18;
    let offset: i64 = 1 << 30;

    let in_h = 4usize;
    let in_w = 4usize;
    let c_in = 1usize;
    let c_out = 2usize;
    let kh_size = 2usize;
    let kw_size = 2usize;
    let stride = 1usize;
    let out_h = (in_h - kh_size) / stride + 1;
    let out_w = (in_w - kw_size) / stride + 1;
    let patch_size = c_in * kh_size * kw_size;
    let num_patches = out_h * out_w;

    let input: Vec<i64> = (1..=16).map(|x| x as i64).collect();
    let kernel: Vec<i64> = vec![alpha, 0, 0, alpha, 0, alpha, alpha, 0];
    let bias: Vec<i64> = vec![0, 0];

    let pad_patches = next_pow2(num_patches);
    let pad_psize = next_pow2(patch_size);
    let pad_cout = next_pow2(c_out);

    let mut im2col = vec![0i64; pad_patches * pad_psize];
    let mut wiring: Vec<(u32, u32)> = Vec::new();
    for oh in 0..out_h {
        for ow in 0..out_w {
            let patch = oh * out_w + ow;
            for c in 0..c_in {
                for kr in 0..kh_size {
                    for kc in 0..kw_size {
                        let col = c * kh_size * kw_size + kr * kw_size + kc;
                        let ih = oh * stride + kr;
                        let iw = ow * stride + kc;
                        let src = c * in_h * in_w + ih * in_w + iw;
                        let dest = patch * pad_psize + col;
                        im2col[dest] = input[src];
                        wiring.push((dest as u32, src as u32));
                    }
                }
            }
        }
    }

    let kernel_t = transpose_matrix(&kernel, c_out, patch_size);
    let kernel_padded = pad_matrix(&kernel_t, patch_size, c_out, pad_psize, pad_cout);
    let mm_result = padded_matmul(&im2col, pad_patches, pad_psize, &kernel_padded, pad_cout);

    let bias_bc: Vec<i64> = (0..pad_patches * pad_cout)
        .map(|i| {
            let j = i % pad_cout;
            if j < c_out {
                bias[j]
            } else {
                0
            }
        })
        .collect();
    let mm_with_bias: Vec<i64> = mm_result
        .iter()
        .zip(bias_bc.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (quotients, remainders) = rescale_array(&mm_with_bias, alpha, offset);

    let input_vars = log2(next_pow2(c_in * in_h * in_w));
    let im2col_row_vars = log2(pad_patches);
    let im2col_col_vars = log2(pad_psize);
    let im2col_vars = im2col_row_vars + im2col_col_vars;
    let out_col_vars = log2(pad_cout);
    let weight_vars = im2col_col_vars + out_col_vars;
    let result_vars = im2col_row_vars + out_col_vars;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", input_vars, &public);
    let weight_node = builder.add_input_shred("weight", weight_vars, &public);
    let bias_bc_node = builder.add_input_shred("bias_bc", result_vars, &public);
    let expected_node = builder.add_input_shred("expected", result_vars, &public);
    let q_node = builder.add_input_shred("quotient", result_vars, &committed);
    let r_node = builder.add_input_shred("remainder", result_vars, &committed);

    let im2col_node = builder.add_identity_gate_node(&input_node, wiring, im2col_vars, None);
    let mm_node = builder.add_matmult_node(
        &im2col_node,
        (im2col_row_vars, im2col_col_vars),
        &weight_node,
        (im2col_col_vars, out_col_vars),
    );

    let alpha_fr = i64_to_fr(alpha);
    let rescale_chk = builder.add_sector(
        mm_node.expr() + bias_bc_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_chk);
    let out_chk = builder.add_sector(q_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    let input_padded = pad_to_size(&input, 1 << input_vars);
    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    prover_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    prover_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input("quotient", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input(
        "remainder",
        MultilinearExtension::new(to_fr_vec(&remainders)),
    );

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    verifier_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    verifier_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_maxpool_prove_verify() {
    let input: Vec<i64> = vec![3, 1, 7, 2, 5, 8, 4, 6, 9, 2, 11, 1, 3, 10, 5, 12];

    let pool_h = 2usize;
    let pool_w = 2usize;
    let stride = 2usize;
    let in_h = 4usize;
    let in_w = 4usize;
    let out_h = in_h / stride;
    let out_w = in_w / stride;
    let num_windows = out_h * out_w;
    let window_size = pool_h * pool_w;

    let mut max_values = vec![0i64; num_windows];
    let mut window_elems: Vec<Vec<i64>> = vec![vec![0i64; num_windows]; window_size];
    let mut gate_wiring: Vec<Vec<(u32, u32)>> = vec![Vec::new(); window_size];

    for oh in 0..out_h {
        for ow in 0..out_w {
            let w_idx = oh * out_w + ow;
            let mut elems = Vec::new();
            for ph in 0..pool_h {
                for pw in 0..pool_w {
                    let elem_idx = ph * pool_w + pw;
                    let ih = oh * stride + ph;
                    let iw = ow * stride + pw;
                    let src = ih * in_w + iw;
                    window_elems[elem_idx][w_idx] = input[src];
                    elems.push(input[src]);
                    gate_wiring[elem_idx].push((w_idx as u32, src as u32));
                }
            }
            max_values[w_idx] = *elems.iter().max().unwrap();
        }
    }

    let deltas: Vec<Vec<i64>> = (0..window_size)
        .map(|i| {
            (0..num_windows)
                .map(|w| max_values[w] - window_elems[i][w])
                .collect()
        })
        .collect();

    let input_vars = 4usize;
    let output_vars = 2usize;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", input_vars, &public);
    let expected_node = builder.add_input_shred("expected", output_vars, &public);

    let max_node = builder.add_input_shred("max_hint", output_vars, &committed);
    let delta_nodes: Vec<_> = (0..window_size)
        .map(|i| builder.add_input_shred(&format!("delta_{}", i), output_vars, &committed))
        .collect();

    let gate_nodes: Vec<_> = (0..window_size)
        .map(|i| {
            builder.add_identity_gate_node(&input_node, gate_wiring[i].clone(), output_vars, None)
        })
        .collect();

    for i in 0..window_size {
        let chk =
            builder.add_sector(max_node.expr() - gate_nodes[i].expr() - delta_nodes[i].expr());
        builder.set_output(&chk);
    }

    let prod = builder.add_sector(AbstractExpression::products(
        delta_nodes.iter().map(|d| d.id()).collect(),
    ));
    builder.set_output(&prod);

    let out_chk = builder.add_sector(max_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    prover_circuit.set_input(
        "expected",
        MultilinearExtension::new(to_fr_vec(&max_values)),
    );
    prover_circuit.set_input(
        "max_hint",
        MultilinearExtension::new(to_fr_vec(&max_values)),
    );
    for i in 0..window_size {
        prover_circuit.set_input(
            &format!("delta_{}", i),
            MultilinearExtension::new(to_fr_vec(&deltas[i])),
        );
    }

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    verifier_circuit.set_input(
        "expected",
        MultilinearExtension::new(to_fr_vec(&max_values)),
    );

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_conv_relu_maxpool_pipeline() {
    let alpha: i64 = 1 << 18;
    let offset: i64 = 1 << 30;

    let in_h = 6usize;
    let in_w = 6usize;
    let c_in = 1usize;
    let c_out = 2usize;
    let kh = 3usize;
    let kw = 3usize;
    let conv_stride = 1usize;
    let conv_oh = (in_h - kh) / conv_stride + 1;
    let conv_ow = (in_w - kw) / conv_stride + 1;
    let patch_size = c_in * kh * kw;
    let num_patches = conv_oh * conv_ow;

    let input: Vec<i64> = (1..=36).map(|x| x as i64).collect();
    let kernel: Vec<i64> = vec![
        alpha, 0, 0, 0, 0, 0, 0, 0, alpha, 0, 0, 0, 0, alpha, 0, 0, 0, 0,
    ];
    let bias: Vec<i64> = vec![0, 0];

    let pp = next_pow2(num_patches);
    let pps = next_pow2(patch_size);
    let pc = next_pow2(c_out);

    let mut im2col_data = vec![0i64; pp * pps];
    let mut im2col_wiring: Vec<(u32, u32)> = Vec::new();
    for oh in 0..conv_oh {
        for ow in 0..conv_ow {
            let p = oh * conv_ow + ow;
            for kr in 0..kh {
                for kc in 0..kw {
                    let col = kr * kw + kc;
                    let src = (oh + kr) * in_w + (ow + kc);
                    let dest = p * pps + col;
                    im2col_data[dest] = input[src];
                    im2col_wiring.push((dest as u32, src as u32));
                }
            }
        }
    }

    let kernel_t = transpose_matrix(&kernel, c_out, patch_size);
    let kernel_padded = pad_matrix(&kernel_t, patch_size, c_out, pps, pc);
    let mm = padded_matmul(&im2col_data, pp, pps, &kernel_padded, pc);
    let bias_bc: Vec<i64> = (0..pp * pc)
        .map(|i| {
            let j = i % pc;
            if j < c_out {
                bias[j]
            } else {
                0
            }
        })
        .collect();
    let mm_biased: Vec<i64> = mm.iter().zip(bias_bc.iter()).map(|(m, b)| m + b).collect();
    let (conv_q, conv_r) = rescale_array(&mm_biased, alpha, offset);

    let relu_out: Vec<i64> = conv_q.iter().map(|&x| x.max(0)).collect();
    let relu_di: Vec<i64> = relu_out
        .iter()
        .zip(conv_q.iter())
        .map(|(o, x)| o - x)
        .collect();
    let relu_dz: Vec<i64> = relu_out.clone();

    let pool_h = 2usize;
    let pool_w = 2usize;
    let pool_stride = 2usize;
    let pool_oh = conv_oh / pool_stride;
    let pool_ow = conv_ow / pool_stride;
    let num_pool_out = pool_oh * pool_ow * c_out;
    let pad_pool = next_pow2(num_pool_out);
    let pool_out_vars = log2(pad_pool);
    let window_size = pool_h * pool_w;

    let mut pool_max = vec![0i64; pad_pool];
    let mut pool_window_elems: Vec<Vec<i64>> = vec![vec![0i64; pad_pool]; window_size];
    let mut pool_gate_wiring: Vec<Vec<(u32, u32)>> = vec![Vec::new(); window_size];

    for c in 0..c_out {
        for poh in 0..pool_oh {
            for pow in 0..pool_ow {
                let dest_idx = (poh * pool_ow + pow) * c_out + c;
                let mut elems = Vec::new();
                for ph in 0..pool_h {
                    for pw in 0..pool_w {
                        let elem_pos = ph * pool_w + pw;
                        let soh = poh * pool_stride + ph;
                        let sow = pow * pool_stride + pw;
                        let src_idx = (soh * conv_ow + sow) * pc + c;
                        pool_window_elems[elem_pos][dest_idx] = relu_out[src_idx];
                        elems.push(relu_out[src_idx]);
                        pool_gate_wiring[elem_pos].push((dest_idx as u32, src_idx as u32));
                    }
                }
                pool_max[dest_idx] = *elems.iter().max().unwrap();
            }
        }
    }

    let pool_deltas: Vec<Vec<i64>> = (0..window_size)
        .map(|i| {
            (0..pad_pool)
                .map(|w| pool_max[w] - pool_window_elems[i][w])
                .collect()
        })
        .collect();

    let input_vars = log2(next_pow2(in_h * in_w));
    let im2col_rv = log2(pp);
    let im2col_cv = log2(pps);
    let im2col_vars = im2col_rv + im2col_cv;
    let ocv = log2(pc);
    let wt_vars = im2col_cv + ocv;
    let res_vars = im2col_rv + ocv;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", input_vars, &public);
    let wt_node = builder.add_input_shred("weight", wt_vars, &public);
    let bias_node = builder.add_input_shred("bias_bc", res_vars, &public);
    let zero_node = builder.add_input_shred("zero", res_vars, &public);
    let expected_node = builder.add_input_shred("expected", pool_out_vars, &public);

    let cq_node = builder.add_input_shred("conv_q", res_vars, &committed);
    let cr_node = builder.add_input_shred("conv_r", res_vars, &committed);
    let relu_node = builder.add_input_shred("relu_max", res_vars, &committed);
    let rdi_node = builder.add_input_shred("relu_di", res_vars, &committed);
    let rdz_node = builder.add_input_shred("relu_dz", res_vars, &committed);
    let pmax_node = builder.add_input_shred("pool_max", pool_out_vars, &committed);
    let pd_nodes: Vec<_> = (0..window_size)
        .map(|i| builder.add_input_shred(&format!("pd_{}", i), pool_out_vars, &committed))
        .collect();

    let im2col_node = builder.add_identity_gate_node(&input_node, im2col_wiring, im2col_vars, None);
    let mm_node = builder.add_matmult_node(
        &im2col_node,
        (im2col_rv, im2col_cv),
        &wt_node,
        (im2col_cv, ocv),
    );

    let alpha_fr = i64_to_fr(alpha);
    let rc = builder.add_sector(
        mm_node.expr() + bias_node.expr()
            - AbstractExpression::scaled(cq_node.expr(), alpha_fr)
            - cr_node.expr(),
    );
    builder.set_output(&rc);

    let r1 = builder.add_sector(relu_node.expr() - cq_node.expr() - rdi_node.expr());
    builder.set_output(&r1);
    let r2 = builder.add_sector(relu_node.expr() - zero_node.expr() - rdz_node.expr());
    builder.set_output(&r2);
    let rp = builder.add_sector(AbstractExpression::products(vec![
        rdi_node.id(),
        rdz_node.id(),
    ]));
    builder.set_output(&rp);

    let pool_gates: Vec<_> = (0..window_size)
        .map(|i| {
            builder.add_identity_gate_node(
                &relu_node,
                pool_gate_wiring[i].clone(),
                pool_out_vars,
                None,
            )
        })
        .collect();
    for i in 0..window_size {
        let c = builder.add_sector(pmax_node.expr() - pool_gates[i].expr() - pd_nodes[i].expr());
        builder.set_output(&c);
    }
    let pp_constraint = builder.add_sector(AbstractExpression::products(
        pd_nodes.iter().map(|d| d.id()).collect(),
    ));
    builder.set_output(&pp_constraint);
    let oc = builder.add_sector(pmax_node.expr() - expected_node.expr());
    builder.set_output(&oc);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    let input_padded = pad_to_size(&input, 1 << input_vars);
    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    prover_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    prover_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    prover_circuit.set_input(
        "zero",
        MultilinearExtension::new(vec![Fr::from(0u64); 1 << res_vars]),
    );
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&pool_max)));
    prover_circuit.set_input("conv_q", MultilinearExtension::new(to_fr_vec(&conv_q)));
    prover_circuit.set_input("conv_r", MultilinearExtension::new(to_fr_vec(&conv_r)));
    prover_circuit.set_input("relu_max", MultilinearExtension::new(to_fr_vec(&relu_out)));
    prover_circuit.set_input("relu_di", MultilinearExtension::new(to_fr_vec(&relu_di)));
    prover_circuit.set_input("relu_dz", MultilinearExtension::new(to_fr_vec(&relu_dz)));
    prover_circuit.set_input("pool_max", MultilinearExtension::new(to_fr_vec(&pool_max)));
    for i in 0..window_size {
        prover_circuit.set_input(
            &format!("pd_{}", i),
            MultilinearExtension::new(to_fr_vec(&pool_deltas[i])),
        );
    }

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    verifier_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    verifier_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    verifier_circuit.set_input(
        "zero",
        MultilinearExtension::new(vec![Fr::from(0u64); 1 << res_vars]),
    );
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&pool_max)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_lenet_conv1_prove_verify() {
    use jstprove_remainder::onnx::parser;
    use std::path::Path;

    let alpha: i64 = 1 << 18;
    let offset: i64 = 1 << 30;

    let parsed = parser::parse_onnx(Path::new("models/lenet.onnx")).unwrap();
    let conv1_w_raw: Vec<i64> = parsed.initializers["conv1.weight"]
        .float_data
        .iter()
        .map(|&f| (f * alpha as f64).round() as i64)
        .collect();
    let conv1_b_raw: Vec<i64> = parsed.initializers["conv1.bias"]
        .float_data
        .iter()
        .map(|&f| (f * (alpha as f64).powi(2)).round() as i64)
        .collect();

    let c_in = 3usize;
    let in_h = 8usize;
    let in_w = 8usize;
    let c_out = 6usize;
    let kh_s = 5usize;
    let kw_s = 5usize;
    let conv_oh = in_h - kh_s + 1;
    let conv_ow = in_w - kw_s + 1;
    let patch_size = c_in * kh_s * kw_s;
    let num_patches = conv_oh * conv_ow;

    let input: Vec<i64> = (0..c_in * in_h * in_w)
        .map(|i| ((i as i64 * 13 + 7) % 201) - 100)
        .collect();

    let pp = next_pow2(num_patches);
    let pps = next_pow2(patch_size);
    let pc = next_pow2(c_out);
    let input_size = next_pow2(c_in * in_h * in_w);

    let mut im2col_data = vec![0i64; pp * pps];
    let mut im2col_wiring: Vec<(u32, u32)> = Vec::new();
    for oh in 0..conv_oh {
        for ow in 0..conv_ow {
            let p = oh * conv_ow + ow;
            for c in 0..c_in {
                for kr in 0..kh_s {
                    for kc in 0..kw_s {
                        let col = c * kh_s * kw_s + kr * kw_s + kc;
                        let src = c * in_h * in_w + (oh + kr) * in_w + (ow + kc);
                        let dest = p * pps + col;
                        im2col_data[dest] = input[src];
                        im2col_wiring.push((dest as u32, src as u32));
                    }
                }
            }
        }
    }

    let kernel_t = transpose_matrix(&conv1_w_raw, c_out, patch_size);
    let kernel_padded = pad_matrix(&kernel_t, patch_size, c_out, pps, pc);
    let mm = padded_matmul(&im2col_data, pp, pps, &kernel_padded, pc);

    let bias_bc: Vec<i64> = (0..pp * pc)
        .map(|i| {
            let j = i % pc;
            if j < c_out {
                conv1_b_raw[j]
            } else {
                0
            }
        })
        .collect();
    let mm_biased: Vec<i64> = mm.iter().zip(bias_bc.iter()).map(|(m, b)| m + b).collect();
    let (conv_q, conv_r) = rescale_array(&mm_biased, alpha, offset);

    let relu_out: Vec<i64> = conv_q.iter().map(|&x| x.max(0)).collect();
    let relu_di: Vec<i64> = relu_out
        .iter()
        .zip(conv_q.iter())
        .map(|(o, x)| o - x)
        .collect();
    let relu_dz: Vec<i64> = relu_out.clone();

    let pool_oh = conv_oh / 2;
    let pool_ow = conv_ow / 2;
    let pool_total = next_pow2(pc * pool_oh * pool_ow);
    let pool_vars = log2(pool_total);
    let window_size = 4usize;

    let mut pool_max = vec![0i64; pool_total];
    let mut pool_we: Vec<Vec<i64>> = vec![vec![0i64; pool_total]; window_size];
    let mut pool_gw: Vec<Vec<(u32, u32)>> = vec![Vec::new(); window_size];

    for c in 0..c_out {
        for poh in 0..pool_oh {
            for pow_i in 0..pool_ow {
                let dest = c * pool_oh * pool_ow + poh * pool_ow + pow_i;
                let mut elems = Vec::new();
                for ph in 0..2usize {
                    for pw in 0..2usize {
                        let ei = ph * 2 + pw;
                        let src = ((poh * 2 + ph) * conv_ow + (pow_i * 2 + pw)) * pc + c;
                        pool_we[ei][dest] = relu_out[src];
                        elems.push(relu_out[src]);
                        pool_gw[ei].push((dest as u32, src as u32));
                    }
                }
                pool_max[dest] = *elems.iter().max().unwrap();
            }
        }
    }

    let pool_deltas: Vec<Vec<i64>> = (0..window_size)
        .map(|i| {
            (0..pool_total)
                .map(|w| pool_max[w] - pool_we[i][w])
                .collect()
        })
        .collect();

    let iv = log2(input_size);
    let irv = log2(pp);
    let icv = log2(pps);
    let imv = irv + icv;
    let ocv = log2(pc);
    let rv = irv + ocv;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let inp = builder.add_input_shred("input", iv, &public);
    let wtn = builder.add_input_shred("weight", icv + ocv, &public);
    let bn = builder.add_input_shred("bias_bc", rv, &public);
    let zn = builder.add_input_shred("zero", rv, &public);
    let en = builder.add_input_shred("expected", pool_vars, &public);

    let cqn = builder.add_input_shred("conv_q", rv, &committed);
    let crn = builder.add_input_shred("conv_r", rv, &committed);
    let rn = builder.add_input_shred("relu_max", rv, &committed);
    let rdi = builder.add_input_shred("relu_di", rv, &committed);
    let rdz = builder.add_input_shred("relu_dz", rv, &committed);
    let pmn = builder.add_input_shred("pool_max", pool_vars, &committed);
    let pdn: Vec<_> = (0..window_size)
        .map(|i| builder.add_input_shred(&format!("pd_{}", i), pool_vars, &committed))
        .collect();

    let im2col_n = builder.add_identity_gate_node(&inp, im2col_wiring, imv, None);
    let mm_n = builder.add_matmult_node(&im2col_n, (irv, icv), &wtn, (icv, ocv));

    let alpha_fr = i64_to_fr(alpha);
    let rc = builder.add_sector(
        mm_n.expr() + bn.expr() - AbstractExpression::scaled(cqn.expr(), alpha_fr) - crn.expr(),
    );
    builder.set_output(&rc);

    let c1 = builder.add_sector(rn.expr() - cqn.expr() - rdi.expr());
    builder.set_output(&c1);
    let c2 = builder.add_sector(rn.expr() - zn.expr() - rdz.expr());
    builder.set_output(&c2);
    let c3 = builder.add_sector(AbstractExpression::products(vec![rdi.id(), rdz.id()]));
    builder.set_output(&c3);

    let pgates: Vec<_> = (0..window_size)
        .map(|i| builder.add_identity_gate_node(&rn, pool_gw[i].clone(), pool_vars, None))
        .collect();
    for i in 0..window_size {
        let s = builder.add_sector(pmn.expr() - pgates[i].expr() - pdn[i].expr());
        builder.set_output(&s);
    }
    let pc_constraint = builder.add_sector(AbstractExpression::products(
        pdn.iter().map(|d| d.id()).collect(),
    ));
    builder.set_output(&pc_constraint);
    let oc = builder.add_sector(pmn.expr() - en.expr());
    builder.set_output(&oc);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    let input_padded = pad_to_size(&input, input_size);
    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    prover_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    prover_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    prover_circuit.set_input(
        "zero",
        MultilinearExtension::new(vec![Fr::from(0u64); 1 << rv]),
    );
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&pool_max)));
    prover_circuit.set_input("conv_q", MultilinearExtension::new(to_fr_vec(&conv_q)));
    prover_circuit.set_input("conv_r", MultilinearExtension::new(to_fr_vec(&conv_r)));
    prover_circuit.set_input("relu_max", MultilinearExtension::new(to_fr_vec(&relu_out)));
    prover_circuit.set_input("relu_di", MultilinearExtension::new(to_fr_vec(&relu_di)));
    prover_circuit.set_input("relu_dz", MultilinearExtension::new(to_fr_vec(&relu_dz)));
    prover_circuit.set_input("pool_max", MultilinearExtension::new(to_fr_vec(&pool_max)));
    for i in 0..window_size {
        prover_circuit.set_input(
            &format!("pd_{}", i),
            MultilinearExtension::new(to_fr_vec(&pool_deltas[i])),
        );
    }

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    verifier_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    verifier_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    verifier_circuit.set_input(
        "zero",
        MultilinearExtension::new(vec![Fr::from(0u64); 1 << rv]),
    );
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&pool_max)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_conv2d_with_padding_prove_verify() {
    let alpha: i64 = 1 << 18;
    let offset: i64 = 1 << 30;

    let in_h = 3usize;
    let in_w = 3usize;
    let c_in = 1usize;
    let c_out = 1usize;
    let kh_size = 3usize;
    let kw_size = 3usize;
    let stride = 1usize;
    let pad_top = 1usize;
    let pad_left = 1usize;
    let pad_bottom = 1usize;
    let pad_right = 1usize;

    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    let out_h = (padded_h - kh_size) / stride + 1;
    let out_w = (padded_w - kw_size) / stride + 1;
    assert_eq!(out_h, 3);
    assert_eq!(out_w, 3);
    let patch_size = c_in * kh_size * kw_size;
    let num_patches = out_h * out_w;

    let input: Vec<i64> = (1..=9).map(|x| x as i64).collect();
    let kernel: Vec<i64> = vec![0, 0, 0, 0, alpha, 0, 0, 0, 0];
    let bias: Vec<i64> = vec![0];

    let pad_patches_sz = next_pow2(num_patches);
    let pad_psize = next_pow2(patch_size);
    let pad_cout = next_pow2(c_out);

    let mut im2col = vec![0i64; pad_patches_sz * pad_psize];
    let mut wiring: Vec<(u32, u32)> = Vec::new();
    for oh in 0..out_h {
        for ow in 0..out_w {
            let patch = oh * out_w + ow;
            for c in 0..c_in {
                for kr in 0..kh_size {
                    for kc in 0..kw_size {
                        let abs_h = oh * stride + kr;
                        let abs_w = ow * stride + kc;
                        if abs_h < pad_top || abs_w < pad_left {
                            continue;
                        }
                        let ih = abs_h - pad_top;
                        let iw = abs_w - pad_left;
                        if ih >= in_h || iw >= in_w {
                            continue;
                        }
                        let col = c * kh_size * kw_size + kr * kw_size + kc;
                        let src = c * in_h * in_w + ih * in_w + iw;
                        let dest = patch * pad_psize + col;
                        im2col[dest] = input[src];
                        wiring.push((dest as u32, src as u32));
                    }
                }
            }
        }
    }

    let kernel_t = transpose_matrix(&kernel, c_out, patch_size);
    let kernel_padded = pad_matrix(&kernel_t, patch_size, c_out, pad_psize, pad_cout);
    let mm_result = padded_matmul(&im2col, pad_patches_sz, pad_psize, &kernel_padded, pad_cout);

    let bias_bc: Vec<i64> = (0..pad_patches_sz * pad_cout)
        .map(|i| {
            let j = i % pad_cout;
            if j < c_out {
                bias[j]
            } else {
                0
            }
        })
        .collect();
    let mm_with_bias: Vec<i64> = mm_result
        .iter()
        .zip(bias_bc.iter())
        .map(|(m, b)| m + b)
        .collect();
    let (quotients, remainders) = rescale_array(&mm_with_bias, alpha, offset);

    assert_eq!(&quotients[0..9], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let input_vars = log2(next_pow2(c_in * in_h * in_w));
    let im2col_row_vars = log2(pad_patches_sz);
    let im2col_col_vars = log2(pad_psize);
    let im2col_vars = im2col_row_vars + im2col_col_vars;
    let out_col_vars = log2(pad_cout);
    let weight_vars = im2col_col_vars + out_col_vars;
    let result_vars = im2col_row_vars + out_col_vars;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", input_vars, &public);
    let weight_node = builder.add_input_shred("weight", weight_vars, &public);
    let bias_bc_node = builder.add_input_shred("bias_bc", result_vars, &public);
    let expected_node = builder.add_input_shred("expected", result_vars, &public);
    let q_node = builder.add_input_shred("quotient", result_vars, &committed);
    let r_node = builder.add_input_shred("remainder", result_vars, &committed);

    let im2col_node = builder.add_identity_gate_node(&input_node, wiring, im2col_vars, None);
    let mm_node = builder.add_matmult_node(
        &im2col_node,
        (im2col_row_vars, im2col_col_vars),
        &weight_node,
        (im2col_col_vars, out_col_vars),
    );

    let alpha_fr = i64_to_fr(alpha);
    let rescale_chk = builder.add_sector(
        mm_node.expr() + bias_bc_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_chk);
    let out_chk = builder.add_sector(q_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    let input_padded = pad_to_size(&input, 1 << input_vars);
    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    prover_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    prover_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input("quotient", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input(
        "remainder",
        MultilinearExtension::new(to_fr_vec(&remainders)),
    );

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    verifier_circuit.set_input(
        "weight",
        MultilinearExtension::new(to_fr_vec(&kernel_padded)),
    );
    verifier_circuit.set_input("bias_bc", MultilinearExtension::new(to_fr_vec(&bias_bc)));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_batchnorm_prove_verify() {
    use jstprove_remainder::gadgets::rescale;
    use jstprove_remainder::padding::{next_power_of_two, num_vars_for};

    let alpha: i64 = 2i64.pow(18);
    let offset = 1i64 << 30;

    let c = 2usize;
    let h = 2usize;
    let w = 2usize;
    let hw = h * w;
    let total = c * hw;

    let input_raw: Vec<i64> = vec![
        1 * alpha,
        2 * alpha,
        3 * alpha,
        4 * alpha,
        5 * alpha,
        6 * alpha,
        7 * alpha,
        8 * alpha,
    ];
    let padded_size = next_power_of_two(total);
    let nv = num_vars_for(total);
    let mut input_padded = input_raw.clone();
    input_padded.resize(padded_size, 0);

    let scale = vec![2.0f64, 0.5];
    let bias = vec![0.0f64, 1.0];
    let mean = vec![0.0f64, 0.0];
    let var = vec![1.0f64, 1.0];
    let epsilon = 1e-5f64;

    let mut mul_per_ch = Vec::with_capacity(c);
    let mut add_per_ch = Vec::with_capacity(c);
    for i in 0..c {
        let m = scale[i] / (var[i] + epsilon).sqrt();
        mul_per_ch.push((m * alpha as f64).round() as i64);
        add_per_ch.push(((bias[i] - mean[i] * m) * (alpha as f64).powi(2)).round() as i64);
    }

    let mut mul_broadcast = vec![0i64; padded_size];
    let mut add_broadcast = vec![0i64; padded_size];
    for ch in 0..c {
        for s in 0..hw {
            let idx = ch * hw + s;
            if idx < padded_size {
                mul_broadcast[idx] = mul_per_ch[ch];
                add_broadcast[idx] = add_per_ch[ch];
            }
        }
    }

    let product: Vec<i64> = input_padded
        .iter()
        .zip(mul_broadcast.iter())
        .map(|(&x, &m)| {
            let prod = x as i128 * m as i128;
            i64::try_from(prod).unwrap()
        })
        .collect();
    let with_add: Vec<i64> = product
        .iter()
        .zip(add_broadcast.iter())
        .map(|(&p, &a)| p + a)
        .collect();

    let (quotients, remainders) = rescale::compute_rescale_array(&with_add, alpha, offset).unwrap();

    let alpha_fr = i64_to_fr(alpha);

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", nv, &public);
    let mul_node = builder.add_input_shred("mul", nv, &public);
    let add_node = builder.add_input_shred("add", nv, &public);
    let expected_node = builder.add_input_shred("expected", nv, &public);
    let q_node = builder.add_input_shred("quotient", nv, &committed);
    let r_node = builder.add_input_shred("remainder", nv, &committed);

    let rescale_chk = builder.add_sector(
        AbstractExpression::products(vec![input_node.id(), mul_node.id()]) + add_node.expr()
            - AbstractExpression::scaled(q_node.expr(), alpha_fr)
            - r_node.expr(),
    );
    builder.set_output(&rescale_chk);
    let out_chk = builder.add_sector(q_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    prover_circuit.set_input("mul", MultilinearExtension::new(to_fr_vec(&mul_broadcast)));
    prover_circuit.set_input("add", MultilinearExtension::new(to_fr_vec(&add_broadcast)));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input("quotient", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input(
        "remainder",
        MultilinearExtension::new(to_fr_vec(&remainders)),
    );

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input_padded)));
    verifier_circuit.set_input("mul", MultilinearExtension::new(to_fr_vec(&mul_broadcast)));
    verifier_circuit.set_input("add", MultilinearExtension::new(to_fr_vec(&add_broadcast)));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_addsub_prove_verify() {
    let a: Vec<i64> = vec![10, 20, 30, 40];
    let b: Vec<i64> = vec![1, 2, 3, 4];
    let sum: Vec<i64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    let diff: Vec<i64> = sum.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();

    let nv = 2usize;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let a_node = builder.add_input_shred("a", nv, &public);
    let b_node = builder.add_input_shred("b", nv, &public);
    let sum_node = builder.add_input_shred("sum_result", nv, &committed);
    let diff_node = builder.add_input_shred("diff_result", nv, &committed);
    let expected_node = builder.add_input_shred("expected", nv, &public);

    let add_chk = builder.add_sector(a_node.expr() + b_node.expr() - sum_node.expr());
    builder.set_output(&add_chk);

    let sub_chk = builder.add_sector(sum_node.expr() - b_node.expr() - diff_node.expr());
    builder.set_output(&sub_chk);

    let out_chk = builder.add_sector(diff_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("a", MultilinearExtension::new(to_fr_vec(&a)));
    prover_circuit.set_input("b", MultilinearExtension::new(to_fr_vec(&b)));
    prover_circuit.set_input("sum_result", MultilinearExtension::new(to_fr_vec(&sum)));
    prover_circuit.set_input("diff_result", MultilinearExtension::new(to_fr_vec(&diff)));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&diff)));

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("a", MultilinearExtension::new(to_fr_vec(&a)));
    verifier_circuit.set_input("b", MultilinearExtension::new(to_fr_vec(&b)));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&diff)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_rescale_with_logup() {
    let alpha: i64 = 1 << 4;
    let table_nv: usize = 4;
    let table_size: usize = 1 << table_nv;
    let offset: i64 = 1 << 30;

    let input: Vec<i64> = vec![1, 2, 3, 4];
    let weights: Vec<i64> = vec![17, 17];

    let raw = matmul_native(&input, 2, 2, &weights, 1);
    let (quotients, remainders) = rescale_array(&raw, alpha, offset);

    let r_mults = compute_multiplicities(&remainders, table_size).unwrap();
    let table: Vec<i64> = (0..table_size as i64).collect();

    let m_vars = 1usize;
    let k_vars = 1usize;
    let n_vars = 0usize;
    let result_nv = m_vars + n_vars;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", m_vars + k_vars, &public);
    let weight_node = builder.add_input_shred("weights", k_vars + n_vars, &public);
    let expected_node = builder.add_input_shred("expected", result_nv, &public);

    let q_node = builder.add_input_shred("quotient", result_nv, &committed);
    let r_node = builder.add_input_shred("remainder", result_nv, &committed);

    let mm = builder.add_matmult_node(
        &input_node,
        (m_vars, k_vars),
        &weight_node,
        (k_vars, n_vars),
    );
    let alpha_fr = i64_to_fr(alpha);
    let rescale_chk = builder.add_sector(
        mm.expr() - AbstractExpression::scaled(q_node.expr(), alpha_fr) - r_node.expr(),
    );
    builder.set_output(&rescale_chk);

    let out_chk = builder.add_sector(q_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let table_node = builder.add_input_shred("range_table", table_nv, &public);
    let mults_node = builder.add_input_shred("r_mults", table_nv, &committed);
    let fs = builder.add_fiat_shamir_challenge_node(1);
    let lookup = builder.add_lookup_table(&table_node, &fs);
    builder.add_lookup_constraint(&lookup, &r_node, &mults_node);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    prover_circuit.set_input("weights", MultilinearExtension::new(to_fr_vec(&weights)));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input("quotient", MultilinearExtension::new(to_fr_vec(&quotients)));
    prover_circuit.set_input(
        "remainder",
        MultilinearExtension::new(to_fr_vec(&remainders)),
    );
    prover_circuit.set_input("range_table", MultilinearExtension::new(to_fr_vec(&table)));
    prover_circuit.set_input("r_mults", MultilinearExtension::new(to_fr_vec(&r_mults)));

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    verifier_circuit.set_input("weights", MultilinearExtension::new(to_fr_vec(&weights)));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&quotients)));
    verifier_circuit.set_input("range_table", MultilinearExtension::new(to_fr_vec(&table)));

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_relu_with_logup() {
    let delta_nv: usize = 4;
    let delta_table_size: usize = 1 << delta_nv;
    let delta_table: Vec<i64> = (0..delta_table_size as i64).collect();

    let input: Vec<i64> = vec![10, -5, 3, -8];
    let nv = 2usize;
    let relu_out: Vec<i64> = input.iter().map(|&x| x.max(0)).collect();
    let di: Vec<i64> = relu_out
        .iter()
        .zip(input.iter())
        .map(|(o, x)| o - x)
        .collect();
    let dz: Vec<i64> = relu_out.clone();

    let di_mults = compute_multiplicities(&di, delta_table_size).unwrap();
    let dz_mults = compute_multiplicities(&dz, delta_table_size).unwrap();

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", nv, &public);
    let zero_node = builder.add_input_shred("zero", nv, &public);
    let expected_node = builder.add_input_shred("expected", nv, &public);

    let max_node = builder.add_input_shred("max", nv, &committed);
    let di_node = builder.add_input_shred("di", nv, &committed);
    let dz_node = builder.add_input_shred("dz", nv, &committed);

    let c1 = builder.add_sector(max_node.expr() - input_node.expr() - di_node.expr());
    builder.set_output(&c1);
    let c2 = builder.add_sector(max_node.expr() - zero_node.expr() - dz_node.expr());
    builder.set_output(&c2);
    let prod = builder.add_sector(AbstractExpression::products(vec![
        di_node.id(),
        dz_node.id(),
    ]));
    builder.set_output(&prod);
    let out_chk = builder.add_sector(max_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let table_node = builder.add_input_shred("delta_table", delta_nv, &public);
    let di_mults_node = builder.add_input_shred("di_mults", delta_nv, &committed);
    let dz_mults_node = builder.add_input_shred("dz_mults", delta_nv, &committed);

    let fs = builder.add_fiat_shamir_challenge_node(1);
    let lookup = builder.add_lookup_table(&table_node, &fs);
    builder.add_lookup_constraint(&lookup, &di_node, &di_mults_node);
    builder.add_lookup_constraint(&lookup, &dz_node, &dz_mults_node);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    prover_circuit.set_input("zero", MultilinearExtension::new(vec![Fr::from(0u64); 4]));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&relu_out)));
    prover_circuit.set_input("max", MultilinearExtension::new(to_fr_vec(&relu_out)));
    prover_circuit.set_input("di", MultilinearExtension::new(to_fr_vec(&di)));
    prover_circuit.set_input("dz", MultilinearExtension::new(to_fr_vec(&dz)));
    prover_circuit.set_input(
        "delta_table",
        MultilinearExtension::new(to_fr_vec(&delta_table)),
    );
    prover_circuit.set_input("di_mults", MultilinearExtension::new(to_fr_vec(&di_mults)));
    prover_circuit.set_input("dz_mults", MultilinearExtension::new(to_fr_vec(&dz_mults)));

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    verifier_circuit.set_input("zero", MultilinearExtension::new(vec![Fr::from(0u64); 4]));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&relu_out)));
    verifier_circuit.set_input(
        "delta_table",
        MultilinearExtension::new(to_fr_vec(&delta_table)),
    );

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}

#[test]
fn test_gemm_relu_gemm_with_logup() {
    let alpha: i64 = 1 << 4;
    let exponent: usize = 4;
    let offset: i64 = 1 << 30;

    let input: Vec<i64> = vec![3, 1, -3, 2];
    let w1: Vec<i64> = vec![17, 0, 0, 17];
    let w2: Vec<i64> = vec![17, 0, 0, 17];

    let mm1 = matmul_native(&input, 2, 2, &w1, 2);
    let (q1, r1) = rescale_array(&mm1, alpha, offset);

    let relu_out: Vec<i64> = q1.iter().map(|&x| x.max(0)).collect();
    let di: Vec<i64> = relu_out.iter().zip(q1.iter()).map(|(o, x)| o - x).collect();
    let dz: Vec<i64> = relu_out.clone();

    let mm2 = matmul_native(&relu_out, 2, 2, &w2, 2);
    let (q2, r2) = rescale_array(&mm2, alpha, offset);
    let expected = q2.clone();

    let delta_nv: usize = 3;
    let rescale_table_size: usize = 1 << exponent;
    let delta_table_size: usize = 1 << delta_nv;

    let rescale_table: Vec<i64> = (0..rescale_table_size as i64).collect();
    let delta_table: Vec<i64> = (0..delta_table_size as i64).collect();

    let r1_mults = compute_multiplicities(&r1, rescale_table_size).unwrap();
    let r2_mults = compute_multiplicities(&r2, rescale_table_size).unwrap();
    let di_mults = compute_multiplicities(&di, delta_table_size).unwrap();
    let dz_mults = compute_multiplicities(&dz, delta_table_size).unwrap();

    let m_vars = 1usize;
    let k_vars = 1usize;
    let n_vars = 1usize;
    let nv = m_vars + n_vars;

    let mut builder = CircuitBuilder::<Fr>::new();
    let public = builder.add_input_layer("Public", LayerVisibility::Public);
    let committed = builder.add_input_layer("Committed", LayerVisibility::Committed);

    let input_node = builder.add_input_shred("input", m_vars + k_vars, &public);
    let w1_node = builder.add_input_shred("w1", k_vars + n_vars, &public);
    let zero_node = builder.add_input_shred("zero", nv, &public);
    let w2_node = builder.add_input_shred("w2", k_vars + n_vars, &public);
    let expected_node = builder.add_input_shred("expected", nv, &public);

    let q1_node = builder.add_input_shred("q1", nv, &committed);
    let r1_node = builder.add_input_shred("r1", nv, &committed);
    let max_node = builder.add_input_shred("max", nv, &committed);
    let di_node = builder.add_input_shred("di", nv, &committed);
    let dz_node = builder.add_input_shred("dz", nv, &committed);
    let q2_node = builder.add_input_shred("q2", nv, &committed);
    let r2_node = builder.add_input_shred("r2", nv, &committed);

    let alpha_fr = i64_to_fr(alpha);

    let mm1_node =
        builder.add_matmult_node(&input_node, (m_vars, k_vars), &w1_node, (k_vars, n_vars));
    let rescale1 = builder.add_sector(
        mm1_node.expr() - AbstractExpression::scaled(q1_node.expr(), alpha_fr) - r1_node.expr(),
    );
    builder.set_output(&rescale1);

    let c1 = builder.add_sector(max_node.expr() - q1_node.expr() - di_node.expr());
    builder.set_output(&c1);
    let c2 = builder.add_sector(max_node.expr() - zero_node.expr() - dz_node.expr());
    builder.set_output(&c2);
    let prod = builder.add_sector(AbstractExpression::products(vec![
        di_node.id(),
        dz_node.id(),
    ]));
    builder.set_output(&prod);

    let mm2_node =
        builder.add_matmult_node(&max_node, (m_vars, n_vars), &w2_node, (k_vars, n_vars));
    let rescale2 = builder.add_sector(
        mm2_node.expr() - AbstractExpression::scaled(q2_node.expr(), alpha_fr) - r2_node.expr(),
    );
    builder.set_output(&rescale2);

    let out_chk = builder.add_sector(q2_node.expr() - expected_node.expr());
    builder.set_output(&out_chk);

    let rescale_tbl_node = builder.add_input_shred("rescale_table", exponent, &public);
    let r1_mults_node = builder.add_input_shred("r1_mults", exponent, &committed);
    let r2_mults_node = builder.add_input_shred("r2_mults", exponent, &committed);
    let delta_tbl_node = builder.add_input_shred("delta_table", delta_nv, &public);
    let di_mults_node = builder.add_input_shred("di_mults", delta_nv, &committed);
    let dz_mults_node = builder.add_input_shred("dz_mults", delta_nv, &committed);

    let fs = builder.add_fiat_shamir_challenge_node(1);

    let rescale_lookup = builder.add_lookup_table(&rescale_tbl_node, &fs);
    builder.add_lookup_constraint(&rescale_lookup, &r1_node, &r1_mults_node);
    builder.add_lookup_constraint(&rescale_lookup, &r2_node, &r2_mults_node);

    let delta_lookup = builder.add_lookup_table(&delta_tbl_node, &fs);
    builder.add_lookup_constraint(&delta_lookup, &di_node, &di_mults_node);
    builder.add_lookup_constraint(&delta_lookup, &dz_node, &dz_mults_node);

    let mut prover_circuit = builder.build_without_layer_combination().unwrap();
    let mut verifier_circuit = prover_circuit.clone();

    prover_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    prover_circuit.set_input("w1", MultilinearExtension::new(to_fr_vec(&w1)));
    prover_circuit.set_input(
        "zero",
        MultilinearExtension::new(vec![Fr::from(0u64); 1 << nv]),
    );
    prover_circuit.set_input("w2", MultilinearExtension::new(to_fr_vec(&w2)));
    prover_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&expected)));
    prover_circuit.set_input("q1", MultilinearExtension::new(to_fr_vec(&q1)));
    prover_circuit.set_input("r1", MultilinearExtension::new(to_fr_vec(&r1)));
    prover_circuit.set_input("max", MultilinearExtension::new(to_fr_vec(&relu_out)));
    prover_circuit.set_input("di", MultilinearExtension::new(to_fr_vec(&di)));
    prover_circuit.set_input("dz", MultilinearExtension::new(to_fr_vec(&dz)));
    prover_circuit.set_input("q2", MultilinearExtension::new(to_fr_vec(&q2)));
    prover_circuit.set_input("r2", MultilinearExtension::new(to_fr_vec(&r2)));
    prover_circuit.set_input(
        "rescale_table",
        MultilinearExtension::new(to_fr_vec(&rescale_table)),
    );
    prover_circuit.set_input("r1_mults", MultilinearExtension::new(to_fr_vec(&r1_mults)));
    prover_circuit.set_input("r2_mults", MultilinearExtension::new(to_fr_vec(&r2_mults)));
    prover_circuit.set_input(
        "delta_table",
        MultilinearExtension::new(to_fr_vec(&delta_table)),
    );
    prover_circuit.set_input("di_mults", MultilinearExtension::new(to_fr_vec(&di_mults)));
    prover_circuit.set_input("dz_mults", MultilinearExtension::new(to_fr_vec(&dz_mults)));

    let provable = prover_circuit.gen_provable_circuit().unwrap();
    let (proof_config, proof_transcript) =
        prove_circuit_with_runtime_optimized_config::<Fr, PoseidonSponge<Fr>>(&provable);

    verifier_circuit.set_input("input", MultilinearExtension::new(to_fr_vec(&input)));
    verifier_circuit.set_input("w1", MultilinearExtension::new(to_fr_vec(&w1)));
    verifier_circuit.set_input(
        "zero",
        MultilinearExtension::new(vec![Fr::from(0u64); 1 << nv]),
    );
    verifier_circuit.set_input("w2", MultilinearExtension::new(to_fr_vec(&w2)));
    verifier_circuit.set_input("expected", MultilinearExtension::new(to_fr_vec(&expected)));
    verifier_circuit.set_input(
        "rescale_table",
        MultilinearExtension::new(to_fr_vec(&rescale_table)),
    );
    verifier_circuit.set_input(
        "delta_table",
        MultilinearExtension::new(to_fr_vec(&delta_table)),
    );

    let verifiable = verifier_circuit.gen_verifiable_circuit().unwrap();
    verify_circuit_with_proof_config(&verifiable, &proof_config, proof_transcript);
}
