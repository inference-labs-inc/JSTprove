// Copyright 2025 Irreducible Inc.

use binius_circuits::sha256::Sha256;
use binius_core::{
    constraint_system::{AndConstraint, ConstraintSystem, MulConstraint, ValueVec},
    verify::{eval_operand, verify_constraints},
    word::Word,
};
use binius_field::{AESTowerField8b, BinaryField};
use binius_frontend::{CircuitBuilder, Wire};
use binius_math::{
    BinarySubspace,
    inner_product::{inner_product, inner_product_buffers},
    multilinear::eq::eq_ind_partial_eval,
    univariate::lagrange_evals,
};
use binius_prover::{
    fold_word::fold_words,
    protocols::shift::{OperatorData, build_key_collection, prove},
};
use binius_transcript::ProverTranscript;
use binius_utils::checked_arithmetics::strict_log_2;
use binius_verifier::{
    config::{LOG_WORD_SIZE_BITS, StdChallenger},
    protocols::shift::{OperatorData as VerifierOperatorData, check_eval, verify},
};
use itertools::Itertools;
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256 as Sha256Hasher};

pub fn create_sha256_cs_with_witness() -> (ConstraintSystem, ValueVec) {
    let builder = CircuitBuilder::new();
    let max_len: usize = 64; // Maximum message length in bytes

    // Create wires for the SHA256 circuit
    let len = builder.add_witness(); // Actual message length
    let digest = [
        builder.add_inout(), // Expected digest as 4x64-bit words
        builder.add_inout(),
        builder.add_inout(),
        builder.add_inout(),
    ];
    let message: Vec<Wire> = (0..max_len.div_ceil(8))
        .map(|_| builder.add_witness())
        .collect();

    // Create the SHA256 circuit
    let sha256 = Sha256::new(&builder, len, digest, message);

    let circuit = builder.build();
    let mut witness_filler = circuit.new_witness_filler();

    // Populate with concrete message: "abc"
    let message_bytes = b"abc";
    sha256.populate_len_bytes(&mut witness_filler, message_bytes.len());
    sha256.populate_message(&mut witness_filler, message_bytes);

    // Calculate SHA256 digest of the message dynamically
    let hash = Sha256Hasher::digest(message_bytes);
    let expected_digest: [u8; 32] = hash.into();
    sha256.populate_digest(&mut witness_filler, expected_digest);

    // Get the witness vector
    circuit.populate_wire_witness(&mut witness_filler).unwrap();

    (
        circuit.constraint_system().clone(),
        witness_filler.into_value_vec(),
    )
}

pub fn create_concat_cs_with_witness() -> (ConstraintSystem, ValueVec) {
    use binius_circuits::{concat::Concat, fixed_byte_vec::ByteVec};

    let builder = CircuitBuilder::new();
    let max_n_joined: usize = 32; // Maximum joined size

    // Create wires for concat circuit
    let len_joined = builder.add_inout();
    let joined: Vec<Wire> = (0..max_n_joined / 8).map(|_| builder.add_inout()).collect();

    // Create terms: "Hello" + " " + "World!"
    let terms = vec![
        ByteVec {
            len_bytes: builder.add_witness(),
            data: (0..1).map(|_| builder.add_witness()).collect(),
        },
        ByteVec {
            len_bytes: builder.add_witness(),
            data: (0..1).map(|_| builder.add_witness()).collect(),
        },
        ByteVec {
            len_bytes: builder.add_witness(),
            data: (0..1).map(|_| builder.add_witness()).collect(),
        },
    ];

    // Create the Concat circuit
    let concat = Concat::new(&builder, len_joined, joined, terms);

    let circuit = builder.build();
    let mut witness_filler = circuit.new_witness_filler();

    // Test data
    let term1_data = b"Hello";
    let term2_data = b" ";
    let term3_data = b"World!";
    let joined_data = b"Hello World!";

    // Populate terms
    concat.terms[0].populate_len_bytes(&mut witness_filler, term1_data.len());
    concat.terms[0].populate_data(&mut witness_filler, term1_data);

    concat.terms[1].populate_len_bytes(&mut witness_filler, term2_data.len());
    concat.terms[1].populate_data(&mut witness_filler, term2_data);

    concat.terms[2].populate_len_bytes(&mut witness_filler, term3_data.len());
    concat.terms[2].populate_data(&mut witness_filler, term3_data);

    // Populate joined result
    concat.populate_len_joined_bytes(&mut witness_filler, joined_data.len());
    concat.populate_joined(&mut witness_filler, joined_data);

    // Get the witness vector
    circuit.populate_wire_witness(&mut witness_filler).unwrap();

    (
        circuit.constraint_system().clone(),
        witness_filler.into_value_vec(),
    )
}

pub fn create_slice_cs_with_witness() -> (ConstraintSystem, ValueVec) {
    use binius_circuits::slice::Slice;

    let builder = CircuitBuilder::new();

    // Create wires for slice circuit
    let len_input = builder.add_witness();
    let len_slice = builder.add_witness();
    let input: Vec<Wire> = (0..4).map(|_| builder.add_witness()).collect();
    let slice: Vec<Wire> = (0..2).map(|_| builder.add_witness()).collect();
    let offset = builder.add_witness();

    // Create the Slice circuit
    let slice_circuit = Slice::new(&builder, len_input, len_slice, input, slice, offset);

    let circuit = builder.build();
    let mut witness_filler = circuit.new_witness_filler();

    // Test slicing "Hello World!" from offset 6 with length 5 to get "World"
    let input_data = b"Hello World!";
    let slice_data = b"World";
    let offset_val = 6;

    slice_circuit.populate_len_input(&mut witness_filler, input_data.len());
    slice_circuit.populate_len_slice(&mut witness_filler, slice_data.len());
    slice_circuit.populate_input(&mut witness_filler, input_data);
    slice_circuit.populate_slice(&mut witness_filler, slice_data);
    slice_circuit.populate_offset(&mut witness_filler, offset_val);

    // Get the witness vector
    circuit.populate_wire_witness(&mut witness_filler).unwrap();

    (
        circuit.constraint_system().clone(),
        witness_filler.into_value_vec(),
    )
}

// Compute the image of the witness applied to the AND constraints
pub fn compute_bitand_images(constraints: &[AndConstraint], witness: &ValueVec) -> [Vec<Word>; 3] {
    let (a_image, b_image, c_image) = constraints
        .iter()
        .map(|constraint| {
            let a = eval_operand(witness, &constraint.a);
            let b = eval_operand(witness, &constraint.b);
            let c = eval_operand(witness, &constraint.c);
            (a, b, c)
        })
        .multiunzip();
    [a_image, b_image, c_image]
}

// Compute the image of the witness applied to the MUL constraints
fn compute_intmul_images(constraints: &[MulConstraint], witness: &ValueVec) -> [Vec<Word>; 4] {
    let (a_image, b_image, hi_image, lo_image) = constraints
        .iter()
        .map(|constraint| {
            let a = eval_operand(witness, &constraint.a);
            let b = eval_operand(witness, &constraint.b);
            let hi = eval_operand(witness, &constraint.hi);
            let lo = eval_operand(witness, &constraint.lo);
            (a, b, hi, lo)
        })
        .multiunzip();
    [a_image, b_image, hi_image, lo_image]
}

// Evaluate the image of the witness applied to the AND or MUL constraints
// Univariate point is `r_zhat_prime`, multilinear point tensor-expanded is `r_x_prime_tensor`
fn evaluate_image<F: BinaryField>(
    subspace: &BinarySubspace<F>,
    image: &[Word],
    r_zhat_prime: F,
    r_x_prime_tensor: &[F],
) -> F {
    let l_tilde = lagrange_evals(subspace, r_zhat_prime);
    let univariate = image
        .iter()
        .map(|&word| {
            (0..64)
                .filter(|&i| (word >> i) & Word::ONE == Word::ONE)
                .map(|i| l_tilde[i as usize])
                .sum()
        })
        .collect::<Vec<_>>();
    inner_product(r_x_prime_tensor.iter().copied(), univariate.iter().copied())
}

/// Compute inner product of tensor with all bits from words
pub fn evaluate_witness<F: BinaryField>(words: &[Word], r_j: &[F], r_y: &[F]) -> F {
    let r_j_tensor = eq_ind_partial_eval::<F>(r_j);
    let r_y_tensor = eq_ind_partial_eval::<F>(r_y);

    let r_j_witness = fold_words::<_, F>(words, r_j_tensor.as_ref());

    inner_product_buffers(&r_j_witness, &r_y_tensor)
}

#[test]
fn test_shift_prove_and_verify() {
    use binius_field::{BinaryField128bGhash, PackedBinaryGhash2x128b, Random};
    type F = BinaryField128bGhash;
    type P = PackedBinaryGhash2x128b;
    let mut rng = StdRng::seed_from_u64(0);

    let mut constraint_systems_to_test = vec![
        create_sha256_cs_with_witness(),
        create_slice_cs_with_witness(),
        create_concat_cs_with_witness(),
    ];
    for (constraint_system, _) in constraint_systems_to_test.iter_mut() {
        constraint_system.validate_and_prepare().unwrap();
    }

    for (cs, value_vec) in constraint_systems_to_test.into_iter() {
        // Validate constraints using frontend verifier first
        if let Err(e) = verify_constraints(&cs, &value_vec) {
            panic!("Circuit failed constraint validation: {e}");
        }

        // Sample multilinear challenge point
        let r_x_prime_bitand = {
            let log_bitand_constraint_count = strict_log_2(cs.and_constraints.len()).unwrap();
            (0..log_bitand_constraint_count as u128)
                .map(F::new)
                .collect::<Vec<_>>()
        };
        let r_x_prime_intmul = {
            let log_intmul_constraint_count = strict_log_2(cs.mul_constraints.len()).unwrap();
            (0..log_intmul_constraint_count as u128)
                .map(F::new)
                .collect::<Vec<_>>()
        };

        // Sample univaraite eval point
        let r_zhat_prime_bitand = F::random(&mut rng);
        let r_zhat_prime_intmul = F::random(&mut rng);

        let subspace = BinarySubspace::<AESTowerField8b>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();

        let bitand_evals = compute_bitand_images(&cs.and_constraints, &value_vec).map(|image| {
            evaluate_image(
                &subspace,
                &image,
                r_zhat_prime_bitand,
                eq_ind_partial_eval(&r_x_prime_bitand).as_ref(),
            )
        });

        let intmul_evals = compute_intmul_images(&cs.mul_constraints, &value_vec).map(|image| {
            evaluate_image(
                &subspace,
                &image,
                r_zhat_prime_intmul,
                eq_ind_partial_eval(&r_x_prime_intmul).as_ref(),
            )
        });

        // Build prover's constraint system
        let key_collection = build_key_collection(&cs);

        // Create prover transcript and call the prover
        let mut prover_transcript = ProverTranscript::<StdChallenger>::default();

        let prover_bitand_data = OperatorData {
            evals: bitand_evals.to_vec(),
            r_zhat_prime: r_zhat_prime_bitand,
            r_x_prime: r_x_prime_bitand.clone(),
        };
        let prover_intmul_data = OperatorData {
            evals: intmul_evals.to_vec(),
            r_zhat_prime: r_zhat_prime_intmul,
            r_x_prime: r_x_prime_intmul.clone(),
        };

        let inout_n_vars = strict_log_2(cs.value_vec_layout.offset_witness).unwrap();

        let prover_output = prove::<F, P, _>(
            inout_n_vars,
            &key_collection,
            value_vec.combined_witness(),
            prover_bitand_data.clone(),
            prover_intmul_data.clone(),
            &mut prover_transcript,
        )
        .unwrap();

        // Create verifier transcript and call the verifier
        let mut verifier_transcript = prover_transcript.into_verifier();

        let verifier_bitand_data =
            VerifierOperatorData::new(r_zhat_prime_bitand, r_x_prime_bitand, bitand_evals);
        let verifier_intmul_data =
            VerifierOperatorData::new(r_zhat_prime_intmul, r_x_prime_intmul, intmul_evals);

        let verifier_output = verify(
            &cs,
            value_vec.public(),
            &verifier_bitand_data,
            &verifier_intmul_data,
            &mut verifier_transcript,
        )
        .unwrap();
        verifier_transcript.finalize().unwrap();

        // Check consistency with verifier output
        check_eval(
            &cs,
            &verifier_bitand_data,
            &verifier_intmul_data,
            &subspace,
            &verifier_output,
        )
        .unwrap();

        // Check the claimed eval matches the computed eval
        let expected_eval = evaluate_witness(
            value_vec.combined_witness(),
            verifier_output.r_j(),
            verifier_output.r_y(),
        );
        assert_eq!(expected_eval, verifier_output.witness_eval);

        // Check consistency of prover and verifier outputs
        let eval_point = [verifier_output.r_j(), verifier_output.r_y()].concat();
        assert_eq!(prover_output.challenges, eval_point);
        assert_eq!(prover_output.eval, verifier_output.witness_eval);
    }
}
