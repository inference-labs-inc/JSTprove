use std::borrow::Cow;

use ark_ff_05::PrimeField;
use whir::algebra::embedding::Basefield;
use whir::algebra::fields::{Field64, Field64_2};
use whir::algebra::linear_form::{Evaluate, LinearForm, MultilinearExtension as WhirMLE};
use whir::hash;
use whir::parameters::ProtocolParameters;
use whir::protocols::whir::Config as WhirConfig;
use whir::transcript::{codecs::Empty, DomainSeparator, Proof, ProverState, VerifierState};

pub const WHIR_SECURITY_LEVEL: usize = 128;
pub const WHIR_POW_BITS: usize = 0;
pub const WHIR_FOLDING_FACTOR: usize = 4;
pub const WHIR_STARTING_LOG_INV_RATE: usize = 1;

pub fn goldilocks_u64_to_whir(v: u64) -> Field64 {
    Field64::from_bigint(<Field64 as PrimeField>::BigInt::from(v))
        .expect("valid Goldilocks element")
}

pub fn goldilocks_vec_to_whir(vals: &[u64]) -> Vec<Field64> {
    vals.iter().map(|&v| goldilocks_u64_to_whir(v)).collect()
}

pub fn goldilocks_ext2_to_whir(c0: u64, c1: u64) -> Field64_2 {
    Field64_2::new(goldilocks_u64_to_whir(c0), goldilocks_u64_to_whir(c1))
}

fn make_whir_config(num_vars: usize) -> WhirConfig<Basefield<Field64_2>> {
    let whir_params = ProtocolParameters {
        security_level: WHIR_SECURITY_LEVEL,
        pow_bits: WHIR_POW_BITS,
        initial_folding_factor: WHIR_FOLDING_FACTOR,
        folding_factor: WHIR_FOLDING_FACTOR,
        unique_decoding: false,
        starting_log_inv_rate: WHIR_STARTING_LOG_INV_RATE,
        batch_size: 1,
        hash_id: hash::SHA2,
    };

    let mut config = WhirConfig::<Basefield<Field64_2>>::new(1 << num_vars, &whir_params);
    config.initial_sumcheck.round_pow.threshold = u64::MAX;
    config.initial_skip_pow.threshold = u64::MAX;
    for round in &mut config.round_configs {
        round.sumcheck.round_pow.threshold = u64::MAX;
        round.pow.threshold = u64::MAX;
    }
    config.final_sumcheck.round_pow.threshold = u64::MAX;
    config.final_pow.threshold = u64::MAX;
    config
}

fn make_domain_separator(
    config: &WhirConfig<Basefield<Field64_2>>,
) -> DomainSeparator<'static, Empty> {
    DomainSeparator::protocol(config)
        .session(&"jstprove-whir")
        .instance(Box::leak(Box::new(Empty)))
}

pub fn whir_commit_and_open(
    base_evals: &[u64],
    eval_point_ext2: &[(u64, u64)],
    num_vars: usize,
) -> Proof {
    let config = make_whir_config(num_vars);
    let ds = make_domain_separator(&config);
    let mut prover_state = ProverState::new_std(&ds);

    let whir_vector: Vec<Field64> = goldilocks_vec_to_whir(base_evals);

    let witness = config.commit(&mut prover_state, &[&whir_vector]);

    let whir_point: Vec<Field64_2> = eval_point_ext2
        .iter()
        .rev()
        .map(|&(c0, c1)| goldilocks_ext2_to_whir(c0, c1))
        .collect();

    let linear_form: WhirMLE<Field64_2> = WhirMLE {
        point: whir_point.clone(),
    };

    let eval_value: Field64_2 = linear_form.evaluate(config.embedding(), &whir_vector);

    let prove_forms: Vec<Box<dyn LinearForm<Field64_2>>> =
        vec![Box::new(WhirMLE { point: whir_point })];

    let _ = config.prove(
        &mut prover_state,
        vec![Cow::from(whir_vector)],
        vec![Cow::Owned(witness)],
        prove_forms,
        Cow::from(vec![eval_value]),
    );

    prover_state.proof()
}

pub fn whir_verify(
    eval_point_ext2: &[(u64, u64)],
    claimed_eval: (u64, u64),
    num_vars: usize,
    proof: &Proof,
) -> bool {
    let config = make_whir_config(num_vars);
    let ds = make_domain_separator(&config);

    let mut verifier_state = VerifierState::new_std(&ds, proof);

    let commitment = match config.receive_commitment(&mut verifier_state) {
        Ok(c) => c,
        Err(_) => return false,
    };

    let whir_eval = goldilocks_ext2_to_whir(claimed_eval.0, claimed_eval.1);

    let final_claim = match config.verify(&mut verifier_state, &[&commitment], &[whir_eval]) {
        Ok(fc) => fc,
        Err(_) => return false,
    };

    let whir_point: Vec<Field64_2> = eval_point_ext2
        .iter()
        .rev()
        .map(|&(c0, c1)| goldilocks_ext2_to_whir(c0, c1))
        .collect();

    let linear_form: WhirMLE<Field64_2> = WhirMLE { point: whir_point };

    let verify_forms: Vec<&dyn LinearForm<Field64_2>> = vec![&linear_form];
    final_claim.verify(verify_forms.into_iter()).is_ok()
}
