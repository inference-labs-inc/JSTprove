// Copyright 2025 Irreducible Inc.

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, BinaryField, ExtensionField};
use binius_iop::{
    basefold_compiler::BaseFoldVerifierCompiler,
    channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
};
use binius_ip::channel::IPVerifierChannel;
use binius_math::{
    BinarySubspace,
    inner_product::inner_product,
    ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
    univariate::lagrange_evals,
};
use binius_transcript::{VerifierTranscript, fiat_shamir::Challenger};
use binius_utils::{
    DeserializeBytes,
    checked_arithmetics::{checked_log_2, log2_ceil_usize},
};
use digest::{Digest, Output, core_api::BlockSizeUser};
use itertools::{Itertools, chain};

use super::error::Error;
use crate::{
    and_reduction::verifier::{AndCheckOutput, verify_with_channel},
    config::{
        B1, B128, LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES,
    },
    fri::{ConstantArityStrategy, FRIParams, calculate_n_test_queries},
    hash::PseudoCompressionFunction,
    merkle_tree::BinaryMerkleTreeScheme,
    protocols::{
        intmul::{IntMulOutput, verify as verify_intmul_reduction},
        shift::{self, OperatorData},
    },
    ring_switch,
};

pub const SECURITY_BITS: usize = 96;

/// Struct for verifying instances of a particular constraint system.
///
/// The [`Self::setup`] constructor determines public parameters for proving instances of the given
/// constraint system. Then [`Self::verify`] is called one or more times with individual instances.
#[derive(Debug, Clone)]
pub struct Verifier<MerkleHash, MerkleCompress>
where
    MerkleHash: Digest + BlockSizeUser,
    MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
{
    constraint_system: ConstraintSystem,
    iop_compiler:
        BaseFoldVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>>,
    log_public_words: usize,
}

impl<MerkleHash, MerkleCompress> Verifier<MerkleHash, MerkleCompress>
where
    MerkleHash: Digest + BlockSizeUser,
    MerkleCompress: PseudoCompressionFunction<Output<MerkleHash>, 2>,
    Output<MerkleHash>: DeserializeBytes,
{
    /// Constructs a verifier for a constraint system.
    ///
    /// See [`Verifier`] struct documentation for details.
    pub fn setup(
        mut constraint_system: ConstraintSystem,
        log_inv_rate: usize,
        compression: MerkleCompress,
    ) -> Result<Self, Error> {
        constraint_system.validate_and_prepare()?;

        // Use offset_witness which is guaranteed to be power of two and be at least one full
        // element.
        let n_public = constraint_system.value_vec_layout.offset_witness;
        let log_public_words = log2_ceil_usize(n_public);
        assert!(n_public.is_power_of_two());
        assert!(log_public_words >= LOG_WORDS_PER_ELEM);

        // The number of field elements that constitute the packed witness.
        let log_witness_words =
            log2_ceil_usize(constraint_system.value_vec_len()).max(LOG_WORDS_PER_ELEM);
        let log_witness_elems = log_witness_words - LOG_WORDS_PER_ELEM;

        let log_code_len = log_witness_elems + log_inv_rate;
        let merkle_scheme = BinaryMerkleTreeScheme::new(compression);
        let fri_arity =
            ConstantArityStrategy::with_optimal_arity::<B128, _>(&merkle_scheme, log_code_len)
                .arity;

        let subspace = BinarySubspace::with_dim(log_code_len);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
        let ntt = NeighborsLastSingleThread::new(domain_context);
        let n_test_queries = calculate_n_test_queries(SECURITY_BITS, log_inv_rate);

        // Create oracle spec for the single witness oracle (not ZK)
        let oracle_specs = vec![OracleSpec {
            log_msg_len: log_witness_elems,
        }];

        let iop_compiler = BaseFoldVerifierCompiler::new(
            &ntt,
            merkle_scheme,
            oracle_specs,
            log_inv_rate,
            n_test_queries,
            &ConstantArityStrategy::new(fri_arity),
        );

        Ok(Self {
            constraint_system,
            iop_compiler,
            log_public_words,
        })
    }

    /// Returns log2 of the number of words in the witness.
    pub fn log_witness_words(&self) -> usize {
        self.log_witness_elems() + LOG_WORDS_PER_ELEM
    }

    /// Returns log2 of the number of field elements in the packed trace.
    pub fn log_witness_elems(&self) -> usize {
        let fri_params = self.fri_params();
        let rs_code = fri_params.rs_code();
        rs_code.log_dim() + fri_params.log_batch_size()
    }

    /// Returns the constraint system.
    pub fn constraint_system(&self) -> &ConstraintSystem {
        &self.constraint_system
    }

    /// Returns the chosen FRI parameters.
    pub fn fri_params(&self) -> &FRIParams<B128> {
        // There is exactly one oracle spec (the witness)
        &self.iop_compiler.fri_params()[0]
    }

    /// Returns the [`crate::merkle_tree::MerkleTreeScheme`] instance used.
    pub fn merkle_scheme(&self) -> &BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress> {
        self.iop_compiler.merkle_scheme()
    }

    /// Returns log2 of the number of public constants and input/output words.
    pub fn log_public_words(&self) -> usize {
        self.log_public_words
    }

    /// Returns the IOP compiler for creating verifier channels.
    pub fn iop_compiler(
        &self,
    ) -> &BaseFoldVerifierCompiler<B128, BinaryMerkleTreeScheme<B128, MerkleHash, MerkleCompress>>
    {
        &self.iop_compiler
    }

    pub fn verify<Challenger_: Challenger>(
        &self,
        public: &[Word],
        transcript: &mut VerifierTranscript<Challenger_>,
    ) -> Result<(), Error> {
        // Check that the public input length is correct
        if public.len() != 1 << self.log_public_words() {
            return Err(Error::IncorrectPublicInputLength {
                expected: 1 << self.log_public_words(),
                actual: public.len(),
            });
        }

        // Verifier observes the public input (includes it in Fiat-Shamir).
        transcript.observe().write_slice(public);

        // Create channel and delegate to verify_iop
        let mut channel = self.iop_compiler.create_channel(transcript);
        self.verify_iop(public, &mut channel)
    }

    fn verify_iop<Channel>(&self, public: &[Word], channel: &mut Channel) -> Result<(), Error>
    where
        Channel: IOPVerifierChannel<B128, Elem = B128>,
    {
        let _verify_guard = tracing::info_span!(
            "Verify",
            operation = "verify",
            perfetto_category = "operation"
        )
        .entered();

        let subfield_subspace = BinarySubspace::<B8>::default().isomorphic();
        let extended_subspace = subfield_subspace.reduce_dim(LOG_WORD_SIZE_BITS + 1);
        let domain_subspace = extended_subspace.reduce_dim(LOG_WORD_SIZE_BITS);

        // Receive the trace oracle commitment via channel.
        let trace_oracle = channel.recv_oracle()?;

        // [phase] Verify IntMul Reduction - multiplication constraint verification
        let intmul_guard = tracing::info_span!(
            "[phase] Verify IntMul Reduction",
            phase = "verify_intmul_reduction",
            perfetto_category = "phase",
            n_constraints = self.constraint_system.n_mul_constraints()
        )
        .entered();
        let log_n_constraints = checked_log_2(self.constraint_system.n_mul_constraints());
        let intmul_output =
            verify_intmul_reduction::<B128, _>(LOG_WORD_SIZE_BITS, log_n_constraints, channel)?;
        drop(intmul_guard);

        // [phase] Verify BitAnd Reduction - AND constraint verification
        let bitand_guard = tracing::info_span!(
            "[phase] Verify BitAnd Reduction",
            phase = "verify_bitand_reduction",
            perfetto_category = "phase",
            n_constraints = self.constraint_system.n_and_constraints()
        )
        .entered();
        let bitand_claim = {
            let log_n_constraints = checked_log_2(self.constraint_system.n_and_constraints());
            let AndCheckOutput {
                a_eval,
                b_eval,
                c_eval,
                z_challenge,
                eval_point,
            }: AndCheckOutput<B128> =
                verify_bitand_reduction(log_n_constraints, &extended_subspace, channel)?;
            OperatorData::new(z_challenge, eval_point, [a_eval, b_eval, c_eval])
        };
        drop(bitand_guard);

        // Build `OperatorData` for IntMul using the same `r_zhat_prime`
        // challenge as in BitAnd. Sharing this univariate challenge
        // improves prover ShiftReduction perf.
        let intmul_claim = {
            let IntMulOutput {
                a_evals,
                b_evals,
                c_lo_evals,
                c_hi_evals,
                eval_point,
            } = intmul_output;

            let r_zhat_prime = bitand_claim.r_zhat_prime;
            let l_tilde = lagrange_evals(&domain_subspace, r_zhat_prime);
            let make_final_claim = |evals| inner_product(evals, l_tilde.iter_scalars());
            OperatorData::new(
                r_zhat_prime,
                eval_point,
                [
                    make_final_claim(a_evals),
                    make_final_claim(b_evals),
                    make_final_claim(c_lo_evals),
                    make_final_claim(c_hi_evals),
                ],
            )
        };

        // [phase] Verify Shift Reduction - shift operations and constraint validation
        let constraint_guard = tracing::info_span!(
            "[phase] Verify Shift Reduction",
            phase = "verify_shift_reduction",
            perfetto_category = "phase"
        )
        .entered();
        let shift_output = shift::verify(
            self.constraint_system(),
            public,
            &bitand_claim,
            &intmul_claim,
            channel,
        )?;
        drop(constraint_guard);

        // [phase] Verify Public Input - public input verification
        let public_guard = tracing::info_span!(
            "[phase] Verify Public Input",
            phase = "verify_public_input",
            perfetto_category = "phase"
        )
        .entered();
        shift::check_eval(
            self.constraint_system(),
            &bitand_claim,
            &intmul_claim,
            &domain_subspace,
            &shift_output,
        )?;
        drop(public_guard);

        // [phase] Ring-Switching + Verify PCS Opening
        let pcs_guard = tracing::info_span!(
            "[phase] Verify PCS Opening",
            phase = "verify_pcs_opening",
            perfetto_category = "phase"
        )
        .entered();

        // Ring-switching verification
        let eval_point = [shift_output.r_j(), shift_output.r_y()].concat();
        let ring_switch::RingSwitchVerifyOutput {
            eq_r_double_prime,
            sumcheck_claim,
        } = ring_switch::verify(shift_output.witness_eval(), &eval_point, channel)?;

        // Build the transparent closure for the ring-switching equality indicator
        let log_packing = <B128 as ExtensionField<B1>>::LOG_DEGREE;
        let eval_point_high = eval_point[log_packing..].to_vec();
        let transparent = Box::new(move |point: &[B128]| {
            ring_switch::eval_rs_eq(&eval_point_high, point, eq_r_double_prime.as_ref())
        });

        // Verify oracle relations (runs BaseFold internally and verifies the product check)
        channel.verify_oracle_relations([OracleLinearRelation {
            oracle: trace_oracle,
            transparent,
            claim: sumcheck_claim,
        }])?;

        drop(pcs_guard);

        Ok(())
    }
}

fn verify_bitand_reduction<F, C>(
    log_constraint_count: usize,
    eval_domain: &BinarySubspace<F>,
    channel: &mut C,
) -> Result<AndCheckOutput<C::Elem>, Error>
where
    F: BinaryField + From<B8>,
    C: IPVerifierChannel<F>,
    // Used to make deterministic basis challenges symbolic
    C::Elem: From<F>,
{
    // The structure of the AND reduction requires that it verifies at least 2^3 word-level
    // constraints, you can zero-pad if necessary to reach this minimum
    assert!(log_constraint_count >= checked_log_2(binius_core::consts::MIN_AND_CONSTRAINTS));

    let big_field_zerocheck_challenges = channel.sample_many(log_constraint_count - 3);

    let small_field_zerocheck_challenges = PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES
        .into_iter()
        .map(|b8_val| C::Elem::from(F::from(b8_val)))
        .collect_vec();

    let zerocheck_challenges = chain!(
        small_field_zerocheck_challenges,
        big_field_zerocheck_challenges
    )
    .collect::<Vec<_>>();
    verify_with_channel(&zerocheck_challenges, channel, eval_domain)
}
