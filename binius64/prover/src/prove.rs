// Copyright 2025 Irreducible Inc.

use binius_core::{
    constraint_system::{AndConstraint, MulConstraint, ValueVec},
    verify::eval_operand,
    word::Word,
};
use binius_field::{
    AESTowerField8b as B8, BinaryField, PackedAESBinaryField16x8b, PackedExtension, PackedField,
    UnderlierWithBitOps, WithUnderlier,
};
use binius_iop_prover::{
    basefold_channel::BaseFoldProverChannel, basefold_compiler::BaseFoldProverCompiler,
    channel::IOPProverChannel,
};
use binius_math::{
    BinarySubspace, FieldBuffer,
    inner_product::inner_product,
    ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
    univariate::lagrange_evals,
};
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{SerializeBytes, checked_arithmetics::checked_log_2, rayon::prelude::*};
use binius_verifier::{
    Verifier,
    and_reduction::verifier::AndCheckOutput,
    config::{
        B1, B128, LOG_WORD_SIZE_BITS, LOG_WORDS_PER_ELEM, PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES,
    },
    protocols::{intmul::IntMulOutput, sumcheck::SumcheckOutput},
};
use digest::{Digest, FixedOutputReset, Output, core_api::BlockSizeUser};

use super::error::Error;
use crate::{
    and_reduction::{prover::OblongZerocheckProver, utils::multivariate::OneBitOblongMultilinear},
    hash::{ParallelDigest, parallel_compression::ParallelPseudoCompression},
    merkle_tree::prover::BinaryMerkleTreeProver,
    protocols::{
        intmul::{prove::IntMulProver, witness::Witness as IntMulWitness},
        shift::{
            KeyCollection, OperatorData, build_key_collection, prove as prove_shift_reduction,
        },
    },
    ring_switch,
};

/// Type alias for the prover NTT parameterized by field.
type ProverNTT<F> = NeighborsLastMultiThread<GenericPreExpanded<F>>;

/// Type alias for the prover Merkle tree prover parameterized by field.
type ProverMerkleProver<F, ParallelMerkleHasher, ParallelMerkleCompress> =
    BinaryMerkleTreeProver<F, ParallelMerkleHasher, ParallelMerkleCompress>;

/// Struct for proving instances of a particular constraint system.
///
/// The [`Self::setup`] constructor pre-processes reusable structures for proving instances of the
/// given constraint system. Then [`Self::prove`] is called one or more times with individual
/// instances.
#[derive(Debug)]
pub struct Prover<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
    P: PackedField<Scalar = B128>,
    ParallelMerkleHasher: ParallelDigest,
    ParallelMerkleHasher::Digest: Digest + BlockSizeUser + FixedOutputReset,
    ParallelMerkleCompress: ParallelPseudoCompression<Output<ParallelMerkleHasher::Digest>, 2>,
{
    key_collection: KeyCollection,
    verifier: Verifier<ParallelMerkleHasher::Digest, ParallelMerkleCompress::Compression>,
    #[allow(clippy::type_complexity)]
    basefold_compiler: BaseFoldProverCompiler<
        P,
        ProverNTT<B128>,
        ProverMerkleProver<B128, ParallelMerkleHasher, ParallelMerkleCompress>,
    >,
}

impl<P, MerkleHash, ParallelMerkleCompress, ParallelMerkleHasher>
    Prover<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
    P: PackedField<Scalar = B128>
        + PackedExtension<B128>
        + PackedExtension<B1>
        + WithUnderlier<Underlier: UnderlierWithBitOps>,
    MerkleHash: Digest + BlockSizeUser + FixedOutputReset,
    ParallelMerkleHasher: ParallelDigest<Digest = MerkleHash>,
    ParallelMerkleCompress: ParallelPseudoCompression<Output<MerkleHash>, 2>,
    Output<MerkleHash>: SerializeBytes,
{
    /// Constructs a prover corresponding to a constraint system verifier.
    ///
    /// See [`Prover`] struct documentation for details.
    pub fn setup(
        verifier: Verifier<MerkleHash, ParallelMerkleCompress::Compression>,
        compression: ParallelMerkleCompress,
    ) -> Result<Self, Error> {
        let key_collection = build_key_collection(verifier.constraint_system());
        Self::setup_with_key_collection(verifier, compression, key_collection)
    }

    /// Constructs a prover with a pre-built KeyCollection.
    ///
    /// This allows loading a previously serialized KeyCollection to avoid
    /// the expensive key building phase during setup.
    pub fn setup_with_key_collection(
        verifier: Verifier<MerkleHash, ParallelMerkleCompress::Compression>,
        compression: ParallelMerkleCompress,
        key_collection: KeyCollection,
    ) -> Result<Self, Error> {
        // Get max subspace from verifier's IOP compiler (reuses FRI params)
        let subspace = verifier.iop_compiler().max_subspace();
        let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
        // FIXME TODO For mobile phones, the number of shares should potentially be more than the
        // number of threads, because the threads/cores have different performance (but in the NTT
        // each share has the same amount of work)
        let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;
        let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

        let merkle_prover = BinaryMerkleTreeProver::<_, ParallelMerkleHasher, _>::new(compression);

        // Create prover compiler from verifier compiler (reuses FRI params and oracle specs)
        let basefold_compiler = BaseFoldProverCompiler::from_verifier_compiler(
            verifier.iop_compiler(),
            ntt,
            merkle_prover,
        );

        Ok(Prover {
            key_collection,
            verifier,
            basefold_compiler,
        })
    }

    /// Returns a reference to the KeyCollection.
    ///
    /// This can be used to serialize the KeyCollection for later use.
    pub fn key_collection(&self) -> &KeyCollection {
        &self.key_collection
    }

    pub fn prove<Challenger_: Challenger>(
        &self,
        witness: ValueVec,
        transcript: &mut ProverTranscript<Challenger_>,
    ) -> Result<(), Error> {
        let verifier = &self.verifier;

        // Check that the public input length is correct
        let public = witness.public().to_vec();
        if public.len() != 1 << self.verifier.log_public_words() {
            return Err(Error::ArgumentError {
                arg: "witness".to_string(),
                msg: format!(
                    "witness layout has {} words, expected {}",
                    public.len(),
                    1 << verifier.log_public_words()
                ),
            });
        }

        // Prover observes the public input (includes it in Fiat-Shamir).
        transcript.observe().write_slice(&public);

        // Create channel and delegate to prove_iop
        let channel = BaseFoldProverChannel::from_compiler(&self.basefold_compiler, transcript);
        self.prove_iop(witness, channel)
    }

    fn prove_iop<Channel>(&self, witness: ValueVec, mut channel: Channel) -> Result<(), Error>
    where
        Channel: IOPProverChannel<P>,
    {
        let verifier = &self.verifier;
        let cs = self.verifier.constraint_system();

        let _prove_guard = tracing::info_span!(
            "Prove",
            operation = "prove",
            perfetto_category = "operation",
            n_witness_words = cs.value_vec_layout.committed_total_len,
            n_bitand = cs.and_constraints.len(),
            n_intmul = cs.mul_constraints.len(),
        )
        .entered();

        // [phase] Setup - initialization and constraint system setup
        let setup_guard = tracing::info_span!(
            "[phase] Setup",
            phase = "setup",
            perfetto_category = "phase"
        )
        .entered();
        let witness_packed = pack_witness::<P>(verifier.log_witness_elems(), &witness)?;
        drop(setup_guard);

        // [phase] Witness Commit - witness generation and commitment
        let witness_commit_guard = tracing::info_span!(
            "[phase] Witness Commit",
            phase = "witness_commit",
            perfetto_category = "phase"
        )
        .entered();

        // Commit witness via channel
        let trace_oracle = channel.send_oracle(witness_packed.to_ref());

        drop(witness_commit_guard);

        // [phase] IntMul Reduction - multiplication constraint reduction
        let intmul_guard = tracing::info_span!(
            "[phase] IntMul Reduction",
            phase = "intmul_reduction",
            perfetto_category = "phase",
            n_constraints = cs.mul_constraints.len()
        )
        .entered();
        let mul_witness = build_intmul_witness(&cs.mul_constraints, &witness);
        let intmul_output = prove_intmul_reduction::<_, P, _>(mul_witness, &mut channel)?;
        drop(intmul_guard);

        // [phase] BitAnd Reduction - AND constraint reduction
        let bitand_guard = tracing::info_span!(
            "[phase] BitAnd Reduction",
            phase = "bitand_reduction",
            perfetto_category = "phase",
            n_constraints = cs.and_constraints.len()
        )
        .entered();
        let bitand_claim = {
            let bitand_witness = build_bitand_witness(&cs.and_constraints, &witness);
            let AndCheckOutput {
                a_eval,
                b_eval,
                c_eval,
                z_challenge,
                eval_point,
            } = prove_bitand_reduction::<B128, _>(bitand_witness, &mut channel)?;
            OperatorData {
                evals: vec![a_eval, b_eval, c_eval],
                r_zhat_prime: z_challenge,
                r_x_prime: eval_point,
            }
        };
        drop(bitand_guard);

        // Build `OperatorData` for IntMul using the same `r_zhat_prime`
        // challenge as in BitAnd. Sharing this univariate challenge
        // improves ShiftReduction perf.
        let intmul_claim = {
            let IntMulOutput {
                eval_point,
                a_evals,
                b_evals,
                c_lo_evals,
                c_hi_evals,
            } = intmul_output;

            let r_zhat_prime = bitand_claim.r_zhat_prime;
            let subspace = BinarySubspace::<B8>::with_dim(LOG_WORD_SIZE_BITS).isomorphic();
            let l_tilde = lagrange_evals(&subspace, r_zhat_prime);
            let make_final_claim = |evals| inner_product(evals, l_tilde.iter_scalars());
            OperatorData {
                evals: vec![
                    make_final_claim(a_evals),
                    make_final_claim(b_evals),
                    make_final_claim(c_lo_evals),
                    make_final_claim(c_hi_evals),
                ],
                r_zhat_prime,
                r_x_prime: eval_point,
            }
        };

        // [phase] Shift Reduction - shift operations
        let shift_guard = tracing::info_span!(
            "[phase] Shift Reduction",
            phase = "shift_reduction",
            perfetto_category = "phase"
        )
        .entered();
        let SumcheckOutput {
            challenges: eval_point,
            eval: _,
        } = prove_shift_reduction::<_, P, _>(
            verifier.log_public_words(),
            &self.key_collection,
            witness.combined_witness(),
            bitand_claim,
            intmul_claim,
            &mut channel,
        )?;
        drop(shift_guard);

        // [phase] Ring-Switching + PCS Opening
        let pcs_guard = tracing::info_span!(
            "[phase] PCS Opening",
            phase = "pcs_opening",
            perfetto_category = "phase"
        )
        .entered();

        // Ring-switching reduction
        let ring_switch::RingSwitchOutput {
            rs_eq_ind,
            sumcheck_claim,
        } = ring_switch::prove(&witness_packed, &eval_point, &mut channel);

        // Prove oracle relations via channel (runs BaseFold internally)
        channel.prove_oracle_relations([(trace_oracle, rs_eq_ind, sumcheck_claim)]);

        drop(pcs_guard);

        Ok(())
    }
}

fn pack_witness<P: PackedField<Scalar = B128>>(
    log_witness_elems: usize,
    witness: &ValueVec,
) -> Result<FieldBuffer<P>, Error> {
    // The number of field elements that constitute the packed witness.
    let n_witness_elems = witness.size().div_ceil(1 << LOG_WORDS_PER_ELEM);
    if n_witness_elems > 1 << log_witness_elems {
        return Err(Error::ArgumentError {
            arg: "witness".to_string(),
            msg: "witness element count is incompatible with the constraint system".to_string(),
        });
    }

    let len = 1 << log_witness_elems.saturating_sub(P::LOG_WIDTH);
    let mut padded_witness_elems = Vec::<P>::with_capacity(len);

    let combined_witness = witness.combined_witness();
    padded_witness_elems
        .spare_capacity_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, dst)| {
            // Pack B128 elements into packed elements
            let offset = i << (P::LOG_WIDTH + 1);
            let value = P::from_fn(|j| {
                let word_0 = combined_witness[offset + 2 * j];
                let word_1 = combined_witness[offset + 2 * j + 1];
                B128::new(((word_1.0 as u128) << 64) | (word_0.0 as u128))
            });

            dst.write(value);
        });

    // SAFETY: We just initialized all elements
    unsafe {
        padded_witness_elems.set_len(len);
    };

    let padded_witness_elems =
        FieldBuffer::new(log_witness_elems, padded_witness_elems.into_boxed_slice());
    Ok(padded_witness_elems)
}

fn prove_bitand_reduction<F, Channel>(
    witness: AndCheckWitness,
    channel: &mut Channel,
) -> Result<AndCheckOutput<F>, Error>
where
    F: BinaryField + From<B8>,
    Channel: binius_ip_prover::channel::IPProverChannel<F>,
{
    let prover_message_domain = BinarySubspace::<B8>::with_dim(LOG_WORD_SIZE_BITS + 1);
    let AndCheckWitness {
        mut a,
        mut b,
        mut c,
    } = witness;

    let log_constraint_count = checked_log_2(a.len());

    // The structure of the AND reduction requires that it proves at least 2^3 word-level
    // constraints, you can zero-pad if necessary to reach this minimum
    assert!(log_constraint_count >= checked_log_2(binius_core::consts::MIN_AND_CONSTRAINTS));

    let big_field_zerocheck_challenges = channel.sample_many(log_constraint_count - 3);

    a.resize(1 << log_constraint_count, Word(0));
    b.resize(1 << log_constraint_count, Word(0));
    c.resize(1 << log_constraint_count, Word(0));

    let prover = OblongZerocheckProver::<_, PackedAESBinaryField16x8b>::new(
        OneBitOblongMultilinear {
            log_num_rows: log_constraint_count + LOG_WORD_SIZE_BITS,
            packed_evals: a,
        },
        OneBitOblongMultilinear {
            log_num_rows: log_constraint_count + LOG_WORD_SIZE_BITS,
            packed_evals: b,
        },
        OneBitOblongMultilinear {
            log_num_rows: log_constraint_count + LOG_WORD_SIZE_BITS,
            packed_evals: c,
        },
        big_field_zerocheck_challenges.to_vec(),
        PROVER_SMALL_FIELD_ZEROCHECK_CHALLENGES.to_vec(),
        prover_message_domain.isomorphic(),
    );

    Ok(prover.prove_with_channel(channel)?)
}

fn prove_intmul_reduction<F, P, Channel>(
    witness: MulCheckWitness,
    channel: &mut Channel,
) -> Result<IntMulOutput<F>, Error>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    Channel: binius_ip_prover::channel::IPProverChannel<F>,
{
    let MulCheckWitness { a, b, lo, hi } = witness;

    let mut mulcheck_prover = IntMulProver::new(0, channel);

    // Words must be converted to u64 because
    // `Bitwise` requires `From<u8>` and `Shr<usize>`
    // We could implement these for `Word` in the future.
    let convert_to_u64 = |w: Vec<Word>| w.into_iter().map(|w| w.0).collect::<Vec<u64>>();
    let [a_u64, b_u64, lo_u64, hi_u64] = [a, b, lo, hi].map(convert_to_u64);
    let intmul_witness =
        IntMulWitness::<P, _, _>::new(LOG_WORD_SIZE_BITS, &a_u64, &b_u64, &lo_u64, &hi_u64)?;

    Ok(mulcheck_prover.prove(intmul_witness)?)
}

struct AndCheckWitness {
    a: Vec<Word>,
    b: Vec<Word>,
    c: Vec<Word>,
}

struct MulCheckWitness {
    a: Vec<Word>,
    b: Vec<Word>,
    lo: Vec<Word>,
    hi: Vec<Word>,
}

#[tracing::instrument(skip_all, "Build BitAnd witness", level = "debug")]
fn build_bitand_witness(and_constraints: &[AndConstraint], witness: &ValueVec) -> AndCheckWitness {
    let n_constraints = and_constraints.len();

    let mut a = Vec::with_capacity(n_constraints);
    let mut b = Vec::with_capacity(n_constraints);
    let mut c = Vec::with_capacity(n_constraints);

    (
        and_constraints,
        a.spare_capacity_mut(),
        b.spare_capacity_mut(),
        c.spare_capacity_mut(),
    )
        .into_par_iter()
        .for_each(|(constraint, a_i, b_i, c_i)| {
            a_i.write(eval_operand(witness, &constraint.a));
            b_i.write(eval_operand(witness, &constraint.b));
            c_i.write(eval_operand(witness, &constraint.c));
        });

    // Safety: all entries in a, b, c are initialized in the parallel loop above.
    unsafe {
        a.set_len(n_constraints);
        b.set_len(n_constraints);
        c.set_len(n_constraints);
    }

    AndCheckWitness { a, b, c }
}

#[tracing::instrument(skip_all, "Build IntMul witness", level = "debug")]
fn build_intmul_witness(mul_constraints: &[MulConstraint], witness: &ValueVec) -> MulCheckWitness {
    let n_constraints = mul_constraints.len();

    let mut a = Vec::with_capacity(n_constraints);
    let mut b = Vec::with_capacity(n_constraints);
    let mut lo = Vec::with_capacity(n_constraints);
    let mut hi = Vec::with_capacity(n_constraints);

    (
        mul_constraints,
        a.spare_capacity_mut(),
        b.spare_capacity_mut(),
        lo.spare_capacity_mut(),
        hi.spare_capacity_mut(),
    )
        .into_par_iter()
        .for_each(|(constraint, a_i, b_i, lo_i, hi_i)| {
            a_i.write(eval_operand(witness, &constraint.a));
            b_i.write(eval_operand(witness, &constraint.b));
            lo_i.write(eval_operand(witness, &constraint.lo));
            hi_i.write(eval_operand(witness, &constraint.hi));
        });

    // Safety: all entries in a, b, lo, hi are initialized in the parallel loop above.
    unsafe {
        a.set_len(n_constraints);
        b.set_len(n_constraints);
        lo.set_len(n_constraints);
        hi.set_len(n_constraints);
    }

    MulCheckWitness { a, b, lo, hi }
}
