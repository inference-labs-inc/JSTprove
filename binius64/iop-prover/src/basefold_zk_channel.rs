// Copyright 2026 The Binius Developers

//! BaseFold ZK implementation of the IOP prover channel.
//!
//! This module provides [`BaseFoldZKProverChannel`], which implements [`IOPProverChannel`]
//! using FRI commitment and ZK BaseFold opening protocols. Unlike [`super::basefold_channel`],
//! this channel always applies zero-knowledge blinding to all oracles by generating masks
//! internally.

use binius_field::{BinaryField, PackedField};
use binius_iop::{channel::OracleSpec, fri::FRIParams, merkle_tree::MerkleTreeScheme};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT};
use binius_transcript::{
    ProverTranscript,
    fiat_shamir::{CanSample, Challenger},
};
use binius_utils::SerializeBytes;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    basefold,
    basefold_compiler::BaseFoldZKProverCompiler,
    channel::IOPProverChannel,
    fri::{self, CommitMaskedOutput, FRIFoldProver},
    merkle_tree::MerkleTreeProver,
};

/// Oracle handle returned by [`BaseFoldZKProverChannel::send_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldZKOracle {
    index: usize,
}

/// Committed oracle data stored internally.
struct CommittedOracleData<P: PackedField, Committed> {
    /// The combined (witness || mask) buffer.
    combined: FieldBuffer<P>,
    /// RS-encoded codeword.
    codeword: FieldBuffer<P>,
    /// Merkle commitment data for query proofs.
    committed: Committed,
}

/// A prover channel that uses ZK BaseFold for all oracle commitments and openings.
///
/// This channel owns an [`StdRng`] and generates random masks internally during
/// [`send_oracle`](IOPProverChannel::send_oracle). The caller provides only the raw witness
/// buffer (not doubled). The channel handles:
/// - Generating a random mask of equal length
/// - Interleaving witness and mask for FRI commitment
/// - Running ZK BaseFold proofs in `prove_oracle_relations`
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `P`: The packed field type with `Scalar = F`
/// - `NTT`: The additive NTT for Reed-Solomon encoding
/// - `MerkleProver_`: The Merkle tree prover for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleProver_: MerkleTreeProver<F>,
    Challenger_: Challenger,
{
    transcript: &'a mut ProverTranscript<Challenger_>,
    ntt: &'a NTT,
    merkle_prover: &'a MerkleProver_,
    oracle_specs: Vec<OracleSpec>,
    fri_params: Vec<FRIParams<F>>,
    committed_oracles: Vec<CommittedOracleData<P, MerkleProver_::Committed>>,
    next_oracle_index: usize,
    rng: StdRng,
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_>
    BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
    MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
    Challenger_: Challenger,
{
    /// Creates a new BaseFold ZK prover channel from a compiler with precomputed FRI parameters.
    ///
    /// The `rng` is used to seed an internal `StdRng` for mask generation.
    pub fn from_compiler(
        compiler: &'a BaseFoldZKProverCompiler<P, NTT, MerkleProver_>,
        transcript: &'a mut ProverTranscript<Challenger_>,
        mut rng: impl Rng,
    ) -> Self {
        Self {
            transcript,
            ntt: compiler.ntt(),
            merkle_prover: compiler.merkle_prover(),
            oracle_specs: compiler.oracle_specs().to_vec(),
            fri_params: compiler.fri_params().to_vec(),
            committed_oracles: Vec::new(),
            next_oracle_index: 0,
            rng: StdRng::from_rng(&mut rng),
        }
    }

    /// Returns a reference to the underlying transcript.
    pub fn transcript(&self) -> &ProverTranscript<Challenger_> {
        self.transcript
    }
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_> IPProverChannel<F>
    for BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
    MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
    Challenger_: Challenger,
{
    fn send_one(&mut self, elem: F) {
        self.transcript.message().write_scalar(elem);
    }

    fn send_many(&mut self, elems: &[F]) {
        self.transcript.message().write_scalar_slice(elems);
    }

    fn observe_one(&mut self, val: F) {
        self.transcript.observe().write_scalar(val);
    }

    fn observe_many(&mut self, vals: &[F]) {
        self.transcript.observe().write_scalar_slice(vals);
    }

    fn sample(&mut self) -> F {
        CanSample::sample(&mut self.transcript)
    }
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_> IOPProverChannel<P>
    for BaseFoldZKProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
    MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
    Challenger_: Challenger,
{
    type Oracle = BaseFoldZKOracle;

    fn remaining_oracle_specs(&self) -> &[OracleSpec] {
        &self.oracle_specs[self.next_oracle_index..]
    }

    fn send_oracle(&mut self, buffer: FieldSlice<P>) -> Self::Oracle {
        let remaining = self.remaining_oracle_specs();
        assert!(
            !remaining.is_empty(),
            "send_oracle called but no remaining oracle specs"
        );

        let index = self.next_oracle_index;
        let spec = &remaining[0];
        let fri_params = &self.fri_params[index];

        // ZK channel expects raw witness buffer (NOT doubled).
        assert_eq!(
            buffer.log_len(),
            spec.log_msg_len,
            "oracle buffer log_len mismatch: expected {}, got {}",
            spec.log_msg_len,
            buffer.log_len()
        );

        // Copy the message for later use in prove_zk (commit_masked consumes buffer).
        let log_len = buffer.log_len();
        let message_values: Box<[P]> = buffer.as_ref().into();

        // Generate mask, interleave, and commit via commit_masked.
        let CommitMaskedOutput {
            commitment,
            committed,
            codeword,
            mask,
        } = fri::commit_masked(
            fri_params,
            self.ntt,
            self.merkle_prover,
            buffer,
            &mut self.rng,
        )
        .expect("FRI commit_masked should succeed with valid params");

        // Build the combined (witness || mask) buffer for later use in prove_zk.
        let mut combined_values = Vec::with_capacity(message_values.len() * 2);
        combined_values.extend_from_slice(&message_values);
        combined_values.extend_from_slice(mask.as_ref());
        let combined = FieldBuffer::new(log_len + 1, combined_values.into_boxed_slice());

        // Send commitment via transcript.
        self.transcript.message().write(&commitment);

        self.committed_oracles.push(CommittedOracleData {
            combined,
            codeword,
            committed,
        });

        self.next_oracle_index += 1;

        BaseFoldZKOracle { index }
    }

    fn prove_oracle_relations(
        &mut self,
        oracle_relations: impl IntoIterator<Item = (Self::Oracle, FieldBuffer<P>, P::Scalar)>,
    ) {
        assert!(
            self.remaining_oracle_specs().is_empty(),
            "prove_oracle_relations called but {} oracle specs remaining",
            self.remaining_oracle_specs().len()
        );

        for (oracle, transparent_poly, eval_claim) in oracle_relations {
            let index = oracle.index;
            assert!(
                index < self.committed_oracles.len(),
                "oracle index {index} out of bounds, expected < {}",
                self.committed_oracles.len()
            );

            let fri_params = &self.fri_params[index];
            let committed_data = &self.committed_oracles[index];

            let fri_folder = FRIFoldProver::new(
                fri_params,
                self.ntt,
                self.merkle_prover,
                committed_data.codeword.clone(),
                &committed_data.committed,
            )
            .expect("FRI folder creation should succeed");

            // Always use ZK variant.
            let prover = basefold::prove_zk(
                committed_data.combined.clone(),
                transparent_poly,
                eval_claim,
                fri_folder,
                self.transcript,
            );
            prover
                .prove(self.transcript)
                .expect("BaseFold ZK proof should succeed");
        }
    }
}

#[cfg(test)]
mod tests {
    use binius_field::{BinaryField, BinaryField128bGhash, PackedBinaryGhash1x128b, PackedField};
    use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
    use binius_iop::{
        basefold_compiler::BaseFoldZKVerifierCompiler,
        channel::{IOPVerifierChannel, OracleLinearRelation, OracleSpec},
        fri::MinProofSizeStrategy,
    };
    use binius_math::{
        BinarySubspace, FieldBuffer,
        inner_product::inner_product_buffers,
        multilinear::eq::eq_ind_partial_eval,
        ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
        test_utils::{random_field_buffer, random_scalars},
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::IOPProverChannel;
    use crate::{
        basefold_compiler::BaseFoldZKProverCompiler, merkle_tree::prover::BinaryMerkleTreeProver,
    };

    type StdChallenger = HasherChallenger<StdDigest>;

    const LOG_INV_RATE: usize = 1;
    const SECURITY_BITS: usize = 32;

    fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
        security_bits.div_ceil(log_inv_rate)
    }

    fn make_ntt(
        log_len: usize,
    ) -> NeighborsLastSingleThread<GenericOnTheFly<BinaryField128bGhash>> {
        let subspace = BinarySubspace::with_dim(log_len);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
        NeighborsLastSingleThread::new(domain_context)
    }

    fn make_merkle_prover() -> BinaryMerkleTreeProver<
        BinaryField128bGhash,
        StdDigest,
        ParallelCompressionAdaptor<StdCompression>,
    > {
        BinaryMerkleTreeProver::new(ParallelCompressionAdaptor::new(StdCompression::default()))
    }

    fn generate_zk_oracle_data<F, P, R: Rng>(
        rng: &mut R,
        n_vars: usize,
    ) -> (FieldBuffer<P>, FieldBuffer<P>, F)
    where
        F: BinaryField,
        P: PackedField<Scalar = F>,
    {
        let buffer = random_field_buffer::<P>(&mut *rng, n_vars);
        let evaluation_point = random_scalars::<F>(&mut *rng, n_vars);
        let transparent_poly = eq_ind_partial_eval::<P>(&evaluation_point);
        let evaluation_claim = inner_product_buffers(&buffer, &transparent_poly);
        (buffer, transparent_poly, evaluation_claim)
    }

    #[test]
    fn test_basefold_zk_channel_single_oracle() {
        type F = BinaryField128bGhash;
        type P = PackedBinaryGhash1x128b;

        let mut rng = StdRng::seed_from_u64(0);
        let n_vars = 8;

        let (buffer, transparent_poly, eval_claim) =
            generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars);

        let max_codeword_log_len = (n_vars + 1) + LOG_INV_RATE;
        let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

        let oracle_specs = vec![OracleSpec {
            log_msg_len: n_vars,
        }];

        // === PROVER SIDE ===
        let prover_compiler = BaseFoldZKProverCompiler::<P, _, _>::new(
            make_ntt(max_codeword_log_len),
            make_merkle_prover(),
            oracle_specs.clone(),
            LOG_INV_RATE,
            n_test_queries,
        );

        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let prover_rng = StdRng::seed_from_u64(1);
        let mut prover_channel = prover_compiler.create_channel(&mut prover_transcript, prover_rng);

        let oracle = prover_channel.send_oracle(buffer.to_ref());
        assert_eq!(oracle.index, 0);

        prover_channel.prove_oracle_relations([(oracle, transparent_poly.clone(), eval_claim)]);

        // === VERIFIER SIDE ===
        let ntt = make_ntt(max_codeword_log_len);
        let merkle_prover = make_merkle_prover();
        let verifier_compiler = BaseFoldZKVerifierCompiler::new(
            &ntt,
            merkle_prover.scheme().clone(),
            oracle_specs,
            LOG_INV_RATE,
            n_test_queries,
            &MinProofSizeStrategy,
        );

        let mut verifier_transcript = prover_transcript.into_verifier();
        let mut verifier_channel = verifier_compiler.create_channel(&mut verifier_transcript);

        let v_oracle = verifier_channel.recv_oracle().unwrap();

        verifier_channel
            .verify_oracle_relations([OracleLinearRelation {
                oracle: v_oracle,
                transparent: Box::new(move |point: &[F]| {
                    let eq = eq_ind_partial_eval::<P>(point);
                    inner_product_buffers(&transparent_poly, &eq)
                }),
                claim: eval_claim,
            }])
            .unwrap();
    }

    #[test]
    fn test_basefold_zk_channel_two_oracles() {
        type F = BinaryField128bGhash;
        type P = PackedBinaryGhash1x128b;

        let mut rng = StdRng::seed_from_u64(0);
        let n_vars_1 = 6;
        let n_vars_2 = 8;

        let (buffer_1, transparent_poly_1, eval_claim_1) =
            generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_1);
        let (buffer_2, transparent_poly_2, eval_claim_2) =
            generate_zk_oracle_data::<F, P, _>(&mut rng, n_vars_2);

        let max_codeword_log_len = (n_vars_2 + 1) + LOG_INV_RATE;
        let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

        let oracle_specs = vec![
            OracleSpec {
                log_msg_len: n_vars_1,
            },
            OracleSpec {
                log_msg_len: n_vars_2,
            },
        ];

        // === PROVER SIDE ===
        let prover_compiler = BaseFoldZKProverCompiler::<P, _, _>::new(
            make_ntt(max_codeword_log_len),
            make_merkle_prover(),
            oracle_specs.clone(),
            LOG_INV_RATE,
            n_test_queries,
        );

        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let prover_rng = StdRng::seed_from_u64(1);
        let mut prover_channel = prover_compiler.create_channel(&mut prover_transcript, prover_rng);

        let oracle_1 = prover_channel.send_oracle(buffer_1.to_ref());
        let oracle_2 = prover_channel.send_oracle(buffer_2.to_ref());

        prover_channel.prove_oracle_relations([
            (oracle_1, transparent_poly_1.clone(), eval_claim_1),
            (oracle_2, transparent_poly_2.clone(), eval_claim_2),
        ]);

        // === VERIFIER SIDE ===
        let ntt = make_ntt(max_codeword_log_len);
        let merkle_prover = make_merkle_prover();
        let verifier_compiler = BaseFoldZKVerifierCompiler::new(
            &ntt,
            merkle_prover.scheme().clone(),
            oracle_specs,
            LOG_INV_RATE,
            n_test_queries,
            &MinProofSizeStrategy,
        );

        let mut verifier_transcript = prover_transcript.into_verifier();
        let mut verifier_channel = verifier_compiler.create_channel(&mut verifier_transcript);

        let v_oracle_1 = verifier_channel.recv_oracle().unwrap();
        let v_oracle_2 = verifier_channel.recv_oracle().unwrap();

        let tp1 = transparent_poly_1.clone();
        let tp2 = transparent_poly_2.clone();

        verifier_channel
            .verify_oracle_relations([
                OracleLinearRelation {
                    oracle: v_oracle_1,
                    transparent: Box::new(move |point: &[F]| {
                        let eq = eq_ind_partial_eval::<P>(point);
                        inner_product_buffers(&tp1, &eq)
                    }),
                    claim: eval_claim_1,
                },
                OracleLinearRelation {
                    oracle: v_oracle_2,
                    transparent: Box::new(move |point: &[F]| {
                        let eq = eq_ind_partial_eval::<P>(point);
                        inner_product_buffers(&tp2, &eq)
                    }),
                    claim: eval_claim_2,
                },
            ])
            .unwrap();
    }
}
