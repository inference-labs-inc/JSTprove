// Copyright 2026 The Binius Developers

//! BaseFold-based implementation of the IOP prover channel.
//!
//! This module provides [`BaseFoldProverChannel`], which implements [`IOPProverChannel`] using
//! FRI commitment and BaseFold opening protocols.

use binius_field::{BinaryField, PackedField};
use binius_iop::{channel::OracleSpec, fri::FRIParams, merkle_tree::MerkleTreeScheme};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{FieldBuffer, FieldSlice, ntt::AdditiveNTT};
use binius_transcript::{
    ProverTranscript,
    fiat_shamir::{CanSample, Challenger},
};
use binius_utils::SerializeBytes;

use crate::{
    basefold::BaseFoldProver,
    basefold_compiler::BaseFoldProverCompiler,
    channel::IOPProverChannel,
    fri::{self, CommitOutput, FRIFoldProver},
    merkle_tree::MerkleTreeProver,
};

/// Oracle handle returned by [`BaseFoldProverChannel::send_oracle`].
#[derive(Debug, Clone, Copy)]
pub struct BaseFoldOracle {
    index: usize,
}

/// Committed oracle data stored internally.
struct CommittedOracleData<P: PackedField, Committed> {
    /// The original message buffer.
    message: FieldBuffer<P>,
    /// RS-encoded codeword.
    codeword: FieldBuffer<P>,
    /// Merkle commitment data for query proofs.
    committed: Committed,
}

/// A prover channel that uses BaseFold for oracle commitment and opening.
///
/// This channel wraps a [`ProverTranscript`] and provides oracle operations using
/// FRI commitment (Reed-Solomon encoding + Merkle tree) and BaseFold opening protocols.
///
/// # Type Parameters
///
/// - `F`: The binary field type
/// - `P`: The packed field type with `Scalar = F`
/// - `NTT`: The additive NTT for Reed-Solomon encoding
/// - `MerkleProver_`: The Merkle tree prover for commitments
/// - `Challenger_`: The Fiat-Shamir challenger
pub struct BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleProver_: MerkleTreeProver<F>,
    Challenger_: Challenger,
{
    /// Prover transcript for Fiat-Shamir (borrowed).
    transcript: &'a mut ProverTranscript<Challenger_>,
    /// NTT for RS encoding (borrowed).
    ntt: &'a NTT,
    /// Merkle tree prover (borrowed).
    merkle_prover: &'a MerkleProver_,
    /// Oracle specifications.
    oracle_specs: Vec<OracleSpec>,
    /// Precomputed FRI params per oracle.
    fri_params: Vec<FRIParams<F>>,
    /// Committed oracle data.
    committed_oracles: Vec<CommittedOracleData<P, MerkleProver_::Committed>>,
    /// Next oracle index.
    next_oracle_index: usize,
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_>
    BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
    MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
    Challenger_: Challenger,
{
    /// Creates a new BaseFold prover channel.
    ///
    /// # Arguments
    ///
    /// * `transcript` - The prover transcript for Fiat-Shamir (borrowed mutably)
    /// * `ntt` - The additive NTT for Reed-Solomon encoding (borrowed)
    /// * `merkle_prover` - The Merkle tree prover (borrowed)
    /// * `oracle_specs` - Specifications for each oracle to be committed
    /// * `log_inv_rate` - Log2 of the inverse Reed-Solomon code rate
    /// * `n_test_queries` - Number of FRI test queries for soundness
    ///
    /// # Preconditions
    ///
    /// * The NTT domain must be large enough for all oracles
    pub fn new(
        transcript: &'a mut ProverTranscript<Challenger_>,
        ntt: &'a NTT,
        merkle_prover: &'a MerkleProver_,
        oracle_specs: Vec<OracleSpec>,
        log_inv_rate: usize,
        n_test_queries: usize,
    ) -> Self {
        use binius_iop::fri::MinProofSizeStrategy;

        let fri_params = oracle_specs
            .iter()
            .map(|spec| {
                FRIParams::with_strategy(
                    ntt,
                    merkle_prover.scheme(),
                    spec.log_msg_len,
                    None,
                    log_inv_rate,
                    n_test_queries,
                    &MinProofSizeStrategy,
                )
                .expect("FRI params should be valid for given oracle spec")
            })
            .collect();

        Self {
            transcript,
            ntt,
            merkle_prover,
            oracle_specs,
            fri_params,
            committed_oracles: Vec::new(),
            next_oracle_index: 0,
        }
    }

    /// Returns a reference to the underlying transcript.
    pub fn transcript(&self) -> &ProverTranscript<Challenger_> {
        self.transcript
    }

    /// Creates a new BaseFold prover channel from a compiler with precomputed FRI parameters.
    ///
    /// This constructor borrows the NTT and other parameters from the compiler.
    ///
    /// # Arguments
    ///
    /// * `compiler` - The BaseFold prover compiler with precomputed parameters
    /// * `transcript` - The prover transcript for Fiat-Shamir (borrowed mutably)
    pub fn from_compiler(
        compiler: &'a BaseFoldProverCompiler<P, NTT, MerkleProver_>,
        transcript: &'a mut ProverTranscript<Challenger_>,
    ) -> Self {
        Self {
            transcript,
            ntt: compiler.ntt(),
            merkle_prover: compiler.merkle_prover(),
            oracle_specs: compiler.oracle_specs().to_vec(),
            fri_params: compiler.fri_params().to_vec(),
            committed_oracles: Vec::new(),
            next_oracle_index: 0,
        }
    }
}

impl<'a, F, P, NTT, MerkleScheme, MerkleProver_, Challenger_> IPProverChannel<F>
    for BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
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
    for BaseFoldProverChannel<'a, F, P, NTT, MerkleProver_, Challenger_>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    NTT: AdditiveNTT<Field = F> + Sync,
    MerkleScheme: MerkleTreeScheme<F, Digest: SerializeBytes>,
    MerkleProver_: MerkleTreeProver<F, Scheme = MerkleScheme>,
    Challenger_: Challenger,
{
    type Oracle = BaseFoldOracle;

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

        // Validate buffer length matches spec
        assert_eq!(
            buffer.log_len(),
            spec.log_msg_len,
            "oracle buffer log_len mismatch: expected {}, got {}",
            spec.log_msg_len,
            buffer.log_len()
        );

        // Copy the message buffer before RS encoding (FieldSlice doesn't implement ToOwned)
        let message_values: Vec<F> = buffer.iter_scalars().collect();
        let message = FieldBuffer::from_values(&message_values);

        // RS encode and commit
        let CommitOutput {
            commitment,
            committed,
            codeword,
        } = fri::commit_interleaved(fri_params, self.ntt, self.merkle_prover, buffer)
            .expect("FRI commit should succeed with valid params");

        // Send commitment via transcript
        self.transcript.message().write(&commitment);

        // Store committed oracle data
        self.committed_oracles.push(CommittedOracleData {
            message,
            codeword,
            committed,
        });

        self.next_oracle_index += 1;

        BaseFoldOracle { index }
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

        // Process each oracle relation with its own BaseFold proof
        for (oracle, transparent_poly, eval_claim) in oracle_relations {
            let index = oracle.index;
            assert!(
                index < self.committed_oracles.len(),
                "oracle index {index} out of bounds, expected < {}",
                self.committed_oracles.len()
            );

            let fri_params = &self.fri_params[index];
            let committed_data = &self.committed_oracles[index];

            // Create FRI folder from committed codeword
            let fri_folder = FRIFoldProver::new(
                fri_params,
                self.ntt,
                self.merkle_prover,
                committed_data.codeword.clone(),
                &committed_data.committed,
            )
            .expect("FRI folder creation should succeed");

            // Run BaseFold proof (non-ZK variant).
            let prover = BaseFoldProver::new(
                committed_data.message.clone(),
                transparent_poly,
                eval_claim,
                fri_folder,
            );
            prover
                .prove(self.transcript)
                .expect("BaseFold proof should succeed");
        }
    }
}

#[cfg(test)]
mod tests {
    use binius_field::{BinaryField, BinaryField128bGhash, PackedBinaryGhash1x128b, PackedField};
    use binius_hash::{ParallelCompressionAdaptor, StdCompression, StdDigest};
    use binius_iop::{channel::OracleSpec, fri::MinProofSizeStrategy};
    use binius_math::{
        BinarySubspace, FieldBuffer,
        inner_product::inner_product_buffers,
        multilinear::eq::eq_ind_partial_eval,
        ntt::{NeighborsLastSingleThread, domain_context::GenericOnTheFly},
        test_utils::{random_field_buffer, random_scalars},
    };
    use binius_transcript::{ProverTranscript, fiat_shamir::HasherChallenger};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::{BaseFoldProverChannel, IOPProverChannel};
    use crate::merkle_tree::prover::BinaryMerkleTreeProver;

    type StdChallenger = HasherChallenger<StdDigest>;

    const LOG_INV_RATE: usize = 1;
    const SECURITY_BITS: usize = 32;

    fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
        security_bits.div_ceil(log_inv_rate)
    }

    /// Generate test data for an oracle.
    ///
    /// For ZK oracles, generates a buffer of size `2^(n_vars+1)` where the first half is the
    /// witness and the second half is random blinding. The transparent polynomial has
    /// `n_vars` variables (not n_vars+1).
    ///
    /// Returns (buffer, transparent_poly, eval_claim) where:
    /// - buffer: The oracle data (size 2^n_vars or 2^(n_vars+1) for ZK)
    /// - transparent_poly: eq_ind evaluated at the evaluation point (size 2^n_vars)
    /// - eval_claim: For ZK, inner product of witness (first half) and transparent_poly. For
    ///   non-ZK, inner product of buffer and transparent_poly.
    fn generate_oracle_data<F, P, R: Rng>(
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
    fn test_basefold_channel_two_oracles() {
        type F = BinaryField128bGhash;
        type P = PackedBinaryGhash1x128b;

        let mut rng = StdRng::seed_from_u64(0);

        let n_vars_1 = 6;
        let n_vars_2 = 8;

        let (buffer_1, transparent_poly_1, eval_claim_1) =
            generate_oracle_data::<F, P, _>(&mut rng, n_vars_1);
        let (buffer_2, transparent_poly_2, eval_claim_2) =
            generate_oracle_data::<F, P, _>(&mut rng, n_vars_2);

        let max_codeword_log_len = n_vars_2 + LOG_INV_RATE;
        let merkle_prover = BinaryMerkleTreeProver::<F, StdDigest, _>::new(
            ParallelCompressionAdaptor::new(StdCompression::default()),
        );

        let subspace = BinarySubspace::with_dim(max_codeword_log_len);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
        let ntt = NeighborsLastSingleThread::new(domain_context);

        let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

        let oracle_specs = vec![
            OracleSpec {
                log_msg_len: n_vars_1,
            },
            OracleSpec {
                log_msg_len: n_vars_2,
            },
        ];

        let mut prover_transcript = ProverTranscript::new(StdChallenger::default());
        let mut prover_channel = BaseFoldProverChannel::<_, P, _, _, _>::new(
            &mut prover_transcript,
            &ntt,
            &merkle_prover,
            oracle_specs.clone(),
            LOG_INV_RATE,
            n_test_queries,
        );

        let oracle_1 = prover_channel.send_oracle(buffer_1.to_ref());
        assert_eq!(oracle_1.index, 0);

        let oracle_2 = prover_channel.send_oracle(buffer_2.to_ref());
        assert_eq!(oracle_2.index, 1);

        prover_channel.prove_oracle_relations([
            (oracle_1, transparent_poly_1, eval_claim_1),
            (oracle_2, transparent_poly_2, eval_claim_2),
        ]);
    }

    #[test]
    fn test_basefold_channel_verifier_two_oracles() {
        use binius_iop::basefold_compiler::BaseFoldVerifierCompiler;

        type F = BinaryField128bGhash;

        let n_vars_1 = 6;
        let n_vars_2 = 8;

        let merkle_prover = BinaryMerkleTreeProver::<F, StdDigest, _>::new(
            ParallelCompressionAdaptor::new(StdCompression::default()),
        );
        let merkle_scheme = merkle_prover.scheme().clone();

        let max_codeword_log_len = n_vars_2 + LOG_INV_RATE;
        let subspace = BinarySubspace::with_dim(max_codeword_log_len);
        let domain_context = GenericOnTheFly::generate_from_subspace(&subspace);
        let ntt = NeighborsLastSingleThread::new(domain_context);

        let n_test_queries = calculate_n_test_queries(SECURITY_BITS, LOG_INV_RATE);

        let oracle_specs = vec![
            OracleSpec {
                log_msg_len: n_vars_1,
            },
            OracleSpec {
                log_msg_len: n_vars_2,
            },
        ];

        let prover_transcript = ProverTranscript::<StdChallenger>::new(StdChallenger::default());
        let mut verifier_transcript = prover_transcript.into_verifier();

        let compiler = BaseFoldVerifierCompiler::new(
            &ntt,
            merkle_scheme,
            oracle_specs,
            LOG_INV_RATE,
            n_test_queries,
            &MinProofSizeStrategy,
        );
        let _verifier_channel = compiler.create_channel(&mut verifier_transcript);
    }
}
