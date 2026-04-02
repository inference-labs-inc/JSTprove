// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use binius_field::{BinaryField, Field};
use binius_math::{ntt::AdditiveNTT, reed_solomon::ReedSolomonCode};
use getset::{CopyGetters, Getters};

use super::error::Error;
use crate::merkle_tree::MerkleTreeScheme;

/// Parameters for an FRI interleaved code proximity protocol.
#[derive(Debug, Clone, Getters, CopyGetters)]
pub struct FRIParams<F> {
    /// The Reed-Solomon code the verifier is testing proximity to.
    #[getset(get = "pub")]
    rs_code: ReedSolomonCode<F>,
    /// log2 the interleaved batch size.
    #[getset(get_copy = "pub")]
    log_batch_size: usize,
    /// The reduction arities between each oracle sent to the verifier.
    fold_arities: Vec<usize>,
    /// log2 the dimension of the terminal codeword.
    log_terminal_dim: usize,
    /// The number oracle consistency queries required during the query phase.
    #[getset(get_copy = "pub")]
    n_test_queries: usize,
}

impl<F> FRIParams<F>
where
    F: BinaryField,
{
    pub fn new(
        rs_code: ReedSolomonCode<F>,
        log_batch_size: usize,
        fold_arities: Vec<usize>,
        n_test_queries: usize,
    ) -> Result<Self, Error> {
        let fold_arities_sum = fold_arities.iter().sum();
        let log_terminal_dim = rs_code
            .log_dim()
            .checked_sub(fold_arities_sum)
            .ok_or(Error::InvalidFoldAritySequence)?;

        Ok(Self {
            rs_code,
            log_batch_size,
            fold_arities,
            log_terminal_dim,
            n_test_queries,
        })
    }

    /// Create parameters using the given arity selection strategy.
    ///
    /// ## Arguments
    ///
    /// * `ntt` - the additive NTT used for Reed-Solomon encoding.
    /// * `merkle_scheme` - the Merkle tree scheme used for commitments.
    /// * `log_msg_len` - the binary logarithm of the length of the message to commit.
    /// * `log_batch_size` - if `Some`, fixes the batch size; if `None`, the batch size is chosen
    ///   optimally along with the fold arities.
    /// * `log_inv_rate` - the binary logarithm of the inverse Reed–Solomon code rate.
    /// * `n_test_queries` - the number of test queries for the FRI protocol.
    /// * `strategy` - the strategy for selecting fold arities.
    ///
    /// ## Preconditions
    ///
    /// * If `log_batch_size` is `Some(b)`, then `b <= log_msg_len`.
    /// * `ntt.log_domain_size() >= log_msg_len - log_batch_size.unwrap_or(0) + log_inv_rate`.
    ///
    /// TODO: This should accept an optional argument for `security_bits`. When it is provided, the
    /// parameters should guarantee that the codeword length is not so large as to violate the
    /// required soundness error probability. In other words, this means bounding the maximum
    /// codeword length based on the field size and security bits.
    pub fn with_strategy<NTT, MerkleScheme, Strategy>(
        ntt: &NTT,
        merkle_scheme: &MerkleScheme,
        log_msg_len: usize,
        log_batch_size: Option<usize>,
        log_inv_rate: usize,
        n_test_queries: usize,
        strategy: &Strategy,
    ) -> Result<Self, Error>
    where
        NTT: AdditiveNTT<Field = F>,
        MerkleScheme: MerkleTreeScheme<F>,
        Strategy: AritySelectionStrategy,
    {
        let (log_batch_size, fold_arities) = match log_batch_size {
            Some(log_batch_size) => {
                assert!(log_batch_size <= log_msg_len); // precondition
                let fold_arities = strategy.choose_arities::<F, _>(
                    merkle_scheme,
                    log_msg_len - log_batch_size,
                    log_inv_rate,
                    n_test_queries,
                );
                (log_batch_size, fold_arities)
            }
            None => {
                let mut fold_arities = strategy.choose_arities::<F, _>(
                    merkle_scheme,
                    log_msg_len,
                    log_inv_rate,
                    n_test_queries,
                );
                let log_batch_size = if !fold_arities.is_empty() {
                    fold_arities.remove(0)
                } else {
                    // Edge case: fold to log_dim = 0 code.
                    log_msg_len
                };
                (log_batch_size, fold_arities)
            }
        };

        let log_dim = log_msg_len - log_batch_size;
        let rs_code = ReedSolomonCode::with_ntt_subspace(ntt, log_dim, log_inv_rate);
        Self::new(rs_code, log_batch_size, fold_arities, n_test_queries)
    }

    pub fn n_fold_rounds(&self) -> usize {
        self.log_msg_len()
    }

    /// Number of oracles sent during the fold rounds.
    pub fn n_oracles(&self) -> usize {
        // One for the batched codeword commitment, and one for each subsequent one.
        1 + self.fold_arities.len()
    }

    /// Number of bits in the query indices sampled during the query phase.
    pub fn index_bits(&self) -> usize {
        self.rs_code.log_len()
    }

    /// Number of folding challenges the verifier sends after receiving the last oracle.
    pub fn n_final_challenges(&self) -> usize {
        self.log_terminal_dim
    }

    /// The reduction arities between each oracle sent to the verifier.
    pub fn fold_arities(&self) -> &[usize] {
        &self.fold_arities
    }

    /// The binary logarithm of the length of the initial oracle.
    pub fn log_len(&self) -> usize {
        self.rs_code.log_len() + self.log_batch_size()
    }

    /// The binary logarithm of the length of the initial message.
    pub fn log_msg_len(&self) -> usize {
        self.rs_code.log_dim() + self.log_batch_size()
    }
}

/// This layer allows minimizing the proof size.
pub fn vcs_optimal_layers_depths_iter<'a, F, VCS>(
    fri_params: &'a FRIParams<F>,
    vcs: &'a VCS,
) -> impl Iterator<Item = usize> + 'a
where
    VCS: MerkleTreeScheme<F>,
    F: BinaryField,
{
    iter::once(fri_params.log_batch_size())
        .chain(fri_params.fold_arities().iter().copied())
        .scan(fri_params.log_len(), |log_n_cosets, arity| {
            *log_n_cosets -= arity;
            Some(vcs.optimal_verify_layer(fri_params.n_test_queries(), *log_n_cosets))
        })
}

/// Calculates the number of test queries required to achieve a target soundness error.
///
/// This chooses a number of test queries so that the soundness error of the FRI query phase is
/// at most $2^{-t}$, where $t$ is the threshold `security_bits`. This _does not_ account for the
/// soundness error from the FRI folding phase or any other protocols, only the query phase. This
/// sets the proximity parameter for FRI to the code's unique decoding radius. See [DP24],
/// Section 5.2, for concrete soundness analysis.
///
/// Throws [`Error::ParameterError`] if the security level is unattainable given the code
/// parameters.
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
pub fn calculate_n_test_queries(security_bits: usize, log_inv_rate: usize) -> usize {
    let rate = 2.0f64.powi(-(log_inv_rate as i32));
    let per_query_err = 0.5 * (1f64 + rate);
    (security_bits as f64 / -per_query_err.log2()).ceil() as usize
}

/// Strategy for selecting fold arities in the FRI protocol.
pub trait AritySelectionStrategy {
    fn choose_arities<F, MerkleScheme>(
        &self,
        merkle_scheme: &MerkleScheme,
        log_msg_len: usize,
        log_inv_rate: usize,
        n_test_queries: usize,
    ) -> Vec<usize>
    where
        F: Field,
        MerkleScheme: MerkleTreeScheme<F>;
}

/// Strategy that minimizes proof size using dynamic programming.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinProofSizeStrategy;

impl AritySelectionStrategy for MinProofSizeStrategy {
    fn choose_arities<F, MerkleScheme>(
        &self,
        merkle_scheme: &MerkleScheme,
        log_msg_len: usize,
        log_inv_rate: usize,
        n_test_queries: usize,
    ) -> Vec<usize>
    where
        F: Field,
        MerkleScheme: MerkleTreeScheme<F>,
    {
        // This algorithm uses a dynamic programming approach to determine the sequence of arities
        // that minimizes proof size. For each i in [0, log_msg_len], we determine the minimum
        // proof size attainable when for a batched codeword with message size 2^i. This is
        // determined by minimizing over the first reduction arity, using the values already
        // determined for the smaller values of i.

        // This vec maps log_msg_len values to the minimum proof size attainable for a batched FRI
        // protocol committing a message with that length.
        let mut min_sizes = Vec::<Entry>::with_capacity(log_msg_len);

        #[derive(Debug)]
        struct Entry {
            // The minimum proof size attainable for the indexed value of i.
            proof_size: usize,
            // The first reduction arity to achieve the minimum proof size. If the value is none,
            // then the best reduction sequence is to skip all folding and send the full codeword.
            arity: Option<usize>,
        }

        // The byte-size of an element.
        let value_size = {
            let mut buf = Vec::new();
            F::default()
                .serialize(&mut buf)
                .expect("default element can be serialized to a resizable buffer");
            buf.len()
        };

        for i in 0..=log_msg_len {
            // Length of the batched codeword.
            let log_code_len = i + log_inv_rate;

            let mut last_entry = None;
            for arity in 1..=i {
                // The additional proof bytes for the reduction by arity.
                let reduction_proof_size = {
                    // Each queried coset contains 2^arity values.
                    let leaf_size = value_size << arity;
                    // One coset per test query.
                    let leaves_size = leaf_size * n_test_queries;

                    // Size of the Merkle multi-proof.
                    let optimal_layer =
                        merkle_scheme.optimal_verify_layer(n_test_queries, log_code_len);
                    let merkle_size = merkle_scheme
                        .proof_size(1 << log_code_len, n_test_queries, optimal_layer)
                        .expect("layer computed with optimal_layer must be valid");
                    leaves_size + merkle_size
                };

                let reduced_proof_size = min_sizes[i - arity].proof_size;
                let proof_size = reduction_proof_size + reduced_proof_size;
                let replace = last_entry
                    .as_ref()
                    .is_none_or(|last_entry: &Entry| proof_size <= last_entry.proof_size);
                if replace {
                    last_entry = Some(Entry {
                        proof_size,
                        arity: Some(arity),
                    });
                } else {
                    // The proof size function is concave with respect to arity. Break as soon is
                    // it ascends.
                    break;
                }
            }

            // Determine the proof size if this is the terminal codeword. In that case, the proof
            // simply consists of the 2^(i + log_inv_rate) leaf values.
            let terminal_proof_size = value_size << log_code_len;
            let terminal_entry = Entry {
                proof_size: terminal_proof_size,
                arity: None,
            };

            let optimal_entry = if let Some(last_entry) = last_entry
                && last_entry.proof_size < terminal_entry.proof_size
            {
                last_entry
            } else {
                terminal_entry
            };

            min_sizes.push(optimal_entry);
        }

        let mut fold_arities = Vec::with_capacity(log_msg_len);

        let mut i = log_msg_len;
        let mut entry = &min_sizes[i];
        while let Some(arity) = entry.arity {
            fold_arities.push(arity);
            i -= arity;
            entry = &min_sizes[i];
        }
        fold_arities
    }
}

/// Strategy that uses a constant fold arity.
#[derive(Debug, Clone, Copy)]
pub struct ConstantArityStrategy {
    /// The fold arity to use for each reduction step.
    pub arity: usize,
}

impl ConstantArityStrategy {
    /// Creates a new strategy with the given arity.
    pub fn new(arity: usize) -> Self {
        Self { arity }
    }

    /// Creates a strategy with an estimated optimal arity.
    ///
    /// Uses a heuristic to estimate the optimal FRI folding arity that minimizes proof size.
    ///
    /// ## Arguments
    ///
    /// * `_merkle_scheme` - the Merkle tree scheme (used to infer digest size)
    /// * `approx_log_code_len` - approximate log2 of the codeword length
    pub fn with_optimal_arity<F, MerkleScheme>(
        _merkle_scheme: &MerkleScheme,
        approx_log_code_len: usize,
    ) -> Self
    where
        F: Field,
        MerkleScheme: MerkleTreeScheme<F>,
    {
        let digest_size = std::mem::size_of::<MerkleScheme::Digest>() * 8;
        let field_size = std::mem::size_of::<F>() * 8;

        // Estimate optimal arity using a heuristic based on the approximation of a single
        // query_proof_size, where θ is the arity:
        // ((n-θ) + (n-2θ) + ...) * digest_size + ((n-θ)/θ) * 2^θ * field_size
        let arity = (1..=approx_log_code_len)
			.map(|arity| {
				(
					arity,
					((approx_log_code_len) / 2 * digest_size + (1 << arity) * field_size)
						* (approx_log_code_len - arity)
						/ arity,
				)
			})
			// Scan and terminate when query_proof_size increases.
			.scan(None, |old: &mut Option<(usize, usize)>, new| {
				let should_continue = !matches!(*old, Some(ref old) if new.1 > old.1);
				*old = Some(new);
				should_continue.then_some(new)
			})
			.last()
			.map(|(arity, _)| arity)
			.unwrap_or(1);

        Self { arity }
    }
}

impl AritySelectionStrategy for ConstantArityStrategy {
    fn choose_arities<F, MerkleScheme>(
        &self,
        merkle_scheme: &MerkleScheme,
        log_msg_len: usize,
        log_inv_rate: usize,
        n_test_queries: usize,
    ) -> Vec<usize>
    where
        F: Field,
        MerkleScheme: MerkleTreeScheme<F>,
    {
        let log_code_len = log_msg_len + log_inv_rate;
        let cap_height = merkle_scheme.optimal_verify_layer(n_test_queries, log_code_len);
        let log_terminal_len = cap_height.max(log_inv_rate);

        let mut fold_arities = Vec::new();
        let mut i = log_code_len;
        while i > log_terminal_len {
            if let Some(next_i) = i.checked_sub(self.arity) {
                fold_arities.push(self.arity);
                i = next_i;
            } else {
                break;
            }
        }
        fold_arities
    }
}

#[cfg(test)]
mod tests {
    use binius_field::BinaryField128bGhash as B128;
    use binius_hash::StdCompression;
    use binius_math::ntt::{NeighborsLastReference, domain_context::GaoMateerOnTheFly};

    use super::*;
    use crate::merkle_tree::BinaryMerkleTreeScheme;

    type StdDigest = sha2::Sha256;
    type TestMerkleScheme = BinaryMerkleTreeScheme<B128, StdDigest, StdCompression>;

    fn test_merkle_scheme() -> TestMerkleScheme {
        BinaryMerkleTreeScheme::new(StdCompression::default())
    }

    #[test]
    fn test_calculate_n_test_queries() {
        let security_bits = 96;
        let n_test_queries = calculate_n_test_queries(security_bits, 1);
        assert_eq!(n_test_queries, 232);

        let n_test_queries = calculate_n_test_queries(security_bits, 2);
        assert_eq!(n_test_queries, 142);
    }

    #[test]
    fn test_min_proof_size_strategy() {
        let merkle_scheme = test_merkle_scheme();
        let log_inv_rate = 2;
        let n_test_queries = 128;
        let strategy = MinProofSizeStrategy;

        // log_msg_len = 0: no folding needed, terminal codeword is optimal
        let arities =
            strategy.choose_arities::<B128, _>(&merkle_scheme, 0, log_inv_rate, n_test_queries);
        assert_eq!(arities, vec![]);

        // log_msg_len = 3: no folding needed, terminal codeword is optimal
        let arities =
            strategy.choose_arities::<B128, _>(&merkle_scheme, 3, log_inv_rate, n_test_queries);
        assert_eq!(arities, vec![]);

        // log_msg_len = 24
        let arities =
            strategy.choose_arities::<B128, _>(&merkle_scheme, 24, log_inv_rate, n_test_queries);
        assert_eq!(arities, vec![4, 4, 4, 4]);
    }

    #[test]
    fn test_with_strategy_min_proof_size() {
        let merkle_scheme = test_merkle_scheme();
        let log_inv_rate = 2;
        let n_test_queries = 128;

        let ntt = NeighborsLastReference {
            domain_context: GaoMateerOnTheFly::<B128>::generate(24 + log_inv_rate),
        };

        // log_msg_len = 0
        {
            let fri_params = FRIParams::with_strategy(
                &ntt,
                &merkle_scheme,
                0,
                None,
                log_inv_rate,
                n_test_queries,
                &MinProofSizeStrategy,
            )
            .unwrap();
            assert_eq!(fri_params.fold_arities(), &[]);
            assert_eq!(fri_params.log_batch_size(), 0);
        }

        // log_msg_len = 3
        {
            let fri_params = FRIParams::with_strategy(
                &ntt,
                &merkle_scheme,
                3,
                None,
                log_inv_rate,
                n_test_queries,
                &MinProofSizeStrategy,
            )
            .unwrap();
            assert_eq!(fri_params.fold_arities(), &[]);
            assert_eq!(fri_params.log_batch_size(), 3);
        }

        // log_msg_len = 24
        {
            let fri_params = FRIParams::with_strategy(
                &ntt,
                &merkle_scheme,
                24,
                None,
                log_inv_rate,
                n_test_queries,
                &MinProofSizeStrategy,
            )
            .unwrap();
            assert_eq!(fri_params.fold_arities(), &[4, 4, 4]);
            assert_eq!(fri_params.log_batch_size(), 4);
        }
    }

    #[test]
    fn test_with_strategy_fixed_batch_size() {
        let merkle_scheme = test_merkle_scheme();
        let log_inv_rate = 2;
        let n_test_queries = 128;

        let ntt = NeighborsLastReference {
            domain_context: GaoMateerOnTheFly::<B128>::generate(24 + log_inv_rate),
        };

        // log_msg_len = 3
        {
            let fri_params = FRIParams::with_strategy(
                &ntt,
                &merkle_scheme,
                3,
                Some(1),
                log_inv_rate,
                n_test_queries,
                &MinProofSizeStrategy,
            )
            .unwrap();
            assert_eq!(fri_params.fold_arities(), &[]);
            assert_eq!(fri_params.log_batch_size(), 1);
        }

        // log_msg_len = 24
        {
            let fri_params = FRIParams::with_strategy(
                &ntt,
                &merkle_scheme,
                24,
                Some(1),
                log_inv_rate,
                n_test_queries,
                &MinProofSizeStrategy,
            )
            .unwrap();
            assert_eq!(fri_params.fold_arities(), &[4, 4, 4, 3]);
            assert_eq!(fri_params.log_batch_size(), 1);
        }
    }
}
