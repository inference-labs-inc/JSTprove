// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

//! Spartan-based proof generation for Binius64 constraint systems.
//!
//! This crate provides the [`Prover`] struct for generating zero-knowledge proofs
//! using the Spartan protocol adapted for Binius64's constraint system. It is the
//! prover-side counterpart to `binius_spartan_verifier`.
//!
//! # When to use this crate
//!
//! Use this crate when you have a constraint system built with `binius_spartan_frontend`
//! and need to generate a Spartan-based proof. This is an alternative to the main
//! `binius_prover` crate.
//!
//! # Key types
//!
//! - [`Prover`] - Main proving interface; call [`Prover::setup`] with a verifier, then
//!   [`Prover::prove`] with witness data
//!
//! # Related crates
//!
//! - `binius_spartan_verifier` - Verification counterpart
//! - `binius_spartan_frontend` - Constraint system builder for Spartan
//! - `binius_prover` - Alternative proving backend

#![warn(rustdoc::missing_crate_level_docs)]

mod error;
mod wiring;

use std::iter::{repeat_n, repeat_with};

use binius_field::{BinaryField, Field, PackedExtension, PackedField};
use binius_iop_prover::{basefold_compiler::BaseFoldZKProverCompiler, channel::IOPProverChannel};
use binius_ip_prover::channel::IPProverChannel;
use binius_math::{
    FieldBuffer, FieldSlice,
    ntt::{NeighborsLastMultiThread, domain_context::GenericPreExpanded},
};
use binius_prover::{
    hash::{ParallelDigest, parallel_compression::ParallelPseudoCompression},
    merkle_tree::prover::BinaryMerkleTreeProver,
    protocols::sumcheck::{quadratic_mle::QuadraticMleCheckProver, zk_mlecheck},
};
use binius_spartan_frontend::constraint_system::{BlindingInfo, MulConstraint, WitnessIndex};
use binius_spartan_verifier::Verifier;
use binius_transcript::{ProverTranscript, fiat_shamir::Challenger};
use binius_utils::{
    SerializeBytes,
    checked_arithmetics::checked_log_2,
    rayon::{self, prelude::*},
};
use digest::{Digest, FixedOutputReset, Output, core_api::BlockSizeUser};
pub use error::*;
use itertools::chain;
use rand::CryptoRng;

use crate::wiring::WiringTranspose;

type ProverNTT<F> = NeighborsLastMultiThread<GenericPreExpanded<F>>;
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
    P: PackedField<Scalar: BinaryField>,
    ParallelMerkleHasher: ParallelDigest<Digest: Digest + BlockSizeUser + FixedOutputReset>,
    ParallelMerkleCompress: ParallelPseudoCompression<Output<ParallelMerkleHasher::Digest>, 2>,
{
    verifier:
        Verifier<P::Scalar, ParallelMerkleHasher::Digest, ParallelMerkleCompress::Compression>,
    #[allow(clippy::type_complexity)]
    basefold_compiler: BaseFoldZKProverCompiler<
        P,
        ProverNTT<P::Scalar>,
        ProverMerkleProver<P::Scalar, ParallelMerkleHasher, ParallelMerkleCompress>,
    >,
    wiring_transpose: WiringTranspose,
}

impl<F, P, MerkleHash, ParallelMerkleCompress, ParallelMerkleHasher>
    Prover<P, ParallelMerkleCompress, ParallelMerkleHasher>
where
    F: BinaryField,
    P: PackedField<Scalar = F> + PackedExtension<F>,
    MerkleHash: Digest + BlockSizeUser + FixedOutputReset,
    ParallelMerkleHasher: ParallelDigest<Digest = MerkleHash>,
    ParallelMerkleCompress: ParallelPseudoCompression<Output<MerkleHash>, 2>,
    Output<MerkleHash>: SerializeBytes,
{
    /// Constructs a prover corresponding to a constraint system verifier.
    ///
    /// See [`Prover`] struct documentation for details.
    pub fn setup(
        verifier: Verifier<F, MerkleHash, ParallelMerkleCompress::Compression>,
        compression: ParallelMerkleCompress,
    ) -> Result<Self, Error> {
        let cs = verifier.constraint_system();
        let log_num_shares = binius_utils::rayon::current_num_threads().ilog2() as usize;

        // Get the largest subspace from the verifier compiler for NTT creation
        let subspace = verifier.iop_compiler().max_subspace();
        let domain_context = GenericPreExpanded::generate_from_subspace(subspace);
        let ntt = NeighborsLastMultiThread::new(domain_context, log_num_shares);

        let merkle_prover = BinaryMerkleTreeProver::<_, ParallelMerkleHasher, _>::new(compression);

        // Create the BaseFold ZK compiler from verifier compiler (reuses oracle_specs and
        // fri_params)
        let basefold_compiler = BaseFoldZKProverCompiler::from_verifier_compiler(
            verifier.iop_compiler(),
            ntt,
            merkle_prover,
        );

        // Compute wiring transpose from constraint system
        let wiring_transpose = WiringTranspose::transpose(cs.size(), cs.mul_constraints());

        Ok(Prover {
            verifier,
            basefold_compiler,
            wiring_transpose,
        })
    }

    /// Generates a proof for a witness against the constraint system.
    ///
    /// # Arguments
    ///
    /// * `witness` - The witness values for the constraint system
    /// * `rng` - Random number generator for blinding
    /// * `transcript` - The prover transcript for Fiat-Shamir
    ///
    /// # Preconditions
    ///
    /// * The witness length must match the constraint system size
    pub fn prove<Challenger_: Challenger>(
        &self,
        witness: &[F],
        mut rng: impl CryptoRng,
        transcript: &mut ProverTranscript<Challenger_>,
    ) -> Result<(), Error> {
        let cs = self.verifier.constraint_system();

        // Prover observes the public input (includes it in Fiat-Shamir).
        let public = &witness[..1 << cs.log_public()];
        transcript.observe().write_slice(public);

        // Create ZK channel (owns the RNG for mask generation) and delegate to prove_iop
        let channel = self.basefold_compiler.create_channel(transcript, &mut rng);
        self.prove_iop(witness, rng, channel)
    }

    /// Proves using an IOP channel interface.
    ///
    /// This is an alternative interface for advanced users who want to provide their own
    /// channel implementation. Most users should use [`Self::prove`] instead.
    ///
    /// # Arguments
    ///
    /// * `witness` - The witness values for the constraint system
    /// * `rng` - Random number generator for blinding
    /// * `channel` - The IOP prover channel (public input must be observed on transcript before
    ///   creating the channel)
    pub fn prove_iop<Channel>(
        &self,
        witness: &[F],
        mut rng: impl CryptoRng,
        mut channel: Channel,
    ) -> Result<(), Error>
    where
        Channel: IOPProverChannel<P>,
    {
        let _prove_guard = tracing::info_span!(
            "Prove",
            operation = "prove",
            perfetto_category = "operation"
        )
        .entered();

        let cs = self.verifier.constraint_system();

        // Check that the witness length matches the constraint system
        let expected_size = cs.size();
        if witness.len() != expected_size {
            return Err(Error::ArgumentError {
                arg: "witness".to_string(),
                msg: format!(
                    "witness has {} elements, expected {}",
                    witness.len(),
                    expected_size
                ),
            });
        }

        let log_mul_constraints = checked_log_2(cs.mul_constraints().len());

        // Create mask buffer for the ZK mulcheck mask polynomial.
        let (m_n, m_d) = self.verifier.mask_dims();
        let mask_degree = 2; // quadratic composition
        let log_masks_buffer_size = m_n + m_d;

        let masks_buffer = FieldBuffer::<P>::new(
            log_masks_buffer_size,
            repeat_with(|| P::random(&mut rng))
                .take(1 << log_masks_buffer_size.saturating_sub(P::LOG_WIDTH))
                .collect(),
        );

        let mulcheck_mask =
            zk_mlecheck::Mask::new(log_mul_constraints, mask_degree, masks_buffer.to_ref());

        // Pack witness into field elements and add blinding
        let blinding_info = cs.blinding_info();
        let witness_packed = pack_and_blind_witness::<_, P>(
            cs.log_size() as usize,
            witness,
            blinding_info,
            cs.n_public() as usize,
            cs.n_private() as usize,
            &mut rng,
        );

        // Send the witness and masks oracles to the channel
        let trace_oracle = channel.send_oracle(witness_packed.to_ref());
        let mask_oracle = channel.send_oracle(masks_buffer.to_ref());

        // Prove the multiplication constraints
        let (mulcheck_evals, mask_eval, r_x) = self.prove_mulcheck(
            cs.mul_constraints(),
            witness_packed.to_ref(),
            mulcheck_mask,
            &mut channel,
        )?;

        // Compute wiring claim components
        let r_public = channel.sample_many(cs.log_public() as usize);

        let wiring_relation = wiring::compute_wiring_relation(
            &self.wiring_transpose,
            &witness_packed.to_ref(),
            &r_public,
            &r_x,
            &mulcheck_evals,
            &mut channel,
        );

        // Compute the mask folding polynomial (libra_eval tensor)
        let n_vars = r_x.len();
        let libra_eval_tensor =
            zk_mlecheck::expand_libra_eval::<P>(&r_x, n_vars, mask_degree, m_n, m_d);

        // Prove both oracle relations
        channel.prove_oracle_relations([
            (
                trace_oracle,
                wiring_relation.l_poly,
                wiring_relation.batched_sum,
            ),
            (mask_oracle, libra_eval_tensor, mask_eval),
        ]);

        Ok(())
    }

    fn prove_mulcheck<Data: std::ops::Deref<Target = [P]>>(
        &self,
        mul_constraints: &[MulConstraint<WitnessIndex>],
        witness: FieldSlice<P>,
        mask: zk_mlecheck::Mask<P, Data>,
        channel: &mut impl IPProverChannel<F>,
    ) -> Result<([F; 3], F, Vec<F>), Error> {
        let mulcheck_witness = wiring::build_mulcheck_witness(mul_constraints, witness);

        // Sample random evaluation point for mulcheck
        let r_mulcheck = channel.sample_many(mask.n_vars());

        // Create the QuadraticMleCheckProver for the mul gate: a * b - c
        let mlecheck_prover = QuadraticMleCheckProver::new(
            [mulcheck_witness.a, mulcheck_witness.b, mulcheck_witness.c],
            |[a, b, c]| a * b - c, // composition
            |[a, b, _c]| a * b,    // infinity_composition (quadratic term only)
            r_mulcheck,
            F::ZERO, // eval_claim: zerocheck
        )?;

        // Run the ZK MLE-check protocol
        let mlecheck_output = zk_mlecheck::prove(mlecheck_prover, mask, channel)?;

        // Extract the reduced evaluation point and multilinear evaluations
        let mut r_x = mlecheck_output.challenges;
        r_x.reverse(); // Match verifier's order

        let [a_eval, b_eval, c_eval]: [F; 3] = mlecheck_output
            .multilinear_evals
            .try_into()
            .expect("mlecheck returns 3 evaluations");

        // Write the multilinear evaluations to channel
        channel.send_many(&[a_eval, b_eval, c_eval]);

        let mulcheck_evals = [a_eval, b_eval, c_eval];
        let mask_eval = mlecheck_output.mask_eval;

        Ok((mulcheck_evals, mask_eval, r_x))
    }
}

/// Packs the witness into a [`FieldBuffer`] and adds blinding values for dummy wires.
fn pack_and_blind_witness<F: Field, P: PackedField<Scalar = F>>(
    log_witness_elems: usize,
    witness: &[F],
    blinding_info: &BlindingInfo,
    n_public: usize,
    n_private: usize,
    mut rng: impl CryptoRng,
) -> FieldBuffer<P> {
    let packed_witness = if log_witness_elems < P::LOG_WIDTH {
        let elems_iter = witness.iter().copied();
        let zeros_iter = repeat_n(F::ZERO, (1 << log_witness_elems) - witness.len());

        let elems = P::from_scalars(chain!(elems_iter, zeros_iter));
        vec![elems]
    } else {
        let packed_len = 1 << (log_witness_elems - P::LOG_WIDTH);

        let elems_iter = witness
            .par_chunks(P::WIDTH)
            .map(|chunk| P::from_scalars(chunk.iter().copied()));
        let zeros_iter = rayon::iter::repeat_n(P::zero(), packed_len - elems_iter.len());

        elems_iter.chain(zeros_iter).collect::<Vec<_>>()
    };

    let mut witness_packed = FieldBuffer::new(log_witness_elems, packed_witness.into_boxed_slice());

    // Add blinding values
    let base = n_public + n_private;

    // Set random values for non-constraint dummy wires
    for i in 0..blinding_info.n_dummy_wires {
        witness_packed.set(base + i, F::random(&mut rng));
    }

    // Set random values for dummy constraint wires (A * B = C)
    let constraint_wire_base = base + blinding_info.n_dummy_wires;
    for i in 0..blinding_info.n_dummy_constraints {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = a * b;

        witness_packed.set(constraint_wire_base + 3 * i, a);
        witness_packed.set(constraint_wire_base + 3 * i + 1, b);
        witness_packed.set(constraint_wire_base + 3 * i + 2, c);
    }

    witness_packed
}
