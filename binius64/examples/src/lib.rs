// Copyright 2025 Irreducible Inc.
// Copyright 2026 The Binius Developers

pub mod circuits;
pub mod cli;
pub mod snapshot;

use anyhow::Result;
use binius_core::constraint_system::{ConstraintSystem, ValueVec};
use binius_frontend::{CircuitBuilder, WitnessFiller};
use binius_hash::{
    ParallelCompressionAdaptor, ParallelDigest, ParallelPseudoCompression,
    PseudoCompressionFunction, StdCompression, StdDigest,
    vision_4::{
        compression::VisionCompression, digest::VisionHasherDigest,
        parallel_compression::VisionParallelCompression,
    },
};
use binius_prover::{KeyCollection, OptimalPackedB128, Prover};
use binius_verifier::{
    Verifier,
    config::StdChallenger,
    transcript::{ProverTranscript, VerifierTranscript},
};
use clap::ValueEnum;
pub use cli::Cli;
use digest::{Digest, FixedOutputReset, Output, core_api::BlockSizeUser};

#[derive(Debug, Clone, ValueEnum)]
pub enum CompressionType {
    /// SHA-256 compression function
    Sha256,
    /// Vision 4-element compression function
    Vision4,
}

/// Standard verifier using SHA256 compression
pub type StdVerifier = Verifier<StdDigest, StdCompression>;
/// Standard prover using SHA256 compression
pub type StdProver =
    Prover<OptimalPackedB128, ParallelCompressionAdaptor<StdCompression>, StdDigest>;
/// Vision4 verifier
pub type VisionVerifier = Verifier<VisionHasherDigest, VisionCompression>;
/// Vision4 prover
pub type VisionProver = Prover<OptimalPackedB128, VisionParallelCompression, VisionHasherDigest>;

/// Setup the prover and verifier and use SHA256 for Merkle tree compression.
/// Providing the `key_collection` skips expensive key collection building.
pub fn setup_sha256(
    cs: ConstraintSystem,
    log_inv_rate: usize,
    key_collection: Option<KeyCollection>,
) -> Result<(StdVerifier, StdProver)> {
    let _setup_guard = tracing::info_span!("Setup", log_inv_rate).entered();
    let parallel_compression = ParallelCompressionAdaptor::new(StdCompression::default());
    let compression = parallel_compression.compression().clone();
    let verifier = Verifier::setup(cs, log_inv_rate, compression)?;
    let prover = if let Some(key_collection) = key_collection {
        Prover::setup_with_key_collection(verifier.clone(), parallel_compression, key_collection)?
    } else {
        Prover::setup(verifier.clone(), parallel_compression)?
    };
    Ok((verifier, prover))
}

/// Setup the prover and verifier and use ZK-friendly hash Vision4 for Merkle tree compression.
/// Providing the `key_collection` skips expensive key collection building.
pub fn setup_vision4(
    cs: ConstraintSystem,
    log_inv_rate: usize,
    key_collection: Option<KeyCollection>,
) -> Result<(VisionVerifier, VisionProver)> {
    let _setup_guard = tracing::info_span!("Setup", log_inv_rate).entered();
    let parallel_compression = VisionParallelCompression::default();
    let compression = parallel_compression.compression().clone();
    let verifier = Verifier::setup(cs, log_inv_rate, compression)?;
    let prover = if let Some(key_collection) = key_collection {
        Prover::setup_with_key_collection(verifier.clone(), parallel_compression, key_collection)?
    } else {
        Prover::setup(verifier.clone(), parallel_compression)?
    };
    Ok((verifier, prover))
}

pub fn prove_verify<D, C, ParD, ParC>(
    verifier: &Verifier<D, C>,
    prover: &Prover<OptimalPackedB128, ParC, ParD>,
    witness: ValueVec,
) -> Result<()>
where
    D: Digest + BlockSizeUser + FixedOutputReset,
    C: PseudoCompressionFunction<Output<D>, 2>,
    ParD: ParallelDigest<Digest = D>,
    ParC: ParallelPseudoCompression<Output<D>, 2, Compression = C>,
{
    let challenger = StdChallenger::default();

    let mut prover_transcript = ProverTranscript::new(challenger.clone());
    prover.prove(witness.clone(), &mut prover_transcript)?;

    let proof = prover_transcript.finalize();
    tracing::info!("Proof size: {} KiB", proof.len() / 1024);

    let mut verifier_transcript = VerifierTranscript::new(challenger, proof);
    verifier.verify(witness.public(), &mut verifier_transcript)?;
    verifier_transcript.finalize()?;

    Ok(())
}

/// Trait for standardizing circuit examples in the Binius framework.
///
/// This trait provides a common pattern for implementing circuit examples by separating:
/// - **Circuit parameters** (`Params`): compile-time configuration that affects circuit structure
/// - **Instance data** (`Instance`): runtime data used to populate the witness
/// - **Circuit building**: logic to construct the circuit based on parameters
/// - **Witness population**: logic to fill in witness values based on instance data
///
/// # Example Implementation
///
/// ```rust,ignore
/// struct MyExample {
///     params: MyParams,
///     // Store any gadgets or wire references needed for witness population
/// }
///
/// #[derive(clap::Args)]
/// struct MyParams {
///     #[arg(long)]
///     max_size: usize,
/// }
///
/// #[derive(clap::Args)]
/// struct MyInstance {
///     #[arg(long)]
///     input_value: Option<String>,
/// }
///
/// impl ExampleCircuit for MyExample {
///     type Params = MyParams;
///     type Instance = MyInstance;
///
///     fn build(params: MyParams, builder: &mut CircuitBuilder) -> Result<Self> {
///         // Construct circuit based on parameters
///         Ok(Self { params })
///     }
///
///     fn populate_witness(&self, instance: MyInstance, filler: &mut WitnessFiller) -> Result<()> {
///         // Fill witness values based on instance data
///         Ok(())
///     }
/// }
/// ```
///
/// # Lifecycle
///
/// 1. Parse CLI arguments to get `Params` and `Instance`
/// 2. Call `build()` with parameters to construct the circuit
/// 3. Build the constraint system
/// 4. Set up prover and verifier
/// 5. Call `populate_witness()` to fill witness values
/// 6. Generate and verify proof
pub trait ExampleCircuit: Sized {
    /// Circuit parameters that affect the structure of the circuit.
    /// These are typically compile-time constants or bounds.
    type Params: clap::Args;

    /// Instance data used to populate the witness.
    /// This represents the actual input values for a specific proof.
    type Instance: clap::Args;

    /// Build the circuit with the given parameters.
    ///
    /// This method should:
    /// - Add witnesses, constants, and constraints to the builder
    /// - Store any wire references needed for witness population
    /// - Return a Self instance that can later populate witness values
    fn build(params: Self::Params, builder: &mut CircuitBuilder) -> Result<Self>;

    /// Populate witness values for a specific instance.
    ///
    /// This method should:
    /// - Process the instance data (e.g., parse inputs, compute hashes)
    /// - Fill all witness values using the provided filler
    /// - Validate that instance data is compatible with circuit parameters
    fn populate_witness(&self, instance: Self::Instance, filler: &mut WitnessFiller) -> Result<()>;

    /// Generate a concise parameter summary for perfetto trace filenames.
    ///
    /// This method should return a short string (5-10 chars max) that captures
    /// the most important parameters for this circuit configuration.
    /// Used to differentiate traces with different parameter settings.
    ///
    /// Format suggestions:
    /// - Bytes: "2048b", "4096b"
    /// - Counts: "10p" (permutations), "5s" (signatures)
    ///
    /// Returns None if no meaningful parameters to include in filename.
    fn param_summary(params: &Self::Params) -> Option<String> {
        let _ = params;
        None
    }
}
