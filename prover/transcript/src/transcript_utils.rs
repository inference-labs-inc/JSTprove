use gkr_engine::{MPIEngine, Transcript};

/// broadcast root transcript state. incurs an additional hash if self.world_size > 1
pub fn transcript_root_broadcast(transcript: &mut impl Transcript, mpi_engine: &impl MPIEngine) {
    if mpi_engine.world_size() > 1 {
        let mut state = transcript.hash_and_return_state();
        mpi_engine.root_broadcast_bytes(&mut state);
        transcript.set_state(&state);
    }
}

/// Correspondence to 'transcript_root_broadcast' from the verifier side.
///
/// Note: Currently, the verifier is assumed to run on a single core with no mpi sync,
/// the word 'sync' here refers to the verifier syncing up with the prover's transcript state,
/// which is updated by 'transcript_root_broadcast' if mpi_size > 1.
pub fn transcript_verifier_sync<T>(transcript: &mut T, mpi_world_size: usize)
where
    T: Transcript,
{
    if mpi_world_size > 1 {
        let state = transcript.hash_and_return_state(); // Sync up the Fiat-Shamir randomness
        transcript.set_state(&state);
    }
}
