use std::collections::VecDeque;

use arith::SimdField;
use gkr_engine::{MPIEngine, Transcript};
use itertools::izip;
use serdes::ExpSerde;
use transpose::transpose_inplace;
use tree::{RangePath, Tree};

use crate::{
    orion::{OrionCommitment, OrionResult, OrionSRS, OrionScratchPad},
    traits::TensorCodeIOPPCS,
    PCS_SOUNDNESS_BITS,
};

/*
In the Orion scenario, the codeword originally look like following in the RAM:

p(0):    *-*-*-*-*-*-*-*-*-*- .... -*

         *-*-*-*-*-*-*-*-*-*- .... -*

p(1):    *-*-*-*-*-*-*-*-*-*- .... -*

         *-*-*-*-*-*-*-*-*-*- .... -*

...

p(n - 1): *-*-*-*-*-*-*-*-*-*- .... -*

          *-*-*-*-*-*-*-*-*-*- .... -*

a local transpose on each process allows for visiting in interleaved codeword order:

p(0):     * * * * * * * * * *  ....  *
          |/|/|/|/|/|/|/|/|/|  .... /|
          * * * * * * * * * *  ....  *

p(1):     * * * * * * * * * *  ....  *
          |/|/|/|/|/|/|/|/|/|  .... /|
          * * * * * * * * * *  ....  *

...

p(n - 1): * * * * * * * * * *  ....  *
          |/|/|/|/|/|/|/|/|/|  .... /|
          * * * * * * * * * *  ....  *

a global MPI ALL TO ALL rewinds the order into the following:

          p(0)        p(1)        p(2)     p(n - 1)
          * * * *     * * * *     * *  ....  *
          |/|/|/|     |/|/|/|     |/|  .... /|
          * * * *     * * * *     * *  ....  *
               /           /
              /           /
             /           /
            /           /
           /           /           /
          * * * *     * * * *     * *  ....  *
          |/|/|/|     |/|/|/|     |/|  .... /|
          * * * *     * * * *     * *  ....  *
               /           /
              /           /
             /           /
            /           /
           /           /           /
          * * * *     * * * *     * *  ....  *
          |/|/|/|     |/|/|/|     |/|  .... /|
          * * * *     * * * *     * *  ....  *

rearrange each row of interleaved codeword on each process, we have:

          p(0)        p(1)        p(2)     p(n - 1)
          *-*-*-*     *-*-*-*     *-*- .... -*
          *-*-*-*     *-*-*-*     *-*- .... -*
               /           /
              /           /
             /           /
            /           /
           /           /           /
          *-*-*-*     *-*-*-*     *-*- .... -*
          *-*-*-*     *-*-*-*     *-*- .... -*
               /           /
              /           /
             /           /
            /           /
           /           /           /
          *-*-*-*     *-*-*-*     *-*- .... -*
          *-*-*-*     *-*-*-*     *-*- .... -*

eventually, a final transpose each row lead to results of ordering by *WHOLE* interleaved alphabet:

          p(0)                    p(1)                 ....
          *     *     *     *     *     *     *     *
          |    /|    /|    /|     |    /|    /|    /|
          *   / *   / *   / *     *   / *   / *   / *
          |  /  |  /  |  /  |     |  /  |  /  |  /  |
          * /   * /   * /   *     * /   * /   * /   *
          |/    |/    |/    |     |/    |/    |/    |
          *     *     *     *     *     *     *     *

After all these, we can go onwards to MT commitment, and later open alphabets lies in one of the parties.
 */

#[inline(always)]
pub(crate) fn mpi_commit_encoded<PackF>(
    mpi_engine: &impl MPIEngine,
    pk: &OrionSRS,
    packed_evals: &[PackF],
    scratch_pad: &mut OrionScratchPad,
) -> OrionResult<OrionCommitment>
where
    PackF: SimdField,
{
    let packed_rows = pk.local_num_fs_per_query() / PackF::PACK_SIZE;

    // NOTE: packed codeword buffer and encode over packed field
    let mut codewords = vec![PackF::ZERO; packed_rows * pk.codeword_len()];
    izip!(
        packed_evals.chunks(pk.message_len()),
        codewords.chunks_mut(pk.codeword_len())
    )
    .try_for_each(|(evals, codeword)| pk.code_instance.encode_in_place(evals, codeword))?;

    // NOTE: transpose codeword s.t., the matrix has codewords being columns
    if packed_rows > 1 {
        let mut scratch = vec![PackF::ZERO; std::cmp::max(packed_rows, pk.codeword_len())];
        transpose_inplace(&mut codewords, &mut scratch, pk.codeword_len(), packed_rows);
        drop(scratch)
    }

    // NOTE: commit the interleaved codeword
    // we just directly commit to the packed field elements to leaves
    // Also note, when codeword is not power of 2 length, pad to nearest po2
    // to commit by merkle tree
    if !codewords.len().is_power_of_two() {
        let aligned_po2_len = codewords.len().next_power_of_two();
        codewords.resize(aligned_po2_len, PackF::ZERO);
    }

    // NOTE: ALL-TO-ALL transpose go get other world's slice of codeword
    mpi_engine.all_to_all_transpose(&mut codewords);

    let codeword_po2_len = pk.codeword_len().next_power_of_two();
    let codeword_this_world_len = packed_rows * codeword_po2_len;
    assert_eq!(codewords.len(), codeword_this_world_len);

    let codeword_po2_chunk_len = codeword_po2_len / mpi_engine.world_size();
    let global_packed_rows = packed_rows * mpi_engine.world_size();
    assert_eq!(codeword_po2_len % mpi_engine.world_size(), 0);

    if packed_rows > 1 {
        let codeword_chunk_per_world_len = codeword_po2_chunk_len * packed_rows;

        // NOTE: now transpose back to row order of each world's codeword slice
        let mut scratch = vec![PackF::ZERO; std::cmp::max(codeword_po2_chunk_len, packed_rows)];
        codewords
            .chunks_mut(codeword_chunk_per_world_len)
            .for_each(|c| transpose_inplace(c, &mut scratch, packed_rows, codeword_po2_chunk_len));
        drop(scratch);
    }

    // NOTE: transpose back into column order of the codeword slice
    let mut scratch = vec![PackF::ZERO; std::cmp::max(codeword_po2_chunk_len, global_packed_rows)];
    transpose_inplace(
        &mut codewords,
        &mut scratch,
        codeword_po2_chunk_len,
        global_packed_rows,
    );
    drop(scratch);

    scratch_pad.interleaved_alphabet_commitment =
        Tree::compact_new_with_packed_field_elems(codewords);

    // NOTE: gather local roots and compute the final MT root
    let local_commitment = scratch_pad.interleaved_alphabet_commitment.root();
    let mut leaves = vec![tree::Node::default(); mpi_engine.world_size()];
    mpi_engine.gather_vec(&[local_commitment], &mut leaves);

    {
        let mut leaves_bytes: Vec<u8> = Vec::new();
        leaves.serialize_into(&mut leaves_bytes)?;
        mpi_engine.root_broadcast_bytes(&mut leaves_bytes);

        if !mpi_engine.is_root() {
            leaves = Vec::deserialize_from(leaves_bytes.as_slice())?;
        }
    }

    scratch_pad.merkle_cap = leaves.clone();

    let root = {
        let height = 1 + leaves.len().ilog2();
        let internal = tree::Tree::new_with_leaf_nodes(&leaves, height);
        internal[0]
    };

    Ok(root)
}

#[inline(always)]
pub(crate) fn orion_mpi_mt_openings<T>(
    mpi_engine: &impl MPIEngine,
    pk: &OrionSRS,
    scratch_pad: &OrionScratchPad,
    transcript: &mut T,
) -> Option<Vec<RangePath>>
where
    T: Transcript,
{
    let num_leaves_per_opening = pk.num_leaves_per_mt_query();

    // NOTE: MT opening for point queries
    let query_num = pk.query_complexity(PCS_SOUNDNESS_BITS);
    let query_indices: Vec<usize> = {
        let mut indices = transcript.generate_usize_vector(query_num);
        indices.iter_mut().for_each(|q| *q %= pk.codeword_len());
        indices
    };

    let index_range_per_world = pk.codeword_len().next_power_of_two() / mpi_engine.world_size();
    let index_starts_this_world = index_range_per_world * mpi_engine.world_rank();
    let index_ends_this_world = index_starts_this_world + index_range_per_world;

    let local_paths: Vec<RangePath> = query_indices
        .iter()
        .filter(|&&index| index_starts_this_world <= index && index < index_ends_this_world)
        .map(|index| {
            let left = (index - index_starts_this_world) * num_leaves_per_opening;
            scratch_pad
                .interleaved_alphabet_commitment
                .range_query(left, left + num_leaves_per_opening - 1)
        })
        .collect();

    let mut global_paths: Vec<Vec<RangePath>> = Vec::new();
    mpi_engine.gather_varlen_vec(&local_paths, &mut global_paths);

    if !mpi_engine.is_root() {
        return None;
    }

    let mut global_paths_deque: Vec<VecDeque<RangePath>> =
        global_paths.into_iter().map(VecDeque::from).collect();

    let flattened_paths: Vec<RangePath> = query_indices
        .iter()
        .map(|q| {
            let which_world = q / index_range_per_world;
            global_paths_deque[which_world].pop_front().unwrap()
        })
        .collect();

    flattened_paths.into()
}
