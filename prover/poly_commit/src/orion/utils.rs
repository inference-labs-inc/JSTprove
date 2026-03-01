use arith::{ExtensionField, Field, SimdField};
use gkr_engine::Transcript;
use itertools::izip;
use serdes::{ExpSerde, SerdeError};
use thiserror::Error;
use transpose::{transpose, transpose_inplace};
use tree::{Node, LEAF_BYTES};

use crate::{
    orion::linear_code::{OrionCode, OrionCodeParameter, ORION_CODE_PARAMETER_INSTANCE},
    traits::TensorCodeIOPPCS,
    PCS_SOUNDNESS_BITS,
};

/*
 * PCS ERROR AND RESULT SETUP
 */

#[derive(Debug, Error)]
pub enum OrionPCSError {
    #[error("Orion PCS linear code parameter unmatch error")]
    ParameterUnmatchError,

    #[error("field serde error")]
    SerializationError(#[from] SerdeError),
}

pub type OrionResult<T> = std::result::Result<T, OrionPCSError>;

/*
 * RELEVANT TYPES SETUP
 */

/// returning leaf number, calibrated local number of variables, and message size
/// NOTE(HS) num of local variables is over base field with no SIMD
pub(crate) const fn orion_eval_shape(
    world_size: usize,
    num_local_vars: usize,
    num_bits_base_field: usize,
    field_pack_size: usize,
) -> (usize, usize, usize) {
    // for a global polynomial, compute the boolean hypercube size
    let hypercube_size = world_size * (1 << num_local_vars);

    // compute how many bits the global polynomial occupy
    let polynomial_bits = hypercube_size * num_bits_base_field;
    let polynomial_bits_log2 = polynomial_bits.ilog2();

    // the leaf should be roughly O(\log N) to the global polynomial size
    let leaves_lower_bound = match polynomial_bits_log2 {
        // 128 bytes MT opening
        0..16 => 2,

        // 256 bytes MT opening
        16..22 => 4,

        // 512 bytes MT opening
        22..28 => 8,

        // 1 KB MT opening
        28..34 => 16,

        // 2 KB MT opening
        34.. => 32,
    };

    // now compute the commitment used SIMD field bits,
    // and compared against the suggest leaf size above.
    //
    // the commitment used SIMD field is the smallest unit of memory in commitment,
    // thus each world can contribute as low as 1 commitment used SIMD field element,
    // and thus if the world size is too large, we might need to scale up the final
    // leaf size based on suggested leaf size - and thus may require padding the poly
    // to the right size.
    let num_bits_packed_f = field_pack_size * num_bits_base_field;
    let minimum_pack_fs_per_world_in_query = {
        let leaves_bits_lower_bound = leaves_lower_bound * LEAF_BYTES * 8;
        // NOTE(HS) div_ceil ensures each world contribute at least 1 commitment SIMD field element
        // in the MT query opening
        leaves_bits_lower_bound.div_ceil(num_bits_packed_f * world_size)
    };

    let final_leaves_bits = minimum_pack_fs_per_world_in_query * world_size * num_bits_packed_f;
    let final_leaves_per_query = final_leaves_bits / (LEAF_BYTES * 8);

    // once we have decided the leaf size, we know the message size need to be encoded.
    // the consideration is that - resulting codeword size should be >= MPI world size.
    //
    // Otherwise, the MPI transpose cannot go through - the codeword is too short to be
    // segmented into the number of shares spreaded across MPI words for all-to-all transpose.
    let msg_size = polynomial_bits / final_leaves_bits;

    // we want to know the mininum message size, such that the resulting codeword length
    // (pad to next po2) >= the MPI world size.
    //
    // for small case, when input message is shorter than Expander code length threshold,
    // the code is itself (by Druk-Ishai),
    // otherwise Expander encoding applies, with a code rate > 1 / 2, eventually pad to 2x length.
    //
    // NOTE(HS): here we assume that the message size and the MPI world size are both po2,
    // while Expander code length threshold might be arbitarary - we scale up the threshold
    // to the next po2 to avoid corner case.
    let minimum_msg_size = {
        let min_expander_po2_code_len = ORION_CODE_PARAMETER_INSTANCE
            .length_threshold_g0s
            .next_power_of_two();

        if world_size <= min_expander_po2_code_len {
            world_size
        } else {
            world_size / 2
        }
    };

    // Early exit - if the message size from the global poly is sufficiently large,
    // we decide the global MT opening size, and are good with the local polynomial number of vars.
    if msg_size >= minimum_msg_size {
        return (final_leaves_per_query, num_local_vars, msg_size);
    }

    // Scale up the polynomial local number of variable and plan again for the evaluation shape.
    let scaled_up_num_local_vars = {
        let num_of_field_elems_per_world_in_query =
            minimum_pack_fs_per_world_in_query * field_pack_size;

        let minimum_poly_len = minimum_msg_size * num_of_field_elems_per_world_in_query;
        minimum_poly_len.ilog2() as usize
    };

    orion_eval_shape(
        world_size,
        scaled_up_num_local_vars,
        num_bits_base_field,
        field_pack_size,
    )
}

#[derive(Clone, Debug, Default, ExpSerde)]
pub struct OrionSRS {
    pub num_vars: usize,
    pub num_leaves_per_mt_query: usize,
    pub code_instance: OrionCode,
}

impl TensorCodeIOPPCS for OrionSRS {
    fn message_len(&self) -> usize {
        self.code_instance.msg_len()
    }

    fn codeword_len(&self) -> usize {
        self.code_instance.code_len()
    }

    fn minimum_hamming_weight(&self) -> f64 {
        self.code_instance.hamming_weight()
    }

    fn num_leaves_per_mt_query(&self) -> usize {
        self.num_leaves_per_mt_query
    }
}

impl OrionSRS {
    // NOTE(HS) num local variables here refers to the number of variables for base field elements
    // rather than SIMD field elements, the number of variables returned for calibration is also
    // over base field elements rather than SIMD field elements.
    pub fn from_random(
        world_size: usize,
        num_local_vars: usize,
        num_field_bits: usize,
        field_pack_size: usize,
        code_param_instance: OrionCodeParameter,
        mut rng: impl rand::RngCore,
    ) -> (Self, usize) {
        let (num_leaves_per_mt_query, scaled_num_local_vars, msg_size) =
            orion_eval_shape(world_size, num_local_vars, num_field_bits, field_pack_size);

        let srs_sampled = Self {
            num_vars: scaled_num_local_vars,
            num_leaves_per_mt_query,
            code_instance: OrionCode::new(code_param_instance, msg_size, &mut rng),
        };

        (srs_sampled, scaled_num_local_vars)
    }

    pub fn local_num_fs_per_query(&self) -> usize {
        let local_poly_len = 1 << self.num_vars;
        local_poly_len / self.message_len()
    }
}

pub type OrionCommitment = Node;

#[derive(Clone, Debug, Default, ExpSerde)]
pub struct OrionScratchPad {
    pub interleaved_alphabet_commitment: tree::Tree,
    pub merkle_cap: Vec<Node>,
}

#[derive(Clone, Debug, Default, ExpSerde)]
pub struct OrionProof<EvalF: Field> {
    pub eval_row: Vec<EvalF>,
    pub proximity_rows: Vec<Vec<EvalF>>,
    pub query_openings: Vec<tree::RangePath>,
    pub merkle_cap: Vec<Node>,
}

#[inline(always)]
pub(crate) fn commit_encoded<PackF>(
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
    let mut scratch = vec![PackF::ZERO; std::cmp::max(packed_rows, pk.codeword_len())];
    transpose_inplace(&mut codewords, &mut scratch, pk.codeword_len(), packed_rows);
    drop(scratch);

    // NOTE: commit the interleaved codeword
    // we just directly commit to the packed field elements to leaves
    // Also note, when codeword is not power of 2 length, pad to nearest po2
    // to commit by merkle tree
    if !codewords.len().is_power_of_two() {
        let aligned_po2_len = codewords.len().next_power_of_two();
        codewords.resize(aligned_po2_len, PackF::ZERO);
    }
    scratch_pad.interleaved_alphabet_commitment =
        tree::Tree::compact_new_with_packed_field_elems(codewords);

    scratch_pad.merkle_cap = vec![scratch_pad.interleaved_alphabet_commitment.root()];

    Ok(scratch_pad.interleaved_alphabet_commitment.root())
}

#[inline(always)]
pub(crate) fn orion_mt_openings<T>(
    pk: &OrionSRS,
    transcript: &mut T,
    scratch_pad: &OrionScratchPad,
) -> Vec<tree::RangePath>
where
    T: Transcript,
{
    let leaves_in_range_opening = pk.num_leaves_per_mt_query();

    // NOTE: MT opening for point queries
    let query_num = pk.query_complexity(PCS_SOUNDNESS_BITS);
    let query_indices = transcript.generate_usize_vector(query_num);
    query_indices
        .iter()
        .map(|qi| {
            let index = *qi % pk.codeword_len();
            let left = index * leaves_in_range_opening;
            let right = left + leaves_in_range_opening - 1;

            scratch_pad
                .interleaved_alphabet_commitment
                .range_query(left, right)
        })
        .collect()
}

#[inline(always)]
pub(crate) fn orion_mt_verify(
    vk: &OrionSRS,
    query_indices: &[usize],
    range_openings: &[tree::RangePath],
    merkle_cap: &[Node],
) -> bool {
    let world_size = merkle_cap.len();
    let indices_per_merkle_cap = vk.codeword_len().next_power_of_two() / world_size;

    izip!(query_indices, range_openings).all(|(&qi, range_path)| {
        let index = qi % vk.codeword_len();
        let merkle_cap_index = index / indices_per_merkle_cap;
        let in_sub_tree_index = index % indices_per_merkle_cap;

        range_path.verify(&merkle_cap[merkle_cap_index])
            && in_sub_tree_index == range_path.left / range_path.leaves.len()
    })
}

/*
 * LINEAR OPERATIONS FOR GF2 (LOOKUP TABLE BASED)
 */

pub struct SubsetSumLUTs<F: Field> {
    pub entry_bits: usize,
    pub tables: Vec<Vec<F>>,
}

impl<F: Field> SubsetSumLUTs<F> {
    #[inline(always)]
    pub fn new(entry_bits: usize, table_num: usize) -> Self {
        assert!(entry_bits > 0 && table_num > 0);

        Self {
            entry_bits,
            tables: vec![vec![F::ZERO; 1 << entry_bits]; table_num],
        }
    }

    #[inline(always)]
    pub fn build(&mut self, weights: &[F]) {
        assert_eq!(weights.len(), self.entry_bits * self.tables.len());

        self.tables.iter_mut().for_each(|lut| lut.fill(F::ZERO));

        // NOTE: we are assuming that the table is for {0, 1}-linear combination
        izip!(&mut self.tables, weights.chunks(self.entry_bits)).for_each(
            |(lut_i, sub_weights)| {
                sub_weights.iter().enumerate().for_each(|(i, weight_i)| {
                    let bit_mask = 1 << i;
                    lut_i.iter_mut().enumerate().for_each(|(bit_map, li)| {
                        if bit_map & bit_mask == bit_mask {
                            *li += weight_i;
                        }
                    });
                });
            },
        );
    }

    #[inline(always)]
    pub fn lookup_and_sum<EntryF>(&self, indices: &[EntryF]) -> F
    where
        EntryF: SimdField,
    {
        // NOTE: at least the entry field elem should have a matching field size
        // and the the entry field being a SIMD field with same packing size as
        // the bits for the table entry
        assert_eq!(EntryF::FIELD_SIZE, 1);
        assert_eq!(EntryF::PACK_SIZE, self.entry_bits);
        assert_eq!(indices.len(), self.tables.len());

        izip!(&self.tables, indices)
            .map(|(t_i, index)| t_i[index.as_u32_unchecked() as usize])
            .sum()
    }
}

#[inline(always)]
pub(crate) fn lut_open_linear_combine<F, EvalF, SimdF>(
    com_pack_size: usize,
    packed_evals: &[SimdF],
    eq_col_coeffs: &[EvalF],
    eval_row: &mut [EvalF],
    rand_col_coeffs: &[Vec<EvalF>],
    proximity_rows: &mut [Vec<EvalF>],
) where
    F: Field,
    EvalF: ExtensionField<BaseField = F>,
    SimdF: SimdField<Scalar = F>,
{
    // NOTE: declare the look up tables for column sums
    let table_num = com_pack_size / SimdF::PACK_SIZE;
    let mut luts = SubsetSumLUTs::<EvalF>::new(SimdF::PACK_SIZE, table_num);
    assert_eq!(com_pack_size % SimdF::PACK_SIZE, 0);

    let combination_size = eq_col_coeffs.len();
    let packed_row_size = combination_size / com_pack_size;

    // NOTE: working on evaluation response of tensor code IOP based PCS
    izip!(
        eq_col_coeffs.chunks(com_pack_size),
        packed_evals.chunks(packed_evals.len() / packed_row_size)
    )
    .for_each(|(eq_chunk, packed_evals_chunk)| {
        luts.build(eq_chunk);

        izip!(packed_evals_chunk.chunks(table_num), &mut *eval_row)
            .for_each(|(p_col, eval)| *eval += luts.lookup_and_sum(p_col));
    });

    // NOTE: compose proximity response(s) of tensor code IOP based PCS
    izip!(proximity_rows, rand_col_coeffs).for_each(|(row_buffer, random_coeffs)| {
        izip!(
            random_coeffs.chunks(com_pack_size),
            packed_evals.chunks(packed_evals.len() / packed_row_size)
        )
        .for_each(|(random_chunk, packed_evals_chunk)| {
            luts.build(random_chunk);

            izip!(packed_evals_chunk.chunks(table_num), &mut *row_buffer)
                .for_each(|(p_col, prox)| *prox += luts.lookup_and_sum(p_col));
        });
    });
    drop(luts);
}

#[inline(always)]
pub(crate) fn lut_verify_alphabet_check<F, SimdF, ExtF>(
    codeword: &[ExtF],
    fixed_rl: &[ExtF],
    query_indices: &[usize],
    packed_interleaved_alphabets: &[Vec<SimdF>],
) -> bool
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    ExtF: ExtensionField<BaseField = F>,
{
    // NOTE: build up lookup table
    let tables_num = fixed_rl.len() / SimdF::PACK_SIZE;
    assert_eq!(fixed_rl.len() % SimdF::PACK_SIZE, 0);
    let mut luts = SubsetSumLUTs::<ExtF>::new(SimdF::PACK_SIZE, tables_num);

    luts.build(fixed_rl);

    izip!(query_indices, packed_interleaved_alphabets).all(|(qi, interleaved_alphabet)| {
        let index = qi % codeword.len();
        let alphabet = luts.lookup_and_sum(interleaved_alphabet);
        alphabet == codeword[index]
    })
}

/*
 * LINEAR OPERATIONS FOR MERSENNE31 (SIMD BASED)
 */

#[inline(always)]
fn simd_ext_base_inner_prod<F, ExtF, SimdF>(
    simd_ext_limbs: &[SimdF],
    simd_base_elems: &[SimdF],
) -> ExtF
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    ExtF: ExtensionField<BaseField = F>,
{
    assert_eq!(simd_ext_limbs.len(), simd_base_elems.len() * ExtF::DEGREE);

    let mut ext_limbs = vec![F::ZERO; ExtF::DEGREE];

    izip!(&mut ext_limbs, simd_ext_limbs.chunks(simd_base_elems.len())).for_each(
        |(e, simd_ext_limb)| {
            let simd_sum: SimdF = izip!(simd_ext_limb, simd_base_elems)
                .map(|(a, b)| *a * b)
                .sum();
            *e = simd_sum.horizontal_sum();
        },
    );

    ExtF::from_limbs(&ext_limbs)
}

#[inline(always)]
pub(crate) fn simd_open_linear_combine<F, EvalF, SimdF>(
    com_pack_size: usize,
    packed_evals: &[SimdF],
    eq_col_coeffs: &[EvalF],
    eval_row: &mut [EvalF],
    rand_col_coeffs: &[Vec<EvalF>],
    proximity_rows: &mut [Vec<EvalF>],
) where
    F: Field,
    EvalF: ExtensionField<BaseField = F>,
    SimdF: SimdField<Scalar = F>,
{
    // NOTE: check SIMD inner product numbers for column sums
    let simd_inner_prods = com_pack_size / SimdF::PACK_SIZE;
    assert_eq!(com_pack_size % SimdF::PACK_SIZE, 0);

    let combination_size = eq_col_coeffs.len();
    let packed_row_size = combination_size / com_pack_size;

    let mut buffer = vec![F::ZERO; com_pack_size * EvalF::DEGREE];

    // NOTE: working on evaluation response of tensor code IOP based PCS
    izip!(
        eq_col_coeffs.chunks(com_pack_size),
        packed_evals.chunks(packed_evals.len() / packed_row_size)
    )
    .for_each(|(eq_chunk, packed_evals_chunk)| {
        let eq_limbs: Vec<_> = eq_chunk.iter().flat_map(|e| e.to_limbs()).collect();
        transpose(&eq_limbs, &mut buffer, EvalF::DEGREE, com_pack_size);
        let simd_limbs: Vec<_> = buffer.chunks(SimdF::PACK_SIZE).map(SimdF::pack).collect();

        izip!(packed_evals_chunk.chunks(simd_inner_prods), &mut *eval_row).for_each(
            |(p_col, eval)| *eval += simd_ext_base_inner_prod::<_, EvalF, _>(&simd_limbs, p_col),
        );
    });

    // NOTE: compose proximity response(s) of tensor code IOP based PCS
    izip!(proximity_rows, rand_col_coeffs).for_each(|(prox_row, random_coeffs)| {
        izip!(
            random_coeffs.chunks(com_pack_size),
            packed_evals.chunks(packed_evals.len() / packed_row_size)
        )
        .for_each(|(rand_chunk, packed_evals_chunk)| {
            let rand_limbs: Vec<_> = rand_chunk.iter().flat_map(|e| e.to_limbs()).collect();
            transpose(&rand_limbs, &mut buffer, EvalF::DEGREE, com_pack_size);
            let simd_limbs: Vec<_> = buffer.chunks(SimdF::PACK_SIZE).map(SimdF::pack).collect();

            izip!(packed_evals_chunk.chunks(simd_inner_prods), &mut *prox_row).for_each(
                |(p_col, prox)| {
                    *prox += simd_ext_base_inner_prod::<_, EvalF, _>(&simd_limbs, p_col)
                },
            );
        });
    });
}

#[inline(always)]
pub(crate) fn simd_verify_alphabet_check<F, SimdF, ExtF>(
    codeword: &[ExtF],
    fixed_rl: &[ExtF],
    query_indices: &[usize],
    packed_interleaved_alphabets: &[Vec<SimdF>],
) -> bool
where
    F: Field,
    SimdF: SimdField<Scalar = F>,
    ExtF: ExtensionField<BaseField = F>,
{
    // NOTE: check SIMD inner product numbers for column sums
    assert_eq!(fixed_rl.len() % SimdF::PACK_SIZE, 0);

    let mut scratch = vec![F::ZERO; fixed_rl.len() * ExtF::DEGREE];
    let rl_limbs: Vec<_> = fixed_rl.iter().flat_map(|e| e.to_limbs()).collect();
    transpose(&rl_limbs, &mut scratch, ExtF::DEGREE, fixed_rl.len());
    let simd_limbs: Vec<_> = scratch.chunks(SimdF::PACK_SIZE).map(SimdF::pack).collect();

    izip!(query_indices, packed_interleaved_alphabets).all(|(qi, interleaved_alphabet)| {
        let index = qi % codeword.len();
        let alphabet: ExtF = simd_ext_base_inner_prod(&simd_limbs, interleaved_alphabet);
        alphabet == codeword[index]
    })
}

#[cfg(test)]
mod tests {
    use arith::{Field, SimdField};
    use ark_std::test_rng;
    use gf2::{GF2x8, GF2};
    use gf2_128::{GF2_128x8, GF2_128};
    use itertools::izip;

    use crate::orion::utils::SubsetSumLUTs;

    #[test]
    fn test_lut_simd_inner_prod_consistency() {
        let mut rng = test_rng();

        let weights: Vec<_> = (0..8).map(|_| GF2_128::random_unsafe(&mut rng)).collect();
        let bases: Vec<_> = (0..8).map(|_| GF2::random_unsafe(&mut rng)).collect();

        let simd_weights = GF2_128x8::pack(&weights);
        let simd_bases = GF2x8::pack(&bases);

        let expected_simd_inner_prod: GF2_128 = (simd_weights * simd_bases).horizontal_sum();

        let expected_vanilla_inner_prod: GF2_128 =
            izip!(&weights, &bases).map(|(w, b)| *w * *b).sum();

        assert_eq!(expected_simd_inner_prod, expected_vanilla_inner_prod);

        let mut table = SubsetSumLUTs::new(8, 1);
        table.build(&weights);

        let actual_lut_inner_prod = table.lookup_and_sum(&[simd_bases]);

        assert_eq!(expected_simd_inner_prod, actual_lut_inner_prod)
    }
}
