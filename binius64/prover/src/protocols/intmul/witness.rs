// Copyright 2025 Irreducible Inc.

use std::{iter, marker::PhantomData, mem::MaybeUninit, ops::Deref};

use binius_field::{BinaryField, Field, PackedField};
use binius_math::field_buffer::FieldBuffer;
use binius_utils::{
    bitwise::{BitSelector, Bitwise},
    checked_arithmetics::{checked_log_2, strict_log_2},
    random_access_sequence::RandomAccessSequence,
    rayon::prelude::*,
    strided_array::StridedArray2DViewMut,
};
use derive_more::IntoIterator;
use getset::Getters;
use itertools::{Itertools, iterate};

use super::error::Error;
use crate::protocols::prodcheck::ProdcheckProver;

/// An integer multiplication protocol witness. Created from integer slices, consumed during
/// proving.
///
/// The statement being proven is `a * b = c`, where `c` is represented as a pair `(c_lo, c_hi)`.
/// All four values are of the same bit width that is passed to the prover via `log_bits` parameter
/// (also denoted $m$). In Binius64, `log_bits = 6` for 64-bit multiplicands and 128-bit product.
///
/// A full binary tree (see [`BinaryTree`]) is constructed from each of `a`, `c_lo`, `c_hi`:
///  1) `a` and `c_lo` select a multiplicative group generator $G$
///  2) `c_hi` selects $G^{2^{2^m}}$
///
/// For `b`, we only store the leaves (for prodcheck) and the root (for initial evaluation):
///  3) `b` selects variable base which is equal to the root of the `a` tree
///
/// Protocol proves that ${(G^a)}^b = G^{c\\_lo} \times (G^{2^{2^m}})^{c\\_hi}$, which is equivalent
/// to $a \times b = c$ modulo $2^{2^{m+1}} - 1$. The special case of `0 * 0 = 1` is handled
/// separately.
#[derive(Clone, Getters)]
#[getset(get = "pub")]
pub struct Witness<P: PackedField, B: Bitwise, S: AsRef<[B]> + Sync> {
    pub a: BinaryTree<P, B, S>,
    /// The exponents for `b` (needed for phase 5).
    pub b_exponents: S,
    /// Concatenated b leaves for prodcheck: [L_0, L_1, ..., L_{2^k-1}].
    /// Has log_len = n_vars + log_bits.
    pub b_leaves: FieldBuffer<P>,
    /// The prover for the prodcheck reduction on b_leaves.
    pub b_prodcheck: ProdcheckProver<P>,
    /// The root of the b tree (product of all leaves element-wise).
    pub b_root: FieldBuffer<P>,
    pub c_lo: BinaryTree<P, B, S>,
    pub c_hi: BinaryTree<P, B, S>,
    pub c_root: FieldBuffer<P>,
}

impl<F, P, B, S> Witness<P, B, S>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
    B: Bitwise,
    S: AsRef<[B]> + Sync,
{
    /// Constructs a new integer multiplication witness from the statement.
    ///
    /// For statement of size $2^\ell$ using $2^m$-wide integers, the upper bound on the
    /// witness size is $8^{\ell+m}$ large field elements.
    pub fn new(log_bits: usize, a: S, b: S, c_lo: S, c_hi: S) -> Result<Self, Error> {
        // Statement should be of pow-2 length.
        let Some(n_vars) = strict_log_2(a.as_ref().len()) else {
            return Err(Error::ExponentsPowerOfTwoLengthRequired);
        };

        // All statement slices should be of same length.
        if [&a, &b, &c_lo, &c_hi]
            .iter()
            .any(|exponents| exponents.as_ref().len() != 1 << n_vars)
        {
            return Err(Error::ExponentLengthMismatch);
        }

        let g = F::MULTIPLICATIVE_GENERATOR;
        let g_c_hi = iterate(g, |g| g.square())
            .nth(1 << log_bits)
            .expect("infinite iterator");

        let a = BinaryTree::constant_base(log_bits, g, a);
        let c_lo = BinaryTree::constant_base(log_bits, g, c_lo);
        let c_hi = BinaryTree::constant_base(log_bits, g_c_hi, c_hi);

        // Compute b_leaves as concatenated leaves for prodcheck
        let variable_base = a.root().clone();
        let b_leaves = compute_b_leaves(log_bits, variable_base, &b);

        // Create the prodcheck prover; its products layer becomes b_root
        let (b_prodcheck, b_root) = ProdcheckProver::new(log_bits, b_leaves.clone());

        // The root of a `log_bits + 1` deep tree of the full product `c`.
        let c_root = buffer_bivariate_product(c_lo.root(), c_hi.root());

        Ok(Self {
            a,
            b_exponents: b,
            b_leaves,
            b_prodcheck,
            b_root,
            c_lo,
            c_hi,
            c_root,
        })
    }
}

/// Compute concatenated b_leaves for prodcheck.
///
/// Each leaf `L_z` contains: if bit z of `exponents[i]` is set then `bases[i]^{2^z}` else 1
/// The leaves are concatenated: `[L_0, L_1, ..., L_{2^k-1}]`
fn compute_b_leaves<F, P, B, S>(
    log_bits: usize,
    bases: FieldBuffer<P>,
    exponents: &S,
) -> FieldBuffer<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
    S: AsRef<[B]> + Sync,
{
    let n_vars = bases.log_len();

    if P::LOG_WIDTH <= n_vars {
        // Parallel optimized path
        return compute_b_leaves_parallel(log_bits, &bases, exponents);
    }

    // Fallback: bases is too small to parallelize (n_vars < P::LOG_WIDTH)
    let mut out = FieldBuffer::zeros(n_vars + log_bits);
    let n_elems = 1 << n_vars;

    let one_bit = B::from(1u8);
    for (i, (mut base, &exp)) in iter::zip(bases.iter_scalars(), exponents.as_ref()).enumerate() {
        for z in 0..1 << log_bits {
            let bit = (exp >> z) & one_bit == one_bit;

            out.set(z * n_elems + i, if bit { base } else { F::ONE });

            base = base.square();
        }
    }

    out
}

/// Parallel implementation of compute_b_leaves for when bases is large enough to parallelize.
fn compute_b_leaves_parallel<F, P, B, S>(
    log_bits: usize,
    bases: &FieldBuffer<P>,
    exponents: &S,
) -> FieldBuffer<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
    S: AsRef<[B]> + Sync,
{
    let n_vars = bases.log_len();
    let n_packed = bases.as_ref().len();
    let height = 1 << log_bits;
    let total = n_packed * height;

    let mut out_vec: Vec<P> = Vec::with_capacity(total);

    {
        let spare: &mut [MaybeUninit<P>] = out_vec.spare_capacity_mut();

        let mut strided = StridedArray2DViewMut::without_stride(spare, height, n_packed)
            .expect("dimensions match capacity");

        let one_bit = B::from(1u8);

        (
            strided.par_iter_cols(),
            bases.as_ref(),
            exponents.as_ref().par_chunks(P::WIDTH),
        )
            .into_par_iter()
            .for_each(|(mut col, packed_base, exp_chunk)| {
                // Keep base as packed element for efficient squaring
                let mut packed_base = *packed_base;

                for z in 0..height {
                    // Decompose to scalars, apply bit selection, recompose
                    // TODO: Optimize with bit-masking for selection
                    let scalars = packed_base.iter().zip(exp_chunk).map(|(base, &exp)| {
                        let bit = (exp >> z) & one_bit == one_bit;
                        if bit { base } else { F::ONE }
                    });

                    col[z].write(P::from_scalars(scalars));

                    // Square packed base for next iteration
                    packed_base = packed_base.square();
                }
            });
    }

    // SAFETY: All elements initialized in the parallel loop above
    unsafe { out_vec.set_len(total) };

    FieldBuffer::new(n_vars + log_bits, out_vec.into_boxed_slice())
}

/// A helper structure which handles full GKR binary tree for the bivariate product.
///
/// On the lowest, widest level, the tree contains `2^log_bits` leaves. Each of the leaves
/// is a selected multilinear, meaning that `i`-th multilinear contains field multiplicative
/// identity if the `i`-th bit on the exponent corresponding to the hypercube vertex is zero,
/// and some base otherwise. Base can be constant (powers of some generator) or variable (value
/// is specified per hypercube vertex).
///
/// Upper `log_bits` of the tree are constructed by taking pairwise per-vertex products of
/// multilinears. The root contains a single multilinear, each vertex value of which equals to the
/// base raised to the power of the corresponding exponent.
///
/// Tree is laid out from root to the leaves. `IntoIterator` follows this convention.
#[derive(Clone, Debug, IntoIterator)]
pub struct BinaryTree<P: PackedField, B: Bitwise, S: AsRef<[B]>> {
    exponents: S,
    #[into_iterator(owned)]
    tree: Vec<Vec<FieldBuffer<P>>>,
    _b_marker: PhantomData<B>,
}

impl<F, P, B, S> BinaryTree<P, B, S>
where
    F: Field,
    P: PackedField<Scalar = F>,
    B: Bitwise,
    S: AsRef<[B]> + Sync,
{
    /// Constant base witness construction.
    pub fn constant_base(log_bits: usize, base: F, exponents: S) -> Self {
        let bases = iterate(base, |g| g.square())
            .take(1 << log_bits)
            .collect::<Vec<_>>();

        let widest_layer = bases
            .par_iter()
            .enumerate()
            .map(|(bit_offset, base)| {
                two_valued_field_buffer(bit_offset, &exponents, [F::ONE, *base])
            })
            .collect();

        let tree = build_remaining_tree_layers(log_bits, widest_layer);
        Self {
            exponents,
            tree,
            _b_marker: PhantomData,
        }
    }

    /// Variable base witness construction.
    ///
    /// The `bases` buffer is consumed and modified inplace.
    pub fn variable_base(log_bits: usize, mut bases: FieldBuffer<P>, exponents: S) -> Self {
        let n_vars = checked_log_2(exponents.as_ref().len());
        let p_width = P::WIDTH.min(1 << n_vars);
        assert_eq!(bases.log_len(), n_vars);

        let mut widest_layer = Vec::with_capacity(1 << log_bits);
        for bit_offset in 0..1 << log_bits {
            let bits = BitSelector::new(bit_offset, &exponents);
            let values = bases
                .as_mut()
                .into_par_iter()
                .enumerate()
                .map(|(i, bases_packed)| {
                    let scalars = bases_packed
                        .iter()
                        .take(p_width)
                        .enumerate()
                        .map(|(j, base)| {
                            let is_base = unsafe {
                                // Safety: `bits` is guaranteed to be in-bounds
                                bits.get_unchecked(i << P::LOG_WIDTH | j)
                            };
                            if is_base { base } else { F::ONE }
                        });

                    let result = P::from_scalars(scalars);
                    *bases_packed = bases_packed.square();

                    result
                })
                .collect::<Box<[_]>>();

            widest_layer.push(FieldBuffer::new(n_vars, values));
        }

        let tree = build_remaining_tree_layers(log_bits, widest_layer);
        Self {
            exponents,
            tree,
            _b_marker: PhantomData,
        }
    }

    pub fn log_bits(&self) -> usize {
        self.tree.len() - 1
    }

    pub fn root(&self) -> &FieldBuffer<P> {
        let first_layer = self
            .tree
            .first()
            .expect("at least one layer is always present");
        assert_eq!(first_layer.len(), 1);

        first_layer.first().expect("first_layer.len() == 1")
    }

    pub fn split(mut self) -> (S, FieldBuffer<P>, Vec<Vec<FieldBuffer<P>>>) {
        let rest = self.tree.split_off(1);
        let mut root_layer = self.tree.pop().expect("exactly one element");
        let root = root_layer.pop().expect("exactly one element");
        (self.exponents, root, rest)
    }
}

fn build_remaining_tree_layers<P: PackedField>(
    log_bits: usize,
    widest_layer: Vec<FieldBuffer<P>>,
) -> Vec<Vec<FieldBuffer<P>>> {
    assert_eq!(widest_layer.len(), 1 << log_bits);

    let mut tree = Vec::with_capacity(log_bits + 1);
    tree.push(widest_layer);

    for layer_no in (0..log_bits).rev() {
        let cur_layer = tree.last().expect("always at least one layer in tree");

        assert_eq!(cur_layer.len(), 2 << layer_no);
        let next_layer = cur_layer
            .iter()
            .tuples()
            .map(|(a, b)| buffer_bivariate_product(a, b))
            .collect::<Vec<_>>();

        tree.push(next_layer);
    }

    tree.reverse();
    tree
}

/// Compute the per-vertex bivariate product of two equally sized field buffers.
pub fn buffer_bivariate_product<P: PackedField, Data: Deref<Target = [P]>>(
    a: &FieldBuffer<P, Data>,
    b: &FieldBuffer<P, Data>,
) -> FieldBuffer<P> {
    assert_eq!(a.len(), b.len());
    let product = (a.as_ref(), b.as_ref())
        .into_par_iter()
        .map(|(&a, &b)| a * b)
        .collect::<Box<[P]>>();
    FieldBuffer::new(a.log_len(), product)
}

/// Constructs a field buffer with values selected from `elements` based on the bit values
/// of `exponents`.
pub fn two_valued_field_buffer<F, P, S, B>(
    bit_offset: usize,
    exponents: &S,
    elements: [F; 2],
) -> FieldBuffer<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
    S: AsRef<[B]> + Sync,
    B: Bitwise,
{
    let n_vars = checked_log_2(exponents.as_ref().len());
    let p_width = P::WIDTH.min(1 << n_vars);
    let bits = BitSelector::new(bit_offset, &exponents);
    let values = (0..1 << n_vars.saturating_sub(P::LOG_WIDTH))
        .map(|i| {
            let scalars = (0..p_width).map(|j| {
                // The following code is equivalent to
                // ```
                // if bits.get(i << P::LOG_WIDTH | j) {
                // 	elements[1]
                // } else {
                // 	elements[0]
                // }
                // ```
                unsafe {
                    // Safety:
                    // - `i << P::LOG_WIDTH | j` is guaranteed to be in-bounds
                    // - elements has two values
                    *elements.get_unchecked(bits.get_unchecked(i << P::LOG_WIDTH | j) as usize)
                }
            });
            P::from_scalars(scalars)
        })
        .collect::<Box<[_]>>();

    FieldBuffer::new(n_vars, values)
}

#[cfg(test)]
mod tests {
    use binius_math::test_utils::Packed128b;

    use super::*;

    type P = Packed128b;

    const LOG_BITS: usize = 6;

    fn check_consistency<P: PackedField, B: Bitwise, S: AsRef<[B]> + Sync>(
        witness: &Witness<P, B, S>,
    ) {
        let b_root = witness.b_root();
        let c_root = witness.c_root();
        assert_eq!(b_root, c_root);
    }

    #[test]
    fn test_forwards() {
        let a: u64 = 2;
        let b: u64 = 3;
        let c_lo: u64 = 6; // 2*3 = 6
        let c_hi: u64 = 0; // no high bits

        let witness = Witness::<P, _, [u64; 1]>::new(LOG_BITS, [a], [b], [c_lo], [c_hi]).unwrap();
        check_consistency(&witness);
    }

    #[test]
    fn test_forwards_larger() {
        let a: u64 = 1 << 32;
        let b: u64 = 1 << 33;
        let c_lo: u64 = 0;
        let c_hi: u64 = 2; // 2^32 * 2^33 = 2^65, which is 2 in the high 64 bits

        let witness = Witness::<P, _, [u64; 1]>::new(LOG_BITS, [a], [b], [c_lo], [c_hi]).unwrap();
        check_consistency(&witness);
    }

    #[test]
    fn test_forwards_multiple_random() {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        let mut rng = StdRng::seed_from_u64(0);

        const VECTOR_SIZE: usize = 8;
        let mut a = Vec::with_capacity(VECTOR_SIZE);
        let mut b = Vec::with_capacity(VECTOR_SIZE);
        let mut c_lo = Vec::with_capacity(VECTOR_SIZE);
        let mut c_hi = Vec::with_capacity(VECTOR_SIZE);

        for _ in 0..VECTOR_SIZE {
            let a_i = rng.random_range(1..u64::MAX);
            let b_i = rng.random_range(1..u64::MAX);

            let full_result = (a_i as u128) * (b_i as u128);
            let c_lo_i = full_result as u64;
            let c_hi_i = (full_result >> 64) as u64;

            a.push(a_i);
            b.push(b_i);
            c_lo.push(c_lo_i);
            c_hi.push(c_hi_i);
        }

        let witness = Witness::<P, _, _>::new(LOG_BITS, a, b, c_lo, c_hi).unwrap();
        check_consistency(&witness);
    }
}
