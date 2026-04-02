// Copyright 2025 Irreducible Inc.

use binius_core::word::Word;
use binius_field::{
    BinaryField, Field, PackedField,
    linear_transformation::{
        BytewiseLookupTransformationFactory, LinearTransformationFactory,
        OutputWrappingTransformationFactory, Transformation,
    },
};
use binius_math::FieldBuffer;
use binius_utils::{checked_arithmetics::log2_strict_usize, rayon::prelude::*};

/// Computes a [`FieldBuffer`] where each element is the inner product of the bits of a word and a
/// vector of field elements.
///
/// Returns a buffer where element `i` is the inner product of the bits of word `i` in `words`
/// (mapping bit 0 to [`Field::ZERO`] and bit 1 to [`Field::ONE`]) and the values in `vec`.
///
/// This implementation uses the [Method of Four Russians] to optimize the computation by
/// precomputing a small lookup table and looking up into it using bitwise chunks of the words.
///
/// ## Preconditions
/// * `vec` contains exactly [`binius_core::consts::WORD_SIZE_BITS`] elements
/// * `words` has a power-of-two length
///
/// [Method of Four Russians]: <https://en.wikipedia.org/wiki/Method_of_Four_Russians>
pub fn fold_words<F, P>(words: &[Word], vec: &[F]) -> FieldBuffer<P>
where
    F: BinaryField,
    P: PackedField<Scalar = F>,
{
    fold_words_with_transform_factory(
        &OutputWrappingTransformationFactory::new(BytewiseLookupTransformationFactory),
        words,
        vec,
    )
}

pub fn fold_words_with_transform_factory<F, P, TransformFactory>(
    transform_factory: &TransformFactory,
    words: &[Word],
    vec: &[F],
) -> FieldBuffer<P>
where
    F: Field,
    P: PackedField<Scalar = F>,
    TransformFactory: LinearTransformationFactory<u64, F>,
{
    assert!(words.len().is_power_of_two()); // precondition

    let transform = transform_factory.create(vec);

    let log_n = log2_strict_usize(words.len());

    // Collect the folded results using the WordTransform
    let values = words
        .par_chunks(P::WIDTH)
        .map(|word_chunk| {
            P::from_scalars(word_chunk.iter().map(|&word| transform.transform(&word.0)))
        })
        .collect::<Vec<_>>();

    FieldBuffer::new(log_n, values.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use binius_core::consts::WORD_SIZE_BITS;
    use binius_math::test_utils::random_scalars;
    use binius_verifier::config::B128;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    fn naive_fold_words<F, P>(words: &[Word], vec: &[F]) -> FieldBuffer<P>
    where
        F: Field,
        P: PackedField<Scalar = F>,
    {
        assert_eq!(vec.len(), WORD_SIZE_BITS);
        assert!(words.len().is_power_of_two());

        let log_n = log2_strict_usize(words.len());

        let values = words
            .par_chunks(P::WIDTH)
            .map(|word_chunk| {
                P::from_scalars(word_chunk.iter().map(|&word| {
                    // Decompose word into bits and compute inner product
                    let mut sum = F::ZERO;
                    for bit_idx in 0..WORD_SIZE_BITS {
                        if (word.as_u64() >> bit_idx) & 1 == 1 {
                            sum += vec[bit_idx];
                        }
                    }
                    sum
                }))
            })
            .collect();

        FieldBuffer::new(log_n, values)
    }

    #[test]
    fn test_fold_words_equivalence() {
        let mut rng = StdRng::seed_from_u64(0);

        let log_n = 6;
        let n_words = 1 << log_n;

        let words = (0..n_words)
            .map(|_| Word::from_u64(rng.random::<u64>()))
            .collect::<Vec<_>>();

        let vec = random_scalars(&mut rng, WORD_SIZE_BITS);

        // Compute using both methods
        let result_optimized = fold_words::<B128, B128>(&words, &vec);
        let result_naive = naive_fold_words::<B128, B128>(&words, &vec);

        // Compare results
        assert_eq!(result_optimized, result_naive);
    }
}
