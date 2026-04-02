// Copyright 2024-2025 Irreducible Inc.

use std::iter;

use proptest::prelude::*;

use crate::{
    AESTowerField8b, BinaryField1b, BinaryField128bGhash, ExtensionField, Field,
    PackedAESBinaryField4x8b, PackedAESBinaryField8x8b, PackedAESBinaryField16x8b,
    PackedAESBinaryField32x8b, PackedAESBinaryField64x8b, PackedBinaryField64x1b,
    PackedBinaryField128x1b, PackedBinaryField256x1b, PackedBinaryField512x1b,
    PackedBinaryGhash1x128b, PackedBinaryGhash2x128b, PackedBinaryGhash4x128b, PackedField,
    field::FieldOps,
    underlier::{SmallU, WithUnderlier},
};

#[test]
fn test_field_text_debug() {
    assert_eq!(format!("{:?}", BinaryField1b::ONE), "BinaryField1b(0x1)");
    assert_eq!(
        format!("{:?}", AESTowerField8b::new(127)),
        "AESTowerField8b(0x7f)"
    );
    assert_eq!(
        format!(
            "{:?}",
            PackedBinaryGhash1x128b::broadcast(BinaryField128bGhash::new(
                162259276829213363391578010288127
            ))
        ),
        "Packed1x128([0x000007ffffffffffffffffffffffffff])"
    );
    assert_eq!(
        format!(
            "{:?}",
            PackedAESBinaryField4x8b::broadcast(AESTowerField8b::new(123))
        ),
        "Packed4x8([0x7b,0x7b,0x7b,0x7b])"
    )
}

fn basic_spread<P>(packed: P, log_block_len: usize, block_idx: usize) -> P
where
    P: PackedField,
{
    assert!(log_block_len <= P::LOG_WIDTH);

    let block_len = 1 << log_block_len;
    let repeat = 1 << (P::LOG_WIDTH - log_block_len);
    assert!(block_idx < repeat);

    P::from_scalars(
        packed
            .iter()
            .skip(block_idx * block_len)
            .take(block_len)
            .flat_map(|elem| iter::repeat_n(elem, repeat)),
    )
}

macro_rules! generate_spread_tests_small {
    ($($name:ident, $type:ty, $scalar:ty, $underlier:ty, $width: expr);* $(;)?) => {
        proptest! {
            $(
                #[test]
                #[allow(clippy::modulo_one)]
                fn $name(z in any::<[u8; $width]>()) {
                    let indexed_packed_field = <$type>::from_fn(|i| <$scalar>::from_underlier(<$underlier>::new(z[i])));
                    for log_block_len in 0..=<$type>::LOG_WIDTH {
						for block_idx in 0..(1 <<(<$type>::LOG_WIDTH - log_block_len)) {
							assert_eq!(
								basic_spread(indexed_packed_field, log_block_len, block_idx),
								indexed_packed_field.spread(log_block_len, block_idx)
							);
						}
					}
                }
            )*
        }
    };
}

macro_rules! generate_spread_tests {
    ($($name:ident, $type:ty, $scalar:ty, $underlier:ty, $width: expr);* $(;)?) => {
        proptest! {
            $(
                #[test]
                #[allow(clippy::modulo_one)]
                fn $name(z in any::<[$underlier; $width]>()) {
                    let indexed_packed_field = <$type>::from_fn(|i| <$scalar>::from_underlier(z[i].into()));
                    for log_block_len in 0..=<$type>::LOG_WIDTH {
						for block_idx in 0..(1 <<(<$type>::LOG_WIDTH - log_block_len)) {
							assert_eq!(
								basic_spread(indexed_packed_field, log_block_len, block_idx),
								indexed_packed_field.spread(log_block_len, block_idx)
							);
						}
					}
				}
            )*
        }
    };
}

generate_spread_tests! {
    // 128-bit configurations
    spread_equals_basic_spread_4x128, PackedBinaryGhash4x128b, BinaryField128bGhash, u128, 4;
    spread_equals_basic_spread_2x128, PackedBinaryGhash2x128b, BinaryField128bGhash, u128, 2;
    spread_equals_basic_spread_1x128, PackedBinaryGhash1x128b, BinaryField128bGhash, u128, 1;

    // 8-bit configurations
    spread_equals_basic_spread_64x8, PackedAESBinaryField64x8b, AESTowerField8b, u8, 64;
    spread_equals_basic_spread_32x8, PackedAESBinaryField32x8b, AESTowerField8b, u8, 32;
    spread_equals_basic_spread_16x8, PackedAESBinaryField16x8b, AESTowerField8b, u8, 16;
    spread_equals_basic_spread_8x8, PackedAESBinaryField8x8b, AESTowerField8b, u8, 8;
}

generate_spread_tests_small! {
    // 1-bit configurations
    spread_equals_basic_spread_512x1, PackedBinaryField512x1b, BinaryField1b, SmallU<1>, 512;
    spread_equals_basic_spread_256x1, PackedBinaryField256x1b, BinaryField1b, SmallU<1>, 256;
    spread_equals_basic_spread_128x1, PackedBinaryField128x1b, BinaryField1b, SmallU<1>, 128;
    spread_equals_basic_spread_64x1, PackedBinaryField64x1b, BinaryField1b, SmallU<1>, 64;
}

fn check_field_ops_square_transpose<FSub, F>()
where
    FSub: Field,
    F: ExtensionField<FSub> + FieldOps<Scalar = F>,
{
    let degree = <F as ExtensionField<FSub>>::DEGREE;

    let elems: Vec<F> = (0..degree).map(|i| F::basis(i)).collect();

    let mut elems_transposed = elems.clone();
    <F as FieldOps>::square_transpose::<FSub>(&mut elems_transposed);

    for i in 0..degree {
        for j in 0..degree {
            assert_eq!(
                elems_transposed[i].get_base(j),
                elems[j].get_base(i),
                "mismatch at ({i}, {j})"
            );
        }
    }
}

fn check_packed_field_ops_square_transpose<FSub, P>()
where
    FSub: Field,
    P: FieldOps<Scalar: ExtensionField<FSub>> + PackedField,
{
    let degree = <P::Scalar as ExtensionField<FSub>>::DEGREE;

    let elems: Vec<P> = (0..degree)
        .map(|i| {
            P::from_fn(|_k| {
                <P::Scalar as ExtensionField<FSub>>::from_bases(
                    (0..degree).map(|j| if j == i { FSub::ONE } else { FSub::ZERO }),
                )
            })
        })
        .collect();

    let mut elems_transposed = elems.clone();
    <P as FieldOps>::square_transpose::<FSub>(&mut elems_transposed);

    for i in 0..degree {
        for k in 0..P::WIDTH {
            for j in 0..degree {
                assert_eq!(
                    elems_transposed[i].get(k).get_base(j),
                    elems[j].get(k).get_base(i),
                    "mismatch at slice_idx={i}, lane={k}, base={j}"
                );
            }
        }
    }
}

#[test]
fn test_scalar_field_ops_square_transpose_aes8b_over_1b() {
    check_field_ops_square_transpose::<BinaryField1b, AESTowerField8b>();
}

#[test]
fn test_scalar_field_ops_square_transpose_1b_over_1b() {
    check_field_ops_square_transpose::<BinaryField1b, BinaryField1b>();
}

#[test]
fn test_packed_field_ops_square_transpose_packed_aes4x8b_over_1b() {
    check_packed_field_ops_square_transpose::<BinaryField1b, PackedAESBinaryField4x8b>();
}

#[test]
fn test_packed_field_ops_square_transpose_packed128x1b_over_1b() {
    check_packed_field_ops_square_transpose::<BinaryField1b, PackedBinaryField128x1b>();
}
