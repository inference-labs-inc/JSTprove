// Copyright 2025 Irreducible Inc.
use binius_core::consts::WORD_SIZE_BITS;
use binius_frontend::{CircuitBuilder, Wire};

use crate::{
    bignum::{BigUint, assert_eq, select as select_biguint},
    secp256k1::{
        N_LIMBS, Secp256k1, Secp256k1Affine, coord_lambda, coord_zero,
        select as select_secp256k1_affine,
    },
};

/// Compute scalar multiplication `point * scalar` using the naive double-and-add algorithm.
///
/// This implementation does not use the secp256k1 endomorphism optimization.
///
/// # Parameters
/// - `b`: The circuit builder
/// - `curve`: The secp256k1 curve instance
/// - `bits`: Number of bits to process in the scalar
/// - `scalar`: The scalar to multiply by (as a BigUint, must have enough limbs for bits)
/// - `point`: The point to multiply (in affine coordinates)
///
/// # Returns
/// The result of `point * scalar` in affine coordinates
pub fn scalar_mul_naive(
    b: &CircuitBuilder,
    curve: &Secp256k1,
    bits: usize,
    scalar: &BigUint,
    point: Secp256k1Affine,
) -> Secp256k1Affine {
    // Ensure scalar has enough limbs for the requested bits
    let required_limbs = bits.div_ceil(WORD_SIZE_BITS);
    assert!(
        scalar.limbs.len() >= required_limbs,
        "scalar must have at least {} limbs for {} bits, but has {}",
        required_limbs,
        bits,
        scalar.limbs.len()
    );

    let mut acc = Secp256k1Affine::point_at_infinity(b);

    for bit_index in (0..bits).rev() {
        let limb = bit_index / WORD_SIZE_BITS;
        let bit = bit_index % WORD_SIZE_BITS;

        if bit_index != bits - 1 {
            acc = curve.double(b, &acc);
        }

        let scalar_bit = b.shl(scalar.limbs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

        // Add the selected point to the accumulator
        let acc_plus_point = curve.add_incomplete(b, &acc, &point);

        // Select whether to add point to accumulator based on scalar bit
        acc = select_secp256k1_affine(b, scalar_bit, &acc_plus_point, &acc);
    }

    acc
}

/// Compute scalar multiplication `point * scalar` using the secp256k1 endomorphism optimization.
///
/// This implementation uses the curve's endomorphism to split the scalar into two ~128-bit
/// components, reducing the number of doublings from 256 to 128.
///
/// # Parameters
/// - `b`: The circuit builder
/// - `curve`: The secp256k1 curve instance
/// - `scalar`: The scalar to multiply by (as a BigUint with N_LIMBS)
/// - `point`: The point to multiply (in affine coordinates)
///
/// # Returns
///
/// The result of `point * scalar` in affine coordinates
pub fn scalar_mul(
    b: &CircuitBuilder,
    curve: &Secp256k1,
    scalar: &BigUint,
    point: Secp256k1Affine,
) -> Secp256k1Affine {
    assert_eq!(scalar.limbs.len(), N_LIMBS);

    // Nondeterministically split the scalar, constrain the split
    let (k1_neg, k2_neg, k1_abs, k2_abs) = b.secp256k1_endomorphism_split_hint(&scalar.limbs);

    check_endomorphism_split(b, curve, k1_neg, k2_neg, k1_abs, k2_abs, scalar);

    // Compute the endomorphism of the point
    let point_endo = curve.endomorphism(b, &point);

    // The split returns "signed scalars" (which is required to fit them into 128 bits).
    // Negate the base if needed to only care about positive exponents.
    let p1 = curve.negate_if(b, k1_neg, &point);
    let p2 = curve.negate_if(b, k2_neg, &point_endo);

    // Compute the 4-element lookup table: {0, P1, P2, P1+P2}
    let lookup = vec![
        Secp256k1Affine::point_at_infinity(b),
        p1.clone(),
        p2.clone(),
        curve.add_incomplete(b, &p1, &p2),
    ];

    let mut acc = Secp256k1Affine::point_at_infinity(b);

    for bit_index in (0..128).rev() {
        let limb = bit_index / WORD_SIZE_BITS;
        let bit = bit_index % WORD_SIZE_BITS;

        if bit_index != 127 {
            acc = curve.double(b, &acc);
        }

        // Extract the current bit from each scalar component
        let k1_bit = b.shl(k1_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
        let k2_bit = b.shl(k2_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

        // Perform 2-bit lookup using nested selection
        // This selects one of the 4 lookup table entries based on the two bits
        let mut level = lookup.clone();
        for sel_bit in [k1_bit, k2_bit] {
            let next_level = level
                .chunks(2)
                .map(|pair| {
                    assert_eq!(pair.len(), 2);
                    select_secp256k1_affine(b, sel_bit, &pair[1], &pair[0])
                })
                .collect();
            level = next_level;
        }

        assert_eq!(level.len(), 1);
        acc = curve.add_incomplete(b, &acc, &level[0]);
    }

    acc
}

/// A common trick to save doublings when computing multiexponentiations of the form
/// `G*g_mult + PK*pk_mult` - instead of doing two scalar multiplications separately and
/// adding their results, we share the doubling step of double-and-add.
///
/// For secp256k1, we can go one step further: the curve has an endomorphism `λ (x, y) = (βx, y)`
/// where `λ³=1 (mod n)` and `β³=1 (mod p)` (`n` being the scalar field modulus and `p` coordinate
/// field one). For a 256-bit scalar `k` it is possible to split it into `k1` and `k2` such that
/// `k1 + λ k2 = k (mod n)` and both `k1` and `k2` are no farther than `2^128` from zero.
///
/// Using the above fact, we can "split" both the G and PK 256-bit multiplier scalars into a total
/// of four 128-bit subscalars. Instead of 4-wide lookup in `shamirs_trick_naive`, we do a 16-wide
/// lookup for all subset sums of `{G, G_endo, PK, PK_endo}`, where `*_endo` points are obtained via
/// endomorphism. This halves the total number of doublings and additions at a cost of a larger
/// precomputation, but the eventual savings are still in the order of 2x.
///
/// Returns `G*g_mult + PK*pk_mult`.
pub fn shamirs_trick_endomorphism(
    b: &CircuitBuilder,
    curve: &Secp256k1,
    g_mult: &BigUint,
    pk_mult: &BigUint,
    pk: Secp256k1Affine,
) -> Secp256k1Affine {
    assert_eq!(g_mult.limbs.len(), N_LIMBS);
    assert_eq!(pk_mult.limbs.len(), N_LIMBS);

    // Nondeterministically split both scalars, constrain the splits
    let (g1_mult_neg, g2_mult_neg, g1_mult_abs, g2_mult_abs) =
        b.secp256k1_endomorphism_split_hint(&g_mult.limbs);

    check_endomorphism_split(
        b,
        curve,
        g1_mult_neg,
        g2_mult_neg,
        g1_mult_abs,
        g2_mult_abs,
        g_mult,
    );

    let (pk1_mult_neg, pk2_mult_neg, pk1_mult_abs, pk2_mult_abs) =
        b.secp256k1_endomorphism_split_hint(&pk_mult.limbs);

    check_endomorphism_split(
        b,
        curve,
        pk1_mult_neg,
        pk2_mult_neg,
        pk1_mult_abs,
        pk2_mult_abs,
        pk_mult,
    );

    // Compute the endomorphisms
    let g = Secp256k1Affine::generator(b);
    let g_endo = curve.endomorphism(b, &g);
    let pk_endo = curve.endomorphism(b, &pk);

    // The split returns "signed scalars" (which is required to fit them into 128 bits).
    // Negate the base if needed to only care about positive exponents.
    let g1 = curve.negate_if(b, g1_mult_neg, &g);
    let g2 = curve.negate_if(b, g2_mult_neg, &g_endo);

    let pk1 = curve.negate_if(b, pk1_mult_neg, &pk);
    let pk2 = curve.negate_if(b, pk2_mult_neg, &pk_endo);

    // Compute subset sums of {G, G_endo, PK, PK_endo} using a total of 11 additions
    let mut lookup = Vec::with_capacity(16);
    lookup.push(Secp256k1Affine::point_at_infinity(b));
    for (i, pt) in [g1, g2, pk1, pk2].into_iter().enumerate() {
        lookup.push(pt.clone());
        for j in 1..1 << i {
            lookup.push(curve.add_incomplete(b, &lookup[j], &pt));
        }
    }

    let mut acc = Secp256k1Affine::point_at_infinity(b);

    for bit_index in (0..128).rev() {
        let limb = bit_index / WORD_SIZE_BITS;
        let bit = bit_index % WORD_SIZE_BITS;

        if bit_index != 127 {
            acc = curve.double(b, &acc);
        }

        // This is essentially an inlined multi wire multiplexer, but due to the fact
        // it uses affine point conditional selections and separate wires instead of masks
        // it's simpler to inline it there.
        // TODO: replace it with a multiplexer once the abstraction is mature enough
        let g1_mult_bit = b.shl(g1_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
        let g2_mult_bit = b.shl(g2_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
        let pk1_mult_bit = b.shl(pk1_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
        let pk2_mult_bit = b.shl(pk2_mult_abs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

        let mut level = lookup.clone();
        for sel_bit in [g1_mult_bit, g2_mult_bit, pk1_mult_bit, pk2_mult_bit] {
            let next_level = level
                .chunks(2)
                .map(|pair| {
                    assert_eq!(pair.len(), 2);
                    select_secp256k1_affine(b, sel_bit, &pair[1], &pair[0])
                })
                .collect();

            level = next_level;
        }

        assert_eq!(level.len(), 1);
        acc = curve.add_incomplete(b, &acc, &level[0]);
    }

    acc
}

// Constrain the return value of `CircuitBuilder::secp256k1_endomorphism_split_hint`.
// Verifies that `k1 + λ k2 = k (mod n)` where `n` is scalar field modulus.
fn check_endomorphism_split(
    b: &CircuitBuilder,
    curve: &Secp256k1,
    k1_neg: Wire,
    k2_neg: Wire,
    k1_abs: [Wire; 2],
    k2_abs: [Wire; 2],
    k: &BigUint,
) {
    assert_eq!(k.limbs.len(), N_LIMBS);

    let k1_abs = BigUint {
        limbs: k1_abs.to_vec(),
    }
    .zero_extend(b, N_LIMBS);
    let k2_abs = BigUint {
        limbs: k2_abs.to_vec(),
    }
    .zero_extend(b, N_LIMBS);

    let f_scalar = curve.f_scalar();
    let k1 = select_biguint(
        b,
        k1_neg,
        &f_scalar.sub(b, &coord_zero(b), &k1_abs),
        &k1_abs,
    );
    let k2 = select_biguint(
        b,
        k2_neg,
        &f_scalar.sub(b, &coord_zero(b), &k2_abs),
        &k2_abs,
    );

    assert_eq(
        b,
        "endomorphism split k1 + λk2 = k (mod n)",
        k,
        &f_scalar.add(b, &k1, &f_scalar.mul(b, &k2, &coord_lambda(b))),
    );
}

/// A common trick to save doublings when computing multiexponentiations of the form
/// `G*g_mult + PK*pk_mult` - instead of doing two scalar multiplications separately and
/// adding their results, we share the doubling step of double-and-add.
///
/// This implementation relies on group axioms only. It is currently unused for secp256k1
/// but may prove useful for other curves.
#[allow(unused)]
pub fn shamirs_trick_naive(
    b: &CircuitBuilder,
    curve: &Secp256k1,
    bits: usize,
    g_mult: &BigUint,
    pk_mult: &BigUint,
    pk: Secp256k1Affine,
) -> Secp256k1Affine {
    let g = Secp256k1Affine::generator(b);
    let g_pk = curve.add(b, &g, &pk);

    let mut acc = Secp256k1Affine::point_at_infinity(b);

    for bit_index in (0..bits).rev() {
        let limb = bit_index / WORD_SIZE_BITS;
        let bit = bit_index % WORD_SIZE_BITS;

        if bit_index != bits - 1 {
            acc = curve.double(b, &acc);
        }

        let g_mult_bit = b.shl(g_mult.limbs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);
        let pk_mult_bit = b.shl(pk_mult.limbs[limb], (WORD_SIZE_BITS - 1 - bit) as u32);

        // A 3-to-1 mux
        let x = select_biguint(
            b,
            pk_mult_bit,
            &select_biguint(b, g_mult_bit, &g_pk.x, &pk.x),
            &g.x,
        );
        let y = select_biguint(
            b,
            pk_mult_bit,
            &select_biguint(b, g_mult_bit, &g_pk.y, &pk.y),
            &g.y,
        );

        // Point at infinity flag is a single wire, allowing us to save a BigUint select.
        let is_point_at_infinity = b.band(b.bnot(g_mult_bit), b.bnot(pk_mult_bit));

        // Addition implementation is incomplete (it handles pai, but not doubling). When
        // the mask is zero, pai-to-pai support is needed. The probability of accumulator
        // assuming value G, PK, or G+PK at any point in the computation is vanishingly low.
        // We assert false in this case, resulting in a completeness gap.
        acc = curve.add_incomplete(
            b,
            &acc,
            &Secp256k1Affine {
                x,
                y,
                is_point_at_infinity,
            },
        );
    }

    acc
}

#[cfg(test)]
mod tests {
    use binius_core::word::Word;
    use binius_frontend::CircuitBuilder;
    use k256::{
        ProjectivePoint, Scalar, U256,
        elliptic_curve::{ops::MulByGenerator, scalar::FromUintUnchecked, sec1::ToEncodedPoint},
    };
    use rand::prelude::*;

    use super::*;
    use crate::{
        bignum::{BigUint, assert_eq},
        secp256k1::{Secp256k1, Secp256k1Affine},
    };

    #[test]
    fn test_scalar_mul_naive() {
        let builder = CircuitBuilder::new();
        let curve = Secp256k1::new(&builder);

        // Test with scalar = 69
        let scalar_value = 69u64;

        // Use k256 to compute the expected result
        let k256_scalar = Scalar::from(scalar_value);
        let k256_point = ProjectivePoint::mul_by_generator(&k256_scalar).to_affine();

        // Extract coordinates from k256 result
        let point_bytes = k256_point.to_encoded_point(false).to_bytes();
        // The uncompressed format is: 0x04 || x || y (65 bytes total)
        // We need to extract x and y coordinates (32 bytes each)
        let x_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[1..33]);
        let y_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[33..65]);

        // Create our scalar as BigUint
        let scalar = BigUint::new_constant(&builder, &num_bigint::BigUint::from(scalar_value));

        // Create expected coordinates as BigUint
        let expected_x = BigUint::new_constant(&builder, &x_coord);
        let expected_y = BigUint::new_constant(&builder, &y_coord);

        // Get the generator point
        let generator = Secp256k1Affine::generator(&builder);

        // Perform scalar multiplication with our implementation
        let result = scalar_mul_naive(&builder, &curve, 7, &scalar, generator);

        // Check that the result matches the expected point
        assert_eq(&builder, "result_x", &result.x, &expected_x);
        assert_eq(&builder, "result_y", &result.y, &expected_y);

        // Build and verify the circuit
        let cs = builder.build();
        let mut w = cs.new_witness_filler();
        assert!(cs.populate_wire_witness(&mut w).is_ok());

        // Also verify the point is not at infinity
        assert_eq!(w[result.is_point_at_infinity], Word::ZERO);
    }

    #[test]
    fn test_scalar_mul_with_endomorphism() {
        let builder = CircuitBuilder::new();
        let curve = Secp256k1::new(&builder);

        // Generate a random 256-bit scalar
        let mut rng = StdRng::seed_from_u64(0);
        let mut scalar_bytes = [0u8; 32];
        rng.fill(&mut scalar_bytes);

        // Create the scalar in both k256 and our format
        let k256_uint = U256::from_be_slice(&scalar_bytes);
        let k256_scalar = Scalar::from_uint_unchecked(k256_uint);
        let scalar_bigint = num_bigint::BigUint::from_bytes_be(&scalar_bytes);
        let scalar = BigUint::new_constant(&builder, &scalar_bigint).zero_extend(&builder, N_LIMBS);

        // Use k256 to compute the expected result with the generator point
        let k256_point = ProjectivePoint::mul_by_generator(&k256_scalar).to_affine();

        // Extract coordinates from k256 result
        let point_bytes = k256_point.to_encoded_point(false).to_bytes();
        let x_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[1..33]);
        let y_coord = num_bigint::BigUint::from_bytes_be(&point_bytes[33..65]);

        // Create expected coordinates as BigUint
        let expected_x = BigUint::new_constant(&builder, &x_coord);
        let expected_y = BigUint::new_constant(&builder, &y_coord);

        // Get the generator point
        let generator = Secp256k1Affine::generator(&builder);

        // Perform scalar multiplication with our endomorphism implementation
        let result = scalar_mul(&builder, &curve, &scalar, generator);

        // Check that the result matches the expected point
        assert_eq(&builder, "result_x", &result.x, &expected_x);
        assert_eq(&builder, "result_y", &result.y, &expected_y);

        // Build and verify the circuit
        let cs = builder.build();
        let mut w = cs.new_witness_filler();
        assert!(cs.populate_wire_witness(&mut w).is_ok());

        // Verify the point is not at infinity
        assert_eq!(w[result.is_point_at_infinity], Word::ZERO);
    }
}
