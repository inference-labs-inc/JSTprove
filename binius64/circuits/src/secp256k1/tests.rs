// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::CircuitBuilder;

use super::{curve::Secp256k1, point::Secp256k1Affine};

#[test]
fn test_secp256k1_group_order() {
    let order = num_bigint::BigUint::from_bytes_be(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFE, 0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36,
        0x41, 0x41,
    ]);

    let builder = CircuitBuilder::new();

    let curve = Secp256k1::new(&builder);

    let generator = Secp256k1Affine::generator(&builder);

    let mut acc = Secp256k1Affine::point_at_infinity(&builder);

    for i in (0..256).rev() {
        acc = curve.double(&builder, &acc);

        if order.bit(i) {
            acc = curve.add(&builder, &acc, &generator);
        }
    }

    let cs = builder.build();
    let mut w = cs.new_witness_filler();
    cs.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[acc.is_point_at_infinity] >> 63, Word::ONE);
}

#[test]
fn test_secp256k1_pow2pow137() {
    let builder = CircuitBuilder::new();

    let curve = Secp256k1::new(&builder);

    let generator = Secp256k1Affine::generator(&builder);

    let mut acc = Secp256k1Affine::point_at_infinity(&builder);

    for _i in 0..137 {
        acc = curve.double(&builder, &acc);
        acc = curve.add(&builder, &generator, &acc);
    }

    let cs = builder.build();
    let mut w = cs.new_witness_filler();
    cs.populate_wire_witness(&mut w).unwrap();

    // 0xede2ae24a4f24f0e70e764555e24170cebf045931e5bff973caff9355246e643
    assert_eq!(w[acc.x.limbs[0]], Word(0x3caff9355246e643));
    assert_eq!(w[acc.x.limbs[1]], Word(0xebf045931e5bff97));
    assert_eq!(w[acc.x.limbs[2]], Word(0x70e764555e24170c));
    assert_eq!(w[acc.x.limbs[3]], Word(0xede2ae24a4f24f0e));

    // 0xa551ada705cab99114b2451de109cc6941178def3dd73e644dfb7839703c1219
    assert_eq!(w[acc.y.limbs[0]], Word(0x4dfb7839703c1219));
    assert_eq!(w[acc.y.limbs[1]], Word(0x41178def3dd73e64));
    assert_eq!(w[acc.y.limbs[2]], Word(0x14b2451de109cc69));
    assert_eq!(w[acc.y.limbs[3]], Word(0xa551ada705cab991));
}

#[test]
fn test_secp256k1_generator_double_and_add() {
    let builder = CircuitBuilder::new();

    let curve = Secp256k1::new(&builder);

    let generator = Secp256k1Affine::generator(&builder);
    let double = curve.double(&builder, &generator);
    let triple = curve.add(&builder, &double, &generator);

    let cs = builder.build();
    let mut w = cs.new_witness_filler();
    cs.populate_wire_witness(&mut w).unwrap();

    // 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
    assert_eq!(w[generator.x.limbs[0]], Word(0x59f2815b16f81798));
    assert_eq!(w[generator.x.limbs[1]], Word(0x029bfcdb2dce28d9));
    assert_eq!(w[generator.x.limbs[2]], Word(0x55a06295ce870b07));
    assert_eq!(w[generator.x.limbs[3]], Word(0x79be667ef9dcbbac));

    // 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
    assert_eq!(w[generator.y.limbs[0]], Word(0x9c47d08ffb10d4b8));
    assert_eq!(w[generator.y.limbs[1]], Word(0xfd17b448a6855419));
    assert_eq!(w[generator.y.limbs[2]], Word(0x5da4fbfc0e1108a8));
    assert_eq!(w[generator.y.limbs[3]], Word(0x483ada7726a3c465));

    // 0xc6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
    assert_eq!(w[double.x.limbs[0]], Word(0xabac09b95c709ee5));
    assert_eq!(w[double.x.limbs[1]], Word(0x5c778e4b8cef3ca7));
    assert_eq!(w[double.x.limbs[2]], Word(0x3045406e95c07cd8));
    assert_eq!(w[double.x.limbs[3]], Word(0xc6047f9441ed7d6d));

    // 0x1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a
    assert_eq!(w[double.y.limbs[0]], Word(0x236431a950cfe52a));
    assert_eq!(w[double.y.limbs[1]], Word(0xf7f632653266d0e1));
    assert_eq!(w[double.y.limbs[2]], Word(0xa3c58419466ceaee));
    assert_eq!(w[double.y.limbs[3]], Word(0x1ae168fea63dc339));

    // 0xf9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9
    assert_eq!(w[triple.x.limbs[0]], Word(0x8601f113bce036f9));
    assert_eq!(w[triple.x.limbs[1]], Word(0xb531c845836f99b0));
    assert_eq!(w[triple.x.limbs[2]], Word(0x49344f85f89d5229));
    assert_eq!(w[triple.x.limbs[3]], Word(0xf9308a019258c310));

    // 0x388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672
    assert_eq!(w[triple.y.limbs[0]], Word(0x6cb9fd7584b8e672));
    assert_eq!(w[triple.y.limbs[1]], Word(0x6500a99934c2231b));
    assert_eq!(w[triple.y.limbs[2]], Word(0x0fe337e62a37f356));
    assert_eq!(w[triple.y.limbs[3]], Word(0x388f7b0f632de814));
}
