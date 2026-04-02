// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::CircuitBuilder;
use hex_literal::hex;

use crate::{
    bignum::{BigUint, assert_eq},
    ecdsa::{bitcoin_verify, ecrecover},
    secp256k1::Secp256k1Affine,
};

#[test]
pub fn test_bitcoin_ecdsa_test_vector() {
    let builder = CircuitBuilder::new();

    let [pkx, pky, z, r, s] = [
        hex!("3AF1E1EFA4D1E1AD5CB9E3967E98E901DAFCD37C44CF0BFB6C216997F5EE51DF"),
        hex!("E4ACAC3E6F139E0C7DB2BD736824F51392BDA176965A1C59EB9C3C5FF9E85D7A"),
        hex!("3F891FDA3704F0368DAB65FA81EBE616F4AA2A0854995DA4DC0B59D2CADBD64F"),
        hex!("A5C7B7756D34D8AAF6AA68F0B71644F0BEF90D8BFD126CE951B6060498345089"),
        hex!("BC9644F1625AF13841E589FD00653AE8C763309184EA0DE481E8F06709E5D1CB"),
    ]
    .map(|bytes| {
        let native = num_bigint::BigUint::from_bytes_be(&bytes);
        BigUint::new_constant(&builder, &native)
    });

    let is_point_at_infinity = builder.add_constant(Word::ZERO);
    let pk = Secp256k1Affine {
        x: pkx,
        y: pky,
        is_point_at_infinity,
    };

    let signature_valid = bitcoin_verify(&builder, pk, &z, &r, &s);

    let cs = builder.build();
    let mut w = cs.new_witness_filler();
    cs.populate_wire_witness(&mut w).unwrap();

    assert_eq!(w[signature_valid] >> 63, Word::ONE);
}

#[test]
pub fn test_ecdsa_recover_test_vector() {
    let builder = CircuitBuilder::new();

    let [pkx, pky, z, r, s] = [
        hex!("3AF1E1EFA4D1E1AD5CB9E3967E98E901DAFCD37C44CF0BFB6C216997F5EE51DF"),
        hex!("E4ACAC3E6F139E0C7DB2BD736824F51392BDA176965A1C59EB9C3C5FF9E85D7A"),
        hex!("3F891FDA3704F0368DAB65FA81EBE616F4AA2A0854995DA4DC0B59D2CADBD64F"),
        hex!("A5C7B7756D34D8AAF6AA68F0B71644F0BEF90D8BFD126CE951B6060498345089"),
        hex!("BC9644F1625AF13841E589FD00653AE8C763309184EA0DE481E8F06709E5D1CB"),
    ]
    .map(|bytes| {
        let native = num_bigint::BigUint::from_bytes_be(&bytes);
        BigUint::new_constant(&builder, &native)
    });

    let recid_odd = builder.add_constant(Word::ALL_ONE);
    let pk = ecrecover(&builder, &z, &r, &s, recid_odd);

    assert_eq(&builder, "pkx", &pk.x, &pkx);
    assert_eq(&builder, "pky", &pk.y, &pky);

    let cs = builder.build();
    let mut w = cs.new_witness_filler();
    assert!(cs.populate_wire_witness(&mut w).is_ok());
}
