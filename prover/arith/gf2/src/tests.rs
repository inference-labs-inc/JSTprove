use ark_std::test_rng;
use std::io::Cursor;

use arith::{random_field_tests, random_inversion_tests, random_simd_field_tests, SimdField};

use crate::{GF2x128, GF2x64, GF2x8, GF2};

#[test]
fn test_field() {
    random_field_tests::<GF2>("GF2".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<GF2, _>(&mut rng, "GF2".to_string());
}

#[test]
fn test_simd_field() {
    random_field_tests::<GF2x8>("Vectorized GF2".to_string());
    random_simd_field_tests::<GF2x8>("Vectorized GF2".to_string());

    random_field_tests::<GF2x64>("Vectorized GF2 len 64".to_string());
    random_simd_field_tests::<GF2x64>("Vectorized GF2 len 64".to_string());

    random_field_tests::<GF2x128>("Vectorized GF2 len 128".to_string());
    random_simd_field_tests::<GF2x128>("Vectorized GF2 len 128".to_string());
}

fn custom_serde_vectorize_gf2<F: SimdField<Scalar = GF2>>() {
    let mut rng = test_rng();

    let a = F::random_unsafe(&mut rng);
    let mut buffer = vec![];
    assert!(a.serialize_into(&mut buffer).is_ok());
    let mut cursor = Cursor::new(buffer);
    let b = F::deserialize_from(&mut cursor);
    assert!(b.is_ok());
    let b = b.unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_custom_serde_vectorize_gf2() {
    custom_serde_vectorize_gf2::<GF2x8>();
    custom_serde_vectorize_gf2::<GF2x64>();
    custom_serde_vectorize_gf2::<GF2x128>()
}
