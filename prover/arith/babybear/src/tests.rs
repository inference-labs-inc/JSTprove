use arith::{
    random_extension_field_tests, random_fft_field_tests, random_field_tests,
    random_inversion_tests, random_simd_field_tests, Field, FieldParameters,
};
use ark_std::test_rng;
use serdes::ExpSerde;

use crate::{babybear::BabyBearParameters, BabyBear, BabyBearExt3, BabyBearExt3x16, BabyBearx16};

// CMD: RUSTFLAGS="-C target-feature=+avx512f" cargo test --package arith --lib --
// tests::baby_bear::test_field --exact --show-output
#[test]
fn test_base_field() {
    random_field_tests::<BabyBear>("BabyBear".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<BabyBear, _>(&mut rng, "BabyBear".to_string());

    random_fft_field_tests::<BabyBear>("Baby Bear".to_string());
}

#[test]
fn test_simd_field() {
    random_field_tests::<BabyBearx16>("BabyBearx16".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<BabyBearx16, _>(&mut rng, "BabyBearx16".to_string());

    random_simd_field_tests::<BabyBearx16>("BabyBearx16".to_string());

    random_fft_field_tests::<BabyBearx16>("BabyBearx16".to_string());

    let a = BabyBearx16::from(256u32 + 2);
    let mut buffer = vec![];
    assert!(a.serialize_into(&mut buffer).is_ok());
    let b = BabyBearx16::deserialize_from(buffer.as_slice());
    assert!(b.is_ok());
    let b = b.unwrap();
    assert_eq!(a, b);
}

// CMD: RUSTFLAGS="-C target-feature=+avx512f" cargo test --package arith --lib --
// tests::baby_bear_ext::test_field --exact --show-output
#[test]
fn test_ext_field() {
    // Deg 3
    random_field_tests::<BabyBearExt3>("Baby Bear Ext3".to_string());
    random_extension_field_tests::<BabyBearExt3>("Baby Bear Ext3".to_string());

    random_field_tests::<BabyBearExt3x16>("Simd Baby Bear Ext3".to_string());
    random_extension_field_tests::<BabyBearExt3x16>("Simd Baby Bear Ext3".to_string());
    random_simd_field_tests::<BabyBearExt3x16>("Simd Baby Bear Ext3".to_string());

    random_fft_field_tests::<BabyBearExt3>("Baby Bear Ext3".to_string());
    random_fft_field_tests::<BabyBearExt3x16>("SIMD Baby Bear Ext3".to_string());
    // // Deg 4
    // random_field_tests::<BabyBearExt4>("Baby Bear Ext4".to_string());
    // random_extension_field_tests::<BabyBearExt4>("Baby Bear Ext4".to_string());

    // random_field_tests::<BabyBearExt4x16>("Simd Baby Bear Ext4".to_string());
    // random_extension_field_tests::<BabyBearExt4x16>("Simd Baby Bear Ext4".to_string());
    // random_simd_field_tests::<BabyBearExt4x16>("Simd Baby Bear Ext4".to_string());
}

#[test]
fn baby_bear_two_inverse() {
    let two = BabyBear::new(2);
    let two_inverse = BabyBearParameters::try_inverse(&two).unwrap();
    // Display impl converts to canonical form
    println!(
        "2^-1 (canonical form): {} Montgomery {:?}",
        two_inverse, two_inverse
    );

    // Check correctness
    let two = BabyBear::new(2);
    let two_inverse_canonical: u32 = 1006632961;
    let two_inverse = BabyBear::new(two_inverse_canonical);
    let one = BabyBear::ONE;
    assert_eq!(one, two * two_inverse)
}

#[test]
fn test_exponentiation() {
    use rand::{rngs::OsRng, Rng};
    let mut rng = OsRng;

    for _ in 0..1000 {
        // Use a small base to avoid overflow
        let base_u32: u32 = rng.gen_range(0..=10);
        let base = BabyBear::new(base_u32);
        // Use a small exponent to avoid overflow
        let exponent: u32 = rng.gen_range(0..=5);
        let expected_result = BabyBear::new(base_u32.pow(exponent));
        assert_eq!(base.exp(exponent as u128), expected_result);
    }
}
