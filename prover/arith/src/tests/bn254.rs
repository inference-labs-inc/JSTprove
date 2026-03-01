use ark_std::test_rng;
use halo2curves::bn256::Fr;

use super::{
    random_fft_field_tests, random_field_tests, random_from_limbs_to_limbs_tests,
    random_inversion_tests, random_simd_field_tests,
};

#[test]
fn test_field() {
    random_field_tests::<Fr>("bn254::Fr".to_string());
    random_simd_field_tests::<Fr>("bn254::Fr".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<Fr, _>(&mut rng, "bn254::Fr".to_string());
    random_from_limbs_to_limbs_tests::<Fr, Fr>("bn254::Fr".to_string());
    random_fft_field_tests::<Fr>("bn254::Fr".to_string());
}
