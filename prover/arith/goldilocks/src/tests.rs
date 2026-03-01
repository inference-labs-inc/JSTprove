use arith::{
    random_extension_field_tests, random_fft_field_tests, random_field_tests,
    random_from_limbs_to_limbs_tests, random_inversion_tests, random_simd_field_tests,
    ExtensionField, Field,
};
use ark_std::test_rng;
use ethnum::U256;
use rand::thread_rng;
use serdes::ExpSerde;

use crate::{
    goldilocks::mod_reduce_u64, Goldilocks, GoldilocksExt2, GoldilocksExt2x8, Goldilocksx8,
    EPSILON, GOLDILOCKS_MOD,
};

fn get_avx_version() -> &'static str {
    if cfg!(all(target_arch = "x86_64", target_feature = "avx512f")) {
        return "AVX512";
    } else if cfg!(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        target_feature = "avx2"
    )) {
        return "AVX2 (256-bit)";
    } else if cfg!(target_arch = "aarch64") {
        return "arm64";
    }
    "Unknown"
}

#[test]
fn test_avx_version() {
    let avx_version = get_avx_version();
    println!("Current AVX version: {}", avx_version);
    assert!([
        "arm64",
        "AVX512",
        "AVX2 (256-bit)",
        "AVX (256-bit)",
        "No AVX (Fallback)",
        "Not x86_64 architecture"
    ]
    .contains(&avx_version));
}

#[test]
fn test_base_field() {
    random_field_tests::<Goldilocks>("Goldilocks".to_string());
    random_simd_field_tests::<Goldilocks>("Goldilocks".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<Goldilocks, _>(&mut rng, "Goldilocks".to_string());
    random_fft_field_tests::<Goldilocks>("Goldilocks".to_string());
}

#[test]
fn test_simd_field() {
    random_field_tests::<Goldilocksx8>("Goldilocksx8".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<Goldilocksx8, _>(&mut rng, "Goldilocksx8".to_string());
    random_fft_field_tests::<Goldilocksx8>("Goldilocksx8".to_string());

    random_simd_field_tests::<Goldilocksx8>("Goldilocksx8".to_string());

    let a = Goldilocksx8::from(256u32 + 2);
    let mut buffer = vec![];
    assert!(a.serialize_into(&mut buffer).is_ok());
    let b = Goldilocksx8::deserialize_from(buffer.as_slice());
    assert!(b.is_ok());
    let b = b.unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_ext_field() {
    random_field_tests::<GoldilocksExt2>("Goldilocks Ext2".to_string());
    random_extension_field_tests::<GoldilocksExt2>("Goldilocks Ext2".to_string());
    random_fft_field_tests::<GoldilocksExt2>("Goldilocks Ext2".to_string());
    random_simd_field_tests::<GoldilocksExt2>("Goldilocks Ext2".to_string());

    random_field_tests::<GoldilocksExt2x8>("Goldilocks Ext2x8".to_string());
    random_extension_field_tests::<GoldilocksExt2x8>("Goldilocks Ext2x8".to_string());
    random_fft_field_tests::<GoldilocksExt2x8>("Goldilocks Ext2x8".to_string());
    random_simd_field_tests::<GoldilocksExt2x8>("Goldilocks Ext2x8".to_string());
    random_from_limbs_to_limbs_tests::<Goldilocks, GoldilocksExt2>("Goldilocks Ext2".to_string());
    random_from_limbs_to_limbs_tests::<Goldilocksx8, GoldilocksExt2x8>(
        "Goldilocks Ext2x8".to_string(),
    );
}

/// Compare to test vectors for extension field arithmetic
#[test]
fn test_ext_field_vectors() {
    let a = GoldilocksExt2 {
        v: [Goldilocks::from(1u32), Goldilocks::from(2u32)],
    };
    let b = GoldilocksExt2 {
        v: [Goldilocks::from(3u32), Goldilocks::from(4u32)],
    };

    // Test multiplication: (1 + 2x)(3 + 4x) = (3 + 14) + (6 + 8x)x = 59 + 10x
    let expected_prod = GoldilocksExt2 {
        v: [Goldilocks::from(59u32), Goldilocks::from(10u32)],
    };
    assert_eq!(expected_prod, a * b);

    // Test inverse of a = 1 + 2x
    // (1 + 2x)(a + bx) ≡ 1 mod (x^2 - 7)
    // a + bx = (1 - 14x)/(1 + 4x)
    let expected_inv = a.inv().unwrap();
    assert_eq!(expected_inv * a, GoldilocksExt2::one());

    // Test exponentiation
    let a_pow_5 = a.exp(5);
    assert_eq!(a_pow_5, a * a * a * a * a);
}

#[test]
fn test_mod_reduction() {
    // Test reduction of values less than modulus
    assert_eq!(mod_reduce_u64(5), 5);

    // Test reduction of values equal to modulus
    assert_eq!(mod_reduce_u64(GOLDILOCKS_MOD), 0);

    // Test reduction of values greater than modulus
    assert_eq!(mod_reduce_u64(GOLDILOCKS_MOD + 1), 1);

    // Test reduction of 2^64 - 1
    assert_eq!(mod_reduce_u64(u64::MAX), EPSILON - 1);
}

#[test]
fn test_serialization() {
    let x = Goldilocks::random_unsafe(thread_rng());
    let mut bytes = Vec::new();
    x.serialize_into(&mut bytes).unwrap();
    assert_eq!(bytes.len(), 8);

    let y = Goldilocks::deserialize_from(bytes.as_slice()).unwrap();
    assert_eq!(x, y);

    // Test extension field serialization
    let x_ext = GoldilocksExt2::random_unsafe(thread_rng());
    let mut bytes = Vec::new();
    x_ext.serialize_into(&mut bytes).unwrap();
    assert_eq!(bytes.len(), 16);

    let y_ext = GoldilocksExt2::deserialize_from(bytes.as_slice()).unwrap();
    assert_eq!(x_ext, y_ext);
}

#[test]
fn test_conversions() {
    // Test u32 conversion
    assert_eq!(Goldilocks::from(5u32).v, 5);

    // Test u64 conversion
    assert_eq!(Goldilocks::from(5u64).v, 5);

    // Test u256 conversion
    let u256 = U256::from(5u32);
    assert_eq!(Goldilocks::from_u256(u256).v, 5);

    // Test extension field conversions
    let base = Goldilocks::from(5u32);
    let ext = GoldilocksExt2::from(base);
    assert_eq!(ext.v[0], base);
    assert_eq!(ext.v[1], Goldilocks::zero());
}

#[test]
fn test_exponentiation() {
    let x = Goldilocks::from(2u32);
    let y = x.exp(3);
    assert_eq!(y.v, 8);

    // Test power of 2
    let z = x.exp_power_of_2(2);
    assert_eq!(z.v, 16);

    // Test extension field exponentiation
    let x_ext = GoldilocksExt2::from(Goldilocks::from(2u32));
    let y_ext = x_ext.exp(3);
    assert_eq!(y_ext, x_ext * x_ext * x_ext);
}

#[test]
fn test_special_values() {
    // Test INV_2
    assert_eq!(
        Goldilocks::INV_2 * Goldilocks::from(2u32),
        Goldilocks::one()
    );

    // Test mul_by_5
    let x = Goldilocks::from(2u32);
    assert_eq!(x.mul_by_5().v, 10);

    // Test mul_by_6
    assert_eq!(x.mul_by_6().v, 12);

    // Test mul_by_7
    assert_eq!(x.mul_by_7().v, 14);

    // Test extension field special values
    assert_eq!(
        GoldilocksExt2::X * GoldilocksExt2::X,
        GoldilocksExt2::from(Goldilocks::from(7u32))
    );
}

#[test]
fn test_edge_cases() {
    // Test zero
    assert!(Goldilocks::zero().is_zero());

    // Test modulus
    assert!(Goldilocks { v: GOLDILOCKS_MOD }.is_zero());

    // Test inverse of zero
    assert!(Goldilocks::zero().inv().is_none());

    // Test large numbers
    let large = Goldilocks {
        v: GOLDILOCKS_MOD - 1,
    };
    assert_eq!(large + Goldilocks::one(), Goldilocks::zero());

    // Test U64::MAX; also tests Eq works over normalized values
    let max = Goldilocks { v: u64::MAX };
    let epsilon = Goldilocks { v: EPSILON - 1 };
    assert_eq!(max, epsilon);

    // Test extension field edge cases
    assert!(GoldilocksExt2::zero().is_zero());
    assert!(GoldilocksExt2::zero().inv().is_none());
    let x = GoldilocksExt2::X;
    assert_eq!(x * x, GoldilocksExt2::from(Goldilocks::from(7u32)));
}
