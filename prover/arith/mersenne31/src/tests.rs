use arith::{
    random_extension_field_tests, random_fft_field_tests, random_field_tests,
    random_inversion_tests, random_simd_field_tests, SimdField,
};
use arith::{random_from_limbs_to_limbs_tests, Field};
use ark_std::test_rng;
use ethnum::U256;
use gkr_hashers::{FiatShamirHasher, PoseidonFiatShamirHasher, PoseidonStateTrait};
use serdes::ExpSerde;

use crate::{
    m31::{mod_reduce_u32_safe, M31_MOD},
    M31Ext3, M31Ext3x16, M31Ext6, M31x16, M31,
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
    random_field_tests::<M31>("M31".to_string());
    random_simd_field_tests::<M31>("M31".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<M31, _>(&mut rng, "M31".to_string());
}

#[test]
fn test_simd_field() {
    random_field_tests::<M31x16>("Vectorized M31".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<M31x16, _>(&mut rng, "Vectorized M31".to_string());

    random_simd_field_tests::<M31x16>("Vectorized M31".to_string());

    let a = M31x16::from(256 + 2);
    let mut buffer = vec![];
    assert!(a.serialize_into(&mut buffer).is_ok());
    let b = M31x16::deserialize_from(buffer.as_slice());
    assert!(b.is_ok());
    let b = b.unwrap();
    assert_eq!(a, b);
}

#[test]
fn test_ext_field() {
    random_field_tests::<M31Ext3>("M31 Ext3".to_string());
    random_extension_field_tests::<M31Ext3>("M31 Ext3".to_string());
    random_simd_field_tests::<M31Ext3>("Simd M31 Ext3".to_string());

    random_field_tests::<M31Ext6>("M31 Ext6".to_string());
    random_extension_field_tests::<M31Ext6>("M31 Ext6".to_string());
    random_fft_field_tests::<M31Ext6>("M31 Ext6".to_string());

    random_field_tests::<M31Ext3x16>("Simd M31 Ext3".to_string());
    random_extension_field_tests::<M31Ext3x16>("Simd M31 Ext3".to_string());
    random_simd_field_tests::<M31Ext3x16>("Simd M31 Ext3".to_string());
    random_from_limbs_to_limbs_tests::<M31, M31Ext3>("M31 Ext3".to_string());
    random_from_limbs_to_limbs_tests::<M31x16, M31Ext3x16>("Simd M31 Ext3".to_string());
}

/// Compare to test vectors generated in SageMath
#[test]
fn test_vectors() {
    let a = M31Ext3 {
        v: [M31::from(1), M31::from(2), M31::from(3)],
    };
    let b = M31Ext3 {
        v: [M31::from(4), M31::from(5), M31::from(6)],
    };
    let expected_prod = M31Ext3 {
        v: [M31::from(139), M31::from(103), M31::from(28)],
    };
    assert_eq!(expected_prod, a * b);

    let expected_inv = M31Ext3 {
        v: [
            M31::from(1279570927),
            M31::from(2027416670),
            M31::from(696388467),
        ],
    };
    assert_eq!(expected_inv, a.inv().unwrap());
    let a_pow_11 = M31Ext3 {
        v: [
            M31::from(2145691179),
            M31::from(1848238717),
            M31::from(1954563431),
        ],
    };
    assert_eq!(a_pow_11, a.exp(11));
}

#[test]
fn test_poseidon_m31_fiat_shamir_hash() {
    let perm = PoseidonFiatShamirHasher::<M31x16>::new();

    {
        let state_elems: Vec<u8> = [M31::from(114514); M31x16::RATE]
            .iter()
            .flat_map(|x| {
                let mut xu8 = vec![];
                x.serialize_into(&mut xu8).unwrap();
                xu8
            })
            .collect();
        let mut actual_output_u8 = [0u8; M31x16::STATE_WIDTH * M31::SIZE];
        perm.hash(&mut actual_output_u8, &state_elems);
        let expected_output = M31x16::pack(&vec![
            M31 { v: 1021105124 },
            M31 { v: 1342990709 },
            M31 { v: 1593716396 },
            M31 { v: 2100280498 },
            M31 { v: 330652568 },
            M31 { v: 1371365483 },
            M31 { v: 586650367 },
            M31 { v: 345482939 },
            M31 { v: 849034538 },
            M31 { v: 175601510 },
            M31 { v: 1454280121 },
            M31 { v: 1362077584 },
            M31 { v: 528171622 },
            M31 { v: 187534772 },
            M31 { v: 436020341 },
            M31 { v: 1441052621 },
        ]);
        let mut expected_output_u8 = vec![];
        expected_output
            .serialize_into(&mut expected_output_u8)
            .unwrap();
        assert_eq!(actual_output_u8.to_vec(), expected_output_u8);
    }

    {
        let state_elems: Vec<u8> = [M31::from(114514); M31x16::STATE_WIDTH]
            .iter()
            .flat_map(|x| {
                let mut xu8 = vec![];
                x.serialize_into(&mut xu8).unwrap();
                xu8
            })
            .collect();
        let mut actual_output_u8 = [0u8; M31x16::STATE_WIDTH * M31::SIZE];
        perm.hash(&mut actual_output_u8, &state_elems);
        let expected_output = M31x16::pack(&vec![
            M31 { v: 1510043913 },
            M31 { v: 1840611937 },
            M31 { v: 45881205 },
            M31 { v: 1134797377 },
            M31 { v: 803058407 },
            M31 { v: 1772167459 },
            M31 { v: 846553905 },
            M31 { v: 2143336151 },
            M31 { v: 300871060 },
            M31 { v: 545838827 },
            M31 { v: 1603101164 },
            M31 { v: 396293243 },
            M31 { v: 502075988 },
            M31 { v: 2067011878 },
            M31 { v: 402134378 },
            M31 { v: 535675968 },
        ]);
        let mut expected_output_u8 = vec![];
        expected_output
            .serialize_into(&mut expected_output_u8)
            .unwrap();
        assert_eq!(actual_output_u8.to_vec(), expected_output_u8);
    }
}

#[test]
fn check_normalized_eq() {
    {
        let a = M31 { v: 0 };
        let b = M31 { v: (1 << 31) - 1 };
        assert_eq!(a, b);
    }
    {
        let a = M31 { v: 1 };
        let b = M31 { v: 1 << 31 };
        assert_eq!(a, b);
    }
}

#[test]
fn test_mod_u256_small_values() {
    // Test with values less than M31_MOD
    assert_eq!(M31::from_u256(U256::new(42)), M31 { v: 42 });
    assert_eq!(M31::from_u256(U256::new(1000000)), M31 { v: 1000000 });

    // Test with value equal to M31_MOD
    assert_eq!(M31::from_u256(U256::new(M31_MOD as u128)), M31 { v: 0 });

    // Test with value slightly larger than M31_MOD
    assert_eq!(M31::from_u256(U256::new(M31_MOD as u128 + 1)), M31 { v: 1 });
}

#[test]
fn test_mod_u256_large_values() {
    // Test with a value that spans both low and high words
    // ethnum::U256 uses big-endian representation by default
    let large_value = U256::from_words(1, 0); // 2^128
    assert_eq!(M31::from_u256(large_value), M31 { v: 16 }); // 2^128 mod (2^31-1) = 16

    // Test with maximum possible value
    let max_value = U256::from_words(u128::MAX, u128::MAX);
    let expected = mod_reduce_u32_safe(255); // Theoretical result based on powers of 2
    assert_eq!(M31::from_u256(max_value), M31 { v: expected });
}
