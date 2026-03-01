use arith::{
    random_extension_field_tests, random_field_tests, random_from_limbs_to_limbs_tests,
    random_inversion_tests, random_simd_field_tests,
};
use ark_std::test_rng;
use gf2::{GF2x8, GF2};

#[cfg(target_arch = "aarch64")]
use crate::gf2_ext128::neon::{gfadd, gfmul};

use crate::{GF2_128x8, GF2_128};

#[test]
fn test_simd_field() {
    random_simd_field_tests::<GF2_128x8>("Simd GF2 Ext128".to_string());
}

#[test]
fn test_ext_field() {
    random_field_tests::<GF2_128>("GF2 Ext128".to_string());
    random_field_tests::<GF2_128x8>("Simd GF2 Ext128".to_string());

    random_extension_field_tests::<GF2_128>("GF2 Ext128".to_string());
    random_extension_field_tests::<GF2_128x8>("Simd GF2 Ext128".to_string());

    let mut rng = test_rng();
    random_inversion_tests::<GF2_128, _>(&mut rng, "GF2_128".to_string());

    random_from_limbs_to_limbs_tests::<GF2, GF2_128>("GF2 Ext128".to_string());
    random_from_limbs_to_limbs_tests::<GF2x8, GF2_128x8>("Simd GF2 Ext128".to_string());
}

#[cfg(target_arch = "aarch64")]
#[test]
// known answer test, results cross-checked with avx_gf2_128
fn test_gf_mul_kat() {
    use core::mem::transmute;
    use std::arch::aarch64::*;

    unsafe {
        {
            let a = vdupq_n_u32(0);
            let b = vcombine_u32(vcreate_u32(1234567890), vcreate_u32(0));
            let result = gfmul(a, b);
            assert_eq!(vgetq_lane_u32(result, 0), 0);
            assert_eq!(vgetq_lane_u32(result, 1), 0);
        }
        {
            let zero = transmute(0u128);
            let a = transmute((3u128 << 64) + 5);
            let result = gfmul(a, zero);

            assert_eq!(vgetq_lane_u32(result, 0), 0);
            assert_eq!(vgetq_lane_u32(result, 1), 0);
            assert_eq!(vgetq_lane_u32(result, 2), 0);
            assert_eq!(vgetq_lane_u32(result, 3), 0);
        }

        {
            let one = transmute(1u128);
            let a = transmute((3u128 << 64) + 5);
            let result = gfmul(one, a);

            assert_eq!(vgetq_lane_u32(result, 0), 5);
            assert_eq!(vgetq_lane_u32(result, 1), 0);
            assert_eq!(vgetq_lane_u32(result, 2), 3);
            assert_eq!(vgetq_lane_u32(result, 3), 0);
        }

        {
            let a = transmute((3u128 << 64) + 5);
            let b = transmute((1u128 << 64) + 7);
            let result = gfmul(a, b);

            assert_eq!(vgetq_lane_u32(result, 0), 402);
            assert_eq!(vgetq_lane_u32(result, 1), 0);
            assert_eq!(vgetq_lane_u32(result, 2), 12);
            assert_eq!(vgetq_lane_u32(result, 3), 0);
        }

        {
            let b = transmute((1u128 << 64) + 7);
            let c = transmute((1u128 << 96) + (1 << 64) + (1 << 32) + 1);
            let result = gfmul(b, c);

            assert_eq!(vgetq_lane_u32(result, 0), 128);
            assert_eq!(vgetq_lane_u32(result, 1), 128);
            assert_eq!(vgetq_lane_u32(result, 2), 6);
            assert_eq!(vgetq_lane_u32(result, 3), 6);
        }

        {
            let a = transmute::<_, uint32x4_t>([7u8; 16]);
            let b = transmute::<_, uint32x4_t>([5u8; 16]);
            let result = gfmul(a, b);

            assert_eq!(vgetq_lane_u32(result, 0), 232394202);
            assert_eq!(vgetq_lane_u32(result, 1), 232394202);
            assert_eq!(vgetq_lane_u32(result, 2), 232394202);
            assert_eq!(vgetq_lane_u32(result, 3), 232394202);
        }

        {
            let mut a = [6u8; 16];
            a[8] = 0;
            let a = transmute::<_, uint32x4_t>(a);
            let mut b = [5u8; 16];
            b[4] = 1;
            let b = transmute::<_, uint32x4_t>(b);
            let result = gfmul(a, b);

            assert_eq!(vgetq_lane_u32(result, 0), 508894806);
            assert_eq!(vgetq_lane_u32(result, 1), 1107902981);
            assert_eq!(vgetq_lane_u32(result, 2), 155322701);
            assert_eq!(vgetq_lane_u32(result, 3), 155322714);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gf_mul_rnd() {
    use core::mem::transmute;
    use rand::RngCore;
    use std::arch::aarch64::*;

    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        unsafe {
            {
                // associativity
                let a = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );
                let b = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );
                let c = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );
                let ab = gfmul(a, b);
                let bc = gfmul(b, c);
                let abc = gfmul(ab, c);

                assert_eq!(
                    transmute::<_, u128>(gfmul(a, bc)),
                    transmute::<_, u128>(abc)
                );
            }

            {
                // commutativity
                let a = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );
                let b = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );

                let ab = gfmul(a, b);
                let ba = gfmul(b, a);

                assert_eq!(transmute::<_, u128>(ab), transmute::<_, u128>(ba));
            }

            {
                // distributivity
                let a = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );
                let b = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );
                let c = transmute::<_, uint32x4_t>(
                    (rng.next_u64() as u128) << 64 | rng.next_u64() as u128,
                );

                let a_plus_b = gfadd(a, b);
                let a_plus_b_then_mul_c = gfmul(a_plus_b, c);

                let ac = gfmul(a, c);
                let bc = gfmul(b, c);
                let ac_plus_bc = gfadd(ac, bc);

                assert_eq!(
                    transmute::<_, u128>(a_plus_b_then_mul_c),
                    transmute::<_, u128>(ac_plus_bc)
                );
            }
        }
    }
}
