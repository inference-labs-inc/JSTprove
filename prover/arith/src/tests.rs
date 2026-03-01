use ark_std::{end_timer, start_timer, test_rng};
use field::{
    associativity_tests, commutativity_tests, identity_tests, random_addition_tests,
    random_doubling_tests, random_expansion_tests, random_multiplication_tests,
    random_negation_tests, random_serde_tests, random_squaring_tests, random_subtraction_tests,
};
use rand::RngCore;

use crate::{ExtensionField, FFTField, Field, SimdField};

#[cfg(test)]
mod bn254;

mod field;

#[cfg(target_arch = "x86_64")]
#[test]
fn test_mm256_const_init() {
    use std::arch::x86_64::*;
    use std::mem::transmute;

    let all_equal = unsafe {
        let x = _mm256_set1_epi32(1);
        let y = transmute::<_, __m256i>([1, 1, 1, 1, 1, 1, 1, 1]);
        let cmp = _mm256_cmpeq_epi32(x, y);
        _mm256_testc_si256(cmp, _mm256_set1_epi32(-1))
    };

    assert!(all_equal != 0, "x and y are not equal");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_uint32x4_const_init() {
    use std::arch::aarch64::*;
    use std::mem::transmute;

    let all_equal = unsafe {
        let x = vdupq_n_u32(1);
        let y = transmute::<_, uint32x4_t>([1, 1, 1, 1]);
        let cmp = vceqq_u32(x, y);
        vgetq_lane_u32(cmp, 0) == 0xffffffff
    };

    assert!(all_equal, "x and y are not equal");
}

pub fn random_extension_field_tests<F: ExtensionField>(_name: String) {
    let mut rng = test_rng();
    for _ in 0..1000 {
        {
            let a = F::random_unsafe(&mut rng);
            let s1 = F::BaseField::random_unsafe(&mut rng);
            let s2 = F::BaseField::random_unsafe(&mut rng);

            assert_eq!(
                a.mul_by_base_field(&s1).mul_by_base_field(&s2),
                a.mul_by_base_field(&s2).mul_by_base_field(&s1),
            );
            assert_eq!(
                a.mul_by_base_field(&s1).mul_by_base_field(&s2),
                a.mul_by_base_field(&(s1 * s2)),
            );

            assert_eq!(
                a.add_by_base_field(&s1).add_by_base_field(&s2),
                a.add_by_base_field(&s2).add_by_base_field(&s1),
            );
            assert_eq!(
                a.add_by_base_field(&s1).add_by_base_field(&s2),
                a.add_by_base_field(&(s1 + s2)),
            );
        }

        {
            let a = F::random_unsafe(&mut rng);
            let b = F::random_unsafe(&mut rng);
            let s = F::BaseField::random_unsafe(&mut rng);

            assert_eq!(a.mul_by_base_field(&s) * b, (a * b).mul_by_base_field(&s),);
            assert_eq!(b.mul_by_base_field(&s) * a, (a * b).mul_by_base_field(&s),);

            assert_eq!(a.add_by_base_field(&s) + b, (a + b).add_by_base_field(&s),);
            assert_eq!(b.add_by_base_field(&s) + a, (a + b).add_by_base_field(&s),);
        }

        {
            let a = F::random_unsafe(&mut rng);
            let b = F::X;
            let ax = a.mul_by_x();
            let ab = a * b;
            assert_eq!(ax, ab);
        }
    }
}

pub fn random_field_tests<F: Field>(type_name: String) {
    let mut rng = test_rng();

    random_multiplication_tests::<F, _>(&mut rng, type_name.clone());
    random_addition_tests::<F, _>(&mut rng, type_name.clone());
    random_subtraction_tests::<F, _>(&mut rng, type_name.clone());
    random_negation_tests::<F, _>(&mut rng, type_name.clone());
    random_doubling_tests::<F, _>(&mut rng, type_name.clone());
    random_squaring_tests::<F, _>(&mut rng, type_name.clone());
    random_expansion_tests::<F, _>(&mut rng, type_name.clone()); // also serve as distributivity tests
    random_serde_tests::<F, _>(&mut rng, type_name.clone());
    associativity_tests::<F, _>(&mut rng, type_name.clone());
    commutativity_tests::<F, _>(&mut rng, type_name.clone());
    identity_tests::<F, _>(&mut rng, type_name.clone());
    // inverse_tests::<F, _>(&mut rng, type_name.clone());

    assert!(F::zero().is_zero());
    {
        let mut z = F::zero();
        z = z.neg();
        assert!(z.is_zero());
    }

    // Multiplication by zero
    {
        let mut a = F::random_unsafe(&mut rng);
        a.mul_assign(&F::zero());
        assert!(a.is_zero());
    }

    // Addition by zero
    {
        let mut a = F::random_unsafe(&mut rng);
        let copy = a;
        a.add_assign(&F::zero());
        assert_eq!(a, copy);
    }
}

pub fn random_from_limbs_to_limbs_tests<F: Field, ExtF: ExtensionField<BaseField = F>>(
    type_name: String,
) {
    let mut rng = test_rng();
    let _message = format!("from/to limbs {type_name}");

    (0..1000).for_each(|_| {
        let ext_f = ExtF::random_unsafe(&mut rng);
        let limbs = ext_f.to_limbs();
        let back_to_extf = ExtF::from_limbs(&limbs);
        assert_eq!(ext_f, back_to_extf);
    })
}

pub fn random_inversion_tests<F: Field, R: RngCore>(mut rng: R, type_name: String) {
    assert!(F::zero().inv().is_none());

    let _message = format!("inversion {type_name}");
    let start = start_timer!(|| _message);
    for _ in 0..1000 {
        let mut a = F::random_unsafe(&mut rng);
        if a.is_zero() {
            a = F::one();
        }
        let b = a.inv().unwrap(); // probabilistically nonzero
        a.mul_assign(&b);
        assert_eq!(a, F::one());
    }
    end_timer!(start);
}

pub fn random_simd_field_tests<F: SimdField>(_name: String) {
    let mut rng = test_rng();

    {
        let a = F::random_unsafe(&mut rng);
        let s1 = F::Scalar::random_unsafe(&mut rng);
        let s2 = F::Scalar::random_unsafe(&mut rng);

        assert_eq!(a.scale(&s1).scale(&s2), a.scale(&s2).scale(&s1),);
        assert_eq!(a.scale(&s1).scale(&s2), a.scale(&(s1 * s2)));
    }

    {
        let a = F::random_unsafe(&mut rng);
        let b = F::random_unsafe(&mut rng);
        let s = F::Scalar::random_unsafe(&mut rng);

        assert_eq!(a.scale(&s) * b, (a * b).scale(&s),);
        assert_eq!(b.scale(&s) * a, (a * b).scale(&s),);
    }

    {
        let x = F::random_unsafe(&mut rng);
        let scalars = x.unpack();
        let x_repacked = F::pack(&scalars);
        assert_eq!(x, x_repacked);
    }
}

pub fn random_fft_field_tests<F: Field + FFTField>(_name: String) {
    let mut rng = test_rng();

    for log_degree in [2, 3, 5, 10] {
        let degree = 1 << log_degree;

        {
            // (x+1)(x^2-1) = x^3 + x^2 - x - 1
            let mut a = vec![F::zero(); degree];
            let mut b = vec![F::zero(); degree];
            let mut c = vec![F::zero(); degree];
            a[0] = F::one();
            a[1] = F::one();
            b[0] = -F::one();
            b[2] = F::one();
            c[0] = -F::one();
            c[1] = -F::one();
            c[2] = F::one();
            c[3] = F::one();

            F::fft_in_place(&mut a);
            F::fft_in_place(&mut b);

            a.iter_mut().zip(b.iter()).for_each(|(a, b)| {
                *a *= *b;
            });

            F::ifft_in_place(&mut a);

            assert_eq!(a, c);
        }

        {
            // (x^(n-1) + 1) * (x + 1) = x^(n-1) + x + 2
            let mut a = vec![F::zero(); degree];
            let mut b = vec![F::zero(); degree];
            let mut c = vec![F::zero(); degree];
            a[0] = F::one();
            a[degree - 1] = F::one();
            b[0] = F::one();
            b[1] = F::one();
            c[0] = F::ONE.double();
            c[1] = F::one();
            c[degree - 1] = F::one();

            F::fft_in_place(&mut a);
            F::fft_in_place(&mut b);

            a.iter_mut().zip(b.iter()).for_each(|(a, b)| {
                *a *= *b;
            });

            F::ifft_in_place(&mut a);

            assert_eq!(a, c);
        }
    }

    for i in [1, 2, 3, 5, 10] {
        let degree = 1 << i;

        let mut a = vec![F::zero(); degree];
        let mut b = vec![F::zero(); degree];

        for i in 0..degree {
            a[i] = F::random_unsafe(&mut rng);
            b[i] = F::random_unsafe(&mut rng);
        }

        let mut a2 = a.clone();

        F::fft_in_place(&mut a2);
        let mut a_add_b = a2.clone();
        let mut a_mul_b = a2.clone();

        F::ifft_in_place(&mut a2);
        assert_eq!(a, a2);

        let mut b2 = b.clone();

        F::fft_in_place(&mut b2);
        a_add_b.iter_mut().zip(b2.iter()).for_each(|(c, b)| *c += b);
        a_mul_b.iter_mut().zip(b2.iter()).for_each(|(c, b)| *c *= b);

        F::ifft_in_place(&mut b2);
        assert_eq!(b, b2);

        F::ifft_in_place(&mut a_add_b);
        let a_add_b_2 = a
            .iter()
            .zip(b.iter())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>();
        assert_eq!(a_add_b, a_add_b_2);

        F::ifft_in_place(&mut a_mul_b);
        let a_mul_b_2 = schoolbook_mul(&a, &b);
        assert_eq!(a_mul_b, a_mul_b_2);
    }
}

/// school book multiplication
/// output = a(x) * b(x) mod x^N - 1 mod MODULUS
/// using school-book multiplications
/// This is used to verify the correctness of the FFT multiplication
fn schoolbook_mul<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let degree = a.len();
    assert_eq!(degree, b.len());

    let mut buf = vec![F::ZERO; degree << 1];

    for i in 0..degree {
        for j in 0..degree {
            buf[i + j] += a[i] * b[j];
        }
    }

    for i in 0..degree {
        buf[i] = buf[i] + buf[i + degree];
    }
    buf.truncate(degree);
    buf
}
