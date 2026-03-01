use itertools::izip;

use crate::{bit_reverse, Field};

pub trait FFTField: Field {
    const TWO_ADICITY: usize;

    fn root_of_unity() -> Self;

    fn two_adic_generator(bits: usize) -> Self {
        let mut g = Self::root_of_unity();
        for _ in bits..Self::TWO_ADICITY {
            g = g.square();
        }
        g
    }

    #[inline(always)]
    fn fft(poly: &[Self]) -> Vec<Self> {
        let mut coeffs = poly.to_vec();
        Self::fft_in_place(&mut coeffs);
        coeffs
    }

    #[inline(always)]
    fn ifft(evals: &[Self]) -> Vec<Self> {
        let mut coeffs = evals.to_vec();
        Self::ifft_in_place(&mut coeffs);
        coeffs
    }

    #[inline(always)]
    fn fft_in_place(poly: &mut [Self]) {
        let po2_mul_subgroup_bits = poly.len().ilog2() as usize;
        let omega = Self::two_adic_generator(po2_mul_subgroup_bits);

        radix2_fft_single_threaded(poly, omega)
    }

    #[inline(always)]
    fn ifft_in_place(evals: &mut [Self]) {
        let log_n = evals.len().ilog2() as usize;
        let omega = Self::two_adic_generator(log_n);
        let omega_inv = omega.inv().unwrap();

        let n_inv = Self::ONE.double().exp(log_n as u128).inv().unwrap();

        radix2_fft_single_threaded(evals, omega_inv);
        evals.iter_mut().for_each(|x| *x *= n_inv);
    }
}

#[inline(always)]
fn bit_reverse_swap<F: Copy>(elems: &mut [F]) {
    // NOTE(HS) we are assuming that this method is only used in FFT,
    // then the elems slice here is assumed to be of length power of 2.

    let mut swap_count: usize = 0;
    let n = elems.len();
    let log_n = n.ilog2() as usize;
    let swap_threshold = n >> 1;

    for i in 0..n {
        // NOTE(HS) swap number should be exactly half of the elems
        if swap_count >= swap_threshold {
            break;
        }

        let swap_to = bit_reverse(i, log_n);
        if i < swap_to {
            // NOTE(HS) the invariant here is bit swap won't exceed the range,
            // so we choose to use unchecked to short wire the range check,
            // s.t., less instructions
            unsafe { elems.swap_unchecked(swap_to, i) }
            swap_count += 1;
        }
    }
}

#[inline(always)]
pub fn radix2_fft_single_threaded<F: FFTField>(coeffs: &mut [F], omega: F) {
    assert!(coeffs.len().is_power_of_two());

    bit_reverse_swap(coeffs);

    let n = coeffs.len();
    let log_n = n.ilog2() as usize;

    // precompute twiddle factors
    let twiddles: Vec<_> = (0..(n / 2))
        .scan(F::ONE, |w, _| {
            let tw = *w;
            *w *= &omega;
            Some(tw)
        })
        .collect();

    let mut chunk = 2_usize;
    let mut twiddle_chunk = n / 2;
    for _ in 0..log_n {
        coeffs.chunks_mut(chunk).for_each(|coeffs| {
            let (left, right) = coeffs.split_at_mut(chunk / 2);

            // case when twiddle factor is one
            let (a, left) = left.split_at_mut(1);
            let (b, right) = right.split_at_mut(1);
            let t = b[0];
            b[0] = a[0];
            a[0] += &t;
            b[0] -= &t;

            izip!(left, right).enumerate().for_each(|(i, (a, b))| {
                let mut t = *b;
                t *= &twiddles[(i + 1) * twiddle_chunk];
                *b = *a;
                *a += &t;
                *b -= &t;
            });
        });
        chunk *= 2;
        twiddle_chunk /= 2;
    }
}

#[cfg(test)]
mod fft_test {
    use ark_std::test_rng;
    use halo2curves::bn256::Fr;
    use itertools::izip;

    use crate::{FFTField, Field};

    #[test]
    fn test_bn254_fft() {
        let mut rng = test_rng();

        (1..10).for_each(|bits| {
            let length = 1 << bits;

            let mut coeffs: Vec<_> = (0..length).map(|_| Fr::random_unsafe(&mut rng)).collect();
            let coeffs_cloned = coeffs.clone();

            Fr::fft_in_place(&mut coeffs);
            Fr::ifft_in_place(&mut coeffs);

            izip!(&coeffs, &coeffs_cloned).for_each(|(a, b)| assert_eq!(a, b));
        });
    }
}
