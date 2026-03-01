use arith::Field;
use halo2curves::bn256::Fr;
use tiny_keccak::{Hasher, Keccak};

use crate::FiatShamirHasher;

#[derive(Debug, Clone, Default)]
pub struct MiMC5FiatShamirHasher<F: Field> {
    constants: Vec<F>,
}

impl<F: Field> MiMC5FiatShamirHasher<F> {
    fn hash_u8_to_state(&self, input: &[u8]) -> F {
        let mut h = F::ZERO;
        let chunks = input.chunks_exact(F::SIZE);
        let mut remainder = chunks.remainder().to_vec();
        for chunk in chunks {
            let x = F::from_uniform_bytes(chunk);
            let r = self.mimc5_hash(&h, &x);
            h += r + x;
        }

        if !remainder.is_empty() {
            remainder.resize(F::SIZE, 0);

            let x = F::from_uniform_bytes(&remainder);
            let r = self.mimc5_hash(&h, &x);
            h += r + x;
        }

        h
    }
}

impl<F: Field> FiatShamirHasher for MiMC5FiatShamirHasher<F> {
    const NAME: &'static str = "MiMC5_Field_Hasher";

    const DIGEST_SIZE: usize = F::SIZE;

    fn new() -> Self {
        let constants = generate_mimc_constants::<F>();
        Self { constants }
    }

    fn hash(&self, output: &mut [u8], input: &[u8]) {
        assert!(output.len() == F::SIZE);
        let res = self.hash_u8_to_state(input);
        res.to_bytes(output);
    }

    fn hash_inplace(&self, buffer: &mut [u8]) {
        assert!(buffer.len() == F::SIZE);
        let res = self.hash_u8_to_state(buffer);
        res.to_bytes(buffer);
    }
}

impl<F: Field> MiMC5FiatShamirHasher<F> {
    #[inline(always)]
    pub fn pow5(x: F) -> F {
        let x2 = x * x;
        let x4 = x2 * x2;
        x4 * x
    }

    #[inline(always)]
    pub fn mimc5_hash(&self, h: &F, x_in: &F) -> F {
        let mut x = *x_in;

        self.constants.iter().for_each(|c| {
            x = Self::pow5(x + h + c);
        });
        x + h
    }
}

const SEED: &str = "seed";

pub fn generate_mimc_constants<F: Field>() -> Vec<F> {
    let n_rounds: usize = match F::NAME {
        Fr::NAME => 110,
        _ => unimplemented!("unimplemented rounds for MiMC5 Field Hasher"),
    };
    get_constants(SEED, n_rounds)
}

pub fn get_constants<F: Field>(seed: &str, n_rounds: usize) -> Vec<F> {
    let mut keccak = Keccak::v256();
    let mut h = [0u8; 32];
    keccak.update(seed.as_bytes());
    keccak.finalize(&mut h);

    (0..n_rounds)
        .map(|_| {
            let mut keccak = Keccak::v256();
            keccak.update(&h);
            keccak.finalize(&mut h);

            // big endian -> little endian, in order to match the one in gnark
            // or probably we can change the implementation there
            let mut h_reverse = h;
            h_reverse.reverse();

            F::from_uniform_bytes(&h_reverse)
        })
        .collect()
}
