use super::utils::{get_constants, get_mds_matrix};

const M31_MODULUS: u32 = 2147483647; // Example modulus for Mersenne 31

fn m31_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    sum % M31_MODULUS
}

fn m31_mul(a: u32, b: u32) -> u32 {
    let product = (a as u128 * b as u128) % M31_MODULUS as u128;
    product as u32
}
fn power_5(base: u32) -> u32 {
    let pow2 = m31_mul(base, base);
    let pow4 = m31_mul(pow2, pow2);
    m31_mul(pow4, base)
}

pub struct PoseidonParams {
    pub mds_matrix: Vec<Vec<u32>>,
    pub round_constants: Vec<Vec<u32>>,

    pub rate: usize,
    pub width: usize,
    pub full_rounds: usize,
    pub partial_rounds: usize,
}

impl PoseidonParams {
    pub fn new(rate: usize, width: usize, full_rounds: usize, partial_rounds: usize) -> Self {
        let mut round_constants = get_constants(width, partial_rounds + full_rounds);
        let mut mds_matrix = get_mds_matrix(width);
        //mod all constants by M31_MODULUS
        (0..round_constants.len()).for_each(|i| {
            (0..round_constants[i].len()).for_each(|j| {
                round_constants[i][j] %= M31_MODULUS;
            });
        });
        //mod all mds_matrix by M31_MODULUS
        (0..mds_matrix.len()).for_each(|i| {
            (0..mds_matrix[i].len()).for_each(|j| {
                mds_matrix[i][j] %= M31_MODULUS;
            });
        });
        Self {
            mds_matrix,
            round_constants,
            rate,
            width,
            full_rounds,
            partial_rounds,
        }
    }

    fn add_round_constants(&self, state: &mut [u32], constants: &[u32]) {
        for i in 0..self.width {
            state[i] = m31_add(state[i], constants[i]);
        }
    }

    fn apply_mds_matrix(&self, state: &mut [u32]) {
        let prev_state = state.to_vec();

        (0..self.width).for_each(|i| {
            let mut inner_product = 0;
            (0..self.width).for_each(|j| {
                let unit = m31_mul(prev_state[j], self.mds_matrix[i][j]);
                inner_product = m31_add(inner_product, unit);
            });
            state[i] = inner_product;
        })
    }

    fn partial_full_sbox(&self, state: &mut [u32]) {
        state[0] = power_5(state[0]);
    }

    fn apply_full_sbox(&self, state: &mut [u32]) {
        state.iter_mut().for_each(|s| *s = power_5(*s))
    }

    pub fn permute(&self, state: &mut [u32]) {
        let half_full_rounds = self.full_rounds / 2;
        let partial_ends = half_full_rounds + self.partial_rounds;

        assert_eq!(self.width, state.len());
        (0..half_full_rounds).for_each(|i| {
            self.add_round_constants(state, &self.round_constants[i]);
            self.apply_mds_matrix(state);
            self.apply_full_sbox(state);
        });
        (half_full_rounds..partial_ends).for_each(|i| {
            self.add_round_constants(state, &self.round_constants[i]);
            self.apply_mds_matrix(state);
            self.partial_full_sbox(state)
        });
        (partial_ends..half_full_rounds + partial_ends).for_each(|i| {
            self.add_round_constants(state, &self.round_constants[i]);
            self.apply_mds_matrix(state);
            self.apply_full_sbox(state)
        });
    }

    pub fn hash_to_state(&self, inputs: &[u32]) -> Vec<u32> {
        let mut elts = inputs.to_vec();
        elts.resize(elts.len().next_multiple_of(self.rate), 0);

        let mut res = vec![0u32; self.width];

        elts.chunks(self.rate).for_each(|chunk| {
            let mut state_elts = vec![0; self.width - self.rate];
            state_elts.extend_from_slice(chunk);

            (0..self.width).for_each(|i| res[i] = m31_add(res[i], state_elts[i]));
            self.permute(&mut res)
        });

        res
    }

    pub fn hash(&self, inputs: &[u32]) -> Vec<u32> {
        self.hash_to_state(inputs)[..self.rate].to_vec()
    }
}
