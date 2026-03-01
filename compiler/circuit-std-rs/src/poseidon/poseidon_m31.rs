use super::utils::{get_constants, get_mds_matrix};
use expander_compiler::frontend::{Config, RootAPI, Variable};

fn power_5<C: Config, B: RootAPI<C>>(api: &mut B, base: Variable) -> Variable {
    let pow2 = api.mul(base, base);
    let pow4 = api.mul(pow2, pow2);
    api.mul(pow4, base)
}

pub struct PoseidonM31Params {
    pub mds_matrix: Vec<Vec<Variable>>,
    pub round_constants: Vec<Vec<Variable>>,

    pub rate: usize,
    pub width: usize,
    pub full_rounds: usize,
    pub partial_rounds: usize,
}

impl PoseidonM31Params {
    pub fn new<C: Config, B: RootAPI<C>>(
        api: &mut B,
        rate: usize,
        width: usize,
        full_rounds: usize,
        partial_rounds: usize,
    ) -> Self {
        let round_constants = get_constants(width, partial_rounds + full_rounds);
        let mds_matrix = get_mds_matrix(width);

        let round_constants_variables = (0..partial_rounds + full_rounds)
            .map(|i| {
                (0..width)
                    .map(|j| api.constant(round_constants[i][j]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mds_matrix_variables = (0..width)
            .map(|i| {
                (0..width)
                    .map(|j| api.constant(mds_matrix[i][j]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self {
            mds_matrix: mds_matrix_variables,
            round_constants: round_constants_variables,
            rate,
            width,
            full_rounds,
            partial_rounds,
        }
    }

    fn add_round_constants<C: Config, B: RootAPI<C>>(
        &self,
        api: &mut B,
        state: &mut [Variable],
        constants: &[Variable],
    ) {
        (0..self.width).for_each(|i| state[i] = api.add(state[i], constants[i]))
    }

    fn apply_mds_matrix<C: Config, B: RootAPI<C>>(&self, api: &mut B, state: &mut [Variable]) {
        let prev_state = state.to_vec();

        (0..self.width).for_each(|i| {
            let mut inner_product = api.constant(0);
            (0..self.width).for_each(|j| {
                let unit = api.mul(prev_state[j], self.mds_matrix[i][j]);
                inner_product = api.add(inner_product, unit);
            });
            state[i] = inner_product;
        })
    }

    fn partial_full_sbox<C: Config, B: RootAPI<C>>(&self, api: &mut B, state: &mut [Variable]) {
        state[0] = power_5(api, state[0])
    }

    fn apply_full_sbox<C: Config, B: RootAPI<C>>(&self, api: &mut B, state: &mut [Variable]) {
        state.iter_mut().for_each(|s| *s = power_5(api, *s))
    }

    pub fn permute<C: Config, B: RootAPI<C>>(&self, api: &mut B, state: &mut [Variable]) {
        let half_full_rounds = self.full_rounds / 2;
        let partial_ends = half_full_rounds + self.partial_rounds;

        assert_eq!(self.width, state.len());

        (0..half_full_rounds).for_each(|i| {
            self.add_round_constants(api, state, &self.round_constants[i]);
            self.apply_mds_matrix(api, state);
            self.apply_full_sbox(api, state)
        });
        (half_full_rounds..partial_ends).for_each(|i| {
            self.add_round_constants(api, state, &self.round_constants[i]);
            self.apply_mds_matrix(api, state);
            self.partial_full_sbox(api, state)
        });
        (partial_ends..half_full_rounds + partial_ends).for_each(|i| {
            self.add_round_constants(api, state, &self.round_constants[i]);
            self.apply_mds_matrix(api, state);
            self.apply_full_sbox(api, state)
        });
    }

    pub fn hash_to_state<C: Config, B: RootAPI<C>>(
        &self,
        api: &mut B,
        inputs: &[Variable],
    ) -> Vec<Variable> {
        let mut elts = inputs.to_vec();
        elts.resize(elts.len().next_multiple_of(self.rate), api.constant(0));

        let mut res = vec![api.constant(0); self.width];

        elts.chunks(self.rate).for_each(|chunk| {
            let mut state_elts = vec![api.constant(0); self.width - self.rate];
            state_elts.extend_from_slice(chunk);

            (0..self.width).for_each(|i| res[i] = api.add(res[i], state_elts[i]));
            self.permute(api, &mut res)
        });

        res
    }
    pub fn hash_to_state_flatten<C: Config, B: RootAPI<C>>(
        &self,
        api: &mut B,
        inputs: &[Variable],
    ) -> Vec<Variable> {
        let mut elts = inputs.to_vec();
        elts.resize(elts.len().next_multiple_of(self.rate), api.constant(0));

        let mut res = vec![api.constant(0); self.width];
        let mut copy_res = vec![api.constant(0); self.width];
        elts.chunks(self.rate).for_each(|chunk| {
            let mut state_elts = vec![api.constant(0); self.width - self.rate];
            state_elts.extend_from_slice(chunk);

            (0..self.width).for_each(|i| res[i] = api.add(copy_res[i], state_elts[i]));
            self.permute(api, &mut res);
            copy_res = api.new_hint("myhint.copyvarshint", &res, res.len());
            assert_vars_is_equal(api, &copy_res, &res);
        });

        res
    }
}
pub fn assert_vars_is_equal<C: Config, B: RootAPI<C>>(api: &mut B, a: &[Variable], b: &[Variable]) {
    a.iter()
        .zip(b.iter())
        .for_each(|(a, b)| api.assert_is_equal(*a, *b))
}
