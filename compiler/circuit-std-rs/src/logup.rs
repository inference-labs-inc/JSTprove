use std::collections::HashMap;

use arith::Field;
use ethnum::U256;
use expander_compiler::frontend::{
    declare_circuit, CircuitField, Config, Define, Error, RootAPI, Variable,
};
use rand::Rng;

use crate::StdCircuit;

#[derive(Clone, Copy, Debug)]
pub struct LogUpParams {
    pub key_len: usize,
    pub value_len: usize,
    pub n_table_rows: usize,
    pub n_queries: usize,
}

declare_circuit!(_LogUpCircuit {
    table_keys: [[Variable]],
    table_values: [[Variable]],

    query_keys: [[Variable]],
    query_results: [[Variable]],

    // counting the number of occurrences for each row of the table
    query_count: [Variable],
});

pub type LogUpCircuit = _LogUpCircuit<Variable>;

#[derive(Clone, Copy, Debug)]
struct Rational {
    numerator: Variable,
    denominator: Variable,
}

fn add_rational<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    v1: &Rational,
    v2: &Rational,
) -> Rational {
    let p1 = builder.mul(v1.numerator, v2.denominator);
    let p2 = builder.mul(v1.denominator, v2.numerator);

    Rational {
        numerator: builder.add(p1, p2),
        denominator: builder.mul(v1.denominator, v2.denominator),
    }
}

fn assert_eq_rational<C: Config, B: RootAPI<C>>(builder: &mut B, v1: &Rational, v2: &Rational) {
    let p1 = builder.mul(v1.numerator, v2.denominator);
    let p2 = builder.mul(v1.denominator, v2.numerator);
    builder.assert_is_equal(p1, p2);
}

fn sum_rational_vec<C: Config, B: RootAPI<C>>(builder: &mut B, vs: &[Rational]) -> Rational {
    if vs.is_empty() {
        return Rational {
            numerator: builder.constant(0),
            denominator: builder.constant(1),
        };
    }

    // Basic version:
    // let mut sum = Rational {
    //     numerator: builder.constant(0),
    //     denominator: builder.constant(1),
    // };
    // for i in 0..vs.len() {
    //     sum = add_rational(builder, &sum, &vs[i]);
    // }
    // sum

    // Fewer-layers version:
    let mut vvs = vs.to_owned();
    let mut n_values_to_sum = vvs.len();
    while n_values_to_sum > 1 {
        let half_size_floor = n_values_to_sum / 2;
        for i in 0..half_size_floor {
            vvs[i] = add_rational(builder, &vvs[i], &vvs[i + half_size_floor])
        }

        if n_values_to_sum & 1 != 0 {
            vvs[half_size_floor] = vvs[n_values_to_sum - 1];
        }

        n_values_to_sum = (n_values_to_sum + 1) / 2;
    }
    vvs[0]
}

fn concat_d1(v1: &[Vec<Variable>], v2: &[Vec<Variable>]) -> Vec<Vec<Variable>> {
    v1.iter()
        .zip(v2.iter())
        .map(|(row_key, row_value)| [row_key.to_vec(), row_value.to_vec()].concat())
        .collect()
}

fn get_column_randomness<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    n_columns: usize,
) -> Vec<Variable> {
    let mut randomness = vec![];
    randomness.push(builder.constant(1));
    for _ in 1..n_columns {
        randomness.push(builder.get_random_value());
    }
    randomness
}

fn combine_columns<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    vec_2d: &[Vec<Variable>],
    randomness: &[Variable],
) -> Vec<Variable> {
    if vec_2d.is_empty() {
        return vec![];
    }

    let column_size = vec_2d[0].len();
    assert!(randomness.len() == column_size);
    vec_2d
        .iter()
        .map(|row| {
            row.iter()
                .zip(randomness)
                .fold(builder.constant(0), |acc, (v, r)| {
                    let prod = builder.mul(v, r);
                    builder.add(acc, prod)
                })
        })
        .collect()
}

fn logup_poly_val<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    vals: &[Variable],
    counts: &[Variable],
    x: &Variable,
) -> Rational {
    let poly_terms = vals
        .iter()
        .zip(counts)
        .map(|(v, c)| Rational {
            numerator: *c,
            denominator: builder.sub(x, v),
        })
        .collect::<Vec<Rational>>();
    sum_rational_vec(builder, &poly_terms)
}

impl<C: Config> Define<C> for LogUpCircuit {
    fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
        let key_len = self.table_keys[0].len();
        let value_len = self.table_values[0].len();

        let alpha = builder.get_random_value();
        let randomness = get_column_randomness(builder, key_len + value_len);

        let table_combined = combine_columns(
            builder,
            &concat_d1(&self.table_keys, &self.table_values),
            &randomness,
        );
        let v_table = logup_poly_val(builder, &table_combined, &self.query_count, &alpha);

        let query_combined = combine_columns(
            builder,
            &concat_d1(&self.query_keys, &self.query_results),
            &randomness,
        );
        let one = builder.constant(1);
        let v_query = logup_poly_val(
            builder,
            &query_combined,
            &vec![one; query_combined.len()],
            &alpha,
        );

        assert_eq_rational(builder, &v_table, &v_query);
    }
}

impl<C: Config> StdCircuit<C> for LogUpCircuit {
    type Params = LogUpParams;
    type Assignment = _LogUpCircuit<CircuitField<C>>;

    fn new_circuit(params: &Self::Params) -> Self {
        let mut circuit = Self::default();

        circuit.table_keys.resize(
            params.n_table_rows,
            vec![Variable::default(); params.key_len],
        );
        circuit.table_values.resize(
            params.n_table_rows,
            vec![Variable::default(); params.value_len],
        );
        circuit
            .query_keys
            .resize(params.n_queries, vec![Variable::default(); params.key_len]);
        circuit.query_results.resize(
            params.n_queries,
            vec![Variable::default(); params.value_len],
        );
        circuit
            .query_count
            .resize(params.n_table_rows, Variable::default());

        circuit
    }

    fn new_assignment(params: &Self::Params, mut rng: impl rand::RngCore) -> Self::Assignment {
        let mut assignment = _LogUpCircuit::<CircuitField<C>>::default();
        assignment.table_keys.resize(params.n_table_rows, vec![]);
        assignment.table_values.resize(params.n_table_rows, vec![]);
        assignment.query_keys.resize(params.n_queries, vec![]);
        assignment.query_results.resize(params.n_queries, vec![]);

        for i in 0..params.n_table_rows {
            for _ in 0..params.key_len {
                assignment.table_keys[i].push(CircuitField::<C>::random_unsafe(&mut rng));
            }

            for _ in 0..params.value_len {
                assignment.table_values[i].push(CircuitField::<C>::random_unsafe(&mut rng));
            }
        }

        assignment.query_count = vec![CircuitField::<C>::ZERO; params.n_table_rows];
        for i in 0..params.n_queries {
            let query_id: usize = rng.gen::<usize>() % params.n_table_rows;
            assignment.query_count[query_id] += CircuitField::<C>::ONE;
            assignment.query_keys[i] = assignment.table_keys[query_id].clone();
            assignment.query_results[i] = assignment.table_values[query_id].clone();
        }

        assignment
    }
}

pub struct LogUpSingleKeyTable {
    pub table: Vec<Vec<Variable>>,
    pub query_keys: Vec<Variable>,
    pub query_results: Vec<Vec<Variable>>,
}

impl LogUpSingleKeyTable {
    pub fn new(_nb_bits: usize) -> Self {
        Self {
            table: vec![],
            query_keys: vec![],
            query_results: vec![],
        }
    }

    pub fn new_table(&mut self, key: Vec<Variable>, value: Vec<Vec<Variable>>) {
        if key.len() != value.len() {
            panic!("key and value should have the same length");
        }
        if !self.table.is_empty() {
            panic!("table already exists");
        }
        for i in 0..key.len() {
            let mut entry = vec![key[i]];
            entry.extend(value[i].clone());
            self.table.push(entry);
        }
    }

    pub fn add_table_row(&mut self, key: Variable, value: Vec<Variable>) {
        let mut entry = vec![key];
        entry.extend(value.clone());
        self.table.push(entry);
    }

    fn add_query(&mut self, key: Variable, value: Vec<Variable>) {
        let mut entry = vec![key];
        entry.extend(value.clone());
        self.query_keys.push(key);
        self.query_results.push(entry);
    }

    pub fn query(&mut self, key: Variable, value: Vec<Variable>) {
        self.add_query(key, value);
    }

    pub fn batch_query(&mut self, keys: Vec<Variable>, values: Vec<Vec<Variable>>) {
        for i in 0..keys.len() {
            self.add_query(keys[i], values[i].clone());
        }
    }

    pub fn final_check<C: Config, B: RootAPI<C>>(&mut self, builder: &mut B) {
        if self.table.is_empty() || self.query_keys.is_empty() {
            panic!("empty table or empty query");
        }

        let value_len = self.table[0].len();

        let alpha = builder.get_random_value();
        let randomness = get_column_randomness(builder, value_len);

        let table_combined = combine_columns(builder, &self.table, &randomness);
        let mut inputs = vec![builder.constant(self.table.len() as u32)];
        //append table keys
        for i in 0..self.table.len() {
            inputs.push(self.table[i][0]);
        }
        //append query keys
        inputs.extend(self.query_keys.clone());

        let query_count = builder.new_hint("myhint.querycountbykeyhint", &inputs, self.table.len());

        let v_table = logup_poly_val(builder, &table_combined, &query_count, &alpha);

        let query_combined = combine_columns(builder, &self.query_results, &randomness);
        let one = builder.constant(1);
        let v_query = logup_poly_val(
            builder,
            &query_combined,
            &vec![one; query_combined.len()],
            &alpha,
        );

        assert_eq_rational(builder, &v_table, &v_query);
    }
}

pub struct LogUpRangeProofTable {
    pub table_keys: Vec<Variable>,
    pub query_keys: Vec<Variable>,
    pub rangeproof_bits: usize,
}

impl LogUpRangeProofTable {
    pub fn new(nb_bits: usize) -> Self {
        Self {
            table_keys: vec![],
            query_keys: vec![],
            rangeproof_bits: nb_bits,
        }
    }

    pub fn initial<C: Config, B: RootAPI<C>>(&mut self, builder: &mut B) {
        for i in 0..1 << self.rangeproof_bits {
            let key = builder.constant(i as u32);
            self.add_table_row(key);
        }
    }

    pub fn add_table_row(&mut self, key: Variable) {
        self.table_keys.push(key);
    }

    pub fn add_query(&mut self, key: Variable) {
        self.query_keys.push(key);
    }

    pub fn rangeproof<C: Config, B: RootAPI<C>>(&mut self, builder: &mut B, a: Variable, n: usize) {
        if n > 128 {
            panic!("n must be less than 128");
        }
        if n <= self.rangeproof_bits {
            self.rangeproof_onechunk(builder, a, n);
            return;
        }
        //add a shift value
        let mut n = n;
        let mut new_a = a;
        let rem = n % self.rangeproof_bits;
        if rem != 0 {
            let shift = self.rangeproof_bits - rem;
            let constant = (1 << shift) - 1;

            let mut mul_factor: u128 = 1;
            mul_factor <<= n;
            let mul_factor_u256 = U256::new(mul_factor);
            let mul_factor_var = CircuitField::<C>::from_u256(mul_factor_u256);
            let mul_factor = builder.constant(mul_factor_var);
            let a_shift = builder.mul(constant, mul_factor);
            new_a = builder.add(a, a_shift);
            n += shift;
        }
        let hint_input = vec![
            builder.constant(n as u32),
            builder.constant(self.rangeproof_bits as u32),
            new_a,
        ];
        let witnesses = builder.new_hint(
            "myhint.rangeproofhint",
            &hint_input,
            n / self.rangeproof_bits,
        );
        let mut sum = witnesses[0];
        for (i, witness) in witnesses.iter().enumerate().skip(1) {
            let constant = 1u128 << (self.rangeproof_bits * i);
            let constant_u256 = U256::new(constant);
            let constant_u256_var = CircuitField::<C>::from_u256(constant_u256);
            let constant = builder.constant(constant_u256_var);
            let mul = builder.mul(witness, constant);
            sum = builder.add(sum, mul);
        }
        builder.assert_is_equal(sum, new_a);
        for witness in witnesses.iter().take(n / self.rangeproof_bits) {
            self.query_range(*witness);
        }
    }

    pub fn rangeproof_onechunk<C: Config, B: RootAPI<C>>(
        &mut self,
        builder: &mut B,
        a: Variable,
        n: usize,
    ) {
        //n must be less than self.rangeproof_bits, not need the hint
        if n > self.rangeproof_bits {
            panic!("n must be less than self.rangeproof_bits");
        }
        //add a shift value
        let mut new_a = a;
        if n % self.rangeproof_bits != 0 {
            let rem = n % self.rangeproof_bits;
            let shift = self.rangeproof_bits - rem;
            let constant = (1 << shift) - 1;
            let mut mul_factor = 0;
            mul_factor <<= n;
            let a_shift = builder.mul(constant, mul_factor);
            new_a = builder.add(a, a_shift);
        }
        self.query_range(new_a);
    }

    pub fn query_range(&mut self, key: Variable) {
        self.query_keys.push(key);
    }

    pub fn final_check<C: Config, B: RootAPI<C>>(&mut self, builder: &mut B) {
        let alpha = builder.get_random_value();
        let inputs = self.query_keys.clone();

        let query_count = builder.new_hint("myhint.querycounthint", &inputs, self.table_keys.len());

        let v_table = logup_poly_val(builder, &self.table_keys, &query_count, &alpha);

        let one = builder.constant(1);
        let v_query = logup_poly_val(
            builder,
            &self.query_keys,
            &vec![one; self.query_keys.len()],
            &alpha,
        );
        assert_eq_rational(builder, &v_table, &v_query);
    }
}

pub fn query_count_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let mut count = vec![0; outputs.len()];
    for input in inputs {
        let query_id = input.to_u256().as_usize();
        count[query_id] += 1;
    }
    for i in 0..outputs.len() {
        outputs[i] = F::from(count[i] as u32);
    }
    Ok(())
}

pub fn query_count_by_key_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let mut outputs_u32 = vec![0; outputs.len()];

    let table_size = inputs[0].to_u256().as_usize();
    let table = &inputs[1..=table_size];
    let query_keys = &inputs[(table_size + 1)..];

    let mut table_map: HashMap<u32, usize> = HashMap::new();
    for key in query_keys {
        let key_value = key.to_u256().as_u32();
        *table_map.entry(key_value).or_insert(0) += 1;
    }

    for (i, value) in table.iter().enumerate() {
        let key_value = value.to_u256().as_u32();
        let count = table_map.get(&key_value).copied().unwrap_or(0);
        outputs_u32[i] = count as u32;
    }
    for i in 0..outputs.len() {
        outputs[i] = F::from(outputs_u32[i]);
    }

    Ok(())
}

pub fn rangeproof_hint<F: Field>(inputs: &[F], outputs: &mut [F]) -> Result<(), Error> {
    let n = inputs[0].to_u256().as_i64();
    let m = inputs[1].to_u256().as_i64();

    let mut a = inputs[2].to_u256(); // Value 'new_a' as U256

    let chunk_mod = U256::from(1_u64) << m;

    for i in 0..n / m {
        let r = a % chunk_mod;
        a /= chunk_mod;
        // println!(
        //     "rangeproof_hint chunk {}: r = {}, a_remaining = {}",
        //     i, r, a
        // ); // Log chunks
        outputs[i as usize] = F::from(r.as_u32());
    }
    Ok(())
}
