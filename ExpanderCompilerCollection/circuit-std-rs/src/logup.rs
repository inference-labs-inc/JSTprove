use arith::Field;
use expander_compiler::frontend::*;
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

    // counting the number of occurences for each row of the table
    query_count: [Variable],
});

pub type LogUpCircuit = _LogUpCircuit<Variable>;

#[derive(Clone, Copy, Debug)]
struct Rational {
    numerator: Variable,
    denominator: Variable,
}

fn add_rational<C: Config>(builder: &mut API<C>, v1: &Rational, v2: &Rational) -> Rational {
    let p1 = builder.mul(v1.numerator, v2.denominator);
    let p2 = builder.mul(v1.denominator, v2.numerator);

    Rational {
        numerator: builder.add(p1, p2),
        denominator: builder.mul(v1.denominator, v2.denominator),
    }
}

fn assert_eq_rational<C: Config>(builder: &mut API<C>, v1: &Rational, v2: &Rational) {
    let p1 = builder.mul(v1.numerator, v2.denominator);
    let p2 = builder.mul(v1.denominator, v2.numerator);
    builder.assert_is_equal(p1, p2);
}

fn sum_rational_vec<C: Config>(builder: &mut API<C>, vs: &[Rational]) -> Rational {
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

// TODO-Feature: poly randomness
fn get_column_randomness<C: Config>(builder: &mut API<C>, n_columns: usize) -> Vec<Variable> {
    let mut randomness = vec![];
    randomness.push(builder.constant(1));
    for _ in 1..n_columns {
        randomness.push(builder.get_random_value());
    }
    randomness
}

fn concat_d1(v1: &[Vec<Variable>], v2: &[Vec<Variable>]) -> Vec<Vec<Variable>> {
    v1.iter()
        .zip(v2.iter())
        .map(|(row_key, row_value)| [row_key.to_vec(), row_value.to_vec()].concat())
        .collect()
}

fn combine_columns<C: Config>(
    builder: &mut API<C>,
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

fn logup_poly_val<C: Config>(
    builder: &mut API<C>,
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
    fn define(&self, builder: &mut API<C>) {
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
    type Assignment = _LogUpCircuit<C::CircuitField>;

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
        let mut assignment = _LogUpCircuit::<C::CircuitField>::default();
        assignment.table_keys.resize(params.n_table_rows, vec![]);
        assignment.table_values.resize(params.n_table_rows, vec![]);
        assignment.query_keys.resize(params.n_queries, vec![]);
        assignment.query_results.resize(params.n_queries, vec![]);

        for i in 0..params.n_table_rows {
            for _ in 0..params.key_len {
                assignment.table_keys[i].push(C::CircuitField::random_unsafe(&mut rng));
            }

            for _ in 0..params.value_len {
                assignment.table_values[i].push(C::CircuitField::random_unsafe(&mut rng));
            }
        }

        assignment.query_count = vec![C::CircuitField::ZERO; params.n_table_rows];
        for i in 0..params.n_queries {
            let query_id: usize = rng.gen::<usize>() % params.n_table_rows;
            assignment.query_count[query_id] += C::CircuitField::ONE;
            assignment.query_keys[i] = assignment.table_keys[query_id].clone();
            assignment.query_results[i] = assignment.table_values[query_id].clone();
        }

        assignment
    }
}
