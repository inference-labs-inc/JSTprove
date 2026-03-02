use crate::StdCircuit;
use arith::Field;
use expander_compiler::frontend::{
    declare_circuit, CircuitField, Config, Define, RootAPI, Variable, M31,
};
use std::convert::From;
use std::ops::{AddAssign, Mul};

#[derive(Clone, Copy, Debug)]
pub struct MatMulParams {
    pub m1: usize,
    pub n1: usize,
    pub m2: usize,
    pub n2: usize,
}

declare_circuit!(_MatMulCircuit {
    // first matrix
    first_mat: [[Variable]],
    // second matrix
    second_mat: [[Variable]],
    // result matrix
    result_mat: [[Variable]],
});

pub type MatMulCircuit = _MatMulCircuit<Variable>;

impl<C: Config> Define<C> for MatMulCircuit {
    fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
        // [m1,n1] represents the first matrix's dimension
        let m1 = self.first_mat.len();
        let n1 = self.first_mat[0].len();

        // [m2,n2] represents the second matrix's dimension
        let m2 = self.second_mat.len();
        let n2 = self.second_mat[0].len();

        // [r1,r2] represents the result matrix's dimension
        let r1 = self.result_mat.len();
        let r2 = self.result_mat[0].len();
        let zero = builder.constant(0);

        builder.assert_is_equal(Variable::from(n1), Variable::from(m2));
        builder.assert_is_equal(Variable::from(r1), Variable::from(m1));
        builder.assert_is_equal(Variable::from(r2), Variable::from(n2));

        let loop_count = if CircuitField::<C>::SIZE == M31::SIZE {
            3
        } else {
            1
        };

        for _ in 0..loop_count {
            let randomness = builder.get_random_value();
            let mut aux_mat = Vec::new();
            let mut challenge = randomness;

            // construct the aux matrix = [1, randomness, randomness^2, ..., randomness^（n-1）]
            aux_mat.push(Variable::from(1));
            for _ in 0..n2 - 1 {
                challenge = builder.mul(challenge, randomness);
                aux_mat.push(challenge);
            }

            let mut aux_second = vec![zero; m2];
            let mut aux_first = vec![zero; m1];
            let mut aux_res = vec![zero; m1];

            // calculate second_mat * aux_mat,
            self.matrix_multiply(builder, &mut aux_second, &aux_mat, &self.second_mat);
            // calculate result_mat * aux_second
            self.matrix_multiply(builder, &mut aux_res, &aux_mat, &self.result_mat);
            // calculate first_mat * aux_second
            self.matrix_multiply(builder, &mut aux_first, &aux_second, &self.first_mat);

            // compare aux_first with aux_res
            for i in 0..m1 {
                builder.assert_is_equal(aux_first[i], aux_res[i]);
            }
        }
    }
}

impl MatMulCircuit {
    // calculate origin_mat * aux_mat and store the result into target_mat
    fn matrix_multiply<C: Config>(
        &self,
        builder: &mut impl RootAPI<C>,
        target_mat: &mut [Variable], // target to modify
        aux_mat: &[Variable],
        origin_mat: &[Vec<Variable>],
    ) {
        // for i in 0..target_mat.len{
        //     for j in 0..aux_mat.len {
        //         let mul_result = builder.mul(origin_mat[i][j], aux_mat[j]);
        //         target_mat[i] = builder.add(target_mat[i], mul_result);
        //     }
        // }
        for (i, target_item) in target_mat.iter_mut().enumerate() {
            for (j, item) in aux_mat.iter().enumerate() {
                let mul_result = builder.mul(origin_mat[i][j], item);
                *target_item = builder.add(*target_item, mul_result);
            }
        }
    }
}

impl<C: Config> StdCircuit<C> for MatMulCircuit {
    type Params = MatMulParams;
    type Assignment = _MatMulCircuit<CircuitField<C>>;

    fn new_circuit(params: &Self::Params) -> Self {
        let mut circuit = Self::default();

        circuit
            .first_mat
            .resize(params.m1, vec![Variable::default(); params.n1]);
        circuit
            .second_mat
            .resize(params.m2, vec![Variable::default(); params.n2]);

        circuit
            .result_mat
            .resize(params.m1, vec![Variable::default(); params.n2]);

        circuit
    }

    fn new_assignment(params: &Self::Params, mut rng: impl rand::RngCore) -> Self::Assignment {
        let mut assignment = _MatMulCircuit::<CircuitField<C>>::default();
        assignment
            .first_mat
            .resize(params.m1, vec![CircuitField::<C>::zero(); params.n1]);
        assignment
            .second_mat
            .resize(params.m2, vec![CircuitField::<C>::zero(); params.n2]);
        assignment
            .result_mat
            .resize(params.m1, vec![CircuitField::<C>::zero(); params.n2]);

        for i in 0..params.m1 {
            for j in 0..params.n1 {
                assignment.first_mat[i][j] = CircuitField::<C>::random_unsafe(&mut rng);
            }
        }
        for i in 0..params.m2 {
            for j in 0..params.n2 {
                assignment.second_mat[i][j] = CircuitField::<C>::random_unsafe(&mut rng);
            }
        }

        // initialize the aux matrix with random values.
        // result matrix should be computed
        assignment.result_mat = matrix_multiply::<C>(&assignment.first_mat, &assignment.second_mat);

        assignment
    }
}

// this helper calculates matrix c = a * b;
#[allow(clippy::needless_range_loop)]
fn matrix_multiply<C: Config>(
    a: &[Vec<CircuitField<C>>],
    b: &[Vec<CircuitField<C>>],
) -> Vec<Vec<CircuitField<C>>> {
    let m1 = a.len();
    let n1 = a[0].len();
    let m2 = b.len();
    let n2 = b[0].len();

    assert_eq!(n1, m2, "n1 ! = m2 ");

    // initialize the result matrix
    let mut c = vec![vec![CircuitField::<C>::default(); n2]; m1];

    // FIXME: optimize calculating the multiplication for super large matrix.
    for i in 0..m1 {
        for j in 0..n2 {
            for k in 0..n1 {
                c[i][j].add_assign(a[i][k].mul(b[k][j]));
            }
        }
    }

    c
}
