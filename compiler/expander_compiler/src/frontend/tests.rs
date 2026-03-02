use crate::frontend::M31Config as C;
use crate::{
    compile::CompileOptions,
    field::{FieldArith, M31},
    frontend::{compile, RootAPI},
};

use super::{builder::Variable, circuit::*, variables::DumpLoadTwoVariables};

declare_circuit!(Circuit1 {
    a: Variable,
    b: [PublicVariable; 2],
    c: u64,
    d: [[Variable; 3]; 5],
    e: [[[u64; 4]; 3]; 2],
});

#[test]
fn test_circuit_declaration() {
    use crate::field::M31 as F;
    let c = Circuit1::<F> {
        a: F::one(),
        b: [F::one(), F::zero()],
        c: 1,
        d: [
            [F::one(), F::zero(), F::one()],
            [F::zero(), F::one(), F::zero()],
            [F::one(), F::zero(), F::one()],
            [F::zero(), F::one(), F::zero()],
            [F::one(), F::zero(), F::one()],
        ],
        e: [
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ],
    };
    assert_eq!(c.num_vars(), (1 + 3 * 5, 2));
    let mut vars = vec![];
    let mut public_vars = vec![];
    c.dump_into(&mut vars, &mut public_vars);
    assert_eq!((vars.len(), public_vars.len()), c.num_vars());
    let mut c2 = Circuit1::<F>::default();
    let vars_ref = &mut vars.as_slice();
    let public_vars_ref = &mut public_vars.as_slice();
    c2.load_from(vars_ref, public_vars_ref);
    assert_eq!(vars_ref.len(), 0);
    assert_eq!(public_vars_ref.len(), 0);
    assert_eq!(c.a, c2.a);
    assert_eq!(c.b, c2.b);
    assert_eq!(c.d, c2.d);
}

declare_circuit!(Circuit2 {
    sum: Variable,
    x: [Variable; 2],
});

impl Define<C> for Circuit2<Variable> {
    fn define<Builder: RootAPI<C>>(&self, builder: &mut Builder) {
        let sum = builder.add(self.x[0], self.x[1]);
        let sum = builder.add(sum, 123);
        builder.assert_is_equal(sum, self.sum);
    }
}

#[test]
fn test_circuit_eval_simple() {
    let compile_result = compile(&Circuit2::default(), CompileOptions::default()).unwrap();
    let assignment = Circuit2::<M31> {
        sum: M31::from(126),
        x: [M31::from(1), M31::from(2)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);

    let assignment = Circuit2::<M31> {
        sum: M31::from(127),
        x: [M31::from(1), M31::from(2)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![false]);
}
