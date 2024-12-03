use expander_compiler::frontend::*;

declare_circuit!(Circuit {
    sum: PublicVariable,
    x: [Variable; 2],
});

impl Define<M31Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<M31Config>) {
        let sum = builder.add(self.x[0], self.x[1]);
        let sum = builder.add(sum, 123);
        builder.assert_is_equal(sum, self.sum);
    }
}

#[test]
fn test_circuit_eval_simple() {
    let compile_result = compile(&Circuit::default()).unwrap();
    let assignment = Circuit::<M31> {
        sum: M31::from(126),
        x: [M31::from(1), M31::from(2)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);

    let assignment = Circuit::<M31> {
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
