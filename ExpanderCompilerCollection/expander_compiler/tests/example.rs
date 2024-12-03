use expander_compiler::frontend::*;
use extra::Serde;

declare_circuit!(Circuit {
    x: Variable,
    y: Variable,
});

impl Define<M31Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<M31Config>) {
        builder.assert_is_equal(self.x, self.y);
    }
}

#[test]
fn example_full() {
    let compile_result = compile(&Circuit::default()).unwrap();
    let assignment = Circuit::<M31> {
        x: M31::from(123),
        y: M31::from(123),
    };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);

    let file = std::fs::File::create("circuit.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result
        .layered_circuit
        .serialize_into(writer)
        .unwrap();

    let file = std::fs::File::create("witness.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();

    let file = std::fs::File::create("witness_solver.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    compile_result
        .witness_solver
        .serialize_into(writer)
        .unwrap();
}
