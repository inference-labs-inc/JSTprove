use expander_compiler::frontend::*;
use serdes::ExpSerde;

declare_circuit!(Circuit {
    x: Variable,
    y: Variable,
});

impl Define<M31Config> for Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, api: &mut Builder) {
        api.assert_is_equal(self.x, self.y);
    }
}

#[test]
fn example_full() {
    let compile_result = compile(&Circuit::default(), CompileOptions::default()).unwrap();
    assert_eq!(compile_result.layered_circuit.layer_ids.len(), 2);
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

#[test]
fn example_cross_layer() {
    let compile_result =
        compile_cross_layer(&Circuit::default(), CompileOptions::default()).unwrap();
    assert_eq!(compile_result.layered_circuit.layer_ids.len(), 2);
}
