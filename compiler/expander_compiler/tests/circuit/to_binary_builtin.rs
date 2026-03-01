use expander_compiler::frontend::*;

declare_circuit!(Circuit {
    input: PublicVariable,
});

impl Define<M31Config> for Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let bits = builder.to_binary(self.input, 8);
        let x = builder.from_binary(&bits);
        builder.assert_is_equal(x, self.input);
    }
}

#[test]
fn test_small() {
    let compile_result = compile(&Circuit::default(), CompileOptions::default()).unwrap();
    for i in 0..256 {
        let assignment = Circuit::<M31> {
            input: M31::from(i as u32),
        };
        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut EmptyHintCaller)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![i < 256]);
    }
}

#[test]
#[should_panic]
fn test_big() {
    let compile_result = compile(&Circuit::default(), CompileOptions::default()).unwrap();
    for i in 256..257 {
        let assignment = Circuit::<M31> {
            input: M31::from(i as u32),
        };
        let witness = compile_result
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut EmptyHintCaller)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![i < 256]);
    }
}
