use expander_compiler::frontend::*;
use extra::UnconstrainedAPI;

declare_circuit!(Circuit {
    input: PublicVariable,
});

fn to_binary<C: Config>(api: &mut API<C>, x: Variable, n_bits: usize) -> Vec<Variable> {
    let mut res = Vec::new();
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res.push(api.unconstrained_bit_and(y, 1));
    }
    res
}

fn from_binary<C: Config>(api: &mut API<C>, bits: Vec<Variable>) -> Variable {
    let mut res = api.constant(0);
    for i in 0..bits.len() {
        let coef = 1 << i;
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    res
}

impl Define<M31Config> for Circuit<Variable> {
    fn define(&self, builder: &mut API<M31Config>) {
        let bits = to_binary(builder, self.input, 8);
        let x = from_binary(builder, bits);
        builder.assert_is_equal(x, self.input);
    }
}

#[test]
fn test_300() {
    let compile_result = compile(&Circuit::default()).unwrap();
    for i in 0..300 {
        let assignment = Circuit::<M31> {
            input: M31::from(i as u32),
        };
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let output = compile_result.layered_circuit.run(&witness);
        assert_eq!(output, vec![i < 256]);
    }
}
