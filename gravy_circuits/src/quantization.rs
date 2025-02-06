use expander_compiler::frontend::*;

// Version 1
fn quantize<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_value: Variable,
    scaling_factor: Variable,
) -> Variable {
    let (q, rem) = div_unconstrained(api, input_value, scaling_factor);
    let quotient = div_constrained(api, input_value, scaling_factor, q, rem);
    quotient
}


// fn div_hint(x: &[M31], y: [M31]) -> Result<(), Error> {
//     let t = x[0].to_u256();
//     for (i, k) in y.iter_mut().enumerate() {
//         *k = M31::from_u256(t >> i as u32 & 1);
//     }
//     Ok(())
// }

fn div_unconstrained<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: Variable,
) -> (Variable, Variable) {
    let q = api.unconstrained_int_div(x, y);
    let rem = api.unconstrained_mod(x, y);
    (q, rem)
}

fn div_constrained<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: Variable,
    q: Variable,
    rem: Variable
) -> Variable {
    let temp = api.mul(y, q);
    let temp2 = api.add(temp, rem);
    api.assert_is_equal(x, temp2);
    q
}


pub fn quantize_matrix<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: Variable,
) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for (_, row) in input_matrix.iter().enumerate() {
        let mut row_out: Vec<Variable> = Vec::new();
        for (_, &element) in row.iter().enumerate() {
            row_out.push(quantize(api, element, scaling_factor))
        }
        out.push(row_out);
    }
    out
}