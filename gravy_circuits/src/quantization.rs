use expander_compiler::frontend::*;
use relu::{from_binary, to_binary};

pub mod relu;

// Version 1
fn quantize<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_value: Variable,
    scaling_factor: u32,
    scaling: usize
) -> Variable {
    let (q, rem) = div_unconstrained(api, input_value, scaling_factor);
    let quotient = div_constrained(api, input_value, scaling_factor, q, rem, scaling);
    quotient
}

pub fn scaling_factor_to_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling_power: u32,
) -> Variable {
    let mut out = api.constant(1);
    for _ in 0..scaling_power{
        out = api.mul(2,out);
    }
    out
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
    y: u32,
) -> (Variable, Variable) {
    let q = api.unconstrained_int_div(x, y);
    let rem = api.unconstrained_mod(x, y);
    (q, rem)
}

fn constrain_rem<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling: usize,
    rem: Variable
){
    let bits = to_binary(api, rem, scaling);
    let total = from_binary(api, &bits, scaling);
    api.assert_is_equal(total, rem);
}

fn div_constrained<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: u32,
    q: Variable,
    rem: Variable,
    scaling: usize
) -> Variable {
    let temp = api.mul(y, q);
    let temp2 = api.add(temp, rem);
    api.assert_is_equal(x, temp2);
    constrain_rem(api, scaling, rem);
    q
}


pub fn quantize_matrix<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: u32,
    scaling: usize
) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for (_, row) in input_matrix.iter().enumerate() {
        let mut row_out: Vec<Variable> = Vec::new();
        for (_, &element) in row.iter().enumerate() {
            row_out.push(quantize(api, element, scaling_factor, scaling))
        }
        out.push(row_out);
    }
    out
}