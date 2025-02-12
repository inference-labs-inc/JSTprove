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
    //can maybe extract this
    let middle = C::CircuitField::from_u256(C::CircuitField::MODULUS / 2);
    let negative_one = C::CircuitField::from(0) - C::CircuitField::from(1);


    let is_pos = api.unconstrained_lesser(x, middle);

    //if x is positive
    let q_pos = api.unconstrained_int_div(x, y);
    let q_pos = api.unconstrained_mul(q_pos, is_pos);

    //if x is negative
    // let x_squared = api.unconstrained_pow(x,-1);
    let x_abs = api.unconstrained_mul(x, negative_one);
    let q_neg = api.unconstrained_int_div(x_abs, y);
    let is_neg = api.unconstrained_bit_xor(is_pos, 1);
    let q_neg_temp = api.unconstrained_mul(q_neg, negative_one);
    // Unsure why, but I must subtract one here
    let q_neg_temp = api.unconstrained_add(q_neg_temp, negative_one);
    let q_neg_out = api.unconstrained_mul(q_neg_temp, is_neg);


    let q = api.unconstrained_add(q_pos, q_neg_out);
    //if a and q are pos
    let rem_pos = api.unconstrained_mod(x, y);
    let rem_pos_out = api.unconstrained_mul(rem_pos, is_pos);

    //if a and q are negative then r = a - bq -> r = a + b(-q)
    let rem_neg = api.unconstrained_mul(y, q);
    let rem_neg = api.unconstrained_mul(rem_neg, negative_one);
    let rem_neg = api.unconstrained_add(x, rem_neg);
    let rem_neg_out = api.unconstrained_mul(is_neg, rem_neg);

    let rem_out = api.unconstrained_add(rem_neg_out, rem_pos_out);

    (q, rem_out)
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

pub fn quantize_4d_vector<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Vec<Vec<Variable>>>>,
    scaling_factor: u32,
    scaling: usize
) -> Vec<Vec<Vec<Vec<Variable>>>>{
    let mut out: Vec<Vec<Vec<Vec<Variable>>>> = Vec::new();
    for (_, dim1) in input_matrix.iter().enumerate() {
        let mut dim1_out: Vec<Vec<Vec<Variable>>> = Vec::new();
        for (_, dim2) in dim1.iter().enumerate() {
            let mut dim2_out: Vec<Vec<Variable>> = Vec::new();
            for (_, dim3) in dim2.iter().enumerate() {
                let mut dim3_out: Vec<Variable> = Vec::new();
                for (_, &element) in dim3.iter().enumerate() {
                    dim3_out.push(quantize(api, element, scaling_factor, scaling));
                }
                dim2_out.push(dim3_out);
            }
            dim1_out.push(dim2_out)
        }
        out.push(dim1_out);
    }
    out
}

pub fn quantize_2d_vector<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: u32,
    scaling: usize
) -> Vec<Vec<Variable>>{
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for (_, dim1) in input_matrix.iter().enumerate() {
        let mut dim1_out: Vec<Variable> = Vec::new();
        for (_, &element) in dim1.iter().enumerate() {
            dim1_out.push(quantize(api, element, scaling_factor, scaling));
        }
        out.push(dim1_out);
    }
    out
}