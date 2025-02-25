use expander_compiler::frontend::*;
use relu::{from_binary, to_binary};

pub mod relu;

// Version 1
fn quantize<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_value: Variable,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Variable {
    // let q = div_unconstrained(api, input_value, scaling_factor);
    return div_constrained(
        api,
        input_value,
        scaling_factor,
        // q,
        scaling,
        v_plus_one,
        two_v,
        alpha_two_v,
        is_relu
    );
}

pub fn scaling_factor_to_constant<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling_power: u32,
) -> Variable {
    let mut out = api.constant(1);
    for _ in 0..scaling_power {
        out = api.mul(2, out);
    }
    out
}

fn constrain_rem<C: Config, Builder: RootAPI<C>>(api: &mut Builder, scaling: usize, rem: Variable) {
    let bits = to_binary(api, rem, scaling);
    let total = from_binary(api, &bits, scaling);
    api.assert_is_equal(total, rem);
}

fn div_constrained<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Variable {

    //6. 
    let d_sharp = api.add(alpha_two_v, x);

    //7.
    let q_sharp = api.unconstrained_int_div(d_sharp, y);
    let rem_sharp = api.unconstrained_mod(d_sharp, y);

    //Assert that q_sharp is in correct range
    let bits = to_binary(api, q_sharp, v_plus_one);
    let total = from_binary(api, &bits, v_plus_one);
    api.assert_is_equal(q_sharp, total);

    //Ensure remainder sharp is in correct range
    //Should this be scaling -1?
    constrain_rem(api, scaling, rem_sharp);

    let q = api.sub(q_sharp, two_v);
    if is_relu{
        return api.mul(q, bits[v_plus_one - 1]);
    }
    else{
        return q;
    }
}

pub fn quantize_matrix<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for (_, row) in input_matrix.iter().enumerate() {
        let mut row_out: Vec<Variable> = Vec::new();
        for (_, &element) in row.iter().enumerate() {
            row_out.push(quantize(
                api,
                element,
                scaling_factor,
                scaling,
                v_plus_one,
                two_v,
                alpha_two_v,
                is_relu
            ))
        }
        out.push(row_out);
    }
    out
}

pub fn quantize_4d_vector<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Vec<Vec<Variable>>>>,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let mut out: Vec<Vec<Vec<Vec<Variable>>>> = Vec::new();
    for (_, dim1) in input_matrix.iter().enumerate() {
        let mut dim1_out: Vec<Vec<Vec<Variable>>> = Vec::new();
        for (_, dim2) in dim1.iter().enumerate() {
            let mut dim2_out: Vec<Vec<Variable>> = Vec::new();
            for (_, dim3) in dim2.iter().enumerate() {
                let mut dim3_out: Vec<Variable> = Vec::new();
                for (_, &element) in dim3.iter().enumerate() {
                    dim3_out.push(quantize(
                        api,
                        element,
                        scaling_factor,
                        scaling,
                        v_plus_one,
                        two_v,
                        alpha_two_v,
                        is_relu
                    ));
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
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();
    for (_, dim1) in input_matrix.iter().enumerate() {
        let mut dim1_out: Vec<Variable> = Vec::new();
        for (_, &element) in dim1.iter().enumerate() {
            dim1_out.push(quantize(
                api,
                element,
                scaling_factor,
                scaling,
                v_plus_one,
                two_v,
                alpha_two_v,
                is_relu
            ));
        }
        out.push(dim1_out);
    }
    out
}

pub fn run_if_quantized_4d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling_in: u64,
    quantized: bool,
    out: Vec<Vec<Vec<Vec<Variable>>>>,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    if quantized {
        let scaling_factor = 1 << scaling_in;
        return quantize_4d_vector(
            api,
            out,
            scaling_factor,
            scaling_in as usize,
            v_plus_one,
            two_v,
            alpha_two_v,
            is_relu
        );
    }
    return out;
}

pub fn run_if_quantized_2d<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling_in: u64,
    quantized: bool,
    out: Vec<Vec<Variable>>,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool
) -> Vec<Vec<Variable>> {
    if quantized {
        let scaling_factor = 1 << scaling_in;
        return quantize_2d_vector(
            api,
            out,
            scaling_factor,
            scaling_in as usize,
            v_plus_one,
            two_v,
            alpha_two_v,
            is_relu
        );
    }
    return out;
}

//Old versions
/*
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

    fn div_constrained_old<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: u32,
    q: Variable,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
) -> Variable {
    //5.
    let q_sharp = api.add(q, two_v);
    //6.
    let bits = to_binary(api, q_sharp, v_plus_one);
    let total = from_binary(api, &bits, v_plus_one);
    api.assert_is_equal(q_sharp, total);


    //7. Note, must incorporate 1 << scaling into the parameters... Also temp can be done outside as well
    // let temp = api.mul(two_v, 1 << scaling);
    // let temp: u64 = two_v as u64 * (1 << scaling);

    let d_sharp = api.add(alpha_two_v, x);

    //8
    let q_flat = api.unconstrained_int_div(d_sharp, y);
    let rem_flat = api.unconstrained_mod(d_sharp, y);

    let temp = api.mul(q_flat, y);
    let temp2 = api.add(temp, rem_flat);
    api.assert_is_equal(d_sharp, temp2);
    constrain_rem(api, scaling, rem_flat);

    //9
    api.assert_is_equal(q_flat, q_sharp);
    q
}

    fn div_unconstrained<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: u32,
) -> Variable {
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
    // //if a and q are pos
    // let rem_pos = api.unconstrained_mod(x, y);
    // let rem_pos_out = api.unconstrained_mul(rem_pos, is_pos);

    // //if a and q are negative then r = a - bq -> r = a + b(-q)
    // let rem_neg = api.unconstrained_mul(y, q);
    // let rem_neg = api.unconstrained_mul(rem_neg, negative_one);
    // let rem_neg = api.unconstrained_add(x, rem_neg);
    // let rem_neg_out = api.unconstrained_mul(is_neg, rem_neg);

    // let rem_out = api.unconstrained_add(rem_neg_out, rem_pos_out);

    q
}
 */
