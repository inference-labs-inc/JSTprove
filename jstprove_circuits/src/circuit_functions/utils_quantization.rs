use crate::circuit_functions::activation_relu::{from_binary, to_binary};
use ethnum::U256;
use expander_compiler::frontend::*;

///Constrain size of value to within 2**scaling, using the to and from binary method
fn constrain_size<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling: usize,
    val: Variable,
) {
    let bits = to_binary(api, val, scaling);
    let total = from_binary(api, &bits, scaling);
    api.assert_is_equal(total, val);
}

///Constrain size of value to within 2**scaling, using the to and from binary method
fn div_constrained<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
) -> Variable {
    //6.
    let d_sharp = api.add(alpha_two_v, x);

    //7.
    let q_sharp = api.unconstrained_int_div(d_sharp, y);
    let rem_sharp = api.unconstrained_mod(d_sharp, y);

    //Assert that q_sharp is in correct range
    let bits = to_binary(api, q_sharp, v_plus_one);
    let total = from_binary(api, &bits, v_plus_one);
    // api.display("q_sharp", q_sharp);
    // api.display("total  ", total);


    api.assert_is_equal(q_sharp, total);

    // Ensure remainder sharp is in correct range
    constrain_size(api, scaling, rem_sharp);




    // d_sharp = q_sharp * y + rem_sharp

    let val = api.mul(q_sharp, y);

    let val_2 = api.add(val, rem_sharp);

    api.assert_is_equal(val_2, d_sharp);




    let q = api.sub(q_sharp, CircuitField::<C>::from_u256(U256::from(two_v.into())));
    // If relu optimization, multiply by most significant bit as bits have
    if is_relu {
        return api.mul(q, bits[v_plus_one - 1]);
    } else {
        return q;
    }
}

/// Quantize a single value using to_binary and from_binary method
fn quantize<C: Config,T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_value: Variable,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
) -> Variable {
    return div_constrained(
        api,
        input_value,
        scaling_factor,
        scaling,
        v_plus_one,
        two_v,
        alpha_two_v,
        is_relu,
    );
}

/// Quantize a 2d vector, by computing element by element division. Can potentially try memorized simple call here for optimizations?
pub fn quantize_2d_vector<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();
    let two_v: u64 = two_v.into();
    
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
                is_relu,
            ));
        }
        out.push(dim1_out);
    }
    out
}

/// run quantized 2d vector only if quantized boolean is true
pub fn run_if_quantized_2d<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling_in: u64,
    quantized: bool,
    out: Vec<Vec<Variable>>,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
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
            is_relu,
        );
    }
    return out;
}