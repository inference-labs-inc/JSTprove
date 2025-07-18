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

fn quantize<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_value: Variable,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
) -> Variable {
    // Step 6: Compute d♯ = α·2ᵛ + x
    let d_sharp = api.add(alpha_two_v, input_value);

    // Step 7: Perform unconstrained division and modulo
    let q_sharp = api.unconstrained_int_div(d_sharp, scaling_factor);
    let rem_sharp = api.unconstrained_mod(d_sharp, scaling_factor);

    // Assert q♯ is in range via binary representation
    let bits = to_binary(api, q_sharp, v_plus_one);
    let total = from_binary(api, &bits, v_plus_one);
    api.assert_is_equal(q_sharp, total);

    // Constrain rem♯ to be in correct size range
    constrain_size(api, scaling, rem_sharp);

    // Enforce d♯ = q♯ * y + rem♯
    let val = api.mul(q_sharp, scaling_factor);
    let val_2 = api.add(val, rem_sharp);
    api.assert_is_equal(val_2, d_sharp);

    // Shift back by 2ᵛ and optionally multiply by MSB for ReLU
    let q = api.sub(q_sharp, CircuitField::<C>::from_u256(U256::from(two_v.into())));
    if is_relu {
        return api.mul(q, bits[v_plus_one - 1]);
    } else {
        return q;
    }
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

/// Quantize a matrix, by computing element by element division, using lookups. Can potentially try memorized simple call here for optimizations?
pub fn quantize_4d_vector<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Vec<Vec<Variable>>>>,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let mut out: Vec<Vec<Vec<Vec<Variable>>>> = Vec::new();
    let two_v: u64 = two_v.into();
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
                        is_relu,
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

/// run quantized 4d vector only if quantized boolean is true
pub fn run_if_quantized_4d<C: Config, T: Into<u64>, Builder: RootAPI<C>>(
    api: &mut Builder,
    scaling_in: u64,
    quantized: bool,
    out: Vec<Vec<Vec<Vec<Variable>>>>,
    v_plus_one: usize,
    two_v: T,
    alpha_two_v: Variable,
    is_relu: bool,
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
            is_relu,
        );
    }
    return out;
}
