use crate::circuit_functions::relu::{from_binary, to_binary};
use circuit_std_rs::logup::LogUpRangeProofTable;
use ethnum::U256;
use expander_compiler::frontend::*;

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

/// Quantize a single value using lookup table
fn quantize_lookup<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_value: Variable,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool,
    table: &mut LogUpRangeProofTable,
) -> Variable {
    return div_constrained_lookup(
        api,
        input_value,
        scaling_factor,
        scaling,
        v_plus_one,
        two_v,
        alpha_two_v,
        is_relu,
        table,
    );
}
/// Create constant from scaling factor, inside the the circuit 
/// This typically should not be needed, I dont think this needs to be confirmed inside the circuit as it is a circuit parameter
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
    api.display("d_sharp", d_sharp);

    api.assert_is_equal(q_sharp, total);

    // Ensure remainder sharp is in correct range
    constrain_size(api, scaling, rem_sharp);

    let q = api.sub(q_sharp, CircuitField::<C>::from_u256(U256::from(two_v.into())));
    // If relu optimization, multiply by most significant bit as bits have
    if is_relu {
        return api.mul(q, bits[v_plus_one - 1]);
    } else {
        return q;
    }
}
/// Perform division, using rangeproofs for tablesize
fn div_constrained_lookup<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    y: u32,
    _scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool,
    table: &mut LogUpRangeProofTable,
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
    table.rangeproof(api, rem_sharp, y.try_into().unwrap());

    let q = api.sub(q_sharp, two_v);
    if is_relu {
        return api.mul(q, bits[v_plus_one - 1]);
    } else {
        return q;
    }
}

/// Quantize a matrix, by computing element by element division. Can potentially try memorized simple call here for optimizations?
pub fn quantize_matrix<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool,
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
                is_relu,
            ))
        }
        out.push(row_out);
    }
    out
}
/// Quantize a matrix, by computing element by element division. Can potentially try memorized simple call here for optimizations?
pub fn quantize_matrix_lookup<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input_matrix: Vec<Vec<Variable>>,
    scaling_factor: u32,
    scaling: usize,
    v_plus_one: usize,
    two_v: u32,
    alpha_two_v: Variable,
    is_relu: bool,
    table: &mut LogUpRangeProofTable,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();

    for (_, row) in input_matrix.iter().enumerate() {
        let mut row_out: Vec<Variable> = Vec::new();
        for (_, &element) in row.iter().enumerate() {
            let q = quantize_lookup(
                api,
                element,
                scaling_factor,
                scaling,
                v_plus_one,
                two_v,
                alpha_two_v,
                is_relu,
                table,
            );
            row_out.push(q);
        }
        out.push(row_out);
    }
    out
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