use expander_compiler::frontend::*;
use ndarray::ArrayD;
use crate::circuit_functions::core_operations::{unconstrained_to_bits, assert_is_bitstring_and_reconstruct, MaxAssertionContext};


pub fn relu_array<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    array: ArrayD<Variable>,
    n_bits: usize,
) -> ArrayD<Variable>{
    let context = MaxAssertionContext::new(api, n_bits - 1);

    let new_array = array.map(|var| relu(api, var, &context));
    new_array
}

// TODO can make use of memorized calls instead, by flattening the array and expanding?
pub fn relu<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Variable,
    context: &MaxAssertionContext,
) -> Variable {
    let shifted_x = api.add(context.offset, x);
    let n_bits = context
        .shift_exponent
        .checked_add(1)
        .expect("shift_exponent + 1 must fit in usize");

    let bits = unconstrained_to_bits(api, shifted_x, n_bits);
    let recon = assert_is_bitstring_and_reconstruct(api, &bits);
    api.assert_is_equal(shifted_x, recon);

    let sign_bit = bits[context.shift_exponent]; // the (s + 1)-st bit d_s
    // let sign_bit = bits[n_bits - 1];
    let out = api.mul(x, sign_bit);
    out
}


// to_binary value, original method using unconstrained api
pub fn to_binary<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    n_bits: usize,
) -> Vec<Variable> {
    let mut res = Vec::new();
    for i in 0..n_bits {
        let y = api.unconstrained_shift_r(x, i as u32);
        res.push(api.unconstrained_bit_and(y, 1));
    }
    res
}
// Convert to integer to be used in twos comp (add 2^k-1 to the value)
fn to_int_for_twos_comp<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    n_bits: usize,
) -> Variable {
    let add_num = api.unconstrained_pow(2, (n_bits - 1) as u32);
    let new_x = api.unconstrained_add(x, add_num);
    return new_x;
}

// Convert to binary using new method. First convert to positive value, and then take relevant bits to binary
fn to_binary_2s<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    n_bits: usize,
) -> Vec<Variable> {
    // The following code to generate new_x is tested and confirmed works
    let new_x = to_int_for_twos_comp(api, x, n_bits);
    let bits = to_binary(api, new_x, n_bits);
    bits
}

//Assert boolean and add bits confirmation to circuit
/// Check that each element in the bitstring, is in fact a binary value
/// Computes the expected field element representation of the bitstring. 
/// The result of this function should be compared against the value prior to the bitstring, to confirm that the bistring representation is correct
pub fn from_binary<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    bits: &Vec<Variable>,
    n_bits: usize,
) -> Variable {
    let mut res = api.constant(0);
    let length = n_bits;
    for i in 0..length {
        let coef = 1 << i;
        api.assert_is_bool(bits[i]);
        let cur = api.mul(coef, bits[i]);
        res = api.add(res, cur);
    }
    res
}
/// 32 bit simple from_binary
fn from_binary_simple_32<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    bits: &Vec<Variable>,
) -> Vec<Variable> {
    vec![from_binary(api, bits, 32)]
}
/// 64 bit simple from_binary
#[allow(dead_code)]
fn from_binary_simple_64<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    bits: &Vec<Variable>,
) -> Vec<Variable> {
    vec![from_binary(api, bits, 64)]
}

/// Relu on vector with using memorized simple call on the from binary component
fn relu_simple_call_32<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Vec<Variable>,
) -> Vec<Variable> {
    let mut out: Vec<Variable> = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let bits = to_binary_2s(api, x[i], 32);
        let total = from_binary(api, &bits, 32);

        let comp = api.add(x[i], 1 << (32 - 1));
        api.assert_is_equal(total, comp);

        out.push(relu_single(api, x[i], bits[31]));
    }
    out
}

/// Perform RELU on a single value
/// Assume 1 is positive and 0 is positive
fn relu_single<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: Variable,
    sign: Variable,
) -> Variable {
    api.mul(x, sign)
}
/// Entire Second version attempt at relu
fn relu_v2<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input: Variable,
    n_bits: usize,
) -> Variable {
    // Iterate over each input/output pair (one per batch)
    // Get the twos compliment binary representation
    let bits = to_binary_2s(api, input, n_bits);

    //Add constraints to ensure that the bitstring is correct
    let total = api.memorized_simple_call(from_binary_simple_32, &bits);

    // Add 2^(k-1)
    let comp = api.add(input, 1 << (n_bits - 1));
    api.assert_is_equal(total[0], comp);

    // // Perform relu using sign bit
    relu_single(api, input, bits[n_bits - 1])
}
// Entire Naive relu attempt
fn relu_naive<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    input: Variable,
    n_bits: usize,
) -> Variable {
    // Iterate over each input/output pair (one per batch)
    // Get the twos compliment binary representation
    let bits = to_binary_2s(api, input, n_bits);

    //Add constraints to ensure that the bitstring is correct
    let total = from_binary_simple_32(api, &bits);

    // Add 2^(k-1)
    let comp = api.add(input, 1 << (n_bits - 1));
    api.assert_is_equal(total[0], comp);

    // // Perform relu using sign bit
    relu_single(api, input, bits[n_bits - 1])
}
/*

            Above is intermediary helper functions
            Below is the full functions


*/

pub fn relu_3d_naive<
    C: Config,
    Builder: RootAPI<C>,
    const X: usize,
    const Y: usize,
    const Z: usize,
>(
    api: &mut Builder,
    input: [[[Variable; Z]; Y]; X],
    n_bits: usize,
) -> Vec<Vec<Vec<Variable>>> {
    let mut out: Vec<Vec<Vec<Variable>>> = Vec::new();

    for i in 0..input.len() {
        let mut out_1: Vec<Vec<Variable>> = Vec::new();
        for j in 0..input[i].len() {
            let mut out_2: Vec<Variable> = Vec::new();
            for k in 0..input[i][j].len() {
                out_2.push(relu_naive(api, input[i][j][k], n_bits));
            }
            out_1.push(out_2);
        }
        out.push(out_1)
    }
    out
}

// memorized_simple call on last dimension of array
pub fn relu_3d_v2<
    C: Config,
    Builder: RootAPI<C>,
    const X: usize,
    const Y: usize,
    const Z: usize,
>(
    api: &mut Builder,
    input: [[[Variable; Z]; Y]; X],
    n_bits: usize,
) -> Vec<Vec<Vec<Variable>>> {
    let mut out: Vec<Vec<Vec<Variable>>> = Vec::new();

    for i in 0..input.len() {
        let mut out_1: Vec<Vec<Variable>> = Vec::new();
        for j in 0..input[i].len() {
            let mut out_2: Vec<Variable> = Vec::new();
            for k in 0..input[i][j].len() {
                out_2.push(relu_v2(api, input[i][j][k], n_bits));
            }
            out_1.push(out_2);
        }
        out.push(out_1)
    }
    out
}

//Appears slightly worse than relu twos v2
pub fn relu_3d_v3<
    C: Config,
    Builder: RootAPI<C>,
    const X: usize,
    const Y: usize,
    const Z: usize,
>(
    api: &mut Builder,
    input: [[[Variable; Z]; Y]; X],
) -> Vec<Vec<Vec<Variable>>> {
    let mut out: Vec<Vec<Vec<Variable>>> = Vec::new();
    for i in 0..input.len() {
        let mut out_1: Vec<Vec<Variable>> = Vec::new();
        for j in 0..input[i].len() {
            out_1.push(api.memorized_simple_call(relu_simple_call_32, &input[i][j].to_vec()));
        }
        out.push(out_1);
    }
    out
}

pub fn relu_1d_naive<
    C: Config,
    Builder: RootAPI<C>,
    const X: usize,
    const Y: usize,
    const Z: usize,
>(
    api: &mut Builder,
    input: Vec<Variable>,
    n_bits: usize,
) -> Vec<Variable> {
    let mut out: Vec<Variable> = Vec::new();
    for i in 0..input.len() {
        out.push(relu_naive(api, input[i], n_bits));
    }
    out
}

// memorized_simple call on last dimension of array
pub fn relu_1d_v2<
    C: Config,
    Builder: RootAPI<C>,
    const X: usize,
    const Y: usize,
    const Z: usize,
>(
    api: &mut Builder,
    input: Vec<Variable>,
    n_bits: usize,
) -> Vec<Variable> {
    let mut out: Vec<Variable> = Vec::new();
    for i in 0..input.len() {
        out.push(relu_v2(api, input[i], n_bits));
    }
    out
}

// memorized_simple call on last dimension of array
pub fn relu_1d_v3<
    C: Config,
    Builder: RootAPI<C>,
    const X: usize,
    const Y: usize,
    const Z: usize,
>(
    api: &mut Builder,
    input: Vec<Variable>,
    n_bits: usize,
) -> Vec<Variable> {
    if n_bits == 32 {
        return api.memorized_simple_call(relu_simple_call_32, &input.to_vec());
    } else if n_bits == 64 as usize {
        panic!("n_bits must be 32");

        // return api.memorized_simple_call(relu_simple_call_64, &input.to_vec());
    } else {
        panic!("n_bits must be 32");
    }
}


pub fn relu_4d_vec_v2<
    C: Config,
    Builder: RootAPI<C>,
>(
    api: &mut Builder,
    input: Vec<Vec<Vec<Vec<Variable>>>>,
    n_bits: usize,
) -> Vec<Vec<Vec<Vec<Variable>>>> {
    let mut out: Vec<Vec<Vec<Vec<Variable>>>> = Vec::new();

    // Iterating over the 4D vector
    for (_, v3) in input.iter().enumerate() {
        let mut out_0: Vec<Vec<Vec<Variable>>> = Vec::new();
        for (_, v2) in v3.iter().enumerate() {
            let mut out_1: Vec<Vec<Variable>> = Vec::new();
            for (_, v1) in v2.iter().enumerate() {
            let mut out_2: Vec<Variable> = Vec::new();
                for (_, value) in v1.iter().enumerate() {
                    out_2.push(relu_v2(api, *value, n_bits));
                }
            out_1.push(out_2);
            }
        out_0.push(out_1);
        }
        out.push(out_0);
    }
    out
}

pub fn relu_2d_vec_v2<
    C: Config,
    Builder: RootAPI<C>,
>(
    api: &mut Builder,
    input: Vec<Vec<Variable>>,
    n_bits: usize,
) -> Vec<Vec<Variable>> {
    let mut out: Vec<Vec<Variable>> = Vec::new();

    // Iterating over the 4D vector
    for (_, v1) in input.iter().enumerate() {
        let mut out_2: Vec<Variable> = Vec::new();
            for (_, value) in v1.iter().enumerate() {
                out_2.push(relu_v2(api, *value, n_bits));
            }
            out.push(out_2);
        }
    out
}