use arith::Field;
use expander_compiler::frontend::M31;
use num_bigint::BigInt;
use num_traits::ToPrimitive;

pub fn recompose(inputs: Vec<BigInt>, nb_bits: u32) -> BigInt {
    if inputs.is_empty() {
        panic!("zero length slice input");
    }
    let mut res = BigInt::from(0u32);
    for i in 0..inputs.len() {
        res <<= nb_bits;
        res += &inputs[inputs.len() - i - 1];
    }
    res
}
pub fn decompose(input: &BigInt, nb_bits: u32, res: &mut [BigInt]) -> Result<(), String> {
    // limb modulus
    if input.bits() > res.len() as u64 * nb_bits as u64 {
        return Err("decomposed integer does not fit into res".to_string());
    }
    let base = BigInt::from(1u32) << nb_bits;
    let mut tmp = input.clone();
    for cur_res in res {
        *cur_res = &tmp % &base;
        tmp >>= nb_bits;
    }
    Ok(())
}

pub fn m31_to_bigint(input: M31) -> BigInt {
    BigInt::from(input.to_u256().as_u32())
}

pub fn bigint_to_m31(input: &BigInt) -> M31 {
    M31::from(input.to_u32().unwrap())
}

pub fn m31_to_bigint_array(input: &[M31]) -> Vec<BigInt> {
    input.iter().map(|x| m31_to_bigint(*x)).collect()
}
