use ark_bls12_381::Fq;
use ark_ff::Field;
use num_bigint::BigInt;

use crate::gnark::element::Element;
use crate::gnark::emparam::FieldParams;
use crate::gnark::emulated::field_bls12381::e2::GE2;
use crate::gnark::limbs::decompose;
use crate::gnark::limbs::recompose;
use crate::sha256::m31::sha256_var_bytes;
use crate::sha256::m31_utils::from_binary;
use ark_bls12_381::Fq2;
use ark_ff::Zero;
use expander_compiler::frontend::{Config, RootAPI, Variable};

pub fn nb_multiplication_res_limbs(len_left: usize, len_right: usize) -> usize {
    let res = len_left + len_right - 1;
    if len_left + len_right < 1 {
        0
    } else {
        res
    }
}

pub fn sub_padding(
    modulus: &BigInt,
    bits_per_limbs: u32,
    overflow: u32,
    nb_limbs: u32,
) -> Vec<BigInt> {
    if modulus == &BigInt::default() {
        panic!("modulus is zero");
    }
    let mut n_limbs = vec![BigInt::default(); nb_limbs as usize];
    for n_limb in &mut n_limbs {
        *n_limb = BigInt::from(1) << (overflow + bits_per_limbs);
    }
    let mut n = recompose(n_limbs.clone(), bits_per_limbs);
    n %= modulus;
    n = modulus - n;
    let mut pad = vec![BigInt::default(); nb_limbs as usize];
    if let Err(err) = decompose(&n, bits_per_limbs, &mut pad) {
        panic!("decompose: {err}");
    }
    let mut new_pad = vec![BigInt::default(); nb_limbs as usize];
    for i in 0..pad.len() {
        new_pad[i] = pad[i].clone() + n_limbs[i].clone();
    }
    new_pad
}

pub fn get_fq_sign(x: &Fq) -> bool {
    let x_bigint = x
        .to_string()
        .parse::<BigInt>()
        .expect("Invalid decimal string");
    !(x_bigint % 2u32).is_zero()
}
pub fn get_fq2_sign(x: &Fq2) -> bool {
    let x_a0 =
        x.c0.to_string()
            .parse::<BigInt>()
            .expect("Invalid decimal string");
    let x_a1 =
        x.c1.to_string()
            .parse::<BigInt>()
            .expect("Invalid decimal string");
    let z = x_a0.is_zero();
    let sgn0 = !(x_a0 % 2u32).is_zero();
    let sgn1 = !(x_a1 % 2u32).is_zero();
    sgn0 | (z & sgn1)
}
pub fn fq_has_sqrt(x: &Fq) -> (Fq, bool) {
    match x.sqrt() {
        Some(sqrt_x) => (sqrt_x, true),
        None => (*x, false),
    }
}
pub fn fq2_has_sqrt(x: &Fq2) -> (Fq2, bool) {
    match x.sqrt() {
        Some(sqrt_x) => (sqrt_x, true),
        None => (*x, false),
    }
}
pub fn xor_variable<C: Config, B: RootAPI<C>>(
    api: &mut B,
    nbits: usize,
    a: Variable,
    b: Variable,
) -> Variable {
    let bits_a = api.to_binary(a, nbits);
    let bits_b = api.to_binary(b, nbits);
    let mut bits_res = vec![Variable::default(); nbits];
    for i in 0..nbits {
        bits_res[i] = api.xor(bits_a[i], bits_b[i]);
    }
    from_binary(api, bits_res)
}
pub fn expand_msg_xmd_variable<C: Config, B: RootAPI<C>>(
    api: &mut B,
    msg: &[Variable],
    dst: &[Variable],
    len_in_bytes: usize,
) -> Vec<Variable> {
    let ell = (len_in_bytes + 31) / 32;
    if ell > 255 {
        panic!("invalid lenInBytes");
    }
    if dst.len() > 255 {
        panic!("invalid domain size (>255 bytes)");
    }
    let size_domain = dst.len() as u8;
    let mut block_v = vec![Variable::default(); 64];
    for v in &mut block_v {
        *v = api.constant(0);
    }
    let mut input = Vec::new();
    input.extend_from_slice(&block_v);
    input.extend_from_slice(msg);
    input.push(api.constant((len_in_bytes >> 8) as u32));
    input.push(api.constant((len_in_bytes % 256) as u32));
    input.push(api.constant(0));
    input.extend_from_slice(dst);
    input.push(api.constant(size_domain as u32));
    let b0 = sha256_var_bytes(api, &input);
    input.clear();
    input.extend_from_slice(&b0);
    input.push(api.constant(1));
    input.extend_from_slice(dst);
    input.push(api.constant(size_domain as u32));
    let mut b1 = sha256_var_bytes(api, &input);
    let mut res = b1.clone();
    for i in 2..=ell {
        let mut strxor = vec![Variable::default(); 32];
        for j in 0..32 {
            strxor[j] = xor_variable(api, 8, b0[j], b1[j]);
        }
        input.clear();
        input.extend_from_slice(&strxor);
        input.push(api.constant(i as u32));
        input.extend_from_slice(dst);
        input.push(api.constant(size_domain as u32));
        b1 = sha256_var_bytes(api, &input);
        res.extend_from_slice(&b1);
    }
    res
}

pub fn hash_to_fp_variable<C: Config, B: RootAPI<C>>(
    api: &mut B,
    msg: &[Variable],
    dst: &[Variable],
    count: usize,
) -> Vec<Vec<Variable>> {
    const FP_BITS: usize = 381;
    let bytes = 1 + (FP_BITS - 1) / 8;
    let l = 16 + bytes;
    let len_in_bytes = count * l;
    let pseudo_random_bytes = expand_msg_xmd_variable(api, msg, dst, len_in_bytes);
    let mut elems = vec![vec![Variable::default(); l]; count];
    for i in 0..count {
        for j in 0..l {
            elems[i][j] = pseudo_random_bytes[i * l + j];
        }
    }
    elems
}

pub fn print_e2<C: Config, B: RootAPI<C>>(native: &mut B, v: &GE2) {
    for i in 0..48 {
        println!(
            "{}: {:?} {:?}",
            i,
            native.display("", v.a0.limbs[i]),
            native.display("", v.a1.limbs[i])
        );
    }
}
pub fn print_element<C: Config, B: RootAPI<C>, T: FieldParams>(native: &mut B, v: &Element<T>) {
    for i in 0..v.limbs.len() {
        print!("{:?} ", native.display("", v.limbs[i]));
    }
    println!(" ");
}
