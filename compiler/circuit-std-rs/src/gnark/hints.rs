use std::str::FromStr;

use arith::Field as ArithField;
use ark_bls12_381::Fq;
use ark_bls12_381::Fq12;
use ark_bls12_381::Fq2;
use ark_bls12_381::Fq6;
use ark_ff::fields::Field;
use ark_ff::Zero;
use expander_compiler::frontend::{Error, M31};
use num_bigint::BigInt;
use num_bigint::BigUint;
use num_traits::One;
use num_traits::Signed;
use num_traits::ToPrimitive;

use crate::gnark::limbs::{
    bigint_to_m31, decompose, m31_to_bigint, m31_to_bigint_array, recompose,
};
use crate::gnark::utils::{
    fq2_has_sqrt, fq_has_sqrt, get_fq2_sign, get_fq_sign, nb_multiplication_res_limbs,
};

pub fn mul_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    let nb_bits = inputs[0].to_u256().as_usize();
    let nb_limbs = inputs[1].to_u256().as_usize();
    let nb_a_len = inputs[2].to_u256().as_usize();
    let nb_quo_len = inputs[3].to_u256().as_usize();
    let nb_b_len = inputs.len() - 4 - nb_limbs - nb_a_len;
    let mut ptr = 4;
    let plimbs_m31 = &inputs[ptr..ptr + nb_limbs];
    let plimbs_u32: Vec<u32> = (0..nb_limbs)
        .map(|i| plimbs_m31[i].to_u256().as_u32())
        .collect();
    let plimbs: Vec<BigInt> = plimbs_u32.iter().map(|x| BigInt::from(*x)).collect();
    ptr += nb_limbs;
    let alimbs_m31 = &inputs[ptr..ptr + nb_a_len];
    let alimbs_u32: Vec<u32> = (0..nb_a_len)
        .map(|i| alimbs_m31[i].to_u256().as_u32())
        .collect();
    let alimbs: Vec<BigInt> = alimbs_u32.iter().map(|x| BigInt::from(*x)).collect();
    ptr += nb_a_len;
    let blimbs_m31 = &inputs[ptr..ptr + nb_b_len];
    let blimbs_u32: Vec<u32> = (0..nb_b_len)
        .map(|i| blimbs_m31[i].to_u256().as_u32())
        .collect();
    let blimbs: Vec<BigInt> = blimbs_u32.iter().map(|x| BigInt::from(*x)).collect();

    let nb_carry_len = std::cmp::max(
        nb_multiplication_res_limbs(nb_a_len, nb_b_len),
        nb_multiplication_res_limbs(nb_quo_len, nb_limbs),
    ) - 1;

    let p = recompose(plimbs.clone(), nb_bits as u32);
    let a = recompose(alimbs.clone(), nb_bits as u32);
    let b = recompose(blimbs.clone(), nb_bits as u32);

    let ab = a.clone() * b.clone();
    let quo = ab.clone() / p.clone();
    let rem = ab.clone() % p.clone();
    let mut quo_limbs = vec![BigInt::default(); nb_quo_len];
    if let Err(err) = decompose(&quo, nb_bits as u32, &mut quo_limbs) {
        panic!("decompose value: {err}");
    }
    let mut rem_limbs = vec![BigInt::default(); nb_limbs];
    if let Err(err) = decompose(&rem, nb_bits as u32, &mut rem_limbs) {
        panic!("decompose value: {err}");
    }
    let mut xp = vec![BigInt::default(); nb_multiplication_res_limbs(nb_a_len, nb_b_len)];
    let mut yp = vec![BigInt::default(); nb_multiplication_res_limbs(nb_quo_len, nb_limbs)];
    let mut tmp;
    for cur_xp in &mut xp {
        *cur_xp = BigInt::default();
    }
    for cur_yp in &mut yp {
        *cur_yp = BigInt::default();
    }
    // we know compute the schoolbook multiprecision multiplication of a*b and
    // r+k*p
    for i in 0..nb_a_len {
        for j in 0..nb_b_len {
            tmp = alimbs[i].clone();
            tmp *= &blimbs[j];
            xp[i + j] += &tmp;
        }
    }
    for i in 0..nb_limbs {
        yp[i] += &rem_limbs[i];
        for j in 0..nb_quo_len {
            tmp = quo_limbs[j].clone();
            tmp *= &plimbs[i];
            yp[i + j] += &tmp;
        }
    }
    let mut carry = BigInt::default();
    let mut carry_limbs = vec![BigInt::default(); nb_carry_len];
    for i in 0..carry_limbs.len() {
        if i < xp.len() {
            carry += &xp[i];
        }
        if i < yp.len() {
            carry -= &yp[i];
        }
        carry >>= nb_bits as u32;
        //if carry is negative, we need to add 2^nb_bits to it
        carry_limbs[i] = carry.clone();
    }
    //convert limbs to m31 output
    let mut outptr = 0;
    for i in 0..nb_quo_len {
        outputs[outptr + i] = M31::from(quo_limbs[i].to_u64().unwrap() as u32);
    }
    outptr += nb_quo_len;
    for i in 0..nb_limbs {
        outputs[outptr + i] = M31::from(rem_limbs[i].to_u64().unwrap() as u32);
    }
    outptr += nb_limbs;
    for i in 0..nb_carry_len {
        if carry_limbs[i] < BigInt::default() {
            outputs[outptr + i] = -M31::from(carry_limbs[i].abs().to_u64().unwrap() as u32);
        } else {
            outputs[outptr + i] = M31::from(carry_limbs[i].to_u64().unwrap() as u32);
        }
    }
    Ok(())
}
pub fn div_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    let nb_bits = inputs[0].to_u256().as_usize();
    let nb_limbs = inputs[1].to_u256().as_usize();
    let nb_denom_limbs = inputs[2].to_u256().as_usize();
    let nb_nom_limbs = inputs[3].to_u256().as_usize();
    let mut ptr = 4;
    let plimbs_m31 = &inputs[ptr..ptr + nb_limbs];
    let plimbs_u32: Vec<u32> = (0..nb_limbs)
        .map(|i| plimbs_m31[i].to_u256().as_u32())
        .collect();
    let plimbs: Vec<BigInt> = plimbs_u32.iter().map(|x| BigInt::from(*x)).collect();
    ptr += nb_limbs;
    let nomlimbs_m31 = &inputs[ptr..ptr + nb_nom_limbs];
    let nomlimbs_u32: Vec<u32> = (0..nb_nom_limbs)
        .map(|i| nomlimbs_m31[i].to_u256().as_u32())
        .collect();
    let nomlimbs: Vec<BigInt> = nomlimbs_u32.iter().map(|x| BigInt::from(*x)).collect();
    ptr += nb_nom_limbs;
    let denomlimbs_m31 = &inputs[ptr..ptr + nb_denom_limbs];
    let denomlimbs_u32: Vec<u32> = (0..nb_denom_limbs)
        .map(|i| denomlimbs_m31[i].to_u256().as_u32())
        .collect();
    let denomlimbs: Vec<BigInt> = denomlimbs_u32.iter().map(|x| BigInt::from(*x)).collect();

    let p = recompose(plimbs.clone(), nb_bits as u32);
    let nom = recompose(nomlimbs.clone(), nb_bits as u32);
    let denom = recompose(denomlimbs.clone(), nb_bits as u32);
    let mut res = denom.clone().modinv(&p).unwrap();
    res *= &nom;
    res %= &p;
    let mut res_limbs = vec![BigInt::default(); nb_limbs];
    if let Err(err) = decompose(&res, nb_bits as u32, &mut res_limbs) {
        panic!("decompose value: {err}");
    }
    for i in 0..nb_limbs {
        outputs[i] = M31::from(res_limbs[i].to_u64().unwrap() as u32);
    }
    Ok(())
}

pub fn inv_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    let nb_bits = inputs[0].to_u256().as_usize();
    let nb_limbs = inputs[1].to_u256().as_usize();
    let mut ptr = 2;
    let plimbs_m31 = &inputs[ptr..ptr + nb_limbs];
    let plimbs_u32: Vec<u32> = (0..nb_limbs)
        .map(|i| plimbs_m31[i].to_u256().as_u32())
        .collect();
    let plimbs: Vec<BigInt> = plimbs_u32.iter().map(|x| BigInt::from(*x)).collect();
    ptr += nb_limbs;
    let xlimbs_m31 = &inputs[ptr..ptr + nb_limbs];
    let xlimbs_u32: Vec<u32> = (0..nb_limbs)
        .map(|i| xlimbs_m31[i].to_u256().as_u32())
        .collect();
    let xlimbs: Vec<BigInt> = xlimbs_u32.iter().map(|x| BigInt::from(*x)).collect();

    let p = recompose(plimbs.clone(), nb_bits as u32);
    let x = recompose(xlimbs.clone(), nb_bits as u32);
    let res = x.clone().modinv(&p).unwrap();
    let mut res_limbs = vec![BigInt::default(); nb_limbs];
    if let Err(err) = decompose(&res, nb_bits as u32, &mut res_limbs) {
        panic!("decompose value: {err}");
    }
    for i in 0..nb_limbs {
        outputs[i] = M31::from(res_limbs[i].to_u64().unwrap() as u32);
    }
    Ok(())
}
pub fn div_e2_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //divE2Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let b = Fq2::new(
                Fq::from(biguint_inputs[2].clone()),
                Fq::from(biguint_inputs[3].clone()),
            );
            let c = a / b;
            let c0_bigint =
                c.c0.to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c1_bigint =
                c.c1.to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            vec![c0_bigint, c1_bigint]
        },
    ) {
        panic!("divE2Hint: {err}");
    }
    Ok(())
}

pub fn inverse_e2_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //inverseE2Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let c = a.inverse().unwrap();
            let c0_bigint =
                c.c0.to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c1_bigint =
                c.c1.to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            vec![c0_bigint, c1_bigint]
        },
    ) {
        panic!("inverseE2Hint: {err}");
    }
    Ok(())
}

pub fn div_e6_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //divE6Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a_b0 = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let a_b1 = Fq2::new(
                Fq::from(biguint_inputs[2].clone()),
                Fq::from(biguint_inputs[3].clone()),
            );
            let a_b2 = Fq2::new(
                Fq::from(biguint_inputs[4].clone()),
                Fq::from(biguint_inputs[5].clone()),
            );
            let a = Fq6::new(a_b0, a_b1, a_b2);
            let b_b0 = Fq2::new(
                Fq::from(biguint_inputs[6].clone()),
                Fq::from(biguint_inputs[7].clone()),
            );
            let b_b1 = Fq2::new(
                Fq::from(biguint_inputs[8].clone()),
                Fq::from(biguint_inputs[9].clone()),
            );
            let b_b2 = Fq2::new(
                Fq::from(biguint_inputs[10].clone()),
                Fq::from(biguint_inputs[11].clone()),
            );
            let b = Fq6::new(b_b0, b_b1, b_b2);
            let c = a / b;
            let c_c0_c0_bigint =
                c.c0.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_c1_bigint =
                c.c0.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_c0_bigint =
                c.c1.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_c1_bigint =
                c.c1.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c2_c0_bigint =
                c.c2.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c2_c1_bigint =
                c.c2.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");

            vec![
                c_c0_c0_bigint,
                c_c0_c1_bigint,
                c_c1_c0_bigint,
                c_c1_c1_bigint,
                c_c2_c0_bigint,
                c_c2_c1_bigint,
            ]
        },
    ) {
        panic!("divE6Hint: {err}");
    }
    Ok(())
}

pub fn inverse_e6_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //inverseE6Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a_b0 = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let a_b1 = Fq2::new(
                Fq::from(biguint_inputs[2].clone()),
                Fq::from(biguint_inputs[3].clone()),
            );
            let a_b2 = Fq2::new(
                Fq::from(biguint_inputs[4].clone()),
                Fq::from(biguint_inputs[5].clone()),
            );
            let a = Fq6::new(a_b0, a_b1, a_b2);
            let c = a.inverse().unwrap();
            let c_c0_c0_bigint =
                c.c0.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_c1_bigint =
                c.c0.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_c0_bigint =
                c.c1.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_c1_bigint =
                c.c1.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c2_c0_bigint =
                c.c2.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c2_c1_bigint =
                c.c2.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            vec![
                c_c0_c0_bigint,
                c_c0_c1_bigint,
                c_c1_c0_bigint,
                c_c1_c1_bigint,
                c_c2_c0_bigint,
                c_c2_c1_bigint,
            ]
        },
    ) {
        panic!("inverseE6Hint: {err}");
    }
    Ok(())
}

pub fn div_e6_by_6_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //divE6By6Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a_b0 = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let a_b1 = Fq2::new(
                Fq::from(biguint_inputs[2].clone()),
                Fq::from(biguint_inputs[3].clone()),
            );
            let a_b2 = Fq2::new(
                Fq::from(biguint_inputs[4].clone()),
                Fq::from(biguint_inputs[5].clone()),
            );
            let mut a = Fq6::new(a_b0, a_b1, a_b2);
            let six_inv = Fq::from(6u32).inverse().unwrap();
            a.c0.mul_assign_by_fp(&six_inv);
            a.c1.mul_assign_by_fp(&six_inv);
            a.c2.mul_assign_by_fp(&six_inv);
            let c_c0_c0_bigint =
                a.c0.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_c1_bigint =
                a.c0.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_c0_bigint =
                a.c1.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_c1_bigint =
                a.c1.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c2_c0_bigint =
                a.c2.c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c2_c1_bigint =
                a.c2.c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            vec![
                c_c0_c0_bigint,
                c_c0_c1_bigint,
                c_c1_c0_bigint,
                c_c1_c1_bigint,
                c_c2_c0_bigint,
                c_c2_c1_bigint,
            ]
        },
    ) {
        panic!("divE6By6Hint: {err}");
    }
    Ok(())
}

pub fn div_e12_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //divE12Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();

            let a_c0_b0 = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let a_c0_b1 = Fq2::new(
                Fq::from(biguint_inputs[2].clone()),
                Fq::from(biguint_inputs[3].clone()),
            );
            let a_c0_b2 = Fq2::new(
                Fq::from(biguint_inputs[4].clone()),
                Fq::from(biguint_inputs[5].clone()),
            );
            let a_c0 = Fq6::new(a_c0_b0, a_c0_b1, a_c0_b2);
            let a_c1_b0 = Fq2::new(
                Fq::from(biguint_inputs[6].clone()),
                Fq::from(biguint_inputs[7].clone()),
            );
            let a_c1_b1 = Fq2::new(
                Fq::from(biguint_inputs[8].clone()),
                Fq::from(biguint_inputs[9].clone()),
            );
            let a_c1_b2 = Fq2::new(
                Fq::from(biguint_inputs[10].clone()),
                Fq::from(biguint_inputs[11].clone()),
            );
            let a_c1 = Fq6::new(a_c1_b0, a_c1_b1, a_c1_b2);
            let a = Fq12::new(a_c0, a_c1);

            let b_c0_b0 = Fq2::new(
                Fq::from(biguint_inputs[12].clone()),
                Fq::from(biguint_inputs[13].clone()),
            );
            let b_c0_b1 = Fq2::new(
                Fq::from(biguint_inputs[14].clone()),
                Fq::from(biguint_inputs[15].clone()),
            );
            let b_c0_b2 = Fq2::new(
                Fq::from(biguint_inputs[16].clone()),
                Fq::from(biguint_inputs[17].clone()),
            );
            let b_c0 = Fq6::new(b_c0_b0, b_c0_b1, b_c0_b2);
            let b_c1_b0 = Fq2::new(
                Fq::from(biguint_inputs[18].clone()),
                Fq::from(biguint_inputs[19].clone()),
            );
            let b_c1_b1 = Fq2::new(
                Fq::from(biguint_inputs[20].clone()),
                Fq::from(biguint_inputs[21].clone()),
            );
            let b_c1_b2 = Fq2::new(
                Fq::from(biguint_inputs[22].clone()),
                Fq::from(biguint_inputs[23].clone()),
            );
            let b_c1 = Fq6::new(b_c1_b0, b_c1_b1, b_c1_b2);
            let b = Fq12::new(b_c0, b_c1);

            let c = a / b;
            let c_c0_b0_a0_bigint =
                c.c0.c0
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b0_a1_bigint =
                c.c0.c0
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b1_a0_bigint =
                c.c0.c1
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b1_a1_bigint =
                c.c0.c1
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b2_a0_bigint =
                c.c0.c2
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b2_a1_bigint =
                c.c0.c2
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b0_a0_bigint =
                c.c1.c0
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b0_a1_bigint =
                c.c1.c0
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b1_a0_bigint =
                c.c1.c1
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b1_a1_bigint =
                c.c1.c1
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b2_a0_bigint =
                c.c1.c2
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b2_a1_bigint =
                c.c1.c2
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");

            vec![
                c_c0_b0_a0_bigint,
                c_c0_b0_a1_bigint,
                c_c0_b1_a0_bigint,
                c_c0_b1_a1_bigint,
                c_c0_b2_a0_bigint,
                c_c0_b2_a1_bigint,
                c_c1_b0_a0_bigint,
                c_c1_b0_a1_bigint,
                c_c1_b1_a0_bigint,
                c_c1_b1_a1_bigint,
                c_c1_b2_a0_bigint,
                c_c1_b2_a1_bigint,
            ]
        },
    ) {
        panic!("divE12Hint: {err}");
    }
    Ok(())
}

pub fn inverse_e12_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //inverseE12Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();

            let a_c0_b0 = Fq2::new(
                Fq::from(biguint_inputs[0].clone()),
                Fq::from(biguint_inputs[1].clone()),
            );
            let a_c0_b1 = Fq2::new(
                Fq::from(biguint_inputs[2].clone()),
                Fq::from(biguint_inputs[3].clone()),
            );
            let a_c0_b2 = Fq2::new(
                Fq::from(biguint_inputs[4].clone()),
                Fq::from(biguint_inputs[5].clone()),
            );
            let a_c0 = Fq6::new(a_c0_b0, a_c0_b1, a_c0_b2);
            let a_c1_b0 = Fq2::new(
                Fq::from(biguint_inputs[6].clone()),
                Fq::from(biguint_inputs[7].clone()),
            );
            let a_c1_b1 = Fq2::new(
                Fq::from(biguint_inputs[8].clone()),
                Fq::from(biguint_inputs[9].clone()),
            );
            let a_c1_b2 = Fq2::new(
                Fq::from(biguint_inputs[10].clone()),
                Fq::from(biguint_inputs[11].clone()),
            );
            let a_c1 = Fq6::new(a_c1_b0, a_c1_b1, a_c1_b2);
            let a = Fq12::new(a_c0, a_c1);

            let c = a.inverse().unwrap();
            let c_c0_b0_a0_bigint =
                c.c0.c0
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b0_a1_bigint =
                c.c0.c0
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b1_a0_bigint =
                c.c0.c1
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b1_a1_bigint =
                c.c0.c1
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b2_a0_bigint =
                c.c0.c2
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c0_b2_a1_bigint =
                c.c0.c2
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b0_a0_bigint =
                c.c1.c0
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b0_a1_bigint =
                c.c1.c0
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b1_a0_bigint =
                c.c1.c1
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b1_a1_bigint =
                c.c1.c1
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b2_a0_bigint =
                c.c1.c2
                    .c0
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let c_c1_b2_a1_bigint =
                c.c1.c2
                    .c1
                    .to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");

            vec![
                c_c0_b0_a0_bigint,
                c_c0_b0_a1_bigint,
                c_c0_b1_a0_bigint,
                c_c0_b1_a1_bigint,
                c_c0_b2_a0_bigint,
                c_c0_b2_a1_bigint,
                c_c1_b0_a0_bigint,
                c_c1_b0_a1_bigint,
                c_c1_b1_a0_bigint,
                c_c1_b1_a1_bigint,
                c_c1_b2_a0_bigint,
                c_c1_b2_a1_bigint,
            ]
        },
    ) {
        panic!("inverseE12Hint: {err}");
    }
    Ok(())
}
pub fn copy_vars_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    outputs.copy_from_slice(&inputs[..outputs.len()]);
    Ok(())
}
pub fn copy_element_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //copyE2Hint
        |inputs| inputs,
    ) {
        panic!("copyElementHint: {err}");
    }
    Ok(())
}
pub fn copy_e2_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //copyE2Hint
        |inputs| inputs,
    ) {
        panic!("copyE2Hint: {err}");
    }
    Ok(())
}
pub fn get_element_sqrt_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //getElementSqrtHint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a = Fq::from(biguint_inputs[0].clone());
            let (sqrt, is_square) = fq_has_sqrt(&a);
            let sqrt_bigint = sqrt
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            vec![BigInt::from(is_square), sqrt_bigint]
        },
    ) {
        panic!("getElementSqrtHint: {err}");
    }
    Ok(())
}
pub fn get_e2_sqrt_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //getElementSqrtHint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let a0 = Fq::from(biguint_inputs[0].clone());
            let a1 = Fq::from(biguint_inputs[1].clone());
            let a = Fq2::new(a0, a1);
            let (sqrt, is_square) = fq2_has_sqrt(&a);
            let sqrt0_bigint = sqrt
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let sqrt1_bigint = sqrt
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            vec![BigInt::from(is_square), sqrt0_bigint, sqrt1_bigint]
        },
    ) {
        panic!("getElementSqrtHint: {err}");
    }
    Ok(())
}
pub fn get_sqrt_x0x1_fq_new_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //divE12Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();

            let g_x0 = Fq::from(biguint_inputs[0].clone());
            let g_x1 = Fq::from(biguint_inputs[1].clone());
            let t = Fq::from(biguint_inputs[2].clone());
            let sgn_t = get_fq_sign(&t);
            let (g_x0_sqrt, is_square0) = fq_has_sqrt(&g_x0);
            let (g_x1_sqrt, is_square1) = fq_has_sqrt(&g_x1);
            let mut y;
            if is_square0 {
                y = g_x0_sqrt;
            } else if is_square1 {
                y = g_x1_sqrt;
            } else {
                panic!("At least one should be square");
            }
            let sgn_y = get_fq_sign(&y);
            if sgn_y != sgn_t {
                y = -y;
            }
            let y_bigint = y
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            vec![BigInt::from(is_square0), y_bigint]
        },
    ) {
        panic!("divE2Hint: {err}");
    }
    Ok(())
}
pub fn get_sqrt_x0x1_fq2_new_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //divE12Hint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();

            let g_x0_a0 = Fq::from(biguint_inputs[0].clone());
            let g_x0_a1 = Fq::from(biguint_inputs[1].clone());
            let g_x1_a0 = Fq::from(biguint_inputs[2].clone());
            let g_x1_a1 = Fq::from(biguint_inputs[3].clone());
            let t_a0 = Fq::from(biguint_inputs[4].clone());
            let t_a1 = Fq::from(biguint_inputs[5].clone());

            let g_x0 = Fq2::new(Fq::from(g_x0_a0), Fq::from(g_x0_a1));
            let g_x1 = Fq2::new(Fq::from(g_x1_a0), Fq::from(g_x1_a1));
            let t = Fq2::new(Fq::from(t_a0), Fq::from(t_a1));
            let sgn_t = get_fq2_sign(&t);
            let (g_x0_sqrt, is_square0) = fq2_has_sqrt(&g_x0);
            let (g_x1_sqrt, is_square1) = fq2_has_sqrt(&g_x1);
            let mut y;
            if is_square0 {
                y = g_x0_sqrt;
            } else if is_square1 {
                y = g_x1_sqrt;
            } else {
                panic!("At least one should be square");
            }
            let sgn_y = get_fq2_sign(&y);
            if sgn_y != sgn_t {
                y.c0 = -y.c0;
                y.c1 = -y.c1;
            }
            let y0_c0_bigint =
                y.c0.to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            let y0_c1_bigint =
                y.c1.to_string()
                    .parse::<BigInt>()
                    .expect("Invalid decimal string");
            vec![BigInt::from(is_square0), y0_c0_bigint, y0_c1_bigint]
        },
    ) {
        panic!("divE2Hint: {err}");
    }
    Ok(())
}
pub fn copy_e12_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //copyE12Hint
        |inputs| inputs,
    ) {
        panic!("copyE12Hint: {err}");
    }
    Ok(())
}
pub fn final_exp_hint(inputs: &[M31], outputs: &mut [M31]) -> Result<(), Error> {
    if let Err(err) = unwrap_hint(
        true,
        true,
        inputs,
        outputs,
        //finalExpHint
        |inputs| {
            let biguint_inputs = inputs
                .iter()
                .map(|x| x.to_biguint().unwrap())
                .collect::<Vec<_>>();
            let mut miller_loop = Fq12::default();
            miller_loop.c0.c0.c0 = Fq::from(biguint_inputs[0].clone());
            miller_loop.c0.c0.c1 = Fq::from(biguint_inputs[1].clone());
            miller_loop.c0.c1.c0 = Fq::from(biguint_inputs[2].clone());
            miller_loop.c0.c1.c1 = Fq::from(biguint_inputs[3].clone());
            miller_loop.c0.c2.c0 = Fq::from(biguint_inputs[4].clone());
            miller_loop.c0.c2.c1 = Fq::from(biguint_inputs[5].clone());
            miller_loop.c1.c0.c0 = Fq::from(biguint_inputs[6].clone());
            miller_loop.c1.c0.c1 = Fq::from(biguint_inputs[7].clone());
            miller_loop.c1.c1.c0 = Fq::from(biguint_inputs[8].clone());
            miller_loop.c1.c1.c1 = Fq::from(biguint_inputs[9].clone());
            miller_loop.c1.c2.c0 = Fq::from(biguint_inputs[10].clone());
            miller_loop.c1.c2.c1 = Fq::from(biguint_inputs[11].clone());

            let mut root_pth_inverse = Fq12::default();
            let mut root_27th_inverse = Fq12::default();
            let order3rd;
            let mut order3rd_power = BigInt::default();
            let mut exponent: BigInt;
            let mut exponent_inv;
            let poly_factor =
                BigInt::from_str("5044125407647214251").expect("Invalid string for BigInt");
            let final_exp_factor= BigInt::from_str("2366356426548243601069753987687709088104621721678962410379583120840019275952471579477684846670499039076873213559162845121989217658133790336552276567078487633052653005423051750848782286407340332979263075575489766963251914185767058009683318020965829271737924625612375201545022326908440428522712877494557944965298566001441468676802477524234094954960009227631543471415676620753242466901942121887152806837594306028649150255258504417829961387165043999299071444887652375514277477719817175923289019181393803729926249507024121957184340179467502106891835144220611408665090353102353194448552304429530104218473070114105759487413726485729058069746063140422361472585604626055492939586602274983146215294625774144156395553405525711143696689756441298365274341189385646499074862712688473936093315628166094221735056483459332831845007196600723053356837526749543765815988577005929923802636375670820616189737737304893769679803809426304143627363860243558537831172903494450556755190448279875942974830469855835666815454271389438587399739607656399812689280234103023464545891697941661992848552456326290792224091557256350095392859243101357349751064730561345062266850238821755009430903520645523345000326783803935359711318798844368754833295302563158150573540616830138810935344206231367357992991289265295323280").expect("Invalid string for BigInt");
            exponent = &final_exp_factor * 27;
            let exp_uint = exponent.to_biguint().unwrap();
            let root = miller_loop.pow(exp_uint.to_u64_digits().iter());
            if root.is_one() {
                root_pth_inverse.set_one();
            } else {
                exponent_inv = exponent.clone().modinv(&poly_factor).unwrap();
                if exponent_inv.abs() > poly_factor {
                    exponent_inv %= &poly_factor;
                }
                exponent = &poly_factor - exponent_inv;
                exponent %= &poly_factor;
                let exp_uint = exponent.to_biguint().unwrap();
                root_pth_inverse = root.pow(exp_uint.to_u64_digits().iter());
            }

            let three = BigUint::from(3u32);
            exponent = &poly_factor * &final_exp_factor;
            let exp_uint = exponent.to_biguint().unwrap();
            let mut root = miller_loop.pow(exp_uint.to_u64_digits().iter());
            if root.is_one() {
                order3rd_power = BigInt::from(0u32);
            }
            root = root.pow(three.to_u64_digits().iter());
            if root.is_one() {
                order3rd_power = BigInt::from(1u32);
            }
            root = root.pow(three.to_u64_digits().iter());
            if root.is_one() {
                order3rd_power = BigInt::from(2u32);
            }
            root = root.pow(three.to_u64_digits().iter());
            if root.is_one() {
                order3rd_power = BigInt::from(3u32);
            }

            if order3rd_power.is_zero() {
                root_27th_inverse.set_one();
            } else {
                let three_bigint = BigInt::from(3u32);
                order3rd = three_bigint.pow(order3rd_power.to_u32().unwrap());
                exponent = &poly_factor * &final_exp_factor;
                let exp_uint = exponent.to_biguint().unwrap();
                root = miller_loop.pow(exp_uint.to_u64_digits().iter());
                exponent_inv = exponent.modinv(&order3rd).unwrap();
                if exponent_inv.abs() > order3rd {
                    exponent_inv %= &order3rd;
                }
                exponent = &order3rd - exponent_inv;
                exponent %= &order3rd;
                let exp_uint = exponent.to_biguint().unwrap();
                root_27th_inverse = root.pow(exp_uint.to_u64_digits().iter());
            }

            let scaling_factor = root_pth_inverse * root_27th_inverse;
            miller_loop *= scaling_factor;

            let lambda= BigInt::from_str("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129030796414117214202539").expect("Invalid string for BigInt");
            exponent = lambda.modinv(&final_exp_factor).unwrap();
            let residue_witness =
                miller_loop.pow(exponent.to_biguint().unwrap().to_u64_digits().iter());

            let res_c0_b0_a0_bigint = residue_witness
                .c0
                .c0
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c0_b0_a1_bigint = residue_witness
                .c0
                .c0
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c0_b1_a0_bigint = residue_witness
                .c0
                .c1
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c0_b1_a1_bigint = residue_witness
                .c0
                .c1
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c0_b2_a0_bigint = residue_witness
                .c0
                .c2
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c0_b2_a1_bigint = residue_witness
                .c0
                .c2
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c1_b0_a0_bigint = residue_witness
                .c1
                .c0
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c1_b0_a1_bigint = residue_witness
                .c1
                .c0
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c1_b1_a0_bigint = residue_witness
                .c1
                .c1
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c1_b1_a1_bigint = residue_witness
                .c1
                .c1
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c1_b2_a0_bigint = residue_witness
                .c1
                .c2
                .c0
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");
            let res_c1_b2_a1_bigint = residue_witness
                .c1
                .c2
                .c1
                .to_string()
                .parse::<BigInt>()
                .expect("Invalid decimal string");

            let sca_c0_b0_a0_bigint = scaling_factor
                .c0
                .c0
                .c0
                .to_string()
                .parse::<BigInt>()
                .unwrap_or_else(|_| BigInt::zero());
            let sca_c0_b0_a1_bigint = scaling_factor
                .c0
                .c0
                .c1
                .to_string()
                .parse::<BigInt>()
                .unwrap_or_else(|_| BigInt::zero());
            let sca_c0_b1_a0_bigint = scaling_factor
                .c0
                .c1
                .c0
                .to_string()
                .parse::<BigInt>()
                .unwrap_or_else(|_| BigInt::zero());
            let sca_c0_b1_a1_bigint = scaling_factor
                .c0
                .c1
                .c1
                .to_string()
                .parse::<BigInt>()
                .unwrap_or_else(|_| BigInt::zero());
            let sca_c0_b2_a0_bigint = scaling_factor
                .c0
                .c2
                .c0
                .to_string()
                .parse::<BigInt>()
                .unwrap_or_else(|_| BigInt::zero());
            let sca_c0_b2_a1_bigint = scaling_factor
                .c0
                .c2
                .c1
                .to_string()
                .parse::<BigInt>()
                .unwrap_or_else(|_| BigInt::zero());

            vec![
                res_c0_b0_a0_bigint,
                res_c0_b0_a1_bigint,
                res_c0_b1_a0_bigint,
                res_c0_b1_a1_bigint,
                res_c0_b2_a0_bigint,
                res_c0_b2_a1_bigint,
                res_c1_b0_a0_bigint,
                res_c1_b0_a1_bigint,
                res_c1_b1_a0_bigint,
                res_c1_b1_a1_bigint,
                res_c1_b2_a0_bigint,
                res_c1_b2_a1_bigint,
                sca_c0_b0_a0_bigint,
                sca_c0_b0_a1_bigint,
                sca_c0_b1_a0_bigint,
                sca_c0_b1_a1_bigint,
                sca_c0_b2_a0_bigint,
                sca_c0_b2_a1_bigint,
            ]
        },
    ) {
        panic!("inverseE12Hint: {err}");
    }
    Ok(())
}

pub fn simple_rangecheck_hint(inputs: &[M31], _outputs: &mut [M31]) -> Result<(), Error> {
    let nb_bits = inputs[0].to_u256().as_u32();
    let number = inputs[1].to_u256().as_f64();
    let number_bit = if number > 1.0 {
        number.log2().ceil() as u32
    } else {
        1
    };
    if number_bit > nb_bits {
        panic!("number is out of range");
    }

    Ok(())
}

pub fn unwrap_hint(
    is_emulated_input: bool,
    is_emulated_output: bool,
    native_inputs: &[M31],
    native_outputs: &mut [M31],
    nonnative_hint: fn(Vec<BigInt>) -> Vec<BigInt>,
) -> Result<(), String> {
    if native_inputs.len() < 2 {
        return Err("hint wrapper header is 2 elements".to_string());
    }
    let i64_max = 1 << 63;
    if native_inputs[0].to_u256() >= i64_max || native_inputs[1].to_u256() >= i64_max {
        return Err("header must be castable to int64".to_string());
    }
    let nb_bits = native_inputs[0].to_u256().as_u32();
    let nb_limbs = native_inputs[1].to_u256().as_usize();
    if native_inputs.len() < 2 + nb_limbs {
        return Err("hint wrapper header is 2+nbLimbs elements".to_string());
    }
    let nonnative_mod_limbs =
        m31_to_bigint_array(native_inputs[2..2 + nb_limbs].to_vec().as_slice());
    let nonnative_mod = recompose(nonnative_mod_limbs, nb_bits);
    let mut nonnative_inputs;
    if is_emulated_input {
        if native_inputs[2 + nb_limbs].to_u256() >= i64_max {
            return Err("number of nonnative elements must be castable to int64".to_string());
        }
        let nb_inputs = native_inputs[2 + nb_limbs].to_u256().as_usize();
        let mut read_ptr = 3 + nb_limbs;
        nonnative_inputs = vec![BigInt::default(); nb_inputs];
        for (i, nonnative_input) in nonnative_inputs.iter_mut().enumerate().take(nb_inputs) {
            if native_inputs.len() < read_ptr + 1 {
                return Err(format!("can not read {i}-th native input"));
            }
            if native_inputs[read_ptr].to_u256() >= i64_max {
                return Err(format!("corrupted {i}-th native input"));
            }
            let current_input_len = native_inputs[read_ptr].to_u256().as_usize();
            if native_inputs.len() < read_ptr + 1 + current_input_len {
                return Err(format!("cannot read {i}-th nonnative element"));
            }
            let tmp_inputs = m31_to_bigint_array(
                native_inputs[read_ptr + 1..read_ptr + 1 + current_input_len]
                    .to_vec()
                    .as_slice(),
            );
            *nonnative_input = recompose(tmp_inputs, nb_bits);
            read_ptr += 1 + current_input_len;
        }
    } else {
        let nb_inputs = native_inputs[2 + nb_limbs..].len();
        let read_ptr = 2 + nb_limbs;
        nonnative_inputs = vec![BigInt::default(); nb_inputs];
        for i in 0..nb_inputs {
            nonnative_inputs[i] = m31_to_bigint(native_inputs[read_ptr + i]);
        }
    }
    let nonnative_outputs = nonnative_hint(nonnative_inputs);
    let mut tmp_outputs = vec![BigInt::default(); nb_limbs * nonnative_outputs.len()];
    if is_emulated_output {
        if native_outputs.len() % nb_limbs != 0 {
            return Err("output count doesn't divide limb count".to_string());
        }
        for i in 0..nonnative_outputs.len() {
            let mod_output = &nonnative_outputs[i] % &nonnative_mod;
            if let Err(e) = decompose(
                &mod_output,
                nb_bits,
                &mut tmp_outputs[i * nb_limbs..(i + 1) * nb_limbs],
            ) {
                return Err(format!("decompose {i}-th element: {e}"));
            }
        }
    } else {
        tmp_outputs[..nonnative_outputs.len()].clone_from_slice(&nonnative_outputs[..]);
    }
    for i in 0..tmp_outputs.len() {
        native_outputs[i] = bigint_to_m31(&tmp_outputs[i]);
    }
    Ok(())
}
