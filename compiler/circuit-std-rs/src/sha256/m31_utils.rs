use expander_compiler::frontend::{
    declare_circuit, Config, Define, M31Config, RootAPI, Variable, M31,
};

#[cfg(test)]
use expander_compiler::frontend::{compile, CompileOptions, EmptyHintCaller};

use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;

pub fn bytes_to_bits<C: Config, B: RootAPI<C>>(api: &mut B, vals: &[Variable]) -> Vec<Variable> {
    let mut ret = api.to_binary(vals[0], 8);
    for val in vals.iter().skip(1) {
        ret = api
            .to_binary(*val, 8)
            .into_iter()
            .chain(ret.into_iter())
            .collect();
    }
    ret
}
pub fn right_shift<C: Config, B: RootAPI<C>>(
    api: &mut B,
    bits: &[Variable],
    shift: usize,
) -> Vec<Variable> {
    if bits.len() != 32 {
        panic!("RightShift: len(bits) != 32");
    }
    let mut shifted_bits = bits[shift..].to_vec();
    for _ in 0..shift {
        shifted_bits.push(api.constant(0));
    }
    shifted_bits
}
pub fn rotate_right(bits: &[Variable], shift: usize) -> Vec<Variable> {
    if bits.len() != 32 {
        panic!("RotateRight: len(bits) != 32");
    }
    let mut rotated_bits = bits[shift..].to_vec();
    rotated_bits.extend_from_slice(&bits[..shift]);
    rotated_bits
}
pub fn sigma0<C: Config, B: RootAPI<C>>(api: &mut B, bits: &[Variable]) -> Vec<Variable> {
    if bits.len() != 32 {
        panic!("Sigma0: len(bits) != 32");
    }
    let bits1 = bits.to_vec();
    let bits2 = bits.to_vec();
    let bits3 = bits.to_vec();
    let v1 = rotate_right(&bits1, 7);
    let v2 = rotate_right(&bits2, 18);
    let v3 = right_shift(api, &bits3, 3);
    let mut ret = vec![];
    for i in 0..32 {
        let tmp = api.xor(v1[i], v2[i]);
        ret.push(api.xor(tmp, v3[i]));
    }
    ret
}
pub fn sigma1<C: Config, B: RootAPI<C>>(api: &mut B, bits: &[Variable]) -> Vec<Variable> {
    if bits.len() != 32 {
        panic!("Sigma1: len(bits) != 32");
    }
    let bits1 = bits.to_vec();
    let bits2 = bits.to_vec();
    let bits3 = bits.to_vec();
    let v1 = rotate_right(&bits1, 17);
    let v2 = rotate_right(&bits2, 19);
    let v3 = right_shift(api, &bits3, 10);
    let mut ret = vec![];
    for i in 0..32 {
        let tmp = api.xor(v1[i], v2[i]);
        ret.push(api.xor(tmp, v3[i]));
    }
    ret
}
pub fn cap_sigma0<C: Config, B: RootAPI<C>>(api: &mut B, bits: &[Variable]) -> Vec<Variable> {
    if bits.len() != 32 {
        panic!("CapSigma0: len(bits) != 32");
    }
    let bits1 = bits.to_vec();
    let bits2 = bits.to_vec();
    let bits3 = bits.to_vec();
    let v1 = rotate_right(&bits1, 2);
    let v2 = rotate_right(&bits2, 13);
    let v3 = rotate_right(&bits3, 22);
    let mut ret = vec![];
    for i in 0..32 {
        let tmp = api.xor(v1[i], v2[i]);
        ret.push(api.xor(tmp, v3[i]));
    }
    ret
}
pub fn cap_sigma1<C: Config, B: RootAPI<C>>(api: &mut B, bits: &[Variable]) -> Vec<Variable> {
    if bits.len() != 32 {
        panic!("CapSigma1: len(bits) != 32");
    }
    let bits1 = bits.to_vec();
    let bits2 = bits.to_vec();
    let bits3 = bits.to_vec();
    let v1 = rotate_right(&bits1, 6);
    let v2 = rotate_right(&bits2, 11);
    let v3 = rotate_right(&bits3, 25);
    let mut ret = vec![];
    for i in 0..32 {
        let tmp = api.xor(v1[i], v2[i]);
        ret.push(api.xor(tmp, v3[i]));
    }
    ret
}
pub fn ch<C: Config, B: RootAPI<C>>(
    api: &mut B,
    x: &[Variable],
    y: &[Variable],
    z: &[Variable],
) -> Vec<Variable> {
    if x.len() != 32 || y.len() != 32 || z.len() != 32 {
        panic!("Ch: len(x) != 32 || len(y) != 32 || len(z) != 32");
    }
    let mut ret = vec![];
    for i in 0..32 {
        let tmp1 = api.and(x[i], y[i]);
        let tmp2 = api.xor(x[i], 1);
        let tmp3 = api.and(tmp2, z[i]);
        ret.push(api.xor(tmp1, tmp3));
    }
    ret
}
pub fn maj<C: Config, B: RootAPI<C>>(
    api: &mut B,
    x: &[Variable],
    y: &[Variable],
    z: &[Variable],
) -> Vec<Variable> {
    if x.len() != 32 || y.len() != 32 || z.len() != 32 {
        panic!("Maj: len(x) != 32 || len(y) != 32 || len(z) != 32");
    }
    let mut ret = vec![];
    for i in 0..32 {
        let tmp1 = api.and(x[i], y[i]);
        let tmp2 = api.and(x[i], z[i]);
        let tmp3 = api.and(y[i], z[i]);
        let tmp4 = api.xor(tmp1, tmp2);
        ret.push(api.xor(tmp3, tmp4));
    }
    ret
}
pub fn big_array_add_no_reduce<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: &[Variable],
    b: &[Variable],
) -> Vec<Variable> {
    if a.len() != b.len() {
        panic!("BigArrayAdd: length of a and b must be equal");
    }
    let mut c = vec![api.constant(0); a.len()];
    for i in 0..a.len() {
        c[i] = api.add(a[i], b[i]);
    }
    c
}
pub fn big_array_add_reduce<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: &[Variable],
    b: &[Variable],
    nb_bits: usize,
) -> Vec<Variable> {
    if a.len() != b.len() {
        panic!("BigArrayAdd: length of a and b must be equal");
    }
    let mut c = vec![api.constant(0); a.len()];
    let mut carry = api.constant(0);
    for i in 0..a.len() {
        c[i] = api.add(a[i], b[i]);
        c[i] = api.add(c[i], carry);
        carry = api.to_binary(c[i], nb_bits + 1)[nb_bits];
        let tmp = api.mul(carry, 1 << nb_bits);
        c[i] = api.sub(c[i], tmp);
    }
    c
}
pub fn big_array_add<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: &[Variable],
    b: &[Variable],
    nb_bits: usize,
) -> Vec<Variable> {
    if a.len() != b.len() {
        panic!("BigArrayAdd: length of a and b must be equal");
    }
    let mut c = vec![api.constant(0); a.len()];
    let mut carry = api.constant(0);
    for i in 0..(a.len() - 1) {
        c[i] = api.add(a[i], b[i]);
        c[i] = api.add(c[i], carry);
        carry = api.to_binary(c[i], nb_bits + 1)[nb_bits];
        let tmp = api.mul(carry, 1 << nb_bits);
        c[i] = api.sub(c[i], tmp);
    }
    c[a.len() - 1] = api.add(a[a.len() - 1], b[a.len() - 1]);
    c[a.len() - 1] = api.add(c[a.len() - 1], carry);
    c
}
pub fn sha_m31_26_add<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: &[Variable],
    b: &[Variable],
    nb_bits: usize,
) -> [Variable; 2] {
    if a.len() != b.len() || a.len() != 2 {
        panic!("BigArrayAdd: length of a and b must be equal");
    }
    let mut c = [api.constant(0); 2];
    let mut carry = api.constant(0);
    c[0] = api.add(a[0], b[0]);
    c[0] = api.add(c[0], carry);
    carry = api.to_binary(c[0], nb_bits + 1)[nb_bits];
    let tmp = api.mul(carry, 1 << nb_bits);
    c[0] = api.sub(c[0], tmp);
    c[1] = api.add(a[1], b[1]);
    c[1] = api.add(c[1], carry);
    carry = api.to_binary(c[1], 7)[6];
    let tmp = api.mul(carry, 1 << 6);
    c[1] = api.sub(c[1], tmp);
    c
}
pub fn bit_array_to_m31<C: Config, B: RootAPI<C>>(api: &mut B, bits: &[Variable]) -> [Variable; 2] {
    if bits.len() >= 60 {
        panic!("BitArrayToM31: length of bits must be less than 60");
    }
    [
        from_binary(api, bits[..30].to_vec()),
        from_binary(api, bits[30..].to_vec()),
    ]
}

pub fn bit_array_to_m31_26<C: Config, B: RootAPI<C>>(
    api: &mut B,
    bits: &[Variable],
) -> [Variable; 2] {
    if bits.len() >= 54 {
        panic!("BitArrayToM31: length of bits must be less than 60");
    }
    [
        from_binary(api, bits[..26].to_vec()),
        from_binary(api, bits[26..].to_vec()),
    ]
}

pub fn big_endian_m31_array_put_uint32<C: Config, B: RootAPI<C>>(
    api: &mut B,
    b: &mut [Variable],
    x: [Variable; 2],
) {
    let mut quo = x[0];
    for i in (1..=3).rev() {
        let (q, r) = idiv_mod_bit(api, quo, 8);
        b[i] = r;
        quo = q;
    }
    let shift = api.mul(x[1], 1 << 6);
    b[0] = api.add(quo, shift);
}

pub fn m31_26_array_put_uint32<C: Config, B: RootAPI<C>>(
    api: &mut B,
    b: &mut [Variable],
    x: [Variable; 2],
) {
    let mut quo = x[0];
    for i in (1..=3).rev() {
        let (q, r) = idiv_mod_bit(api, quo, 8);
        b[i] = r;
        quo = q;
    }
    let shift = api.mul(x[1], 1 << 2);
    b[0] = api.add(quo, shift);
}
pub fn big_endian_put_uint64<C: Config, B: RootAPI<C>>(
    api: &mut B,
    b: &mut [Variable],
    x: Variable,
) {
    let mut quo = x;
    for i in (1..=7).rev() {
        let (q, r) = idiv_mod_bit(api, quo, 8);
        b[i] = r;
        quo = q;
    }
    b[0] = quo;
}
pub fn m31_to_bit_array<C: Config, B: RootAPI<C>>(api: &mut B, m31: &[Variable]) -> Vec<Variable> {
    let mut bits = vec![];
    for val in m31 {
        bits.extend_from_slice(&api.to_binary(*val, 30));
    }
    bits
}
pub fn m31_26_to_bit_array_seperate<C: Config, B: RootAPI<C>>(
    api: &mut B,
    m31: &[Variable],
    overflow: usize,
) -> Vec<Variable> {
    let mut bits = vec![];
    let bits30 = api.to_binary(m31[0], 30);
    bits.extend_from_slice(&bits30[..26]);
    let carry = from_binary(api, bits30[26..].to_vec());
    let high = api.add(m31[1], carry);
    bits.extend_from_slice(&api.to_binary(high, 6 + overflow + 1));
    bits
}
pub fn from_binary<C: Config, B: RootAPI<C>>(api: &mut B, bits: Vec<Variable>) -> Variable {
    let mut res = api.constant(0);
    for (i, bit) in bits.iter().enumerate() {
        let coef = 1 << i;
        let cur = api.mul(coef, *bit);
        res = api.add(res, cur);
    }
    res
}

pub fn big_is_zero<C: Config, B: RootAPI<C>>(api: &mut B, k: usize, in_: &[Variable]) -> Variable {
    let mut total = api.constant(k as u32);
    for val in in_.iter().take(k) {
        let tmp = api.is_zero(val);
        total = api.sub(total, tmp);
    }
    api.is_zero(total)
}

pub fn bigint_to_m31_array<C: Config, B: RootAPI<C>>(
    api: &mut B,
    x: BigInt,
    n_bits: usize,
    limb_len: usize,
) -> Vec<Variable> {
    let mut res = vec![];
    let mut a = x.clone();
    let mut mask = BigInt::from(1) << n_bits;
    mask -= 1;
    for _ in 0..limb_len {
        let tmp = a.clone() & mask.clone();
        let tmp = api.constant(tmp.to_u32().unwrap());
        res.push(tmp);
        a >>= n_bits;
    }
    res
}
pub fn big_less_than<C: Config, B: RootAPI<C>>(
    api: &mut B,
    n: usize,
    k: usize,
    a: &[Variable],
    b: &[Variable],
) -> Variable {
    let mut lt = vec![];
    let mut eq = vec![];
    for i in 0..k {
        lt.push(my_is_less(api, n, a[i], b[i]));
        let diff = api.sub(a[i], b[i]);
        eq.push(api.is_zero(diff));
    }
    let mut ors = vec![Variable::default(); k - 1];
    let mut ands = vec![Variable::default(); k - 1];
    let mut eq_ands = vec![Variable::default(); k - 1];
    for i in (0..k - 1).rev() {
        if i == k - 2 {
            ands[i] = api.and(eq[k - 1], lt[k - 2]);
            eq_ands[i] = api.and(eq[k - 1], eq[k - 2]);
            ors[i] = api.or(lt[k - 1], ands[k - 2]);
        } else {
            ands[i] = api.and(eq_ands[i + 1], lt[i]);
            eq_ands[i] = api.and(eq_ands[i + 1], eq[i]);
            ors[i] = api.or(ors[i + 1], ands[i]);
        }
    }
    ors[0]
}
pub fn my_is_less<C: Config, B: RootAPI<C>>(
    api: &mut B,
    n: usize,
    a: Variable,
    b: Variable,
) -> Variable {
    let neg_b = api.neg(b);
    let tmp = api.add(a, 1 << n);
    let tmp = api.add(tmp, neg_b);
    let bi1 = api.to_binary(tmp, n + 1);
    let one = api.constant(1);
    api.sub(one, bi1[n])
}

pub fn idiv_mod_bit<C: Config, B: RootAPI<C>>(
    builder: &mut B,
    a: Variable,
    b: u64,
) -> (Variable, Variable) {
    let bits = builder.to_binary(a, 30);
    let quotient = from_binary(builder, bits[b as usize..].to_vec());
    let remainder = from_binary(builder, bits[..b as usize].to_vec());
    (quotient, remainder)
}

pub fn string_to_m31_array(s: &str, nb_bits: u32) -> [M31; 48] {
    let mut big =
        BigInt::parse_bytes(s.as_bytes(), 10).unwrap_or_else(|| panic!("Failed to parse BigInt"));
    let mut res = [M31::from(0); 48];
    let base = BigInt::from(1) << nb_bits;
    for cur_res in &mut res {
        let tmp = &big % &base;
        *cur_res = M31::from(tmp.to_u32().unwrap());
        big >>= nb_bits;
    }
    res
}

declare_circuit!(IDIVMODBITCircuit {
    value: PublicVariable,
    quotient: Variable,
    remainder: Variable,
});

impl Define<M31Config> for IDIVMODBITCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let (quotient, remainder) = idiv_mod_bit(builder, self.value, 8);
        builder.assert_is_equal(quotient, self.quotient);
        builder.assert_is_equal(remainder, self.remainder);
    }
}
#[test]
fn test_idiv_mod_bit() {
    //compile and test
    let compile_result = compile(&IDIVMODBITCircuit::default(), CompileOptions::default()).unwrap();
    let assignment = IDIVMODBITCircuit::<M31> {
        value: M31::from(3845),
        quotient: M31::from(15),
        remainder: M31::from(5),
    };
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, &mut EmptyHintCaller)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}

declare_circuit!(BITCONVERTCircuit {
    big_int: PublicVariable,
    big_int_bytes: [Variable; 8],
    big_int_m31: [Variable; 2],
    big_int_m31_bytes: [Variable; 4],
});

impl Define<M31Config> for BITCONVERTCircuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, builder: &mut Builder) {
        let mut big_int_bytes = [builder.constant(0); 8];
        big_endian_put_uint64(builder, &mut big_int_bytes, self.big_int);
        for (i, big_int_byte) in big_int_bytes.iter().enumerate() {
            builder.assert_is_equal(big_int_byte, self.big_int_bytes[i]);
        }
        let mut big_int_m31 = [builder.constant(0); 4];
        big_endian_m31_array_put_uint32(builder, &mut big_int_m31, self.big_int_m31);
        for (i, val) in big_int_m31.iter().enumerate() {
            builder.assert_is_equal(val, self.big_int_m31_bytes[i]);
        }
    }
}
#[test]
fn test_bit_convert() {
    //compile and test
    let compile_result = compile(&BITCONVERTCircuit::default(), CompileOptions::default()).unwrap();
    let assignment = BITCONVERTCircuit::<M31> {
        big_int: M31::from(3845),
        big_int_bytes: [
            M31::from(0),
            M31::from(0),
            M31::from(0),
            M31::from(0),
            M31::from(0),
            M31::from(0),
            M31::from(15),
            M31::from(5),
        ],
        big_int_m31: [M31::from(3845), M31::from(0)],
        big_int_m31_bytes: [M31::from(0), M31::from(0), M31::from(15), M31::from(5)],
    };
    let witness = compile_result
        .witness_solver
        .solve_witness_with_hints(&assignment, &mut EmptyHintCaller)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}
