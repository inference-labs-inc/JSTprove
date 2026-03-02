use expander_compiler::frontend::{Config, RootAPI, Variable};

pub type Sha256Word = [Variable; 32];

// parse the u32 into 32 bits, big-endian
pub fn u32_to_bit<C: Config, Builder: RootAPI<C>>(api: &mut Builder, value: u32) -> [Variable; 32] {
    (0..32)
        .map(|i| api.constant((value >> (31 - i)) & 1))
        .collect::<Vec<Variable>>()
        .try_into()
        .expect("Iterator should have exactly 32 elements")
}

pub fn u64_to_bit<C: Config, Builder: RootAPI<C>>(api: &mut Builder, value: u64) -> [Variable; 64] {
    (0..64)
        .map(|i| api.constant(((value >> (63 - i)) & 1) as u32))
        .collect::<Vec<Variable>>()
        .try_into()
        .expect("Iterator should have exactly 64 elements")
}

pub fn rotate_right(bits: &Sha256Word, k: usize) -> Sha256Word {
    assert!(bits.len() & (bits.len() - 1) == 0);
    let n = bits.len();
    let s = n - k;
    let mut new_bits = bits[s..].to_vec();
    new_bits.append(&mut bits[0..s].to_vec());
    new_bits.try_into().unwrap()
}

pub fn shift_right<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    bits: &Sha256Word,
    k: usize,
) -> Sha256Word {
    assert!(bits.len() & (bits.len() - 1) == 0);
    let n = bits.len();
    let s = n - k;
    let mut new_bits = vec![api.constant(0); k];
    new_bits.append(&mut bits[0..s].to_vec());
    new_bits.try_into().unwrap()
}

// Ch function: (x AND y) XOR (NOT x AND z)
pub fn ch<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Sha256Word,
    y: &Sha256Word,
    z: &Sha256Word,
) -> Sha256Word {
    let xy = and(api, x, y);
    let not_x = not(api, x);
    let not_xz = and(api, &not_x, z);

    xor(api, &xy, &not_xz)
}

// Maj function: (x AND y) XOR (x AND z) XOR (y AND z)
pub fn maj<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Sha256Word,
    y: &Sha256Word,
    z: &Sha256Word,
) -> Sha256Word {
    let xy = and(api, x, y);
    let xz = and(api, x, z);
    let yz = and(api, y, z);
    let tmp = xor(api, &xy, &xz);

    xor(api, &tmp, &yz)
}

// sigma0 function: ROTR(x, 7) XOR ROTR(x, 18) XOR SHR(x, 3)
pub fn lower_case_sigma0<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    word: &Sha256Word,
) -> Sha256Word {
    let rot7 = rotate_right(word, 7);
    let rot18 = rotate_right(word, 18);
    let shft3 = shift_right(api, word, 3);
    let tmp = xor(api, &rot7, &rot18);

    xor(api, &tmp, &shft3)
}

pub fn lower_case_sigma1<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    word: &Sha256Word,
) -> Sha256Word {
    let rot17 = rotate_right(word, 17);
    let rot19 = rotate_right(word, 19);
    let shft10 = shift_right(api, word, 10);
    let tmp = xor(api, &rot17, &rot19);

    xor(api, &tmp, &shft10)
}

// Sigma0 function: ROTR(x, 2) XOR ROTR(x, 13) XOR ROTR(x, 22)
pub fn capital_sigma0<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Sha256Word,
) -> Sha256Word {
    let rot2 = rotate_right(x, 2);
    let rot13 = rotate_right(x, 13);
    let rot22 = rotate_right(x, 22);
    let tmp = xor(api, &rot2, &rot13);

    xor(api, &tmp, &rot22)
}

// Sigma1 function: ROTR(x, 6) XOR ROTR(x, 11) XOR ROTR(x, 25)
pub fn capital_sigma1<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    x: &Sha256Word,
) -> Sha256Word {
    let rot6 = rotate_right(x, 6);
    let rot11 = rotate_right(x, 11);
    let rot25 = rotate_right(x, 25);
    let tmp = xor(api, &rot6, &rot11);

    xor(api, &tmp, &rot25)
}

pub fn add_const<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Sha256Word,
    b: u32,
) -> Sha256Word {
    let n = a.len();
    let mut c = *a;
    let mut ci = api.constant(0);
    for i in (0..n).rev() {
        if (b >> (31 - i)) & 1 == 1 {
            let p = api.add(a[i], 1);
            c[i] = api.add(p, ci);

            ci = api.mul(ci, p);
            ci = api.add(ci, a[i]);
        } else {
            c[i] = api.add(c[i], ci);
            ci = api.mul(ci, a[i]);
        }
    }
    c
}

// The brentkung addition algorithm, recommended
pub fn add_brentkung<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Sha256Word,
    b: &Sha256Word,
) -> Sha256Word {
    // temporary solution to change endianness, big -> little
    let mut a = *a;
    let mut b = *b;
    a.reverse();
    b.reverse();

    let mut c = vec![api.constant(0); 32];
    let mut ci = api.constant(0);

    for i in 0..8 {
        let start = i * 4;
        let end = start + 4;

        let (sum, ci_next) = brent_kung_adder_4_bits(api, &a[start..end], &b[start..end], ci);
        ci = ci_next;

        c[start..end].copy_from_slice(&sum);
    }

    // temporary solution to change endianness, little -> big
    c.reverse();
    c.try_into().unwrap()
}

fn brent_kung_adder_4_bits<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &[Variable],
    b: &[Variable],
    carry_in: Variable,
) -> ([Variable; 4], Variable) {
    let mut g = [api.constant(0); 4];
    let mut p = [api.constant(0); 4];

    // Step 1: Generate and propagate
    for i in 0..4 {
        g[i] = api.mul(a[i], b[i]);
        p[i] = api.add(a[i], b[i]);
    }

    // Step 2: Prefix computation
    let p1g0 = api.mul(p[1], g[0]);
    let p0p1 = api.mul(p[0], p[1]);
    let p2p3 = api.mul(p[2], p[3]);

    let g10 = api.add(g[1], p1g0);
    let g20 = api.mul(p[2], g10);
    let g20 = api.add(g[2], g20);
    let g30 = api.mul(p[3], g20);
    let g30 = api.add(g[3], g30);

    // Step 3: Calculate carries
    let mut c = [api.constant(0); 5];
    c[0] = carry_in;
    let tmp = api.mul(p[0], c[0]);
    c[1] = api.add(g[0], tmp);
    let tmp = api.mul(p0p1, c[0]);
    c[2] = api.add(g10, tmp);
    let tmp = api.mul(p[2], c[0]);
    let tmp = api.mul(p0p1, tmp);
    c[3] = api.add(g20, tmp);
    let tmp = api.mul(p0p1, p2p3);
    let tmp = api.mul(tmp, c[0]);
    c[4] = api.add(g30, tmp);

    // Step 4: Calculate sum
    let mut sum = [api.constant(0); 4];
    for i in 0..4 {
        sum[i] = api.add(p[i], c[i]);
    }

    (sum, c[4])
}

pub fn add<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Sha256Word,
    b: &Sha256Word,
) -> Sha256Word {
    add_brentkung(api, a, b)
}

pub fn sum_all<C: Config, Builder: RootAPI<C>>(api: &mut Builder, vs: &[Sha256Word]) -> Sha256Word {
    let mut n_values_to_sum = vs.len();
    let mut vvs = vs.to_vec();

    // Sum all values in a binary tree fashion to produce fewer layers in the circuit
    while n_values_to_sum > 1 {
        let half_size_floor = n_values_to_sum / 2;
        for i in 0..half_size_floor {
            vvs[i] = add(api, &vvs[i], &vvs[i + half_size_floor])
        }

        if n_values_to_sum & 1 != 0 {
            vvs[half_size_floor] = vvs[n_values_to_sum - 1];
        }

        n_values_to_sum = (n_values_to_sum + 1) / 2;
    }

    vvs[0]
}

fn bit_add_with_carry<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: Variable,
    b: Variable,
    carry: Variable,
) -> (Variable, Variable) {
    let sum = api.add(a, b);
    let sum = api.add(sum, carry);

    // a * (b + (b + 1) * carry) + (a + 1) * b * carry
    // = a * b + a * b * carry + a * b * carry + a * carry + b * carry
    let ab = api.mul(a, b);
    let ac = api.mul(a, carry);
    let bc = api.mul(b, carry);
    let abc = api.mul(ab, carry);

    let carry_next = api.add(ab, abc);
    let carry_next = api.add(carry_next, abc);
    let carry_next = api.add(carry_next, ac);
    let carry_next = api.add(carry_next, bc);

    (sum, carry_next)
}

// The vanilla addition algorithm, not recommended
pub fn add_vanilla<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Sha256Word,
    b: &Sha256Word,
) -> Sha256Word {
    let mut c = vec![api.constant(0); 32];

    let mut carry = api.constant(0);
    for i in (0..32).rev() {
        (c[i], carry) = bit_add_with_carry(api, a[i], b[i], carry);
    }
    c.try_into().unwrap()
}

pub fn xor<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Sha256Word,
    b: &Sha256Word,
) -> Sha256Word {
    let mut bits_res = [api.constant(0); 32];
    for i in 0..32 {
        bits_res[i] = api.add(a[i], b[i]);
    }
    bits_res
}

pub fn and<C: Config, Builder: RootAPI<C>>(
    api: &mut Builder,
    a: &Sha256Word,
    b: &Sha256Word,
) -> Sha256Word {
    let mut bits_res = [api.constant(0); 32];
    for i in 0..32 {
        bits_res[i] = api.mul(a[i], b[i]);
    }
    bits_res
}

pub fn not<C: Config, Builder: RootAPI<C>>(api: &mut Builder, a: &Sha256Word) -> Sha256Word {
    let mut bits_res = [api.constant(0); 32];
    for i in 0..32 {
        bits_res[i] = api.sub(1, a[i]);
    }
    bits_res
}
