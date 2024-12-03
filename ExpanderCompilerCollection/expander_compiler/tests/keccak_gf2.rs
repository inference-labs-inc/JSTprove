use expander_compiler::frontend::*;
use internal::Serde;
use rand::{thread_rng, Rng};
use tiny_keccak::Hasher;

const N_HASHES: usize = 1;

fn rc() -> Vec<u64> {
    vec![
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808A,
        0x8000000080008000,
        0x000000000000808B,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008A,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000A,
        0x000000008000808B,
        0x800000000000008B,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800A,
        0x800000008000000A,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ]
}

fn xor_in<C: Config>(
    api: &mut API<C>,
    mut s: Vec<Vec<Variable>>,
    buf: Vec<Vec<Variable>>,
) -> Vec<Vec<Variable>> {
    for y in 0..5 {
        for x in 0..5 {
            if x + 5 * y < buf.len() {
                s[5 * x + y] = xor(api, s[5 * x + y].clone(), buf[x + 5 * y].clone())
            }
        }
    }
    s
}

fn keccak_f<C: Config>(api: &mut API<C>, mut a: Vec<Vec<Variable>>) -> Vec<Vec<Variable>> {
    let mut b = vec![vec![api.constant(0); 64]; 25];
    let mut c = vec![vec![api.constant(0); 64]; 5];
    let mut d = vec![vec![api.constant(0); 64]; 5];
    let mut da = vec![vec![api.constant(0); 64]; 5];
    let rc = rc();

    for i in 0..24 {
        for j in 0..5 {
            let t1 = xor(api, a[j * 5 + 1].clone(), a[j * 5 + 2].clone());
            let t2 = xor(api, a[j * 5 + 3].clone(), a[j * 5 + 4].clone());
            c[j] = xor(api, t1, t2);
        }

        for j in 0..5 {
            d[j] = xor(
                api,
                c[(j + 4) % 5].clone(),
                rotate_left::<C>(&c[(j + 1) % 5], 1),
            );
            da[j] = xor(
                api,
                a[((j + 4) % 5) * 5].clone(),
                rotate_left::<C>(&a[((j + 1) % 5) * 5], 1),
            );
        }

        for j in 0..25 {
            let tmp = xor(api, da[j / 5].clone(), a[j].clone());
            a[j] = xor(api, tmp, d[j / 5].clone());
        }

        /*Rho and pi steps*/
        b[0] = a[0].clone();

        b[8] = rotate_left::<C>(&a[1], 36);
        b[11] = rotate_left::<C>(&a[2], 3);
        b[19] = rotate_left::<C>(&a[3], 41);
        b[22] = rotate_left::<C>(&a[4], 18);

        b[2] = rotate_left::<C>(&a[5], 1);
        b[5] = rotate_left::<C>(&a[6], 44);
        b[13] = rotate_left::<C>(&a[7], 10);
        b[16] = rotate_left::<C>(&a[8], 45);
        b[24] = rotate_left::<C>(&a[9], 2);

        b[4] = rotate_left::<C>(&a[10], 62);
        b[7] = rotate_left::<C>(&a[11], 6);
        b[10] = rotate_left::<C>(&a[12], 43);
        b[18] = rotate_left::<C>(&a[13], 15);
        b[21] = rotate_left::<C>(&a[14], 61);

        b[1] = rotate_left::<C>(&a[15], 28);
        b[9] = rotate_left::<C>(&a[16], 55);
        b[12] = rotate_left::<C>(&a[17], 25);
        b[15] = rotate_left::<C>(&a[18], 21);
        b[23] = rotate_left::<C>(&a[19], 56);

        b[3] = rotate_left::<C>(&a[20], 27);
        b[6] = rotate_left::<C>(&a[21], 20);
        b[14] = rotate_left::<C>(&a[22], 39);
        b[17] = rotate_left::<C>(&a[23], 8);
        b[20] = rotate_left::<C>(&a[24], 14);

        /*Xi state*/

        for j in 0..25 {
            let t = not(api, b[(j + 5) % 25].clone());
            let t = and(api, t, b[(j + 10) % 25].clone());
            a[j] = xor(api, b[j].clone(), t);
        }

        /*Last step*/

        for j in 0..64 {
            if rc[i] >> j & 1 == 1 {
                a[0][j] = api.sub(1, a[0][j]);
            }
        }
    }

    a
}

fn xor<C: Config>(api: &mut API<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    let nbits = a.len();
    let mut bits_res = vec![api.constant(0); nbits];
    for i in 0..nbits {
        bits_res[i] = api.add(a[i].clone(), b[i].clone());
    }
    bits_res
}

fn and<C: Config>(api: &mut API<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    let nbits = a.len();
    let mut bits_res = vec![api.constant(0); nbits];
    for i in 0..nbits {
        bits_res[i] = api.mul(a[i].clone(), b[i].clone());
    }
    bits_res
}

fn not<C: Config>(api: &mut API<C>, a: Vec<Variable>) -> Vec<Variable> {
    let mut bits_res = vec![api.constant(0); a.len()];
    for i in 0..a.len() {
        bits_res[i] = api.sub(1, a[i].clone());
    }
    bits_res
}

fn rotate_left<C: Config>(bits: &Vec<Variable>, k: usize) -> Vec<Variable> {
    let n = bits.len();
    let s = k & (n - 1);
    let mut new_bits = bits[(n - s) as usize..].to_vec();
    new_bits.append(&mut bits[0..(n - s) as usize].to_vec());
    new_bits
}

fn copy_out_unaligned(s: Vec<Vec<Variable>>, rate: usize, output_len: usize) -> Vec<Variable> {
    let mut out = vec![];
    let w = 8;
    let mut b = 0;
    while b < output_len {
        for y in 0..5 {
            for x in 0..5 {
                if x + 5 * y < rate / w && b < output_len {
                    out.append(&mut s[5 * x + y].clone());
                    b += 8;
                }
            }
        }
    }
    out
}

declare_circuit!(Keccak256Circuit {
    p: [[Variable; 64 * 8]; N_HASHES],
    out: [[PublicVariable; 256]; N_HASHES],
});

fn compute_keccak<C: Config>(api: &mut API<C>, p: &Vec<Variable>) -> Vec<Variable> {
    let mut ss = vec![vec![api.constant(0); 64]; 25];
    let mut new_p = p.clone();
    let mut append_data = vec![0; 136 - 64];
    append_data[0] = 1;
    append_data[135 - 64] = 0x80;
    for i in 0..136 - 64 {
        for j in 0..8 {
            new_p.push(api.constant(((append_data[i] >> j) & 1) as u32));
        }
    }
    let mut p = vec![vec![api.constant(0); 64]; 17];
    for i in 0..17 {
        for j in 0..64 {
            p[i][j] = new_p[i * 64 + j].clone();
        }
    }
    ss = xor_in(api, ss, p);
    ss = keccak_f(api, ss);
    copy_out_unaligned(ss, 136, 32)
}

impl Define<GF2Config> for Keccak256Circuit<Variable> {
    fn define(&self, api: &mut API<GF2Config>) {
        for i in 0..N_HASHES {
            // You can use api.memorized_simple_call for sub-circuits
            // let out = api.memorized_simple_call(compute_keccak, &self.p[i].to_vec());
            let out = compute_keccak(api, &self.p[i].to_vec());
            for j in 0..256 {
                api.assert_is_equal(out[j].clone(), self.out[i][j].clone());
            }
        }
    }
}

#[test]
fn keccak_gf2_main() {
    let compile_result = compile(&Keccak256Circuit::default()).unwrap();
    let CompileResult {
        witness_solver,
        layered_circuit,
    } = compile_result;

    let mut assignment = Keccak256Circuit::<GF2>::default();
    for k in 0..N_HASHES {
        let mut data = vec![0u8; 64];
        for i in 0..64 {
            data[i] = thread_rng().gen();
        }
        let mut hash = tiny_keccak::Keccak::v256();
        hash.update(&data);
        let mut output = [0u8; 32];
        hash.finalize(&mut output);
        for i in 0..64 {
            for j in 0..8 {
                assignment.p[k][i * 8 + j] = ((data[i] >> j) as u32 & 1).into();
            }
        }
        for i in 0..32 {
            for j in 0..8 {
                assignment.out[k][i * 8 + j] = ((output[i] >> j) as u32 & 1).into();
            }
        }
    }
    let witness = witness_solver.solve_witness(&assignment).unwrap();
    let res = layered_circuit.run(&witness);
    assert_eq!(res, vec![true]);
    println!("test 1 passed");

    for k in 0..N_HASHES {
        assignment.p[k][0] = assignment.p[k][0] - GF2::from(1);
    }
    let witness = witness_solver.solve_witness(&assignment).unwrap();
    let res = layered_circuit.run(&witness);
    assert_eq!(res, vec![false]);
    println!("test 2 passed");

    let mut assignments = Vec::new();
    for _ in 0..16 {
        for k in 0..N_HASHES {
            assignment.p[k][0] = assignment.p[k][0] - GF2::from(1);
        }
        assignments.push(assignment.clone());
    }
    let witness = witness_solver.solve_witnesses(&assignments).unwrap();
    let res = layered_circuit.run(&witness);
    let mut expected_res = vec![false; 16];
    for i in 0..8 {
        expected_res[i * 2] = true;
    }
    assert_eq!(res, expected_res);
    println!("test 3 passed");

    let assignments_correct: Vec<Keccak256Circuit<GF2>> =
        (0..8).map(|i| assignments[i * 2].clone()).collect();
    let witness = witness_solver
        .solve_witnesses(&assignments_correct)
        .unwrap();

    let file = std::fs::File::create("circuit_gf2.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    layered_circuit.serialize_into(writer).unwrap();

    let file = std::fs::File::create("witness_gf2.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    witness.serialize_into(writer).unwrap();

    let file = std::fs::File::create("witness_gf2_solver.txt").unwrap();
    let writer = std::io::BufWriter::new(file);
    witness_solver.serialize_into(writer).unwrap();

    println!("dumped to files");
}
