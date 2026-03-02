use expander_compiler::field::FieldArith;
use expander_compiler::frontend::*;
use expander_compiler::zkcuda::proving_system::Expander;
use expander_compiler::zkcuda::proving_system::ParallelizedExpander;
use expander_compiler::zkcuda::proving_system::ProvingSystem;
use expander_compiler::zkcuda::{context::*, kernel::*};
use rand::{Rng, SeedableRng};
use tiny_keccak::Hasher;

// We will show two ways to archive the same goal: compute multiple Keccak hashes in parallel.
// The first way is to run multiple kernels in parallel, each of which computes a single Keccak hash. (Recommended)
// The second way is to run a single kernel that computes multiple Keccak hashes.
const N_PARALLEL: usize = 4;

const CHECK_BITS: usize = 256;
const PARTITION_BITS: usize = 30;
const CHECK_PARTITIONS: usize = (CHECK_BITS + PARTITION_BITS - 1) / PARTITION_BITS;

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

fn compress_bits(b: Vec<usize>) -> Vec<usize> {
    if b.len() != CHECK_BITS {
        panic!("gg");
    }
    let mut res = vec![0; CHECK_PARTITIONS];
    for i in (0..b.len()).step_by(PARTITION_BITS) {
        let r = b.len().min(i + PARTITION_BITS);
        for j in i..r {
            res[i / PARTITION_BITS] += b[j] << (j - i);
        }
    }
    res
}

fn compress_bits_var<C: Config>(api: &mut API<C>, mut a: Vec<Variable>) -> Vec<Variable> {
    if a.len() != CHECK_BITS || CircuitField::<C>::FIELD_SIZE <= PARTITION_BITS {
        panic!("gg");
    }
    for i in 0..a.len() {
        a[i] = from_my_bit_form(api, a[i].clone());
    }
    let mut compressed = Vec::new();
    for i in (0..a.len()).step_by(PARTITION_BITS) {
        let r = a.len().min(i + PARTITION_BITS);
        let mut sum = api.constant(0);
        for j in i..r {
            let t = api.mul(a[j].clone(), 1 << (j - i));
            sum = api.add(sum, t);
        }
        compressed.push(sum);
    }
    compressed
}

fn from_my_bit_form<C: Config>(api: &mut API<C>, x: Variable) -> Variable {
    let t = api.sub(1, x);
    api.div(t, 2, true)
}

fn to_my_bit_form<C: Config>(x: usize) -> CircuitField<C> {
    if x == 0 {
        CircuitField::<C>::one()
    } else {
        assert_eq!(x, 1);
        -CircuitField::<C>::one()
    }
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
                a[0][j] = api.sub(0, a[0][j]);
            }
        }
    }

    a
}

fn xor<C: Config>(api: &mut API<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    let nbits = a.len();
    let mut bits_res = vec![api.constant(0); nbits];
    for i in 0..nbits {
        bits_res[i] = api.mul(a[i].clone(), b[i].clone());
    }
    bits_res
}

fn and<C: Config>(api: &mut API<C>, a: Vec<Variable>, b: Vec<Variable>) -> Vec<Variable> {
    let nbits = a.len();
    let mut bits_res = vec![api.constant(0); nbits];
    for i in 0..nbits {
        let t = api.mul(a[i].clone(), b[i].clone());
        let t = api.sub(0, t);
        let t = api.add(t, b[i].clone());
        let t = api.add(t, a[i].clone());
        let t = api.add(t, 1);
        bits_res[i] = api.div(t, 2, true);
    }
    bits_res
}

fn not<C: Config>(api: &mut API<C>, a: Vec<Variable>) -> Vec<Variable> {
    let mut bits_res = vec![api.constant(0); a.len()];
    for i in 0..a.len() {
        bits_res[i] = api.sub(0, a[i].clone());
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

fn compute_keccak_inner<C: Config>(api: &mut API<C>, p: &Vec<Variable>) -> Vec<Variable> {
    for x in p.iter() {
        let x_sqr = api.mul(x, x);
        api.assert_is_equal(x_sqr, 1);
    }

    let mut ss = vec![vec![api.constant(1); 64]; 25];
    let mut new_p = p.clone();
    let mut append_data = vec![0; 136 - 64];
    append_data[0] = 1;
    append_data[135 - 64] = 0x80;
    for i in 0..136 - 64 {
        for j in 0..8 {
            new_p.push(api.constant(to_my_bit_form::<C>((append_data[i] >> j) & 1)));
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
    compress_bits_var(api, copy_out_unaligned(ss, 136, 32))
}

#[kernel]
fn compute_keccak<C: Config>(
    api: &mut API<C>,
    p: &[InputVariable; 64 * 8],
    out: &mut [OutputVariable; CHECK_PARTITIONS],
) {
    let outc = api.memorized_simple_call(compute_keccak_inner, p);
    for i in 0..CHECK_PARTITIONS {
        out[i] = outc[i];
    }
}

#[kernel]
fn compute_multiple_keccak<C: Config>(
    api: &mut API<C>,
    p: &[[InputVariable; 64 * 8]; N_PARALLEL],
    out: &mut [[OutputVariable; CHECK_PARTITIONS]; N_PARALLEL],
) {
    for i in 0..N_PARALLEL {
        compute_keccak(api, &p[i], &mut out[i]);
    }
}

fn zkcuda_keccak_1_helper<P: ProvingSystem<M31Config>>() {
    let kernel: KernelPrimitive<M31Config> = compile_compute_keccak().unwrap();
    println!("compile ok");

    let mut ctx: Context<M31Config> = Context::default();
    let mut p: Vec<Vec<M31>> = vec![];
    let mut data: Vec<Vec<u8>> = vec![];
    let mut expected_res: Vec<Vec<M31>> = vec![];
    let mut rng = rand::rngs::StdRng::seed_from_u64(1237);
    for i in 0..N_PARALLEL {
        p.push(vec![]);
        data.push(vec![]);
        for _ in 0..64 {
            data[i].push(rng.gen());
        }
        let mut hash = tiny_keccak::Keccak::v256();
        hash.update(&data[i]);
        let mut output = [0u8; 32];
        hash.finalize(&mut output);
        for j in 0..64 {
            for k in 0..8 {
                p[i].push(to_my_bit_form::<M31Config>((data[i][j] >> k) as usize & 1));
            }
        }
        let mut out_bits = vec![0; 256];
        for j in 0..32 {
            for k in 0..8 {
                out_bits[j * 8 + k] = (output[j] >> k) as usize & 1;
            }
        }
        let out_compressed = compress_bits(out_bits);
        assert_eq!(out_compressed.len(), CHECK_PARTITIONS);
        expected_res.push(vec![]);
        for j in 0..CHECK_PARTITIONS {
            expected_res[i].push(M31::from(out_compressed[j] as u32));
        }
    }

    println!("prepare data ok");
    let p = ctx.copy_to_device(&p);
    println!("copy to device ok");
    let mut out = None;
    call_kernel!(ctx, kernel, N_PARALLEL, p, mut out).unwrap();
    println!("call kernel ok");
    let out: Vec<Vec<M31>> = ctx.copy_to_host(out);
    println!("copy to host ok");
    assert_eq!(out, expected_res);
    assert_eq!(out[0][0], expected_res[0][0]);

    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    let (prover_setup, verifier_setup) = P::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    println!("proof generation ok");
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
    println!("verify ok");
    P::post_process();
}

fn zkcuda_keccak_2_helper<P: ProvingSystem<M31Config>>() {
    let kernel: KernelPrimitive<M31Config> = compile_compute_multiple_keccak().unwrap();
    println!("compile ok");

    let mut ctx: Context<M31Config> = Context::default();
    let mut p: Vec<Vec<M31>> = vec![];
    let mut data: Vec<Vec<u8>> = vec![];
    let mut expected_res: Vec<Vec<M31>> = vec![];
    let mut rng = rand::rngs::StdRng::seed_from_u64(1237);
    for i in 0..N_PARALLEL {
        p.push(vec![]);
        data.push(vec![]);
        for _ in 0..64 {
            data[i].push(rng.gen());
        }
        let mut hash = tiny_keccak::Keccak::v256();
        hash.update(&data[i]);
        let mut output = [0u8; 32];
        hash.finalize(&mut output);
        for j in 0..64 {
            for k in 0..8 {
                p[i].push(to_my_bit_form::<M31Config>((data[i][j] >> k) as usize & 1));
            }
        }
        let mut out_bits = vec![0; 256];
        for j in 0..32 {
            for k in 0..8 {
                out_bits[j * 8 + k] = (output[j] >> k) as usize & 1;
            }
        }
        let out_compressed = compress_bits(out_bits);
        assert_eq!(out_compressed.len(), CHECK_PARTITIONS);
        expected_res.push(vec![]);
        for j in 0..CHECK_PARTITIONS {
            expected_res[i].push(M31::from(out_compressed[j] as u32));
        }
    }

    println!("prepare data ok");
    let p = ctx.copy_to_device(&vec![p]);
    println!("copy to device ok");
    let mut out = None;
    call_kernel!(ctx, kernel, 1, p, mut out).unwrap();
    println!("call kernel ok");
    let out: Vec<Vec<Vec<M31>>> = ctx.copy_to_host(out);
    println!("copy to host ok");
    assert_eq!(out[0], expected_res);
    assert_eq!(out[0][0][0], expected_res[0][0]);

    let computation_graph = ctx.compile_computation_graph().unwrap();
    ctx.solve_witness().unwrap();
    let (prover_setup, verifier_setup) = P::setup(&computation_graph);
    let proof = P::prove(
        &prover_setup,
        &computation_graph,
        ctx.export_device_memories(),
    );
    println!("proof generation ok");
    assert!(P::verify(&verifier_setup, &computation_graph, &proof));
    println!("verify ok");
    P::post_process();
}

#[test]
fn zkcuda_keccak_single_core() {
    zkcuda_keccak_1_helper::<Expander<M31Config>>();
    zkcuda_keccak_2_helper::<Expander<M31Config>>();
}

#[test]
fn zkcuda_keccak_multi_core() {
    zkcuda_keccak_1_helper::<ParallelizedExpander<M31Config>>();
    zkcuda_keccak_2_helper::<ParallelizedExpander<M31Config>>();
}
