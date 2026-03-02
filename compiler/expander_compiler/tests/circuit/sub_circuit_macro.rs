use expander_compiler::frontend::*;
use rand::{Rng, SeedableRng};

declare_circuit!(Circuit { x: [Variable; 100] });

// A simple sub-circuit with bunch of inputs and outputs
#[memorized]
fn work<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: Variable,
    b: &Variable,
    c: Vec<Variable>,
    d: &Vec<Variable>,
    e: Vec<Vec<Variable>>,
    f: &Vec<Vec<Variable>>,
    g: u32,
    h: &u32,
    i: Vec<Vec<u32>>,
) -> Vec<Vec<Variable>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let mut res = api.constant(0);
    res = api.add(res, a);
    res = api.mul(res, 10007);
    res = api.add(res, b);
    res = api.mul(res, 10007);
    for x in c.iter() {
        res = api.add(res, x);
        res = api.mul(res, 10007);
    }
    for x in d.iter() {
        res = api.add(res, x);
        res = api.mul(res, 10007);
    }
    for x in e.iter() {
        for y in x.iter() {
            if rng.gen_ratio(1, 8) {
                res = api.mul(res, y);
            }
            res = api.add(res, y);
            res = api.mul(res, 10007);
        }
    }
    for x in f.iter() {
        for y in x.iter() {
            if rng.gen_ratio(1, 8) {
                res = api.mul(res, y);
            }
            res = api.add(res, y);
            res = api.mul(res, 10007);
        }
    }
    res = api.add(res, g);
    res = api.mul(res, 10007);
    res = api.add(res, *h);
    res = api.mul(res, 10007);
    for x in i.iter() {
        for y in x.iter() {
            res = api.add(res, *y);
            res = api.mul(res, 10007);
        }
    }
    let mut resv: Vec<Vec<Variable>> = Vec::new();
    for _ in 0..g as usize {
        resv.push(vec![res]);
    }
    resv.push(vec![res; *h as usize]);
    resv
}

fn call_work_and_check2<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: Variable,
    b: &Variable,
    c: Vec<Variable>,
    d: &Vec<Variable>,
    e: Vec<Vec<Variable>>,
    f: &Vec<Vec<Variable>>,
    g: u32,
    h: &u32,
    i: Vec<Vec<u32>>,
) {
    let res1 = work(api, a, b, c.clone(), d, e.clone(), f, g, h, i.clone());
    let res2 = memorized_work(api, a, b, c.clone(), d, e.clone(), f, g, h, i.clone());
    for (x, y) in res1.iter().zip(res2.iter()) {
        for (x, y) in x.iter().zip(y.iter()) {
            api.assert_is_equal(*x, *y);
        }
    }
}

fn call_work_and_check<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: Variable,
    b: &Variable,
    c: Vec<Variable>,
    d: &Vec<Variable>,
    e: Vec<Vec<Variable>>,
    f: &Vec<Vec<Variable>>,
    g: u32,
    h: &u32,
    i: Vec<Vec<u32>>,
) {
    for _ in 0..5 {
        call_work_and_check2(api, a, b, c.clone(), d, e.clone(), f, g, h, i.clone());
        call_work_and_check2(api, *b, &a, c.clone(), d, e.clone(), f, g, h, i.clone());
    }
}

fn call_tests<C: Config, B: RootAPI<C>>(api: &mut B, x: &[Variable]) {
    call_work_and_check(
        api,
        x[0],
        &x[1],
        vec![x[2]],
        &vec![],
        vec![],
        &vec![],
        3,
        &2,
        vec![vec![3]],
    );
    call_work_and_check(
        api,
        x[0],
        &x[1],
        vec![x[2]],
        &vec![],
        vec![],
        &vec![],
        3,
        &2,
        vec![vec![4]],
    );
    call_work_and_check(
        api,
        x[0],
        &x[1],
        vec![x[2], x[3]],
        &vec![],
        vec![],
        &vec![],
        3,
        &2,
        vec![vec![3]],
    );
    // do some random tests, make sure vec shapes are different
    let mut rng = rand::rngs::StdRng::seed_from_u64(10);
    for _ in 0..100 {
        let a = x[rng.gen_range(0..100)];
        let b = x[rng.gen_range(0..100)];
        let c_len = rng.gen_range(0..10);
        let c: Vec<Variable> = (0..c_len).map(|_| x[rng.gen_range(0..100)]).collect();
        let d_len = rng.gen_range(0..10);
        let d: Vec<Variable> = (0..d_len).map(|_| x[rng.gen_range(0..100)]).collect();
        let e_len = rng.gen_range(0..10);
        let mut e: Vec<Vec<Variable>> = Vec::new();
        for _ in 0..e_len {
            let e_inner_len = rng.gen_range(0..10);
            let e_inner: Vec<Variable> =
                (0..e_inner_len).map(|_| x[rng.gen_range(0..100)]).collect();
            e.push(e_inner);
        }
        let f_len = rng.gen_range(0..10);
        let mut f: Vec<Vec<Variable>> = Vec::new();
        for _ in 0..f_len {
            let f_inner_len = rng.gen_range(0..10);
            let f_inner: Vec<Variable> =
                (0..f_inner_len).map(|_| x[rng.gen_range(0..100)]).collect();
            f.push(f_inner);
        }
        let g = rng.gen::<u32>() % 10;
        let h = rng.gen::<u32>() % 10;
        let i_len = rng.gen_range(0..10);
        let mut i: Vec<Vec<u32>> = Vec::new();
        for _ in 0..i_len {
            let i_inner_len = rng.gen_range(0..10);
            let i_inner: Vec<u32> = (0..i_inner_len).map(|_| rng.gen::<u32>()).collect();
            i.push(i_inner);
        }
        call_work_and_check(api, a, &b, c.clone(), &d, e.clone(), &f, g, &h, i.clone());
    }
}

impl Define<M31Config> for Circuit<Variable> {
    fn define<Builder: RootAPI<M31Config>>(&self, api: &mut Builder) {
        call_tests(api, &self.x);
    }
}

#[test]
fn sub_circuit_macro() {
    let compile_result = compile(&Circuit::default(), CompileOptions::default()).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(10);
    let x: [M31; 100] = (0..100)
        .map(|_| M31::from(rng.gen::<u32>()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let assignment = Circuit::<M31> { x };
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let output = compile_result.layered_circuit.run(&witness);
    assert_eq!(output, vec![true]);
}
