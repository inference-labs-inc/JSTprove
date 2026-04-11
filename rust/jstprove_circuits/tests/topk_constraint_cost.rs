use expander_compiler::frontend::*;

use jstprove_circuits::circuit_functions::gadgets::LogupRangeCheckContext;
use jstprove_circuits::circuit_functions::hints::topk::TOPK_HINT_KEY;

struct TopKBench {
    n: usize,
    k: usize,
    n_bits: usize,
}

impl TopKBench {
    fn run_bench(&self) {
        let n = self.n;
        let k = self.k;
        let n_bits = self.n_bits;

        declare_circuit!(TopKCircuit {
            lane: [Variable; 2048],
        });

        struct Params {
            n: usize,
            k: usize,
            n_bits: usize,
        }

        thread_local! {
            static PARAMS: std::cell::RefCell<Params> = const { std::cell::RefCell::new(Params { n: 0, k: 0, n_bits: 0 }) };
        }

        PARAMS.with(|p| {
            *p.borrow_mut() = Params { n, k, n_bits };
        });

        impl Define<BN254Config> for TopKCircuit<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                PARAMS.with(|p| {
                    let p = p.borrow();
                    let n = p.n;
                    let k = p.k;
                    let n_bits = p.n_bits;
                    let shift_n_bits = n_bits + 1;

                    let offset_val = 1u64 << n_bits;
                    let offset = api.constant(BN254Fr::from_u256(ethnum::U256::from(offset_val)));
                    let scale_var =
                        api.constant(BN254Fr::from_u256(ethnum::U256::from(1u64 << 18)));

                    let lane: Vec<Variable> = self.lane[..n].to_vec();

                    let mut hint_inputs = Vec::with_capacity(n + 1);
                    hint_inputs.extend_from_slice(&lane);
                    hint_inputs.push(scale_var);

                    let hint_out = api.new_hint(TOPK_HINT_KEY, &hint_inputs, 2 * k);
                    let values = &hint_out[..k];

                    let mut logup_ctx = LogupRangeCheckContext::new_default();

                    let mut membership_prods: Vec<Variable> =
                        (0..k).map(|_| api.constant(1u32)).collect();
                    let min_val = values[k - 1];

                    for &elem in lane.iter() {
                        let mut qi = api.constant(1u32);
                        for j in 0..k {
                            let factor = api.sub(values[j], elem);
                            membership_prods[j] = api.mul(membership_prods[j], factor);
                            qi = api.mul(qi, factor);
                        }

                        let is_sel = api.is_zero(qi);
                        let one_c = api.constant(1u32);
                        let not_sel = api.sub(one_c, is_sel);

                        let delta = api.sub(min_val, elem);
                        let check_val = api.mul(not_sel, delta);

                        let _ = logup_ctx.range_check::<BN254Config, Builder>(
                            api,
                            check_val,
                            shift_n_bits,
                        );
                    }

                    for prod in membership_prods.iter().take(k) {
                        api.assert_is_zero(*prod);
                    }

                    for j in 0..k.saturating_sub(1) {
                        let diff = api.sub(values[j], values[j + 1]);
                        let shifted = api.add(diff, offset);
                        let _ = logup_ctx.range_check::<BN254Config, Builder>(
                            api,
                            shifted,
                            shift_n_bits,
                        );
                    }
                });
            }
        }

        let compile_result = compile(&TopKCircuit::default(), CompileOptions::default()).unwrap();
        let layered_stats = compile_result.layered_circuit.get_stats();

        println!(
            "TopK(N={:>5}, K={:>3}): \
             Layered mul={:>8}, add={:>8}, total_gates={:>8}, cost={:>10}",
            n,
            k,
            layered_stats.num_expanded_mul,
            layered_stats.num_expanded_add,
            layered_stats.num_total_gates,
            layered_stats.total_cost,
        );
    }
}

fn run_constrained_max_bench(n: usize, n_bits: usize) {
    declare_circuit!(MaxCircuit {
        lane: [Variable; 2048],
    });

    struct MaxParams {
        n: usize,
        n_bits: usize,
    }
    thread_local! {
        static MAX_PARAMS: std::cell::RefCell<MaxParams> = const { std::cell::RefCell::new(MaxParams { n: 0, n_bits: 0 }) };
    }
    MAX_PARAMS.with(|p| {
        *p.borrow_mut() = MaxParams { n, n_bits };
    });

    impl Define<BN254Config> for MaxCircuit<Variable> {
        fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
            MAX_PARAMS.with(|p| {
                let p = p.borrow();
                let shift_exp = p.n_bits.saturating_sub(1);
                let offset_val = 1u64 << shift_exp;
                let offset = api.constant(BN254Fr::from_u256(ethnum::U256::from(offset_val)));

                let lane: Vec<Variable> = self.lane[..p.n].to_vec();
                let mut shifted = Vec::with_capacity(p.n);
                for &x in &lane {
                    shifted.push(api.add(x, offset));
                }
                let max_shifted = api.unconstrained_greater_eq(shifted[0], shifted[1]);
                let _ = max_shifted;

                let max_offset =
                    jstprove_circuits::circuit_functions::hints::max_min_clip::unconstrained_max(
                        api, &shifted,
                    )
                    .unwrap();
                let max_raw = api.sub(max_offset, offset);

                let check_bits = shift_exp + 1;
                let mut logup_ctx = LogupRangeCheckContext::new_default();
                let mut prod = api.constant(1);
                for &x in &lane {
                    let delta = api.sub(max_raw, x);
                    let _ = logup_ctx.range_check::<BN254Config, Builder>(api, delta, check_bits);
                    prod = api.mul(prod, delta);
                }
                api.assert_is_zero(prod);
            });
        }
    }

    let compile_result = compile(&MaxCircuit::default(), CompileOptions::default()).unwrap();
    let s = compile_result.layered_circuit.get_stats();
    println!(
        "  Max(N={:>5}):       Layered mul={:>8}, add={:>8}, total_gates={:>8}, cost={:>10}",
        n, s.num_expanded_mul, s.num_expanded_add, s.num_total_gates, s.total_cost,
    );
}

#[test]
#[ignore]
fn topk_constraint_cost_sweep() {
    println!();
    println!("--- Baseline: constrained_max (ReduceMax pattern) ---");
    for &n in &[100, 1000] {
        run_constrained_max_bench(n, 32);
    }
    println!();
    println!("--- TopK (merged single-pass) ---");
    for &(n, k) in &[
        (100, 1),
        (100, 5),
        (100, 10),
        (1000, 1),
        (1000, 5),
        (1000, 10),
        (2048, 5),
    ] {
        TopKBench { n, k, n_bits: 32 }.run_bench();
    }
}
