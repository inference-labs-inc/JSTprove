use arith::{FFTField, Field};
use expander_compiler::frontend::{
    CircuitField, CompileOptions, Config, Define, FieldArith, GoldilocksBasefoldConfig, RootAPI,
    Variable,
};
use ndarray::ArrayD;

type C = GoldilocksBasefoldConfig;
type F = CircuitField<C>;

static SWEEP_H: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SWEEP_W: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SWEEP_KH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SWEEP_KW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SWEEP_CIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SWEEP_COUT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SWEEP_NTT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn load(a: &std::sync::atomic::AtomicUsize) -> usize {
    a.load(std::sync::atomic::Ordering::Relaxed)
}

expander_compiler::frontend::declare_circuit!(ConvBench { vals: [Variable] });

impl Define<C> for ConvBench<Variable> {
    fn define<Builder: RootAPI<C>>(&self, api: &mut Builder) {
        let h = load(&SWEEP_H);
        let w = load(&SWEEP_W);
        let kh = load(&SWEEP_KH);
        let kw = load(&SWEEP_KW);
        let c_in = load(&SWEEP_CIN);
        let c_out = load(&SWEEP_COUT);
        let force_ntt = SWEEP_NTT.load(std::sync::atomic::Ordering::Relaxed);

        let h_out = h - kh + 1;
        let w_out = w - kw + 1;
        let zero = api.constant(0);

        let input_size = c_in * h * w;
        let input_arr = ArrayD::from_shape_fn(ndarray::IxDyn(&[1, c_in, h, w]), |idx| {
            self.vals[idx[1] * h * w + idx[2] * w + idx[3]]
        });
        let weight_arr = ArrayD::from_shape_fn(ndarray::IxDyn(&[c_out, c_in, kh, kw]), |idx| {
            self.vals
                [input_size + idx[0] * c_in * kh * kw + idx[1] * kh * kw + idx[2] * kw + idx[3]]
        });

        if force_ntt {
            let nh = (h + kh - 1).next_power_of_two();
            let nw = (w + kw - 1).next_power_of_two();

            let mut input_ntt = Vec::with_capacity(c_in);
            for ci in 0..c_in {
                let mut buf = vec![zero; nh * nw];
                for r in 0..h {
                    for col in 0..w {
                        buf[r * nw + col] = input_arr[[0, ci, r, col]];
                    }
                }
                ntt_2d_bench(api, &mut buf, nh, nw);
                input_ntt.push(buf);
            }

            for co in 0..c_out {
                let mut accum = vec![zero; nh * nw];
                for ci in 0..c_in {
                    let mut w_buf = vec![zero; nh * nw];
                    for r in 0..kh {
                        for col in 0..kw {
                            w_buf[r * nw + col] = weight_arr[[co, ci, r, col]];
                        }
                    }
                    ntt_2d_bench(api, &mut w_buf, nh, nw);
                    for i in 0..nh * nw {
                        let prod = api.mul(input_ntt[ci][i], w_buf[i]);
                        accum[i] = api.add(accum[i], prod);
                    }
                }
                intt_2d_bench(api, &mut accum, nh, nw);
                for r in 0..h_out {
                    for col in 0..w_out {
                        api.assert_is_non_zero(accum[r * nw + col]);
                    }
                }
            }
        } else {
            for co in 0..c_out {
                for ci in 0..c_in {
                    for r in 0..h_out {
                        for col in 0..w_out {
                            let mut sum = zero;
                            for kr in 0..kh {
                                for kc in 0..kw {
                                    let prod = api.mul(
                                        input_arr[[0, ci, r + kr, col + kc]],
                                        weight_arr[[co, ci, kr, kc]],
                                    );
                                    sum = api.add(sum, prod);
                                }
                            }
                            api.assert_is_non_zero(sum);
                        }
                    }
                }
            }
        }
    }
}

fn ntt_1d_bench<Builder: RootAPI<C>>(api: &mut Builder, data: &mut [Variable], omega: F) {
    let n = data.len();
    if !n.is_power_of_two() || n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS as usize - log_n);
        if i < j {
            data.swap(i, j);
        }
    }
    for s in 0..log_n {
        let half = 1 << s;
        let full = half << 1;
        let exp = n / full;
        let mut w_step = F::ONE;
        let mut base = omega;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                w_step *= base;
            }
            base *= base;
            e >>= 1;
        }
        let mut tw = F::ONE;
        let mut twiddles = Vec::with_capacity(half);
        for _ in 0..half {
            twiddles.push(tw);
            tw *= w_step;
        }
        for k in (0..n).step_by(full) {
            for j in 0..half {
                let u = data[k + j];
                let t = api.mul(data[k + j + half], twiddles[j]);
                data[k + j] = api.add(u, t);
                data[k + j + half] = api.sub(u, t);
            }
        }
    }
}

fn ntt_2d_bench<Builder: RootAPI<C>>(
    api: &mut Builder,
    data: &mut [Variable],
    rows: usize,
    cols: usize,
) {
    let omega_cols = F::two_adic_generator(cols.trailing_zeros() as usize);
    for r in 0..rows {
        ntt_1d_bench(api, &mut data[r * cols..(r + 1) * cols], omega_cols);
    }
    let omega_rows = F::two_adic_generator(rows.trailing_zeros() as usize);
    let mut col_buf = vec![Variable::default(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = data[r * cols + c];
        }
        ntt_1d_bench(api, &mut col_buf, omega_rows);
        for r in 0..rows {
            data[r * cols + c] = col_buf[r];
        }
    }
}

fn intt_2d_bench<Builder: RootAPI<C>>(
    api: &mut Builder,
    data: &mut [Variable],
    rows: usize,
    cols: usize,
) {
    let omega_cols_inv = F::two_adic_generator(cols.trailing_zeros() as usize)
        .inv()
        .unwrap();
    for r in 0..rows {
        ntt_1d_bench(api, &mut data[r * cols..(r + 1) * cols], omega_cols_inv);
    }
    let omega_rows_inv = F::two_adic_generator(rows.trailing_zeros() as usize)
        .inv()
        .unwrap();
    let mut col_buf = vec![Variable::default(); rows];
    for c in 0..cols {
        for r in 0..rows {
            col_buf[r] = data[r * cols + c];
        }
        ntt_1d_bench(api, &mut col_buf, omega_rows_inv);
        for r in 0..rows {
            data[r * cols + c] = col_buf[r];
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    let n_inv = F::from((rows * cols) as u32).inv().unwrap();
    for x in data.iter_mut() {
        *x = api.mul(*x, n_inv);
    }
}

fn run_config(
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    use_ntt: bool,
) -> (usize, usize) {
    SWEEP_H.store(h, std::sync::atomic::Ordering::Relaxed);
    SWEEP_W.store(w, std::sync::atomic::Ordering::Relaxed);
    SWEEP_KH.store(kh, std::sync::atomic::Ordering::Relaxed);
    SWEEP_KW.store(kw, std::sync::atomic::Ordering::Relaxed);
    SWEEP_CIN.store(c_in, std::sync::atomic::Ordering::Relaxed);
    SWEEP_COUT.store(c_out, std::sync::atomic::Ordering::Relaxed);
    SWEEP_NTT.store(use_ntt, std::sync::atomic::Ordering::Relaxed);

    let total_vars = c_in * h * w + c_out * c_in * kh * kw;
    let circuit = ConvBench {
        vals: vec![Variable::default(); total_vars],
    };
    let compile_result = expander_compiler::frontend::compile::<C, ConvBench<Variable>>(
        &circuit,
        CompileOptions::default(),
    )
    .unwrap();

    let circuit = compile_result.layered_circuit.export_to_expander_flatten();
    let mut total_mul = 0usize;
    let mut total_add = 0usize;
    for layer in &circuit.layers {
        total_mul += layer.mul.len();
        total_add += layer.add.len();
    }
    (total_mul, total_add)
}

fn main() {
    println!(
        "{:<6} {:<6} {:<4} {:<4} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "H", "W", "kh", "kw", "naive_mul", "naive_add", "ntt_mul", "ntt_add", "mul_Δ", "winner"
    );
    println!("{}", "-".repeat(86));

    let configs: Vec<(usize, usize, usize, usize)> = vec![
        (8, 8, 3, 3),
        (16, 16, 3, 3),
        (32, 32, 3, 3),
        (64, 64, 3, 3),
        (16, 16, 5, 5),
        (32, 32, 5, 5),
        (64, 64, 5, 5),
        (16, 16, 7, 7),
        (32, 32, 7, 7),
        (64, 64, 7, 7),
    ];

    let c_in = 3;
    let c_out = 8;

    for &(h, w, kh, kw) in &configs {
        let (naive_mul, naive_add) = run_config(c_in, c_out, h, w, kh, kw, false);
        let (ntt_mul, ntt_add) = run_config(c_in, c_out, h, w, kh, kw, true);

        let ratio = if naive_mul > 0 {
            format!("{:.1}x", naive_mul as f64 / ntt_mul.max(1) as f64)
        } else {
            "N/A".into()
        };

        let winner = if ntt_mul < naive_mul && ntt_add + ntt_mul < naive_add + naive_mul {
            "NTT"
        } else if ntt_mul < naive_mul {
            "NTT?"
        } else {
            "naive"
        };

        println!(
            "{:<6} {:<6} {:<4} {:<4} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
            h, w, kh, kw, naive_mul, naive_add, ntt_mul, ntt_add, ratio, winner
        );
    }
}
