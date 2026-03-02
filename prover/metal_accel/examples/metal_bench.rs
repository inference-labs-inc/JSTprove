use metal_accel::{MetalAccelerator, MetalBufferPool};
use std::time::Instant;

const GL_P: u64 = 0xFFFFFFFF00000001;

fn gl_add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    if carry || sum >= GL_P {
        sum.wrapping_sub(GL_P)
    } else {
        sum
    }
}

fn gl_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a.wrapping_sub(b).wrapping_add(GL_P)
    }
}

fn gl_mul(a: u64, b: u64) -> u64 {
    let full = a as u128 * b as u128;
    let lo = full as u64;
    let hi = (full >> 64) as u64;
    gl_reduce128(lo, hi)
}

fn gl_reduce128(lo: u64, hi: u64) -> u64 {
    let hi_lo = hi & 0xFFFFFFFF;
    let hi_hi = hi >> 32;
    let shifted = hi_lo << 32;

    let (s1, c1) = lo.overflowing_add(shifted);
    let (s2, c2) = s1.overflowing_sub(hi_lo);
    let (s3, c3) = s2.overflowing_sub(hi_hi);

    let carry = c1 as i64 - c2 as i64 - c3 as i64;
    let mut result = s3;
    if carry > 0 {
        let adj = carry as u64 * 0xFFFFFFFF;
        let (r, overflow) = result.overflowing_add(adj);
        result = r;
        if overflow {
            result = result.wrapping_sub(GL_P);
        }
    } else if carry < 0 {
        let adj = (-carry) as u64 * 0xFFFFFFFF;
        if result >= adj {
            result -= adj;
        } else {
            result = result.wrapping_add(GL_P).wrapping_sub(adj);
            if result >= GL_P {
                result -= GL_P;
            }
        }
    }
    if result >= GL_P {
        result -= GL_P;
    }
    result
}

fn cpu_build_eq_half(r: &[u64], mul_factor: u64, eq_evals: &mut [u64]) {
    eq_evals[0] = mul_factor;
    let mut cur = 1usize;
    for &r_i in r {
        for j in 0..cur {
            let prod = gl_mul(eq_evals[j], r_i);
            eq_evals[j + cur] = prod;
            eq_evals[j] = gl_sub(eq_evals[j], prod);
        }
        cur <<= 1;
    }
}

fn cpu_eq_eval_at(r: &[u64], mul_factor: u64, eq_evals: &mut [u64]) {
    let first_half_bits = r.len() / 2;
    let first_half_mask = (1usize << first_half_bits) - 1;
    let half_size = 1usize << (r.len() / 2 + 1);
    let mut first_half = vec![0u64; half_size];
    let mut second_half = vec![0u64; half_size];
    cpu_build_eq_half(&r[..first_half_bits], mul_factor, &mut first_half);
    cpu_build_eq_half(&r[first_half_bits..], 1, &mut second_half);
    for i in 0..(1usize << r.len()) {
        eq_evals[i] = gl_mul(
            first_half[i & first_half_mask],
            second_half[i >> first_half_bits],
        );
    }
}

fn cpu_poly_eval(bk_f: &[u64], bk_hg: &[u64], eval_size: usize) -> [u64; 3] {
    let mut p0 = 0u64;
    let mut p1 = 0u64;
    let mut p2 = 0u64;
    for i in 0..eval_size {
        let f0 = bk_f[i * 2];
        let f1 = bk_f[i * 2 + 1];
        let h0 = bk_hg[i * 2];
        let h1 = bk_hg[i * 2 + 1];
        p0 = gl_add(p0, gl_mul(h0, f0));
        p1 = gl_add(p1, gl_mul(h1, f1));
        p2 = gl_add(p2, gl_mul(gl_add(h0, h1), gl_add(f0, f1)));
    }
    [p0, p1, p2]
}

fn cpu_fold(bk: &mut [u64], r: u64, eval_size: usize) {
    for i in 0..eval_size {
        let v0 = bk[i * 2];
        let v1 = bk[i * 2 + 1];
        bk[i] = gl_add(v0, gl_mul(gl_sub(v1, v0), r));
    }
}

fn rand_field_vec(n: usize) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        v.push(h.finish() % GL_P);
    }
    v
}

fn bench_eq_eval(accel: &MetalAccelerator, log_n: usize) {
    let n = 1 << log_n;
    let r: Vec<u64> = rand_field_vec(log_n);

    let mut cpu_result = vec![0u64; n];
    let start = Instant::now();
    for _ in 0..5 {
        cpu_eq_eval_at(&r, 1, &mut cpu_result);
    }
    let cpu_us = start.elapsed().as_micros() / 5;

    let pool = MetalBufferPool::new(accel.device(), n);
    let start = Instant::now();
    for _ in 0..5 {
        metal_accel::metal_eq_eval_at(
            accel,
            &r,
            1,
            &pool.eq_first_half,
            &pool.eq_second_half,
            &pool.eq_evals_rz0,
        );
    }
    let metal_us = start.elapsed().as_micros() / 5;

    let gpu_ptr = pool.eq_evals_rz0.contents() as *const u64;
    let gpu_val = unsafe { *gpu_ptr };
    let match_str = if gpu_val == cpu_result[0] {
        "MATCH"
    } else {
        "MISMATCH"
    };

    let speedup = cpu_us as f64 / metal_us as f64;
    println!(
        "  eq_eval  2^{log_n:>2} ({n:>8}):  CPU {cpu_us:>8}us  Metal {metal_us:>8}us  {speedup:>6.2}x  [{match_str}]"
    );
}

fn bench_poly_eval(accel: &MetalAccelerator, log_n: usize) {
    let n = 1 << log_n;
    let eval_size = n / 2;
    let bk_f = rand_field_vec(n);
    let bk_hg = rand_field_vec(n);
    let gate_exists: Vec<u32> = (0..n).map(|_| 1u32).collect();

    let start = Instant::now();
    let mut cpu_result = [0u64; 3];
    for _ in 0..5 {
        cpu_result = cpu_poly_eval(&bk_f, &bk_hg, eval_size);
    }
    let cpu_us = start.elapsed().as_micros() / 5;

    let opts = metal::MTLResourceOptions::StorageModeShared;
    let f_buf =
        accel
            .device()
            .new_buffer_with_data(bk_f.as_ptr() as *const _, (n * 8) as u64, opts);
    let hg_buf =
        accel
            .device()
            .new_buffer_with_data(bk_hg.as_ptr() as *const _, (n * 8) as u64, opts);
    let ge_buf =
        accel
            .device()
            .new_buffer_with_data(gate_exists.as_ptr() as *const _, (n * 4) as u64, opts);
    let max_blocks = (eval_size + 255) / 256;
    let block_buf = accel.device().new_buffer((max_blocks * 3 * 8) as u64, opts);
    let out_buf = accel.device().new_buffer(24, opts);

    let start = Instant::now();
    let mut gpu_result = [0u64; 3];
    for _ in 0..5 {
        gpu_result = metal_accel::metal_poly_eval(
            accel,
            &f_buf,
            &hg_buf,
            &ge_buf,
            &block_buf,
            &out_buf,
            eval_size as u32,
        );
    }
    let metal_us = start.elapsed().as_micros() / 5;

    let match_str = if gpu_result == cpu_result {
        "MATCH"
    } else {
        eprintln!("    CPU: {:?}", cpu_result);
        eprintln!("    GPU: {:?}", gpu_result);
        "MISMATCH"
    };
    let speedup = cpu_us as f64 / metal_us as f64;
    println!(
        "  poly_eval 2^{log_n:>2} ({n:>8}):  CPU {cpu_us:>8}us  Metal {metal_us:>8}us  {speedup:>6.2}x  [{match_str}]"
    );
}

fn bench_fold(accel: &MetalAccelerator, log_n: usize) {
    let n = 1 << log_n;
    let eval_size = n / 2;
    let r = rand_field_vec(1)[0];

    let mut cpu_data = rand_field_vec(n);
    let cpu_copy = cpu_data.clone();

    let start = Instant::now();
    for _ in 0..5 {
        cpu_data.copy_from_slice(&cpu_copy);
        cpu_fold(&mut cpu_data, r, eval_size);
    }
    let cpu_us = start.elapsed().as_micros() / 5;

    let opts = metal::MTLResourceOptions::StorageModeShared;
    let gpu_in =
        accel
            .device()
            .new_buffer_with_data(cpu_copy.as_ptr() as *const _, (n * 8) as u64, opts);
    let gpu_out = accel.device().new_buffer((n * 8) as u64, opts);
    let r_buf = accel.device().new_buffer(8, opts);
    let r_ptr = r_buf.contents() as *mut u64;
    unsafe {
        *r_ptr = r;
    }

    let start = Instant::now();
    for _ in 0..5 {
        let ptr = gpu_in.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_copy.as_ptr() as *const u8, ptr, n * 8);
        }
        metal_accel::metal_fold_f(accel, &gpu_in, &gpu_out, &r_buf, eval_size as u32);
    }
    let metal_us = start.elapsed().as_micros() / 5;

    let gpu_ptr = gpu_out.contents() as *const u64;
    let gpu_val = unsafe { *gpu_ptr };
    let match_str = if gpu_val == cpu_data[0] {
        "MATCH"
    } else {
        "MISMATCH"
    };

    let speedup = cpu_us as f64 / metal_us as f64;
    println!(
        "  fold    2^{log_n:>2} ({n:>8}):  CPU {cpu_us:>8}us  Metal {metal_us:>8}us  {speedup:>6.2}x  [{match_str}]"
    );
}

fn main() {
    let accel = MetalAccelerator::new().expect("Metal GPU not available");
    println!("Metal device: {}", accel.device().name());
    println!();

    println!("=== EQ Polynomial Evaluation ===");
    for log_n in [12, 14, 16, 18, 20] {
        bench_eq_eval(&accel, log_n);
    }

    println!();
    println!("=== Polynomial Evaluation (map-reduce) ===");
    for log_n in [12, 14, 16, 18, 20] {
        bench_poly_eval(&accel, log_n);
    }

    println!();
    println!("=== Domain Folding ===");
    for log_n in [12, 14, 16, 18, 20] {
        bench_fold(&accel, log_n);
    }
}
