#![allow(clippy::all, clippy::pedantic)]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal_bench only runs on macOS");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use halo2curves::bn256::Fr;
    use halo2curves::ff::Field;
    use metal_accel::{MetalAccelerator, MetalBufferPool, BN254_ELEM_SIZE, BN254_R};
    use std::time::Instant;

    fn fr_to_limbs(f: &Fr) -> [u64; 4] {
        use halo2curves::serde::SerdeObject;
        let repr: [u8; 32] = f.to_raw_bytes();
        [
            u64::from_le_bytes(repr[0..8].try_into().unwrap()),
            u64::from_le_bytes(repr[8..16].try_into().unwrap()),
            u64::from_le_bytes(repr[16..24].try_into().unwrap()),
            u64::from_le_bytes(repr[24..32].try_into().unwrap()),
        ]
    }

    fn rand_fr_vec(n: usize) -> Vec<Fr> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        (0..n)
            .map(|i| {
                let mut limbs = [0u64; 4];
                for (j, limb) in limbs.iter_mut().enumerate() {
                    let mut h = DefaultHasher::new();
                    (i, j).hash(&mut h);
                    *limb = h.finish();
                }
                limbs[3] &= 0x0FFFFFFFFFFFFFFF;
                Fr::from_raw(limbs)
            })
            .collect()
    }

    fn fr_vec_to_limbs(v: &[Fr]) -> Vec<[u64; 4]> {
        v.iter().map(fr_to_limbs).collect()
    }

    fn cpu_build_eq_half(r: &[Fr], mul_factor: Fr, eq_evals: &mut [Fr]) {
        eq_evals[0] = mul_factor;
        let mut cur = 1usize;
        for &r_i in r {
            for j in 0..cur {
                let prod = eq_evals[j] * r_i;
                eq_evals[j + cur] = prod;
                eq_evals[j] -= prod;
            }
            cur <<= 1;
        }
    }

    fn cpu_eq_eval_at(r: &[Fr], mul_factor: Fr, eq_evals: &mut [Fr]) {
        let first_half_bits = r.len() / 2;
        let first_half_mask = (1usize << first_half_bits) - 1;
        let half_size = 1usize << (r.len() / 2 + 1);
        let mut first_half = vec![Fr::ZERO; half_size];
        let mut second_half = vec![Fr::ZERO; half_size];
        cpu_build_eq_half(&r[..first_half_bits], mul_factor, &mut first_half);
        cpu_build_eq_half(&r[first_half_bits..], Fr::ONE, &mut second_half);
        for i in 0..(1usize << r.len()) {
            eq_evals[i] = first_half[i & first_half_mask] * second_half[i >> first_half_bits];
        }
    }

    fn cpu_poly_eval(bk_f: &[Fr], bk_hg: &[Fr], eval_size: usize) -> [Fr; 3] {
        let mut p0 = Fr::ZERO;
        let mut p1 = Fr::ZERO;
        let mut p2 = Fr::ZERO;
        for i in 0..eval_size {
            let f0 = bk_f[i * 2];
            let f1 = bk_f[i * 2 + 1];
            let h0 = bk_hg[i * 2];
            let h1 = bk_hg[i * 2 + 1];
            p0 += h0 * f0;
            p1 += h1 * f1;
            p2 += (h0 + h1) * (f0 + f1);
        }
        [p0, p1, p2]
    }

    fn cpu_fold(bk: &mut [Fr], r: Fr, eval_size: usize) {
        for i in 0..eval_size {
            let v0 = bk[i * 2];
            let v1 = bk[i * 2 + 1];
            bk[i] = v0 + (v1 - v0) * r;
        }
    }

    fn upload_fr_slice(device: &metal::Device, data: &[[u64; 4]]) -> metal::Buffer {
        let opts = metal::MTLResourceOptions::StorageModeShared;
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * BN254_ELEM_SIZE) as u64,
            opts,
        )
    }

    fn bench_eq_eval(accel: &MetalAccelerator, log_n: usize) {
        let n = 1 << log_n;
        let r = rand_fr_vec(log_n);

        let mut cpu_result = vec![Fr::ZERO; n];
        let start = Instant::now();
        for _ in 0..5 {
            cpu_eq_eval_at(&r, Fr::ONE, &mut cpu_result);
        }
        let cpu_us = start.elapsed().as_micros() / 5;

        let r_limbs: Vec<[u64; 4]> = fr_vec_to_limbs(&r);
        let pool = MetalBufferPool::new(accel.device(), n);
        let start = Instant::now();
        for _ in 0..5 {
            metal_accel::metal_eq_eval_at(
                accel,
                &r_limbs,
                &BN254_R,
                &pool.eq_first_half,
                &pool.eq_second_half,
                &pool.eq_evals_rz0,
            );
        }
        let metal_us = start.elapsed().as_micros() / 5;

        let gpu_ptr = pool.eq_evals_rz0.contents() as *const [u64; 4];
        let gpu_slice: &[[u64; 4]] = unsafe { std::slice::from_raw_parts(gpu_ptr, n) };
        let cpu_limbs: Vec<[u64; 4]> = cpu_result.iter().map(fr_to_limbs).collect();
        let match_str = if gpu_slice == cpu_limbs.as_slice() {
            "MATCH"
        } else {
            if gpu_slice.len() != cpu_limbs.len() {
                eprintln!(
                    "    length mismatch: GPU={} CPU={}",
                    gpu_slice.len(),
                    cpu_limbs.len()
                );
            } else if let Some(i) = gpu_slice
                .iter()
                .zip(cpu_limbs.iter())
                .position(|(g, c)| g != c)
            {
                eprintln!("    first mismatch at index {i}");
                eprintln!("    CPU[{i}]: {:?}", cpu_limbs[i]);
                eprintln!("    GPU[{i}]: {:?}", gpu_slice[i]);
            }
            "MISMATCH"
        };

        let speedup_str = if metal_us == 0 {
            "   N/A".to_string()
        } else {
            format!("{:>6.2}x", cpu_us as f64 / metal_us as f64)
        };
        println!(
            "  eq_eval  2^{log_n:>2} ({n:>8}):  CPU {cpu_us:>8}us  Metal {metal_us:>8}us  {speedup_str}  [{match_str}]"
        );
    }

    fn bench_poly_eval(accel: &MetalAccelerator, log_n: usize) {
        let n = 1 << log_n;
        let eval_size = n / 2;
        let bk_f = rand_fr_vec(n);
        let bk_hg = rand_fr_vec(n);
        let gate_exists: Vec<u32> = vec![1u32; n];

        let start = Instant::now();
        let mut cpu_result = [Fr::ZERO; 3];
        for _ in 0..5 {
            cpu_result = cpu_poly_eval(&bk_f, &bk_hg, eval_size);
        }
        let cpu_us = start.elapsed().as_micros() / 5;

        let f_limbs = fr_vec_to_limbs(&bk_f);
        let hg_limbs = fr_vec_to_limbs(&bk_hg);
        let f_buf = upload_fr_slice(accel.device(), &f_limbs);
        let hg_buf = upload_fr_slice(accel.device(), &hg_limbs);
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let ge_buf = accel.device().new_buffer_with_data(
            gate_exists.as_ptr() as *const _,
            (n * 4) as u64,
            opts,
        );
        let max_blocks = (eval_size + 255) / 256;
        let block_buf = accel
            .device()
            .new_buffer((max_blocks * 3 * BN254_ELEM_SIZE) as u64, opts);
        let out_buf = accel
            .device()
            .new_buffer((3 * BN254_ELEM_SIZE) as u64, opts);

        let start = Instant::now();
        let mut gpu_result = [[0u64; 4]; 3];
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

        let cpu_limbs: [[u64; 4]; 3] = [
            fr_to_limbs(&cpu_result[0]),
            fr_to_limbs(&cpu_result[1]),
            fr_to_limbs(&cpu_result[2]),
        ];
        let match_str = if gpu_result == cpu_limbs {
            "MATCH"
        } else {
            eprintln!("    CPU: {:?}", cpu_limbs);
            eprintln!("    GPU: {:?}", gpu_result);
            "MISMATCH"
        };
        let speedup_str = if metal_us == 0 {
            "   N/A".to_string()
        } else {
            format!("{:>6.2}x", cpu_us as f64 / metal_us as f64)
        };
        println!(
            "  poly_eval 2^{log_n:>2} ({n:>8}):  CPU {cpu_us:>8}us  Metal {metal_us:>8}us  {speedup_str}  [{match_str}]"
        );
    }

    fn bench_fold(accel: &MetalAccelerator, log_n: usize) {
        let n = 1 << log_n;
        let eval_size = n / 2;
        let r_vec = rand_fr_vec(1);
        let r = r_vec[0];

        let mut cpu_data = rand_fr_vec(n);
        let cpu_copy = cpu_data.clone();

        let start = Instant::now();
        for _ in 0..5 {
            cpu_data.copy_from_slice(&cpu_copy);
            cpu_fold(&mut cpu_data, r, eval_size);
        }
        let cpu_us = start.elapsed().as_micros() / 5;

        let copy_limbs = fr_vec_to_limbs(&cpu_copy);
        let r_limbs = fr_to_limbs(&r);
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let gpu_in = upload_fr_slice(accel.device(), &copy_limbs);
        let gpu_out = accel
            .device()
            .new_buffer((n * BN254_ELEM_SIZE) as u64, opts);
        let r_buf = accel.device().new_buffer(BN254_ELEM_SIZE as u64, opts);
        let r_ptr = r_buf.contents() as *mut [u64; 4];
        unsafe {
            *r_ptr = r_limbs;
        }

        let start = Instant::now();
        for _ in 0..5 {
            let ptr = gpu_in.contents() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    copy_limbs.as_ptr() as *const u8,
                    ptr,
                    n * BN254_ELEM_SIZE,
                );
            }
            metal_accel::metal_fold_f(accel, &gpu_in, &gpu_out, &r_buf, eval_size as u32);
        }
        let metal_us = start.elapsed().as_micros() / 5;

        let gpu_ptr = gpu_out.contents() as *const [u64; 4];
        let gpu_slice: &[[u64; 4]] = unsafe { std::slice::from_raw_parts(gpu_ptr, eval_size) };
        let cpu_limbs: Vec<[u64; 4]> = cpu_data[..eval_size].iter().map(fr_to_limbs).collect();
        let match_str = if gpu_slice == cpu_limbs.as_slice() {
            "MATCH"
        } else {
            if gpu_slice.len() != cpu_limbs.len() {
                eprintln!(
                    "    length mismatch: GPU={} CPU={}",
                    gpu_slice.len(),
                    cpu_limbs.len()
                );
            } else if let Some(i) = gpu_slice
                .iter()
                .zip(cpu_limbs.iter())
                .position(|(g, c)| g != c)
            {
                eprintln!("    first mismatch at index {i}");
                eprintln!("    CPU[{i}]: {:?}", cpu_limbs[i]);
                eprintln!("    GPU[{i}]: {:?}", gpu_slice[i]);
            }
            "MISMATCH"
        };

        let speedup_str = if metal_us == 0 {
            "   N/A".to_string()
        } else {
            format!("{:>6.2}x", cpu_us as f64 / metal_us as f64)
        };
        println!(
            "  fold    2^{log_n:>2} ({n:>8}):  CPU {cpu_us:>8}us  Metal {metal_us:>8}us  {speedup_str}  [{match_str}]"
        );
    }

    pub fn run() {
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
}
