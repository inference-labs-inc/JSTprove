use metal::{Buffer, MTLSize};

use crate::buffers::BN254_ELEM_SIZE;
use crate::device::MetalAccelerator;
use crate::device::THREADGROUP_SIZE;

pub const BN254_R: [u64; 4] = [
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
];

fn write_u32_constant(device: &metal::Device, val: u32) -> Buffer {
    let buf = device.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
    let ptr = buf.contents() as *mut u32;
    unsafe {
        *ptr = val;
    }
    buf
}

pub fn metal_eq_eval_at(
    accel: &MetalAccelerator,
    r: &[[u64; 4]],
    mul_factor: &[u64; 4],
    eq_first_half: &metal::BufferRef,
    eq_second_half: &metal::BufferRef,
    eq_evals: &metal::BufferRef,
) {
    assert!(
        r.len() < 63,
        "r.len() ({}) must be < 63 so each half is < 32",
        r.len()
    );

    let first_half_bits = r.len() / 2;
    let r_first = &r[0..first_half_bits];
    let r_second = &r[first_half_bits..];

    let cmd_buffer = accel.queue.new_command_buffer();

    build_eq_half_batched(accel, r_first, mul_factor, eq_first_half, cmd_buffer);
    build_eq_half_batched(accel, r_second, &BN254_R, eq_second_half, cmd_buffer);

    {
        let total = 1u64 << r.len();
        let bits_const = write_u32_constant(&accel.device, first_half_bits as u32);
        let pipeline = accel.pipeline("cross_prod_eq");
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(eq_evals), 0);
        encoder.set_buffer(1, Some(eq_first_half), 0);
        encoder.set_buffer(2, Some(eq_second_half), 0);
        encoder.set_buffer(3, Some(&bits_const), 0);
        let tg_size = THREADGROUP_SIZE.min(total);
        let grid = MTLSize::new(total, 1, 1);
        let tg = MTLSize::new(tg_size, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}

fn build_eq_half_batched(
    accel: &MetalAccelerator,
    r: &[[u64; 4]],
    mul_factor: &[u64; 4],
    eq_buf: &metal::BufferRef,
    cmd_buffer: &metal::CommandBufferRef,
) {
    assert!(
        r.len() < 32,
        "r.len() ({}) must be < 32 to avoid u32 shift overflow",
        r.len()
    );

    let ptr = eq_buf.contents() as *mut [u64; 4];
    unsafe {
        *ptr = *mul_factor;
    }

    if r.is_empty() {
        return;
    }

    let r_buf = accel.device.new_buffer(
        (r.len() * BN254_ELEM_SIZE) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let r_ptr = r_buf.contents() as *mut [u64; 4];
    for (i, r_i) in r.iter().enumerate() {
        unsafe {
            *r_ptr.add(i) = *r_i;
        }
    }

    let cur_buf = accel.device.new_buffer(
        (r.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let cur_ptr = cur_buf.contents() as *mut u32;
    let mut cur = 1u32;
    for i in 0..r.len() {
        unsafe {
            *cur_ptr.add(i) = cur;
        }
        cur = cur
            .checked_shl(1)
            .expect("build_eq_half: cur_eval_num overflow");
    }

    let pipeline = accel.pipeline("build_eq_step");
    cur = 1u32;
    for i in 0..r.len() {
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(eq_buf), 0);
        encoder.set_buffer(1, Some(&r_buf), (i * BN254_ELEM_SIZE) as u64);
        encoder.set_buffer(2, Some(&cur_buf), (i * 4) as u64);
        let tg_size = THREADGROUP_SIZE.min(cur as u64);
        let grid = MTLSize::new(cur as u64, 1, 1);
        let tg = MTLSize::new(tg_size, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
        cur = cur
            .checked_shl(1)
            .expect("build_eq_half: cur_eval_num overflow");
    }
}

pub fn metal_cross_prod_eq(
    accel: &MetalAccelerator,
    eq_evals: &metal::BufferRef,
    first_half: &metal::BufferRef,
    second_half: &metal::BufferRef,
    first_half_bits: u32,
    total: u64,
) {
    let bits_const = write_u32_constant(&accel.device, first_half_bits);
    accel.dispatch_1d(
        "cross_prod_eq",
        total,
        &[eq_evals, first_half, second_half, &bits_const],
    );
}

pub fn metal_vec_add(
    accel: &MetalAccelerator,
    dst: &metal::BufferRef,
    src: &metal::BufferRef,
    count: u32,
) {
    if count == 0 {
        return;
    }
    let n_const = write_u32_constant(&accel.device, count);
    accel.dispatch_1d("vec_add", count as u64, &[dst, src, &n_const]);
}

pub fn metal_poly_eval(
    accel: &MetalAccelerator,
    bk_f: &metal::BufferRef,
    bk_hg: &metal::BufferRef,
    gate_exists: &metal::BufferRef,
    block_results: &metal::BufferRef,
    output: &metal::BufferRef,
    eval_size: u32,
) -> [[u64; 4]; 3] {
    let num_threadgroups = ((eval_size as u64) + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    let total_threads = num_threadgroups * THREADGROUP_SIZE;
    let eval_const = write_u32_constant(&accel.device, eval_size);
    let nb_const = write_u32_constant(&accel.device, num_threadgroups as u32);

    let cmd_buffer = accel.queue.new_command_buffer();

    {
        let pipeline = accel.pipeline("poly_eval_kernel");
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(bk_f), 0);
        encoder.set_buffer(1, Some(bk_hg), 0);
        encoder.set_buffer(2, Some(gate_exists), 0);
        encoder.set_buffer(3, Some(block_results), 0);
        encoder.set_buffer(4, Some(&eval_const), 0);
        let grid = MTLSize::new(total_threads, 1, 1);
        let tg = MTLSize::new(THREADGROUP_SIZE, 1, 1);
        encoder.dispatch_threads(grid, tg);
        encoder.end_encoding();
    }

    {
        let reduce_pipeline = accel.pipeline("reduce_blocks");
        let encoder2 = cmd_buffer.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(reduce_pipeline);
        encoder2.set_buffer(0, Some(block_results), 0);
        encoder2.set_buffer(1, Some(output), 0);
        encoder2.set_buffer(2, Some(&nb_const), 0);
        let tg = MTLSize::new(THREADGROUP_SIZE, 1, 1);
        let reduce_grid = MTLSize::new(THREADGROUP_SIZE, 1, 1);
        encoder2.dispatch_threads(reduce_grid, tg);
        encoder2.end_encoding();
    }

    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    let ptr = output.contents() as *const [u64; 4];
    unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] }
}

pub fn metal_fold_all(
    accel: &MetalAccelerator,
    bk_f_in: &metal::BufferRef,
    bk_f_out: &metal::BufferRef,
    bk_hg_in: &metal::BufferRef,
    bk_hg_out: &metal::BufferRef,
    gate_exists_in: &metal::BufferRef,
    gate_exists_out: &metal::BufferRef,
    challenge: &metal::BufferRef,
    eval_size: u32,
) {
    if eval_size == 0 {
        return;
    }
    let eval_const = write_u32_constant(&accel.device, eval_size);
    let cmd_buffer = accel.queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();

    let tg_size = THREADGROUP_SIZE.min(eval_size as u64);
    let grid = MTLSize::new(eval_size as u64, 1, 1);
    let tg = MTLSize::new(tg_size, 1, 1);

    encoder.set_compute_pipeline_state(accel.pipeline("fold_f"));
    encoder.set_buffer(0, Some(bk_f_in), 0);
    encoder.set_buffer(1, Some(bk_f_out), 0);
    encoder.set_buffer(2, Some(challenge), 0);
    encoder.set_buffer(3, Some(&eval_const), 0);
    encoder.dispatch_threads(grid, tg);

    encoder.set_compute_pipeline_state(accel.pipeline("fold_hg"));
    encoder.set_buffer(0, Some(bk_hg_in), 0);
    encoder.set_buffer(1, Some(bk_hg_out), 0);
    encoder.set_buffer(2, Some(gate_exists_in), 0);
    encoder.set_buffer(3, Some(gate_exists_out), 0);
    encoder.set_buffer(4, Some(challenge), 0);
    encoder.set_buffer(5, Some(&eval_const), 0);
    encoder.dispatch_threads(grid, tg);

    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}

pub fn metal_fold_f(
    accel: &MetalAccelerator,
    bk_f_in: &metal::BufferRef,
    bk_f_out: &metal::BufferRef,
    challenge: &metal::BufferRef,
    eval_size: u32,
) {
    let eval_const = write_u32_constant(&accel.device, eval_size);
    accel.dispatch_1d(
        "fold_f",
        eval_size as u64,
        &[bk_f_in, bk_f_out, challenge, &eval_const],
    );
}

pub fn metal_fold_hg(
    accel: &MetalAccelerator,
    bk_hg_in: &metal::BufferRef,
    bk_hg_out: &metal::BufferRef,
    gate_exists_in: &metal::BufferRef,
    gate_exists_out: &metal::BufferRef,
    challenge: &metal::BufferRef,
    eval_size: u32,
) {
    let eval_const = write_u32_constant(&accel.device, eval_size);
    accel.dispatch_1d(
        "fold_hg",
        eval_size as u64,
        &[
            bk_hg_in,
            bk_hg_out,
            gate_exists_in,
            gate_exists_out,
            challenge,
            &eval_const,
        ],
    );
}
