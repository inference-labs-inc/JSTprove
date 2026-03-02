use metal::{Buffer, Device, MTLResourceOptions};

pub const BN254_ELEM_SIZE: usize = 4 * std::mem::size_of::<u64>();

pub struct MetalBufferPool {
    max_input_size: usize,
    pub hg_evals: Buffer,
    pub v_evals: Buffer,
    pub eq_evals_rz0: Buffer,
    pub eq_evals_rx: Buffer,
    pub eq_first_half: Buffer,
    pub eq_second_half: Buffer,
    pub gate_exists: Buffer,
    pub hg_locks: Buffer,
    pub mul_gates: Option<Buffer>,
    pub add_gates: Option<Buffer>,
    pub block_results: Buffer,
    pub output: Buffer,
    pub challenge: Buffer,
    pub fold_scratch: Buffer,
    pub fold_ge_scratch: Buffer,
}

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

impl MetalBufferPool {
    pub fn new(device: &Device, max_input_size: usize) -> Self {
        let max_bytes = max_input_size
            .checked_mul(BN254_ELEM_SIZE)
            .expect("max_input_size * BN254_ELEM_SIZE overflows usize");
        let half_size = (max_input_size as f64).sqrt().ceil() as usize;
        let half_size = half_size
            .checked_add(1)
            .expect("half_size + 1 overflows usize");
        let half_bytes = half_size
            .checked_mul(BN254_ELEM_SIZE)
            .expect("half_size * BN254_ELEM_SIZE overflows usize");
        let max_blocks = max_input_size
            .checked_add(255)
            .expect("max_input_size + 255 overflows usize")
            / 256;
        let ge_bytes = max_input_size
            .checked_mul(4)
            .expect("max_input_size * 4 overflows usize");

        MetalBufferPool {
            max_input_size,
            hg_evals: device.new_buffer(max_bytes as u64, SHARED),
            v_evals: device.new_buffer(max_bytes as u64, SHARED),
            eq_evals_rz0: device.new_buffer(max_bytes as u64, SHARED),
            eq_evals_rx: device.new_buffer(max_bytes as u64, SHARED),
            eq_first_half: device.new_buffer(half_bytes as u64, SHARED),
            eq_second_half: device.new_buffer(half_bytes as u64, SHARED),
            gate_exists: device.new_buffer(ge_bytes as u64, SHARED),
            hg_locks: device.new_buffer(ge_bytes as u64, SHARED),
            mul_gates: None,
            add_gates: None,
            block_results: device.new_buffer(
                (max_blocks
                    .checked_mul(3)
                    .expect("max_blocks * 3 overflows")
                    .checked_mul(BN254_ELEM_SIZE)
                    .expect("max_blocks * 3 * BN254_ELEM_SIZE overflows")) as u64,
                SHARED,
            ),
            output: device.new_buffer((3 * BN254_ELEM_SIZE) as u64, SHARED),
            challenge: device.new_buffer(BN254_ELEM_SIZE as u64, SHARED),
            fold_scratch: device.new_buffer(max_bytes as u64, SHARED),
            fold_ge_scratch: device.new_buffer(ge_bytes as u64, SHARED),
        }
    }

    pub fn max_input_size(&self) -> usize {
        self.max_input_size
    }

    pub fn upload_mul_gates(&mut self, device: &Device, data: &[u8]) {
        let buf = device.new_buffer_with_data(data.as_ptr() as *const _, data.len() as u64, SHARED);
        self.mul_gates = Some(buf);
    }

    pub fn upload_add_gates(&mut self, device: &Device, data: &[u8]) {
        let buf = device.new_buffer_with_data(data.as_ptr() as *const _, data.len() as u64, SHARED);
        self.add_gates = Some(buf);
    }

    pub fn write_challenge(&self, val: &[u64; 4]) {
        let ptr = self.challenge.contents() as *mut [u64; 4];
        unsafe {
            *ptr = *val;
        }
    }

    pub fn read_output(&self) -> [[u64; 4]; 3] {
        let ptr = self.output.contents() as *const [u64; 4];
        unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] }
    }

    pub fn zero_buffer(buf: &Buffer, num_bytes: usize) {
        let buf_len = buf.length() as usize;
        assert!(
            num_bytes <= buf_len,
            "zero_buffer: num_bytes ({num_bytes}) exceeds buffer length ({buf_len})"
        );
        let ptr = buf.contents() as *mut u8;
        assert!(
            !ptr.is_null(),
            "zero_buffer: buffer contents pointer is null"
        );
        unsafe {
            std::ptr::write_bytes(ptr, 0, num_bytes);
        }
    }
}
