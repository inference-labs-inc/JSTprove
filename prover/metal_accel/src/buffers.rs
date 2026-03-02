use metal::{Buffer, Device, MTLResourceOptions};

pub struct MetalBufferPool {
    pub hg_evals: Buffer,
    pub v_evals: Buffer,
    pub eq_evals_rz0: Buffer,
    pub eq_evals_rx: Buffer,
    pub eq_first_half: Buffer,
    pub eq_second_half: Buffer,
    pub gate_exists: Buffer,
    pub mul_gates: Option<Buffer>,
    pub add_gates: Option<Buffer>,
    pub block_results: Buffer,
    pub output: Buffer,
    pub challenge: Buffer,
}

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

impl MetalBufferPool {
    pub fn new(device: &Device, max_input_size: usize) -> Self {
        let elem_size = std::mem::size_of::<u64>();
        let max_bytes = max_input_size * elem_size;
        let half_size = (max_input_size as f64).sqrt().ceil() as usize + 1;
        let half_bytes = half_size * elem_size;
        let max_blocks = (max_input_size + 255) / 256;

        MetalBufferPool {
            hg_evals: device.new_buffer(max_bytes as u64, SHARED),
            v_evals: device.new_buffer(max_bytes as u64, SHARED),
            eq_evals_rz0: device.new_buffer(max_bytes as u64, SHARED),
            eq_evals_rx: device.new_buffer(max_bytes as u64, SHARED),
            eq_first_half: device.new_buffer(half_bytes as u64, SHARED),
            eq_second_half: device.new_buffer(half_bytes as u64, SHARED),
            gate_exists: device.new_buffer((max_input_size * 4) as u64, SHARED),
            mul_gates: None,
            add_gates: None,
            block_results: device.new_buffer((max_blocks * 3 * elem_size) as u64, SHARED),
            output: device.new_buffer((3 * elem_size) as u64, SHARED),
            challenge: device.new_buffer(elem_size as u64, SHARED),
        }
    }

    pub fn upload_mul_gates(&mut self, device: &Device, data: &[u8]) {
        let buf = device.new_buffer_with_data(data.as_ptr() as *const _, data.len() as u64, SHARED);
        self.mul_gates = Some(buf);
    }

    pub fn upload_add_gates(&mut self, device: &Device, data: &[u8]) {
        let buf = device.new_buffer_with_data(data.as_ptr() as *const _, data.len() as u64, SHARED);
        self.add_gates = Some(buf);
    }

    pub fn write_challenge(&self, val: u64) {
        let ptr = self.challenge.contents() as *mut u64;
        unsafe {
            *ptr = val;
        }
    }

    pub fn read_output(&self) -> [u64; 3] {
        let ptr = self.output.contents() as *const u64;
        unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] }
    }

    pub fn zero_buffer(buf: &Buffer, num_bytes: usize) {
        let ptr = buf.contents() as *mut u8;
        unsafe {
            std::ptr::write_bytes(ptr, 0, num_bytes);
        }
    }
}
