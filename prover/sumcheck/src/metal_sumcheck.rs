use std::cell::RefCell;

use metal_accel::{MetalAccelerator, MetalBufferPool};

pub struct MetalSumcheckCtx {
    pub accel: MetalAccelerator,
    pub pool: MetalBufferPool,
}

thread_local! {
    pub static METAL_CTX: RefCell<Option<MetalSumcheckCtx>> = const { RefCell::new(None) };
}

pub fn init_metal_ctx(max_input_size: usize) {
    METAL_CTX.with(|cell| {
        let mut ctx = cell.borrow_mut();
        let needs_init = match ctx.as_ref() {
            None => true,
            Some(existing) => existing.pool.max_input_size() < max_input_size,
        };
        if needs_init {
            let accel = match ctx.take() {
                Some(existing) => existing.accel,
                None => match MetalAccelerator::new() {
                    Some(a) => a,
                    None => return,
                },
            };
            let pool = MetalBufferPool::new(accel.device(), max_input_size);
            *ctx = Some(MetalSumcheckCtx { accel, pool });
            log::info!("Metal sumcheck initialized (max_input_size={max_input_size})");
        }
    });
}

pub fn metal_available() -> bool {
    METAL_CTX.with(|cell| cell.borrow().is_some())
}
