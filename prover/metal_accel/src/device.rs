use metal::{CompileOptions, ComputePipelineState, Device, MTLSize};
use std::collections::HashMap;

const SHADER_SOURCE: &str = concat!(
    include_str!("../shaders/field_goldilocks.metal"),
    "\n",
    include_str!("../shaders/eq_eval.metal"),
    "\n",
    include_str!("../shaders/gate_accumulate.metal"),
    "\n",
    include_str!("../shaders/poly_eval.metal"),
    "\n",
    include_str!("../shaders/sumcheck_fold.metal"),
);

pub const THREADGROUP_SIZE: u64 = 256;
pub const GPU_DISPATCH_THRESHOLD: usize = 1 << 12;

pub struct MetalAccelerator {
    pub(crate) device: Device,
    pub(crate) queue: metal::CommandQueue,
    pub(crate) pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalAccelerator {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = match device.new_library_with_source(SHADER_SOURCE, &opts) {
            Ok(lib) => lib,
            Err(e) => {
                log::error!("Metal shader compilation failed: {e}");
                return None;
            }
        };

        let kernel_names = &[
            "build_eq_step",
            "cross_prod_eq",
            "accumulate_mul_gates",
            "accumulate_add_gates",
            "poly_eval_kernel",
            "reduce_blocks",
            "fold_f",
            "fold_hg",
        ];

        let mut pipelines = HashMap::new();
        for &name in kernel_names {
            let func = library
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("Metal function '{name}' not found: {e}"));
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .unwrap_or_else(|e| panic!("Pipeline creation failed for '{name}': {e}"));
            pipelines.insert(name, pipeline);
        }

        log::info!(
            "Metal accelerator initialized: {} ({}MB)",
            device.name(),
            device.recommended_max_working_set_size() / (1024 * 1024)
        );

        Some(MetalAccelerator {
            device,
            queue,
            pipelines,
        })
    }

    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    pub(crate) fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("pipeline '{name}' not registered"))
    }

    pub(crate) fn dispatch_1d(
        &self,
        pipeline_name: &str,
        total_threads: u64,
        buffers: &[&metal::BufferRef],
    ) {
        let pipeline = self.pipeline(pipeline_name);
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }

        let tg_size = THREADGROUP_SIZE.min(total_threads);
        let grid_size = MTLSize::new(total_threads, 1, 1);
        let tg = MTLSize::new(tg_size, 1, 1);
        encoder.dispatch_threads(grid_size, tg);
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
}
