use metal::{CompileOptions, ComputePipelineState, Device, MTLSize};
use std::collections::HashMap;

const SHADER_SOURCE_BN254: &str = concat!(
    include_str!("../shaders/field_bn254.metal"),
    "\n",
    include_str!("../shaders/eq_eval.metal"),
    "\n",
    include_str!("../shaders/gate_accumulate.metal"),
    "\n",
    include_str!("../shaders/poly_eval.metal"),
    "\n",
    include_str!("../shaders/sumcheck_fold.metal"),
);

const SHADER_SOURCE_GOLDILOCKS: &str = concat!(
    include_str!("../shaders/field_goldilocks.metal"),
    "\n",
    include_str!("../shaders/eq_eval_gl.metal"),
    "\n",
    include_str!("../shaders/poly_eval_gl.metal"),
    "\n",
    include_str!("../shaders/sumcheck_fold_gl.metal"),
);

pub const THREADGROUP_SIZE: u64 = 256;
pub const GPU_DISPATCH_THRESHOLD: usize = 1 << 18;

pub struct MetalAccelerator {
    pub(crate) device: Device,
    pub(crate) queue: metal::CommandQueue,
    pub(crate) pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl MetalAccelerator {
    pub fn new() -> Option<Self> {
        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("Metal: no system default device");
                return None;
            }
        };
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);

        let bn254_library = match device.new_library_with_source(SHADER_SOURCE_BN254, &opts) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("Metal BN254 shader compilation failed: {e}");
                return None;
            }
        };

        let gl_library = match device.new_library_with_source(SHADER_SOURCE_GOLDILOCKS, &opts) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("Metal Goldilocks shader compilation failed: {e}");
                return None;
            }
        };

        let bn254_kernels: &[&str] = &[
            "build_eq_step",
            "cross_prod_eq",
            "vec_add",
            "poly_eval_kernel",
            "reduce_blocks",
            "fold_f",
            "fold_hg",
        ];

        let gl_kernels: &[&str] = &[
            "gl_build_eq_step",
            "gl_cross_prod_eq",
            "gl_vec_add",
            "gl_poly_eval_kernel",
            "gl_reduce_blocks",
            "gl_fold_f",
            "gl_fold_hg",
        ];

        let mut pipelines = HashMap::new();

        for (&name, library) in bn254_kernels
            .iter()
            .map(|n| (n, &bn254_library))
            .chain(gl_kernels.iter().map(|n| (n, &gl_library)))
        {
            let func = match library.get_function(name, None) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Metal function '{name}' not found: {e}");
                    return None;
                }
            };
            let pipeline = match device.new_compute_pipeline_state_with_function(&func) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Pipeline creation failed for '{name}': {e}");
                    return None;
                }
            };
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

    pub fn device(&self) -> &Device {
        &self.device
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
        if total_threads == 0 {
            return;
        }
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
