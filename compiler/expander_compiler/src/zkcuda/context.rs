use arith::SimdField;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serdes::ExpSerde;

use crate::{
    circuit::config::{CircuitField, Config, SIMDField},
    field::FieldArith,
    hints::registry::{EmptyHintCaller, HintCaller},
    utils::{error::Error, pool::Pool},
    zkcuda::shape::keep_shape_until,
};

use super::{
    kernel::{compile_primitive, Kernel, KernelPrimitive},
    shape::{
        keep_shape_products_until, keep_shape_since, merge_shape_products, prefix_products,
        prefix_products_to_shape, shape_padded_mapping, shape_prepend, shape_vec_len,
        shape_vec_padded_len, BitOrder, Reshape, Shape, ShapeHistory, Transpose,
    },
    vec_shaped::{
        flatten_shaped, flatten_shaped_pack_simd, unflatten_shaped, unflatten_shaped_unpack_simd,
        VecShaped,
    },
};

pub use macros::call_kernel;

struct DeviceMemory<C: Config> {
    values: Vec<SIMDField<C>>,
    required_shape_products: Vec<usize>,
}

#[derive(Clone, Debug, ExpSerde)]
pub struct DeviceMemoryHandleRaw {
    id: usize,
    shape_history: ShapeHistory,
}

pub type DeviceMemoryHandle = Option<DeviceMemoryHandleRaw>;

#[derive(Clone, ExpSerde)]
pub struct KernelCall {
    kernel_id: usize,
    num_parallel: usize,
    input_handles: Vec<DeviceMemoryHandle>,
    output_handles: Vec<DeviceMemoryHandle>,
    is_broadcast: Vec<bool>,
}

#[derive(PartialEq, Eq, Clone, Debug, ExpSerde)]
pub struct ProofTemplate {
    pub kernel_id: usize,
    pub commitment_indices: Vec<usize>,
    pub commitment_bit_orders: Vec<BitOrder>,
    pub parallel_count: usize,
    pub is_broadcast: Vec<bool>,
}

impl ProofTemplate {
    pub fn kernel_id(&self) -> usize {
        self.kernel_id
    }
    pub fn commitment_indices(&self) -> &[usize] {
        &self.commitment_indices
    }
    pub fn commitment_bit_orders(&self) -> &[BitOrder] {
        &self.commitment_bit_orders
    }
    pub fn parallel_count(&self) -> usize {
        self.parallel_count
    }
    pub fn is_broadcast(&self) -> &[bool] {
        &self.is_broadcast
    }
}

#[derive(Default, Clone, Debug, ExpSerde)]
pub struct ComputationGraph<C: Config> {
    pub kernels: Vec<Kernel<C>>,
    pub commitments_lens: Vec<usize>,
    pub proof_templates: Vec<ProofTemplate>,
}

impl<C: Config> ComputationGraph<C> {
    pub fn kernels(&self) -> &[Kernel<C>] {
        &self.kernels
    }
    pub fn commitments_lens(&self) -> &[usize] {
        &self.commitments_lens
    }
    pub fn proof_templates(&self) -> &[ProofTemplate] {
        &self.proof_templates
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ContextState {
    ComputationGraphNotDone,
    ComputationGraphDone,
    WitnessDone,
}

pub struct Context<C: Config, H: HintCaller<CircuitField<C>> = EmptyHintCaller> {
    kernel_primitives: Pool<KernelPrimitive<C>>,
    kernels: Pool<Kernel<C>>,
    device_memories: Vec<DeviceMemory<C>>,
    kernel_calls: Vec<KernelCall>,
    proof_templates: Vec<ProofTemplate>,
    hint_caller: H,
    // current state of the context
    state: ContextState,
}

impl<C: Config> Default for Context<C> {
    fn default() -> Self {
        Self::new(EmptyHintCaller)
    }
}

fn ensure_handle(handle: DeviceMemoryHandle) -> DeviceMemoryHandleRaw {
    match handle {
        Some(handle) => handle,
        None => panic!("Empty DeviceMemoryHandle"),
    }
}

fn pack_vec<C: Config>(v: &[CircuitField<C>]) -> Vec<SIMDField<C>> {
    v.iter()
        .map(|x| {
            let mut v = Vec::with_capacity(SIMDField::<C>::PACK_SIZE);
            for _ in 0..SIMDField::<C>::PACK_SIZE {
                v.push(*x);
            }
            SIMDField::<C>::pack(&v)
        })
        .collect::<Vec<_>>()
}

fn unpack_vec<C: Config>(v: &[SIMDField<C>]) -> Vec<CircuitField<C>> {
    v.iter().map(|x| x.unpack()[0]).collect()
}

// returns Option<is_broadcast>
fn check_shape_compat(
    kernel_shape: &Shape,
    io_shape: &Shape,
    parallel_count: usize,
) -> Option<bool> {
    if kernel_shape.len() == io_shape.len() {
        if *kernel_shape == *io_shape {
            Some(true)
        } else {
            None
        }
    } else if kernel_shape.len() + 1 == io_shape.len() {
        if io_shape.iter().skip(1).eq(kernel_shape.iter()) {
            if io_shape[0] == parallel_count {
                Some(false)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

impl Reshape for DeviceMemoryHandle {
    fn reshape(&self, new_shape: &[usize]) -> Self {
        let handle = ensure_handle(self.clone());
        Some(DeviceMemoryHandleRaw {
            id: handle.id,
            shape_history: handle.shape_history.reshape(new_shape),
        })
    }
}

impl Transpose for DeviceMemoryHandle {
    fn transpose(&self, axes: &[usize]) -> Self {
        let handle = ensure_handle(self.clone());
        Some(DeviceMemoryHandleRaw {
            id: handle.id,
            shape_history: handle.shape_history.transpose(axes),
        })
    }
}

fn make_device_mem<C: Config>(
    device_memories: &mut Vec<DeviceMemory<C>>,
    values: Vec<SIMDField<C>>,
    shape: Shape,
) -> DeviceMemoryHandle {
    let t = shape_vec_len(&shape);
    let required_shape_products = if t == 1 { vec![1] } else { vec![1, t] };
    device_memories.push(DeviceMemory {
        values,
        required_shape_products,
    });
    Some(DeviceMemoryHandleRaw {
        id: device_memories.len() - 1,
        shape_history: ShapeHistory::new(shape),
    })
}

impl<C: Config, H: HintCaller<CircuitField<C>>> Context<C, H> {
    pub fn new(hint_caller: H) -> Self {
        Context {
            kernel_primitives: Pool::new(),
            kernels: Pool::new(),
            device_memories: vec![],
            kernel_calls: vec![],
            proof_templates: vec![],
            hint_caller,
            state: ContextState::ComputationGraphNotDone,
        }
    }

    pub fn copy_to_device<T: VecShaped<CircuitField<C>>>(
        &mut self,
        host_memory: &T,
    ) -> DeviceMemoryHandle {
        let (flat, shape) = flatten_shaped(host_memory);
        let simd_flat = pack_vec::<C>(&flat);
        make_device_mem(&mut self.device_memories, simd_flat, shape)
    }

    pub fn copy_to_device_and_pack_simd<T: VecShaped<CircuitField<C>>>(
        &mut self,
        host_memory: &T,
    ) -> DeviceMemoryHandle {
        let (flat, shape) = flatten_shaped_pack_simd(host_memory);
        make_device_mem(&mut self.device_memories, flat, shape)
    }

    pub fn copy_simd_to_device<T: VecShaped<SIMDField<C>>>(
        &mut self,
        host_memory: &T,
    ) -> DeviceMemoryHandle {
        let (flat, shape) = flatten_shaped(host_memory);
        make_device_mem(&mut self.device_memories, flat, shape)
    }

    pub fn copy_to_host<T: VecShaped<CircuitField<C>> + Default>(
        &self,
        device_memory_handle: DeviceMemoryHandle,
    ) -> T {
        let device_memory_handle = ensure_handle(device_memory_handle);
        let permuted_values = device_memory_handle
            .shape_history
            .permute_vec(&self.device_memories[device_memory_handle.id].values);
        unflatten_shaped(
            &unpack_vec::<C>(&permuted_values),
            &device_memory_handle.shape_history.shape(),
        )
    }

    pub fn copy_to_host_and_unpack_simd<T: VecShaped<CircuitField<C>> + Default>(
        &self,
        device_memory_handle: DeviceMemoryHandle,
    ) -> T {
        let device_memory_handle = ensure_handle(device_memory_handle);
        let permuted_values = device_memory_handle
            .shape_history
            .permute_vec(&self.device_memories[device_memory_handle.id].values);
        unflatten_shaped_unpack_simd(
            &permuted_values,
            &device_memory_handle.shape_history.shape(),
        )
    }

    pub fn copy_simd_to_host<T: VecShaped<SIMDField<C>> + Default>(
        &self,
        device_memory_handle: DeviceMemoryHandle,
    ) -> T {
        let device_memory_handle = ensure_handle(device_memory_handle);
        let permuted_values = device_memory_handle
            .shape_history
            .permute_vec(&self.device_memories[device_memory_handle.id].values);
        unflatten_shaped(
            &permuted_values,
            &device_memory_handle.shape_history.shape(),
        )
    }

    fn ir_copy_from_device_memory(
        &self,
        values: &[SIMDField<C>],
        s: &mut [SIMDField<C>],
        is_broadcast: bool,
        parallel_index: usize,
        chunk_size: Option<usize>,
    ) {
        if is_broadcast {
            s.copy_from_slice(values);
        } else {
            let chunk_size = chunk_size.unwrap();
            s.copy_from_slice(
                &values[chunk_size * parallel_index..chunk_size * (parallel_index + 1)],
            );
        }
    }

    pub fn call_kernel(
        &mut self,
        kernel: &KernelPrimitive<C>,
        num_parallel: usize,
        ios: &mut [DeviceMemoryHandle],
    ) -> Result<(), Error> {
        assert_eq!(self.state, ContextState::ComputationGraphNotDone);
        if kernel.io_shapes().len() != ios.len() {
            panic!("Invalid number of inputs/outputs");
        }
        let mut is_broadcast = Vec::with_capacity(ios.len());
        for (i, ((kernel_shape, io), spec)) in kernel
            .io_shapes()
            .iter()
            .zip(ios.iter())
            .zip(kernel.io_specs().iter())
            .enumerate()
        {
            if !spec.is_input {
                is_broadcast.push(false);
                continue;
            }
            /*println!(
                "Checking shape compatibility for input/output {}: kernel_shape={:?}, io_shape={:?}, num_parallel={}",
                i, kernel_shape, io, num_parallel
            );*/
            let io_shape = if let Some(handle) = io {
                handle.shape_history.shape()
            } else {
                panic!("Missing input at index {i}")
            };
            match check_shape_compat(kernel_shape, &io_shape, num_parallel) {
                Some(ib) => {
                    let isl = io
                        .as_ref()
                        .unwrap()
                        .shape_history
                        .get_initial_split_list(!ib);
                    let t = io.as_ref().unwrap().id;
                    self.device_memories[t].required_shape_products = merge_shape_products(
                        &isl,
                        &self.device_memories[t].required_shape_products,
                    );
                    is_broadcast.push(ib)
                }
                None => {
                    panic!(
                        "Incompatible shapes: want {:?}, got {:?}, num_parallel={} (Hint: if you want to broadcast, use {:?}, otherwise use {:?})",
                        kernel_shape,
                        io_shape,
                        num_parallel,
                        kernel_shape,
                        shape_prepend(kernel_shape, num_parallel)
                    );
                }
            }
        }
        for (io_spec, ib) in kernel.io_specs().iter().zip(is_broadcast.iter()) {
            if io_spec.is_output && *ib {
                panic!("Output is broadcasted, but it shouldn't be");
            }
        }

        let kernel_id = self.kernel_primitives.add(kernel);

        let mut outputs_tmp = vec![Vec::new(); kernel.io_specs().len()];
        let mut ir_inputs_all = vec![Vec::new(); kernel.io_specs().len()];
        let mut chunk_sizes: Vec<Option<usize>> = vec![None; kernel.io_specs().len()];
        for (((input, &ib), ir_inputs), chunk_size) in ios
            .iter()
            .zip(is_broadcast.iter())
            .zip(ir_inputs_all.iter_mut())
            .zip(chunk_sizes.iter_mut())
        {
            if input.is_none() {
                continue;
            }
            let handle = ensure_handle(input.clone());
            let values = handle
                .shape_history
                .permute_vec(&self.device_memories[handle.id].values);
            if !ib {
                *chunk_size = Some(values.len() / num_parallel);
            }
            *ir_inputs = values;
        }
        let mut ir_inputs_per_parallel = Vec::new();
        for parallel_i in 0..num_parallel {
            let mut ir_inputs = vec![SIMDField::<C>::zero(); kernel.ir_for_calling().input_size()];
            for (i, ((input, input_start), input_end)) in ios
                .iter()
                .zip(kernel.ir_input_offsets().iter())
                .zip(kernel.ir_input_offsets().iter().skip(1))
                .enumerate()
            {
                if input.is_none() {
                    continue;
                }
                self.ir_copy_from_device_memory(
                    &ir_inputs_all[i],
                    &mut ir_inputs[*input_start..*input_end],
                    is_broadcast[i],
                    parallel_i,
                    chunk_sizes[i],
                );
            }
            ir_inputs_per_parallel.push(ir_inputs);
        }
        let ir_outputs_per_parallel: Vec<Result<Vec<SIMDField<C>>, Error>> = ir_inputs_per_parallel
            .into_par_iter()
            .map(|ir_inputs| {
                kernel
                    .ir_for_calling()
                    .eval_safe_simd(ir_inputs, &[], &self.hint_caller)
            })
            .collect();
        for ir_outputs in ir_outputs_per_parallel {
            let ir_outputs = ir_outputs?;
            for (((spec, output_start), output_end), out) in kernel
                .io_specs()
                .iter()
                .zip(kernel.ir_output_offsets().iter())
                .zip(kernel.ir_output_offsets().iter().skip(1))
                .zip(outputs_tmp.iter_mut())
            {
                if !spec.is_output {
                    continue;
                }
                out.extend_from_slice(&ir_outputs[*output_start..*output_end]);
            }
        }
        let input_handles = ios.to_vec();
        let mut output_handles = vec![None; kernel.io_specs().len()];

        for ((((output, out2), spec), ov), shape) in ios
            .iter_mut()
            .zip(output_handles.iter_mut())
            .zip(kernel.io_specs().iter())
            .zip(outputs_tmp.into_iter())
            .zip(kernel.io_shapes().iter())
        {
            if !spec.is_output {
                *output = None;
                continue;
            }
            let handle = make_device_mem(
                &mut self.device_memories,
                ov,
                shape_prepend(shape, num_parallel),
            );
            let id = handle.as_ref().unwrap().id;
            self.device_memories[id].required_shape_products = merge_shape_products(
                &handle
                    .as_ref()
                    .unwrap()
                    .shape_history
                    .get_initial_split_list(true),
                &self.device_memories[id].required_shape_products,
            );
            *output = handle.clone();
            *out2 = handle;
        }
        self.kernel_calls.push(KernelCall {
            kernel_id,
            num_parallel,
            input_handles,
            output_handles,
            is_broadcast,
        });
        Ok(())
    }

    fn get_current_device_memory_shapes(&self) -> Vec<Shape> {
        self.device_memories
            .iter()
            .map(|dm| prefix_products_to_shape(&dm.required_shape_products))
            .collect()
    }

    fn propagate_and_get_shapes(&mut self) -> Vec<Shape> {
        let mut dm_shapes = self.get_current_device_memory_shapes();
        loop {
            let get_pad_shape = |x: &DeviceMemoryHandle| {
                x.as_ref().map(|handle| {
                    handle
                        .shape_history
                        .get_transposed_shape_and_bit_order(&dm_shapes[handle.id])
                        .0
                })
            };
            for kernel_call in self.kernel_calls.iter() {
                let kernel_primitive = self.kernel_primitives.get(kernel_call.kernel_id);
                let mut all_shapes = Vec::new();
                let mut all_handles = Vec::new();
                for ((spec, input_handle), &ib) in kernel_primitive
                    .io_specs()
                    .iter()
                    .zip(kernel_call.input_handles.iter())
                    .zip(kernel_call.is_broadcast.iter())
                {
                    if !spec.is_input || ib {
                        continue;
                    }
                    let pad_shape = get_pad_shape(input_handle).unwrap();
                    all_shapes.push(pad_shape);
                    all_handles.push(ensure_handle(input_handle.clone()));
                }
                for ((spec, output_handle), &ib) in kernel_primitive
                    .io_specs()
                    .iter()
                    .zip(kernel_call.output_handles.iter())
                    .zip(kernel_call.is_broadcast.iter())
                {
                    if !spec.is_output || ib {
                        continue;
                    }
                    let pad_shape = get_pad_shape(output_handle).unwrap();
                    all_shapes.push(pad_shape);
                    all_handles.push(ensure_handle(output_handle.clone()));
                }
                let mut required_shape_products = prefix_products(&[kernel_call.num_parallel]);
                for shape in all_shapes.iter() {
                    let products = keep_shape_products_until(
                        &prefix_products(shape),
                        kernel_call.num_parallel,
                    );
                    required_shape_products =
                        merge_shape_products(&required_shape_products, &products);
                }
                for handle in all_handles.iter() {
                    let dm = &mut self.device_memories[handle.id];
                    let total = shape_vec_len(&handle.shape_history.shape());
                    for &x in required_shape_products.iter() {
                        if x != 1 && x != kernel_call.num_parallel {
                            let sh_tmp = handle.shape_history.reshape(&[x, total / x]);
                            dm.required_shape_products = merge_shape_products(
                                &sh_tmp.get_initial_split_list(true),
                                &dm.required_shape_products,
                            );
                        }
                    }
                }
            }
            let new_dm_shapes = self.get_current_device_memory_shapes();
            if new_dm_shapes == dm_shapes {
                return dm_shapes;
            }
            dm_shapes = new_dm_shapes;
        }
    }

    fn compile_or_load_computation_graph(
        &mut self,
        cg: Option<ComputationGraph<C>>,
    ) -> Result<Option<ComputationGraph<C>>, Error> {
        assert_eq!(
            self.state,
            ContextState::ComputationGraphNotDone,
            "Computation graph is already done, please compile or load it only once."
        );
        self.state = ContextState::ComputationGraphDone;

        let dm_shapes = self.propagate_and_get_shapes();

        let (mut cg_kernels, cg_proof_templates, cg_commitments_lens) = if let Some(cg) = cg {
            for (i, kernel) in cg.kernels.iter().enumerate() {
                assert_eq!(self.kernels.add(kernel), i);
            }
            assert!(cg.commitments_lens.len() >= self.device_memories.len());
            for (dm_shape, cm_len) in dm_shapes.iter().zip(cg.commitments_lens.iter()) {
                assert_eq!(shape_vec_padded_len(dm_shape), *cm_len);
            }
            (
                Some(cg.kernels),
                Some(cg.proof_templates),
                Some(cg.commitments_lens),
            )
        } else {
            (None, None, None)
        };
        let mut commitments_lens: Vec<usize> =
            dm_shapes.iter().map(|x| shape_vec_padded_len(x)).collect();

        let get_pad_shape = |x: &DeviceMemoryHandle| {
            x.as_ref().map(|handle| {
                handle
                    .shape_history
                    .get_transposed_shape_and_bit_order(&dm_shapes[handle.id])
            })
        };
        let mut dm_max = self.device_memories.len();
        for kernel_call in self.kernel_calls.iter() {
            let pad_shapes_input = kernel_call
                .input_handles
                .iter()
                .map(get_pad_shape)
                .collect::<Vec<_>>();
            let pad_shapes_output = kernel_call
                .output_handles
                .iter()
                .map(get_pad_shape)
                .collect::<Vec<_>>();
            let kernel_primitive = self.kernel_primitives.get(kernel_call.kernel_id);
            let kernel = if let Some(cg_kernels) = cg_kernels.as_mut() {
                cg_kernels.drain(..1).next().unwrap()
            } else {
                let mut psi = Vec::new();
                for (s, &ib) in pad_shapes_input.iter().zip(kernel_call.is_broadcast.iter()) {
                    psi.push(s.as_ref().map(|t| {
                        if ib {
                            t.0.clone()
                        } else {
                            keep_shape_since(&t.0, kernel_call.num_parallel)
                        }
                    }));
                }
                let mut pso = Vec::new();
                for (s, &ib) in pad_shapes_output
                    .iter()
                    .zip(kernel_call.is_broadcast.iter())
                {
                    pso.push(s.as_ref().map(|t| {
                        if ib {
                            t.0.clone()
                        } else {
                            keep_shape_since(&t.0, kernel_call.num_parallel)
                        }
                    }));
                }
                compile_primitive(kernel_primitive, &psi, &pso)?
            };

            let mut commitment_indices: Vec<usize> = Vec::new();
            let mut commitment_bit_orders: Vec<BitOrder> = Vec::new();
            let mut any_shape = None;
            let mut is_broadcast = Vec::new();
            for (((spec, pad_shape), handle), &ib) in kernel_primitive
                .io_specs()
                .iter()
                .zip(&pad_shapes_input)
                .zip(kernel_call.input_handles.iter())
                .zip(kernel_call.is_broadcast.iter())
            {
                if spec.is_input {
                    let shape = pad_shape.as_ref().unwrap();
                    commitment_indices.push(handle.as_ref().unwrap().id);
                    commitment_bit_orders.push(shape.1.clone());
                    is_broadcast.push(ib);
                    if !ib {
                        any_shape = Some(shape.0.clone());
                    }
                }
            }
            for (((spec, pad_shape), handle), &ib) in kernel_primitive
                .io_specs()
                .iter()
                .zip(&pad_shapes_output)
                .zip(kernel_call.output_handles.iter())
                .zip(kernel_call.is_broadcast.iter())
            {
                if spec.is_output {
                    let shape = pad_shape.as_ref().unwrap();
                    commitment_indices.push(handle.as_ref().unwrap().id);
                    commitment_bit_orders.push(shape.1.clone());
                    is_broadcast.push(ib);
                    if !ib {
                        any_shape = Some(shape.0.clone());
                    }
                }
            }

            let any_shape = any_shape.unwrap();
            let any_shape = keep_shape_until(&any_shape, kernel_call.num_parallel);
            let dim0_len = shape_vec_padded_len(&any_shape);

            if kernel.hint_solver().is_some() {
                // if the kernel has a hint solver, we need to add another input
                let n = kernel.layered_circuit_input().last().unwrap().len * dim0_len;
                commitment_indices.push(dm_max);
                dm_max += 1;
                commitment_bit_orders.push((0..n.trailing_zeros() as usize).collect());
                commitments_lens.push(n);
                is_broadcast.push(false);
            }

            let kernel_id = self.kernels.add(&kernel);
            self.proof_templates.push(ProofTemplate {
                kernel_id,
                commitment_indices,
                commitment_bit_orders,
                parallel_count: dim0_len,
                is_broadcast,
            });
        }

        if let Some(cg_kernels) = cg_kernels {
            assert!(cg_kernels.is_empty());
            assert_eq!(cg_proof_templates.unwrap(), self.proof_templates);
            assert_eq!(cg_commitments_lens.unwrap(), commitments_lens);
            Ok(None)
        } else {
            Ok(Some(ComputationGraph {
                kernels: self.kernels.vec().clone(),
                commitments_lens,
                proof_templates: self.proof_templates.clone(),
            }))
        }
    }

    pub fn compile_computation_graph(&mut self) -> Result<ComputationGraph<C>, Error> {
        Ok(self.compile_or_load_computation_graph(None)?.unwrap())
    }

    pub fn load_computation_graph(&mut self, cg: ComputationGraph<C>) -> Result<(), Error> {
        let _ = self.compile_or_load_computation_graph(Some(cg))?;
        Ok(())
    }

    // actually, this function computes hints
    pub fn solve_witness(&mut self) -> Result<(), Error> {
        match self.state {
            ContextState::ComputationGraphNotDone => {
                panic!("Please compile computation graph first.");
            }
            ContextState::ComputationGraphDone => {}
            ContextState::WitnessDone => {
                panic!("Witness already solved.");
            }
        }
        self.state = ContextState::WitnessDone;

        for (kernel_call, proof_template) in
            self.kernel_calls.iter().zip(self.proof_templates.iter())
        {
            let kernel = self.kernels.get(proof_template.kernel_id);
            if kernel.hint_solver().is_none() {
                continue; // no need to solve hints
            }
            let hint_solver = kernel.hint_solver().unwrap();
            let kernel_primitive = self.kernel_primitives.get(kernel_call.kernel_id);

            let mut ir_inputs_all = vec![Vec::new(); kernel_primitive.io_specs().len()];
            let mut ir_outputs_all = vec![Vec::new(); kernel_primitive.io_specs().len()];
            let mut input_chunk_sizes: Vec<Option<usize>> =
                vec![None; kernel_primitive.io_specs().len()];
            let mut output_chunk_sizes: Vec<Option<usize>> =
                vec![None; kernel_primitive.io_specs().len()];
            let mut any_shape = None;
            for (((input, &ib), ir_inputs), chunk_size) in kernel_call
                .input_handles
                .iter()
                .zip(kernel_call.is_broadcast.iter())
                .zip(ir_inputs_all.iter_mut())
                .zip(input_chunk_sizes.iter_mut())
            {
                if input.is_none() {
                    continue;
                }
                let handle = ensure_handle(input.clone());
                if any_shape.is_none() {
                    any_shape = Some(handle.shape_history.shape());
                }
                let values = handle
                    .shape_history
                    .permute_vec(&self.device_memories[handle.id].values);
                if !ib {
                    *chunk_size = Some(values.len() / kernel_call.num_parallel);
                }
                *ir_inputs = values;
            }
            for (((output, &ib), ir_inputs), chunk_size) in kernel_call
                .output_handles
                .iter()
                .zip(kernel_call.is_broadcast.iter())
                .zip(ir_outputs_all.iter_mut())
                .zip(output_chunk_sizes.iter_mut())
            {
                if output.is_none() {
                    continue;
                }
                let handle = ensure_handle(output.clone());
                if any_shape.is_none() {
                    any_shape = Some(handle.shape_history.shape());
                }
                let values = handle
                    .shape_history
                    .permute_vec(&self.device_memories[handle.id].values);
                assert!(!ib);
                *chunk_size = Some(values.len() / kernel_call.num_parallel);
                *ir_inputs = values;
            }

            let mut hints_inputs_per_parallel = Vec::new();
            for parallel_i in 0..kernel_call.num_parallel {
                let mut inputs = vec![SIMDField::<C>::zero(); hint_solver.input_size()];

                for ((((spec, ir_inputs), input_start), input_end), chunk_size) in kernel_primitive
                    .io_specs()
                    .iter()
                    .zip(ir_inputs_all.iter())
                    .zip(kernel_primitive.ir_input_offsets().iter())
                    .zip(kernel_primitive.ir_input_offsets().iter().skip(1))
                    .zip(input_chunk_sizes.iter())
                {
                    if !spec.is_input {
                        continue;
                    }
                    self.ir_copy_from_device_memory(
                        ir_inputs,
                        &mut inputs[*input_start..*input_end],
                        chunk_size.is_none(),
                        parallel_i,
                        *chunk_size,
                    );
                }
                for ((((spec, ir_outputs), output_start), output_end), chunk_size) in
                    kernel_primitive
                        .io_specs()
                        .iter()
                        .zip(ir_outputs_all.iter())
                        .zip(kernel_primitive.ir_output_offsets().iter())
                        .zip(kernel_primitive.ir_output_offsets().iter().skip(1))
                        .zip(output_chunk_sizes.iter())
                {
                    if !spec.is_output {
                        continue;
                    }
                    self.ir_copy_from_device_memory(
                        ir_outputs,
                        &mut inputs[*output_start..*output_end],
                        chunk_size.is_none(),
                        parallel_i,
                        *chunk_size,
                    );
                }
                hints_inputs_per_parallel.push(inputs);
            }
            let hints_per_parallel: Vec<Result<Vec<SIMDField<C>>, Error>> =
                hints_inputs_per_parallel
                    .into_par_iter()
                    .map(|inputs| hint_solver.eval_safe_simd(inputs, &[], &self.hint_caller))
                    .collect();
            let mut hints_all = Vec::new();
            for hints in hints_per_parallel {
                hints_all.extend(hints?);
            }

            let hints_len = hints_all.len();
            let hints_id = make_device_mem(&mut self.device_memories, hints_all, vec![hints_len])
                .unwrap()
                .id;
            // we need to assign correct shape to it
            let any_shape = any_shape.unwrap();
            let mut any_shape_products =
                keep_shape_products_until(&prefix_products(&any_shape), kernel_call.num_parallel);
            if kernel_call.num_parallel != hints_len {
                any_shape_products.push(hints_len);
            }
            self.device_memories[hints_id].required_shape_products = merge_shape_products(
                &any_shape_products,
                &self.device_memories[hints_id].required_shape_products,
            );
        }

        Ok(())
    }

    pub fn export_device_memories(&self) -> Vec<Vec<SIMDField<C>>> {
        assert_eq!(
            self.state,
            ContextState::WitnessDone,
            "Please finish computation graph and witness solving before exporting device memories."
        );
        self.device_memories
            .iter()
            .map(|dm| {
                let shape = prefix_products_to_shape(&dm.required_shape_products);
                let im = shape_padded_mapping(&shape);
                im.map_inputs(&dm.values)
            })
            .collect()
    }
}
