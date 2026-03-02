use gkr_engine::MPISharedMemory;

use crate::{
    circuit::layered::{
        Allocation, Circuit, GateAdd, GateConst, GateMul, NormalInputType, NormalInputUsize,
        Segment,
    },
    frontend::Config,
    zkcuda::{
        context::{ComputationGraph, ProofTemplate},
        kernel::{Kernel, LayeredCircuitInputVec},
        shape::BitOrder,
    },
};

impl<C: Config> MPISharedMemory for ComputationGraph<C> {
    fn bytes_size(&self) -> usize {
        self.kernels.len().bytes_size()
            + self.kernels.iter().map(|k| k.bytes_size()).sum::<usize>()
            + self.commitments_lens.bytes_size()
            + self.proof_templates.len().bytes_size()
            + self
                .proof_templates
                .iter()
                .map(|pt| pt.bytes_size())
                .sum::<usize>()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        self.kernels.len().to_memory(ptr);
        self.kernels.iter().for_each(|k| k.to_memory(ptr));
        self.commitments_lens.to_memory(ptr);
        self.proof_templates.len().to_memory(ptr);
        self.proof_templates.iter().for_each(|pt| pt.to_memory(ptr));
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let kernels_len = usize::new_from_memory(ptr);
        let kernels = (0..kernels_len)
            .map(|_| Kernel::<C>::new_from_memory(ptr))
            .collect::<Vec<_>>();
        let commitments_lens = Vec::<usize>::new_from_memory(ptr);
        let proof_templates_len = usize::new_from_memory(ptr);
        let proof_templates = (0..proof_templates_len)
            .map(|_| ProofTemplate::new_from_memory(ptr))
            .collect::<Vec<_>>();

        ComputationGraph {
            kernels,
            commitments_lens,
            proof_templates,
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.kernels
            .into_iter()
            .for_each(|k| k.discard_control_of_shared_mem());
        self.commitments_lens.discard_control_of_shared_mem();
        self.proof_templates
            .into_iter()
            .for_each(|pt| pt.discard_control_of_shared_mem());
    }
}

impl<C: Config> MPISharedMemory for Kernel<C> {
    fn bytes_size(&self) -> usize {
        assert!(
            self.hint_solver.is_none(),
            "Hint solver is not supported in MPISharedMemory for Kernel"
        );
        self.layered_circuit.bytes_size() + self.layered_circuit_input.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        assert!(
            self.hint_solver.is_none(),
            "Hint solver is not supported in MPISharedMemory for Kernel"
        );
        self.layered_circuit.to_memory(ptr);
        self.layered_circuit_input.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let layered_circuit = Circuit::<C, NormalInputType>::new_from_memory(ptr);
        let layered_circuit_input = Vec::<LayeredCircuitInputVec>::new_from_memory(ptr);

        Kernel {
            hint_solver: None, // Hint solver is not supported
            layered_circuit,
            layered_circuit_input,
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.layered_circuit.discard_control_of_shared_mem();
        self.layered_circuit_input.discard_control_of_shared_mem();
        // Hint solver is not supported, so no need to discard control of it
    }
}

impl MPISharedMemory for ProofTemplate {
    fn bytes_size(&self) -> usize {
        self.kernel_id.bytes_size()
            + self.commitment_indices.bytes_size()
            + self.commitment_bit_orders.len().bytes_size()
            + self
                .commitment_bit_orders
                .iter()
                .map(|order| order.bytes_size())
                .sum::<usize>()
            + self.parallel_count.bytes_size()
            + self.is_broadcast.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        self.kernel_id.to_memory(ptr);
        self.commitment_indices.to_memory(ptr);
        self.commitment_bit_orders.len().to_memory(ptr);
        self.commitment_bit_orders
            .iter()
            .for_each(|order| order.to_memory(ptr));
        self.parallel_count.to_memory(ptr);
        self.is_broadcast.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let kernel_id = usize::new_from_memory(ptr);
        let commitment_indices = Vec::<usize>::new_from_memory(ptr);
        let commitment_bit_orders_len = usize::new_from_memory(ptr);
        let commitment_bit_orders = (0..commitment_bit_orders_len)
            .map(|_| BitOrder::new_from_memory(ptr))
            .collect();
        let parallel_count = usize::new_from_memory(ptr);
        let is_broadcast = Vec::<bool>::new_from_memory(ptr);

        ProofTemplate {
            kernel_id,
            commitment_indices,
            commitment_bit_orders,
            parallel_count,
            is_broadcast,
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.commitment_indices.discard_control_of_shared_mem();
        self.commitment_bit_orders
            .into_iter()
            .for_each(|order| order.discard_control_of_shared_mem());
        self.is_broadcast.discard_control_of_shared_mem();
    }
}

impl<C: Config> MPISharedMemory for Circuit<C, NormalInputType> {
    fn bytes_size(&self) -> usize {
        self.num_public_inputs.bytes_size()
            + self.num_actual_outputs.bytes_size()
            + self.expected_num_output_zeroes.bytes_size()
            + self.segments.len().bytes_size()
            + self.segments.iter().map(|s| s.bytes_size()).sum::<usize>()
            + self.layer_ids.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        self.num_public_inputs.to_memory(ptr);
        self.num_actual_outputs.to_memory(ptr);
        self.expected_num_output_zeroes.to_memory(ptr);
        self.segments.len().to_memory(ptr);
        self.segments.iter().for_each(|s| s.to_memory(ptr));
        self.layer_ids.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let num_public_inputs = usize::new_from_memory(ptr);
        let num_actual_outputs = usize::new_from_memory(ptr);
        let expected_num_output_zeroes = usize::new_from_memory(ptr);
        let segments_len = usize::new_from_memory(ptr);
        let segments = (0..segments_len)
            .map(|_| Segment::<C, NormalInputType>::new_from_memory(ptr))
            .collect();
        let layer_ids = Vec::<usize>::new_from_memory(ptr);

        Circuit {
            num_public_inputs,
            num_actual_outputs,
            expected_num_output_zeroes,
            segments,
            layer_ids,
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.segments
            .into_iter()
            .for_each(|s| s.discard_control_of_shared_mem());
        self.layer_ids.discard_control_of_shared_mem();
    }
}

impl<C: Config> MPISharedMemory for Segment<C, NormalInputType> {
    fn bytes_size(&self) -> usize {
        assert!(
            self.gate_customs.is_empty(),
            "GateCustom is not supported in MPISharedMemory for Segment"
        );

        self.num_inputs.v.bytes_size()
            + self.num_outputs.bytes_size()
            + self.child_segs.len().bytes_size()
            + self
                .child_segs
                .iter()
                .map(|x| x.bytes_size())
                .sum::<usize>()
            + self.gate_muls.bytes_size()
            + self.gate_adds.bytes_size()
            + self.gate_consts.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        assert!(
            self.gate_customs.is_empty(),
            "GateCustom is not supported in MPISharedMemory for Segment"
        );

        self.num_inputs.v.to_memory(ptr);
        self.num_outputs.to_memory(ptr);
        self.child_segs.len().to_memory(ptr);
        self.child_segs.iter().for_each(|x| x.to_memory(ptr));
        self.gate_muls.to_memory(ptr);
        self.gate_adds.to_memory(ptr);
        self.gate_consts.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let num_inputs = NormalInputUsize {
            v: usize::new_from_memory(ptr),
        };
        let num_outputs = usize::new_from_memory(ptr);
        let child_segs_len = usize::new_from_memory(ptr);
        let child_segs = (0..child_segs_len)
            .map(|_| <(usize, Vec<Allocation<NormalInputType>>)>::new_from_memory(ptr))
            .collect();
        let gate_muls = Vec::<GateMul<C, NormalInputType>>::new_from_memory(ptr);
        let gate_adds = Vec::<GateAdd<C, NormalInputType>>::new_from_memory(ptr);
        let gate_consts = Vec::<GateConst<C, NormalInputType>>::new_from_memory(ptr);

        Segment {
            num_inputs,
            num_outputs,
            child_segs,
            gate_muls,
            gate_adds,
            gate_consts,
            gate_customs: vec![], // GateCustom is not supported
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.child_segs
            .into_iter()
            .for_each(|seg| seg.discard_control_of_shared_mem());
        self.gate_muls.discard_control_of_shared_mem();
        self.gate_adds.discard_control_of_shared_mem();
        self.gate_consts.discard_control_of_shared_mem();
        // GateCustom is not supported, so no need to discard control of it
    }
}
