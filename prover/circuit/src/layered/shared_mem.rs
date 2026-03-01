use super::circuit::{Circuit, CircuitLayer, StructureInfo};
use super::gates::{GateAdd, GateConst, GateMul, GateUni};

use gkr_engine::{FieldEngine, MPISharedMemory};

impl<C: FieldEngine> MPISharedMemory for CircuitLayer<C> {
    fn bytes_size(&self) -> usize {
        8 + 8
            + self.mul.bytes_size()
            + self.add.bytes_size()
            + self.const_.bytes_size()
            + self.uni.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        self.input_var_num.to_memory(ptr);
        self.output_var_num.to_memory(ptr);
        self.mul.to_memory(ptr);
        self.add.to_memory(ptr);
        self.const_.to_memory(ptr);
        self.uni.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let input_var_num = usize::new_from_memory(ptr);
        let output_var_num = usize::new_from_memory(ptr);
        let mul = Vec::<GateMul<C>>::new_from_memory(ptr);
        let add = Vec::<GateAdd<C>>::new_from_memory(ptr);
        let const_ = Vec::<GateConst<C>>::new_from_memory(ptr);
        let uni = Vec::<GateUni<C>>::new_from_memory(ptr);

        CircuitLayer {
            input_var_num,
            output_var_num,

            input_vals: vec![],
            output_vals: vec![],

            mul,
            add,
            const_,
            uni,

            structure_info: StructureInfo::default(),
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.mul.discard_control_of_shared_mem();
        self.add.discard_control_of_shared_mem();
        self.const_.discard_control_of_shared_mem();
        self.uni.discard_control_of_shared_mem();
    }
}

impl<C: FieldEngine> MPISharedMemory for Circuit<C> {
    fn bytes_size(&self) -> usize {
        self.layers.len().bytes_size()
            + self
                .layers
                .iter()
                .map(|layer| layer.bytes_size())
                .sum::<usize>()
            + self.expected_num_output_zeros.bytes_size()
    }

    fn to_memory(&self, ptr: &mut *mut u8) {
        let len = self.layers.len();
        len.to_memory(ptr);
        self.layers.iter().for_each(|layer| layer.to_memory(ptr));
        self.expected_num_output_zeros.to_memory(ptr);
    }

    fn new_from_memory(ptr: &mut *mut u8) -> Self {
        let len = usize::new_from_memory(ptr);
        let layers = (0..len)
            .map(|_| CircuitLayer::<C>::new_from_memory(ptr))
            .collect();
        let expected_num_output_zeros = usize::new_from_memory(ptr);

        Circuit {
            layers,

            public_input: vec![],
            expected_num_output_zeros,

            rnd_coefs_identified: false,
            rnd_coefs: vec![],
        }
    }

    fn discard_control_of_shared_mem(self) {
        self.layers
            .into_iter()
            .for_each(|layer| layer.discard_control_of_shared_mem());
    }
}
