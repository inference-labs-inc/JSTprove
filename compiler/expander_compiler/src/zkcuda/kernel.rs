use crate::circuit::input_mapping::EMPTY;
use crate::circuit::ir::common::Instruction;
use crate::compile::{
    compile_step_1, compile_step_2, compile_step_3, print_ir_stats, print_layered_circuit_stats,
    CompileOptions,
};
use crate::frontend::{BasicAPI, Error, Variable, API};
use crate::zkcuda::shape::{shape_padded_mapping, shape_vec_len, shape_vec_padded_len, Shape};
use crate::{
    circuit::{
        config::Config,
        input_mapping::InputMapping,
        ir,
        layered::{Circuit as LayeredCircuit, NormalInputType},
    },
    compile::compile_step_4,
};
pub use macros::kernel;

use serdes::ExpSerde;

#[derive(Debug, Clone, Hash, PartialEq, Eq, ExpSerde)]
pub struct KernelPrimitive<C: Config> {
    // The circuit IR for output computation and later compilation
    ir_for_later_compilation: ir::hint_normalized::RootCircuit<C>,
    ir_for_calling: ir::hint_normalized::RootCircuit<C>,
    ir_input_offsets: Vec<usize>,
    ir_output_offsets: Vec<usize>,

    io_specs: Vec<IOVecSpec>,
    io_shapes: Vec<Shape>,
}

impl<C: Config> KernelPrimitive<C> {
    pub fn ir_for_later_compilation(&self) -> &ir::hint_normalized::RootCircuit<C> {
        &self.ir_for_later_compilation
    }
    pub fn ir_for_calling(&self) -> &ir::hint_normalized::RootCircuit<C> {
        &self.ir_for_calling
    }
    pub fn ir_input_offsets(&self) -> &[usize] {
        &self.ir_input_offsets
    }
    pub fn ir_output_offsets(&self) -> &[usize] {
        &self.ir_output_offsets
    }
    pub fn io_specs(&self) -> &[IOVecSpec] {
        &self.io_specs
    }
    pub fn io_shapes(&self) -> &[Shape] {
        &self.io_shapes
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, ExpSerde)]
pub struct Kernel<C: Config> {
    pub hint_solver: Option<ir::hint_normalized::RootCircuit<C>>,
    pub layered_circuit: LayeredCircuit<C, NormalInputType>,
    pub layered_circuit_input: Vec<LayeredCircuitInputVec>,
}

impl<C: Config> Kernel<C> {
    pub fn layered_circuit(&self) -> &LayeredCircuit<C, NormalInputType> {
        &self.layered_circuit
    }
    pub fn layered_circuit_input(&self) -> &[LayeredCircuitInputVec] {
        &self.layered_circuit_input
    }
    pub fn hint_solver(&self) -> Option<&ir::hint_normalized::RootCircuit<C>> {
        self.hint_solver.as_ref()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, ExpSerde)]
pub struct IOVecSpec {
    pub len: usize,
    pub is_input: bool,
    pub is_output: bool,
}

#[derive(Default, Debug, Copy, Clone, Hash, PartialEq, Eq, ExpSerde)]
pub struct LayeredCircuitInputVec {
    pub len: usize,
    pub offset: usize,
}

pub fn compile_with_spec_and_shapes<C, F>(
    f: F,
    io_specs: &[IOVecSpec],
    shapes: &[Vec<usize>],
) -> Result<KernelPrimitive<C>, Error>
where
    C: Config,
    F: Fn(&mut API<C>, &mut Vec<Vec<Variable>>),
{
    let total_inputs = io_specs
        .iter()
        .map(|spec| spec.len * (spec.is_input as usize + spec.is_output as usize))
        .sum();
    let (mut root_builder, input_variables, _) = API::<C>::new(total_inputs, 0);
    let mut io_vars = vec![];
    let mut expected_outputs = vec![];
    let mut inputs_offsets = vec![];
    let mut outputs_offsets = vec![];
    let mut global_input_offset = 0;
    for spec in io_specs {
        let mut cur_inputs = vec![];
        if spec.is_input {
            for i in 0..spec.len {
                cur_inputs.push(input_variables[global_input_offset + i]);
            }
            inputs_offsets.push(global_input_offset);
            global_input_offset += spec.len;
        } else {
            for _ in 0..spec.len {
                cur_inputs.push(root_builder.constant(0));
            }
            inputs_offsets.push(global_input_offset);
        }
        io_vars.push(cur_inputs);
    }
    inputs_offsets.push(global_input_offset);
    let n_in = global_input_offset;
    for spec in io_specs {
        if spec.is_output {
            let mut cur_outputs = vec![];
            for i in 0..spec.len {
                cur_outputs.push(input_variables[global_input_offset + i]);
            }
            expected_outputs.push(cur_outputs);
            outputs_offsets.push(global_input_offset);
            global_input_offset += spec.len;
        } else {
            expected_outputs.push(vec![]);
            outputs_offsets.push(global_input_offset);
        }
    }
    outputs_offsets.push(global_input_offset);
    f(&mut root_builder, &mut io_vars);
    let mut output_vars = vec![];
    for i in 1..=n_in {
        output_vars.push(i);
    }
    for (i, spec) in io_specs.iter().enumerate() {
        if spec.is_output {
            for (x, y) in io_vars[i].iter().zip(expected_outputs[i].iter()) {
                root_builder.assert_is_equal(x, y);
                output_vars.push(x.id());
            }
        }
    }
    let mut r_source = root_builder.build();
    assert_eq!(r_source.circuits[&0].outputs.len(), 0);
    r_source.circuits.get_mut(&0).unwrap().outputs = output_vars.clone();
    let (r, src_im) = compile_step_1(&r_source, CompileOptions::default())?;
    for (i, x) in src_im.mapping().iter().enumerate() {
        assert_eq!(*x, i);
    }
    print_ir_stats(&r);
    let mut r2 = r.clone();
    r2.circuits.get_mut(&0).unwrap().constraints = Vec::new();
    let mut tmp_im = InputMapping::new_identity(r2.input_size());
    let r2 = compile_step_4(r2, &mut tmp_im, CompileOptions::default())?;
    // No inputs should be removed in this step.
    for (i, x) in tmp_im.mapping().iter().take(n_in).enumerate() {
        assert_eq!(i, *x);
    }

    Ok(KernelPrimitive {
        ir_for_later_compilation: r,
        ir_for_calling: r2,
        ir_input_offsets: inputs_offsets,
        ir_output_offsets: outputs_offsets,
        io_specs: io_specs.to_vec(),
        io_shapes: shapes.to_vec(),
    })
}

pub fn compile_primitive<C: Config>(
    kernel: &KernelPrimitive<C>,
    pad_shapes_input: &[Option<Shape>],
    pad_shapes_output: &[Option<Shape>],
) -> Result<Kernel<C>, Error> {
    let prev_total_inputs = kernel.ir_for_later_compilation.input_size();

    // Split the ir into hint_solver and hint less circuit.
    // In compile_with_spec_and_shapes, the circuit has all inputs exported to output.
    // Thus r_hint_less also has all inputs exported to output.
    // Additionally, r_hint_exported has hints in output.
    let (mut r_hint_less, r_hint_exported) =
        kernel.ir_for_later_compilation.remove_and_export_hints();

    // Process the hint solver
    r_hint_exported
        .validate()
        .map_err(|e| e.prepend("hint exported circuit invalid"))?;
    let mut tmp_im = InputMapping::new_identity(r_hint_exported.input_size());
    let mut r_hint_exported_opt =
        compile_step_4(r_hint_exported, &mut tmp_im, CompileOptions::default())?;
    // No inputs should be removed in this step.
    for (i, x) in tmp_im.mapping().iter().enumerate() {
        assert_eq!(i, *x);
    }
    // Only keep the hints in the output.
    let re_c0 = r_hint_exported_opt.circuits.get_mut(&0).unwrap();
    re_c0.outputs.drain(..prev_total_inputs);
    let num_hints = re_c0.outputs.len();

    // Process the hint less circuit
    // First, we need to pad and reorder the inputs.
    let mut reorder_pad_shapes = Vec::with_capacity(pad_shapes_input.len() * 2 + 1);
    for (spec, shape) in kernel.io_specs.iter().zip(pad_shapes_input.iter()) {
        if spec.is_input {
            reorder_pad_shapes.push(shape.as_ref().unwrap().clone());
        }
    }
    for (spec, shape) in kernel.io_specs.iter().zip(pad_shapes_output.iter()) {
        if spec.is_output {
            reorder_pad_shapes.push(shape.as_ref().unwrap().clone());
        }
    }
    if num_hints > 0 {
        reorder_pad_shapes.push(vec![num_hints]);
    }
    let lc_input = reorder_ir_inputs(&mut r_hint_less, &reorder_pad_shapes);
    // Now we need to ensure that every input can't be optimized away.
    let rl_c0 = r_hint_less.circuits.get_mut(&0).unwrap();
    let num_unused_outputs = rl_c0.num_inputs;
    rl_c0.outputs = (1..=rl_c0.num_inputs).collect();
    let (mut r_dest_opt, hl_im) =
        compile_step_2::<C, NormalInputType>(r_hint_less, CompileOptions::default())?;
    for (i, x) in hl_im.mapping().iter().enumerate() {
        assert_eq!(i, *x);
    }
    // compile_step_2 may introduce new outputs at beginning to ensure constraints, we only keep those
    let rd_c0 = r_dest_opt.circuits.get_mut(&0).unwrap();
    rd_c0
        .outputs
        .truncate(rd_c0.outputs.len() - num_unused_outputs);
    let num_inputs_with_hint_padded = rd_c0.num_inputs;

    // Compile to layered circuit
    let (lc, dest_im) = crate::layering::compile(
        &r_dest_opt,
        crate::layering::CompileOptions {
            allow_input_reorder: false,
        },
    );
    for (i, x) in dest_im.mapping().iter().enumerate() {
        if i < num_inputs_with_hint_padded {
            assert_eq!(i, *x);
        } else {
            assert_eq!(*x, EMPTY);
        }
    }
    let lc = compile_step_3(lc, CompileOptions::default())?;
    print_layered_circuit_stats(&lc);

    Ok(Kernel {
        hint_solver: if num_hints > 0 {
            Some(r_hint_exported_opt)
        } else {
            None
        },
        layered_circuit: lc,
        layered_circuit_input: lc_input,
    })
}

fn reorder_ir_inputs<C: Config>(
    r: &mut ir::hint_less::RootCircuit<C>,
    pad_shapes: &[Shape],
) -> Vec<LayeredCircuitInputVec> {
    // sort by size, pad to 2^n, then reorder
    let mut sizes: Vec<(usize, usize)> = pad_shapes
        .iter()
        .enumerate()
        .map(|(i, x)| (shape_vec_padded_len(x), i))
        .collect();
    let sizes_prev: Vec<(usize, usize)> = pad_shapes
        .iter()
        .enumerate()
        .map(|(i, x)| (shape_vec_len(x), i))
        .collect();
    let mut prev_offset = Vec::new();
    let mut cur = 0;
    for (x, _) in sizes_prev.iter() {
        prev_offset.push(cur);
        cur += *x;
    }
    sizes.sort_by(|a, b| b.cmp(a));

    let r0 = r.circuits.get_mut(&0).unwrap();
    let mut var_new_id = vec![0; r0.num_inputs + 1];
    let mut var_max = 0;
    let mut lc_in = vec![LayeredCircuitInputVec::default(); sizes.len()];

    for &(n, i) in sizes.iter() {
        let prev = prev_offset[i];
        lc_in[i].offset = var_max;
        lc_in[i].len = n;
        assert!(var_max % n == 0);
        let im = shape_padded_mapping(&pad_shapes[i]);
        // println!("{:?}", im.mapping());
        for (j, &k) in im.mapping().iter().enumerate() {
            var_new_id[prev + k + 1] = var_max + j + 1;
        }
        var_max += n;
    }
    r0.num_inputs = var_max;
    let mut new_insns = vec![];
    for insn in r0.instructions.iter() {
        new_insns.push(insn.replace_vars(|x| var_new_id[x]));
        for _ in 0..insn.num_outputs() {
            var_max += 1;
            var_new_id.push(var_max);
        }
    }
    r0.instructions = new_insns;
    r0.constraints = r0.constraints.iter().map(|x| var_new_id[*x]).collect();
    r0.outputs = r0.outputs.iter().map(|x| var_new_id[*x]).collect();

    lc_in
}
