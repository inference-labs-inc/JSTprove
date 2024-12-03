use std::collections::HashMap;

use crate::{
    circuit::{config::Config, input_mapping::InputMapping, ir, layered},
    utils::pool::Pool,
};

mod compile;
mod input;
pub mod ir_split;
mod layer_layout;
mod wire;

#[cfg(test)]
mod tests;

pub fn compile<C: Config>(rc: &ir::dest::RootCircuit<C>) -> (layered::Circuit<C>, InputMapping) {
    let mut ctx = compile::CompileContext {
        rc,
        circuits: HashMap::new(),
        order: Vec::new(),
        layer_layout_pool: Pool::new(),
        layer_req_to_layout: HashMap::new(),
        compiled_circuits: Vec::new(),
        conncected_wires: HashMap::new(),
        layout_ids: Vec::new(),
        layers: Vec::new(),
        input_order: Vec::new(),
        root_has_constraints: false,
    };
    ctx.compile();
    let l0_size = ctx.compiled_circuits[ctx.layers[0]].num_inputs;
    let output_zeroes = rc.expected_num_output_zeroes + ctx.root_has_constraints as usize;
    let output_all = rc.circuits[&0].outputs.len() + ctx.root_has_constraints as usize;
    (
        layered::Circuit {
            num_public_inputs: rc.num_public_inputs,
            num_actual_outputs: output_all,
            expected_num_output_zeroes: output_zeroes,
            segments: ctx.compiled_circuits,
            layer_ids: ctx.layers,
        },
        InputMapping::new(l0_size, ctx.input_order),
    )
}
