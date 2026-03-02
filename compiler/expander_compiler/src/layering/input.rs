use std::collections::HashMap;

use crate::circuit::{config::Config, input_mapping::EMPTY, layered::InputType};

use super::{compile::CompileContext, layer_layout::LayerLayoutInner};

impl<'a, C: Config, I: InputType> CompileContext<'a, C, I> {
    pub fn record_input_order(&self) -> Vec<usize> {
        let layout_id = self.layout_ids[0];
        let l = self.layer_layout_pool.get(layout_id);
        let placement = match &l.inner {
            LayerLayoutInner::Dense { placement } => placement,
            _ => {
                panic!("unexpected situation");
            }
        };
        let lc = &self.circuits[&0].lcs[0];
        let mut v = HashMap::new();
        for (i, x) in placement.iter().cloned().enumerate() {
            if x != EMPTY {
                v.insert(*lc.vars.get(x), i);
            }
        }
        let mut gi = Vec::new();
        let circuit = self.circuits[&0].circuit;
        for i in 1..=circuit.num_inputs {
            if let Some(&vi) = v.get(&i) {
                gi.push(vi);
            } else {
                gi.push(EMPTY);
            }
        }
        gi
    }
}
