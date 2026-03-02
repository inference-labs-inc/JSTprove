use gkr_engine::FieldEngine;
use std::{cmp::max, collections::HashMap};

use crate::*;

// recursive format used in compiler
pub type SegmentId = usize;

pub struct Allocation {
    pub i_offset: usize,
    pub o_offset: usize,
}

#[derive(Default)]
pub struct Segment<C: FieldEngine> {
    pub i_var_num: usize,
    pub o_var_num: usize,
    pub child_segs: Vec<(SegmentId, Vec<Allocation>)>,
    pub gate_muls: Vec<GateMul<C>>,
    pub gate_adds: Vec<GateAdd<C>>,
    pub gate_consts: Vec<GateConst<C>>,
    pub gate_uni: Vec<GateUni<C>>,
}

impl<C: FieldEngine> Segment<C> {
    #[inline]
    pub fn contain_gates(&self) -> bool {
        !self.gate_muls.is_empty()
            || !self.gate_adds.is_empty()
            || !self.gate_consts.is_empty()
            || !self.gate_uni.is_empty()
    }

    #[inline]
    pub fn scan_leaf_segments(
        &self,
        rc: &RecursiveCircuit<C>,
        cur_id: SegmentId,
    ) -> HashMap<SegmentId, Vec<Allocation>> {
        let mut ret = HashMap::new();
        if self.contain_gates() {
            ret.insert(
                cur_id,
                vec![Allocation {
                    i_offset: 0,
                    o_offset: 0,
                }],
            );
        }
        for (child_seg_id, child_allocs) in &self.child_segs {
            let leaves = rc.segments[*child_seg_id].scan_leaf_segments(rc, *child_seg_id);
            for (leaf_seg_id, leaf_allocs) in leaves {
                ret.entry(leaf_seg_id).or_insert_with(Vec::new);
                for child_alloc in child_allocs {
                    for leaf_alloc in &leaf_allocs {
                        ret.get_mut(&leaf_seg_id).unwrap().push(Allocation {
                            i_offset: child_alloc.i_offset + leaf_alloc.i_offset,
                            o_offset: child_alloc.o_offset + leaf_alloc.o_offset,
                        });
                    }
                }
            }
        }
        ret
    }
}

#[derive(Default)]
pub struct RecursiveCircuit<C: FieldEngine> {
    pub num_public_inputs: usize,
    pub num_outputs: usize,
    pub expected_num_output_zeros: usize,

    pub segments: Vec<Segment<C>>,
    pub layers: Vec<SegmentId>,
}

impl<C: FieldEngine> RecursiveCircuit<C> {
    pub fn flatten(&self) -> Circuit<C> {
        let mut ret = Circuit::<C> {
            expected_num_output_zeros: self.expected_num_output_zeros,
            ..Default::default()
        };
        // layer-by-layer conversion
        for layer_id in &self.layers {
            let layer_seg = &self.segments[*layer_id];
            let leaves = layer_seg.scan_leaf_segments(self, *layer_id);
            let mut ret_layer = CircuitLayer {
                input_var_num: max(layer_seg.i_var_num, 1), // var_num >= 1
                output_var_num: max(layer_seg.o_var_num, 1), // var_num >= 1
                ..Default::default()
            };
            for (leaf_seg_id, leaf_allocs) in leaves {
                let leaf_seg = &self.segments[leaf_seg_id];
                for alloc in leaf_allocs {
                    for gate in &leaf_seg.gate_muls {
                        let mut gate = *gate;
                        gate.i_ids[0] += alloc.i_offset;
                        gate.i_ids[1] += alloc.i_offset;
                        gate.o_id += alloc.o_offset;
                        ret_layer.mul.push(gate);
                    }
                    for gate in &leaf_seg.gate_adds {
                        let mut gate = *gate;
                        gate.i_ids[0] += alloc.i_offset;
                        gate.o_id += alloc.o_offset;
                        ret_layer.add.push(gate);
                    }
                    for gate in &leaf_seg.gate_consts {
                        let mut gate = *gate;
                        gate.o_id += alloc.o_offset;
                        ret_layer.const_.push(gate);
                    }
                    for gate in &leaf_seg.gate_uni {
                        let mut gate = *gate;
                        gate.i_ids[0] += alloc.i_offset;
                        gate.o_id += alloc.o_offset;
                        ret_layer.uni.push(gate);
                    }
                }
            }
            // debug print layer
            log::trace!(
                "layer {} mul: {} add: {} const:{} uni:{} i_var_num: {} o_var_num: {}",
                ret.layers.len(),
                ret_layer.mul.len(),
                ret_layer.add.len(),
                ret_layer.const_.len(),
                ret_layer.uni.len(),
                ret_layer.input_var_num,
                ret_layer.output_var_num,
            );
            ret.layers.push(ret_layer);
        }

        ret
    }
}
