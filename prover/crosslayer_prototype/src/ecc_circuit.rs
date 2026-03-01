use std::{collections::HashMap, fs, io::Cursor};

use gkr_engine::{ExpErrors, FieldEngine};
use serdes::ExpSerde;

use crate::*;

pub type SegmentId = usize;
pub type CrossLayerInputSize = Vec<usize>;
pub type CrossLayerInputOffset = Vec<usize>;

// This function is used to add the offset of the parent segment and that of its children
// It is now required that the two offsets cross the same number of layers
fn offset_add(
    offset_1: &CrossLayerInputOffset,
    offset_2: &CrossLayerInputOffset,
) -> CrossLayerInputOffset {
    assert_eq!(offset_1.len(), offset_2.len());
    offset_1
        .iter()
        .zip(offset_2)
        .map(|(o1, o2)| o1 + o2)
        .collect()
}

pub struct Allocation {
    pub i_offset: CrossLayerInputOffset,
    pub o_offset: usize,
}

pub type CrossLayerChildSpec = (SegmentId, Vec<Allocation>);

#[derive(Default)]
pub struct CrossLayerSegment<F: FieldEngine> {
    pub input_size: CrossLayerInputSize,
    pub output_size: usize,
    pub child_segs: Vec<CrossLayerChildSpec>,
    pub gate_muls: Vec<SimpleGateMul<F>>,
    pub gate_adds: Vec<SimpleGateAdd<F>>,
    pub gate_csts: Vec<SimpleGateCst<F>>,
    pub gate_relay: Vec<CrossLayerRelay<F>>,
}

impl<F: FieldEngine> CrossLayerSegment<F> {
    #[inline]
    pub fn contain_gates(&self) -> bool {
        !self.gate_muls.is_empty()
            || !self.gate_adds.is_empty()
            || !self.gate_csts.is_empty()
            || !self.gate_relay.is_empty()
    }

    #[inline]
    pub fn scan_leaf_segments(
        &self,
        rc: &CrossLayerRecursiveCircuit<F>,
        cur_id: SegmentId,
    ) -> HashMap<SegmentId, Vec<Allocation>> {
        let mut ret: HashMap<usize, Vec<Allocation>> = HashMap::new();
        if self.contain_gates() {
            ret.insert(
                cur_id,
                vec![Allocation {
                    i_offset: vec![0; self.input_size.len()],
                    o_offset: 0,
                }],
            );
        }
        for (child_seg_id, child_allocs) in &self.child_segs {
            let leaves = rc.segments[*child_seg_id].scan_leaf_segments(rc, *child_seg_id);
            for (leaf_seg_id, leaf_allocs) in leaves {
                ret.entry(leaf_seg_id).or_default();
                for child_alloc in child_allocs {
                    for leaf_alloc in &leaf_allocs {
                        ret.get_mut(&leaf_seg_id).unwrap().push(Allocation {
                            i_offset: offset_add(&child_alloc.i_offset, &leaf_alloc.i_offset),
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
pub struct CrossLayerRecursiveCircuit<F: FieldEngine> {
    pub num_public_inputs: usize,
    pub num_outputs: usize,
    pub expected_num_output_zeros: usize,

    pub segments: Vec<CrossLayerSegment<F>>,
    pub layers: Vec<SegmentId>,
}

impl<F: FieldEngine> CrossLayerRecursiveCircuit<F> {
    pub fn load(filename: &str) -> std::result::Result<Self, ExpErrors> {
        let file_bytes = fs::read(filename)?;
        let cursor = Cursor::new(file_bytes);
        Ok(<Self as ExpSerde>::deserialize_from(cursor)?)
    }

    pub fn flatten(&self) -> CrossLayerCircuit<F> {
        let mut ret = CrossLayerCircuit::<F>::default();

        // denote the input layer as layer 0 here
        assert!(self.segments[self.layers[0]].input_size.len() == 1);
        ret.layers.push(GenericLayer::<F> {
            layer_id: 0,
            layer_size: self.segments[self.layers[0]].input_size[0],
            input_layer_size: 0,
            ..Default::default()
        });

        // layer-by-layer conversion
        for (i, seg_id) in self.layers.iter().enumerate() {
            let layer_seg = &self.segments[*seg_id];
            let leaves = layer_seg.scan_leaf_segments(self, *seg_id);
            let mut ret_layer = GenericLayer::<F> {
                layer_id: i + 1,
                layer_size: layer_seg.output_size,
                input_layer_size: layer_seg.input_size[0],
                ..Default::default()
            };
            for (leaf_seg_id, leaf_allocs) in leaves {
                let leaf_seg = &self.segments[leaf_seg_id];
                for alloc in leaf_allocs {
                    for gate in &leaf_seg.gate_muls {
                        let mut gate = gate.clone();
                        gate.i_ids[0] += alloc.i_offset[0];
                        gate.i_ids[1] += alloc.i_offset[0];
                        gate.o_id += alloc.o_offset;
                        ret_layer.mul_gates.push(gate);
                    }
                    for gate in &leaf_seg.gate_adds {
                        let mut gate = gate.clone();
                        gate.i_ids[0] += alloc.i_offset[0];
                        gate.o_id += alloc.o_offset;
                        ret_layer.add_gates.push(gate);
                    }
                    for gate in &leaf_seg.gate_csts {
                        let mut gate = gate.clone();
                        gate.o_id += alloc.o_offset;
                        ret_layer.const_gates.push(gate);
                    }
                    for gate in &leaf_seg.gate_relay {
                        let mut gate = gate.clone();
                        gate.i_id += alloc.i_offset[gate.i_layer];
                        gate.o_id += alloc.o_offset;
                        // this is due to how offset is represented in ecc
                        gate.i_layer = ret_layer.layer_id - gate.i_layer - 1;
                        ret_layer.relay_gates.push(gate);
                    }
                }
            }
            ret.layers.push(ret_layer);
        }

        ret
    }
}
