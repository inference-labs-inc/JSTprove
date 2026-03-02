use crate::circuit::config::CircuitField;

use super::{Circuit, Config, CrossLayerInputType, Input, InputUsize, NormalInputType};

impl<C: Config> Circuit<C, NormalInputType> {
    pub fn export_to_expander<
        DestConfig: gkr_engine::FieldEngine<CircuitField = CircuitField<C>>,
    >(
        &self,
    ) -> expander_circuit::RecursiveCircuit<DestConfig> {
        let segments = self
            .segments
            .iter()
            .map(|seg| expander_circuit::Segment {
                i_var_num: seg.num_inputs.get(0).trailing_zeros() as usize,
                o_var_num: seg.num_outputs.trailing_zeros() as usize,
                gate_muls: seg
                    .gate_muls
                    .iter()
                    .map(|gate| gate.export_to_expander())
                    .collect(),
                gate_adds: seg
                    .gate_adds
                    .iter()
                    .map(|gate| gate.export_to_expander())
                    .collect(),
                gate_consts: seg
                    .gate_consts
                    .iter()
                    .map(|gate| gate.export_to_expander())
                    .collect(),
                gate_uni: seg
                    .gate_customs
                    .iter()
                    .map(|gate| {
                        let (c, r) = gate.coef.export_to_expander();
                        expander_circuit::GateUni {
                            i_ids: [gate.inputs[0].offset()],
                            o_id: gate.output,
                            coef: c,
                            coef_type: r,
                            gate_type: gate.gate_type,
                        }
                    })
                    .collect(),
                child_segs: seg
                    .child_segs
                    .iter()
                    .map(|seg| {
                        (
                            seg.0,
                            seg.1
                                .iter()
                                .map(|alloc| expander_circuit::Allocation {
                                    i_offset: alloc.input_offset.get(0),
                                    o_offset: alloc.output_offset,
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            })
            .collect();
        expander_circuit::RecursiveCircuit {
            segments,
            layers: self.layer_ids.clone(),
            num_outputs: self.num_actual_outputs,
            num_public_inputs: self.num_public_inputs,
            expected_num_output_zeros: self.expected_num_output_zeroes,
        }
    }

    pub fn export_to_expander_flatten(&self) -> expander_circuit::Circuit<C::FieldConfig> {
        let circuit = self.export_to_expander::<C::FieldConfig>();
        let mut flattened = circuit.flatten();
        flattened.pre_process_gkr();
        flattened
    }
}

impl<C: Config> Circuit<C, CrossLayerInputType> {
    pub fn export_to_expander<
        DestConfig: gkr_engine::FieldEngine<CircuitField = CircuitField<C>>,
    >(
        &self,
    ) -> crosslayer_prototype::CrossLayerRecursiveCircuit<DestConfig> {
        let mut segments = Vec::new();
        for segment in self.segments.iter() {
            let mut gate_adds = Vec::new();
            let mut gate_relays = Vec::new();
            for gate in segment.gate_adds.iter() {
                if gate.inputs[0].layer() == 0 {
                    gate_adds.push(gate.export_to_crosslayer_simple());
                } else {
                    let (c, r) = gate.coef.export_to_expander();
                    assert_eq!(r, expander_circuit::CoefType::Constant);
                    gate_relays.push(crosslayer_prototype::CrossLayerRelay {
                        i_id: gate.inputs[0].offset(),
                        o_id: gate.output,
                        i_layer: gate.inputs[0].layer(),
                        coef: c,
                    });
                }
            }
            assert_eq!(segment.gate_customs.len(), 0);
            segments.push(crosslayer_prototype::CrossLayerSegment {
                input_size: segment.num_inputs.to_vec(),
                output_size: segment.num_outputs,
                child_segs: segment
                    .child_segs
                    .iter()
                    .map(|seg| {
                        (
                            seg.0,
                            seg.1
                                .iter()
                                .map(|alloc| crosslayer_prototype::Allocation {
                                    i_offset: alloc.input_offset.to_vec(),
                                    o_offset: alloc.output_offset,
                                })
                                .collect(),
                        )
                    })
                    .collect(),
                gate_muls: segment
                    .gate_muls
                    .iter()
                    .map(|gate| gate.export_to_crosslayer_simple())
                    .collect(),
                gate_csts: segment
                    .gate_consts
                    .iter()
                    .map(|gate| gate.export_to_crosslayer_simple())
                    .collect(),
                gate_adds,
                gate_relay: gate_relays,
            });
        }
        crosslayer_prototype::CrossLayerRecursiveCircuit {
            num_public_inputs: self.num_public_inputs,
            num_outputs: self.num_actual_outputs,
            expected_num_output_zeros: self.expected_num_output_zeroes,
            layers: self.layer_ids.clone(),
            segments,
        }
    }
}
