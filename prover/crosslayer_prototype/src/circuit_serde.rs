use arith::Field;
use gkr_engine::FieldEngine;
use serdes::{ExpSerde, SerdeResult};
use std::{
    io::{Read, Write},
    vec,
};

use super::{Allocation, CoefType, CrossLayerRecursiveCircuit, CrossLayerSegment, Witness};
use crate::{CrossLayerRelay, SegmentId, SimpleGate};

/// A gate whose inputs are from different layers.
#[derive(Debug, Clone)]
pub struct ECCCrossLayerGate<F: FieldEngine, const INPUT_NUM: usize> {
    pub i_ids: [(usize, usize); INPUT_NUM], // (layer_offset, gate_offset)
    pub o_id: usize,
    pub coef_type: CoefType,
    pub coef: F::CircuitField,
}

impl<F, const INPUT_NUM: usize> ECCCrossLayerGate<F, INPUT_NUM>
where
    F: FieldEngine,
{
    pub fn to_simple_gate_or_relay(
        &self,
    ) -> (Option<SimpleGate<F, INPUT_NUM>>, Option<CrossLayerRelay<F>>) {
        if INPUT_NUM == 1 && self.i_ids[0].0 != 0 {
            (
                None,
                Some(CrossLayerRelay {
                    o_id: self.o_id,
                    i_id: self.i_ids[0].1,
                    i_layer: self.i_ids[0].0,
                    coef: self.coef,
                }),
            )
        } else {
            let mut i_ids = [0; INPUT_NUM];
            for (i, i_id) in i_ids.iter_mut().enumerate() {
                let (_layer_offset, gate_offset) = self.i_ids[i];
                *i_id = gate_offset;
            }

            let gate = SimpleGate {
                i_ids,
                o_id: self.o_id,
                coef_type: self.coef_type.clone(),
                coef: self.coef,
            };
            (Some(gate), None)
        }
    }
}

impl<C: FieldEngine, const INPUT_NUM: usize> ExpSerde for ECCCrossLayerGate<C, INPUT_NUM> {
    fn serialize_into<W: Write>(&self, _writer: W) -> SerdeResult<()> {
        todo!()
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut i_ids = [(0usize, 0usize); INPUT_NUM];
        for (layer_offset, gate_offset) in &mut i_ids {
            *layer_offset = <usize as ExpSerde>::deserialize_from(&mut reader)?;
            *gate_offset = <usize as ExpSerde>::deserialize_from(&mut reader)?;
        }

        let o_id = <usize as ExpSerde>::deserialize_from(&mut reader)?;

        let coef_type_u8 = u8::deserialize_from(&mut reader)?;
        let (coef_type, coef) = match coef_type_u8 {
            1 => (
                CoefType::Constant,
                C::CircuitField::deserialize_from(&mut reader)?,
            ),
            2 => (CoefType::Random, C::CircuitField::ONE),
            3 => {
                if INPUT_NUM > 0 {
                    panic!("Public Input can only be used with constant gates")
                };

                (
                    CoefType::PublicInput(<usize as ExpSerde>::deserialize_from(&mut reader)?),
                    C::CircuitField::ZERO,
                )
            }
            _ => unreachable!(),
        };

        Ok(Self {
            i_ids,
            o_id,
            coef_type,
            coef,
        })
    }
}

impl ExpSerde for Allocation {
    fn serialize_into<W: Write>(&self, _writer: W) -> SerdeResult<()> {
        todo!()
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        Ok(Self {
            i_offset: <Vec<usize> as ExpSerde>::deserialize_from(&mut reader)?,
            o_offset: <usize as ExpSerde>::deserialize_from(&mut reader)?,
        })
    }
}

impl<F: FieldEngine> ExpSerde for CrossLayerSegment<F> {
    fn serialize_into<W: Write>(&self, _writer: W) -> SerdeResult<()> {
        todo!()
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let input_size = <Vec<usize> as ExpSerde>::deserialize_from(&mut reader).unwrap();
        let output_size = <usize as ExpSerde>::deserialize_from(&mut reader).unwrap();
        assert!(input_size.iter().all(|&x| x.is_power_of_two()));
        assert!(output_size.is_power_of_two());

        let child_segs = Vec::<(SegmentId, Vec<Allocation>)>::deserialize_from(&mut reader)?;
        let ecc_gate_muls = Vec::<ECCCrossLayerGate<F, 2>>::deserialize_from(&mut reader)?;
        let ecc_gate_adds = Vec::<ECCCrossLayerGate<F, 1>>::deserialize_from(&mut reader)?;
        let ecc_gate_csts = Vec::<ECCCrossLayerGate<F, 0>>::deserialize_from(&mut reader)?;

        let gate_muls = ecc_gate_muls
            .into_iter()
            .map(|gate| gate.to_simple_gate_or_relay().0.unwrap())
            .collect();

        let mut gate_adds = vec![];
        let mut gate_relay = vec![];
        for g in ecc_gate_adds.iter() {
            let (gate, relay) = g.to_simple_gate_or_relay();
            if let Some(gate) = gate {
                gate_adds.push(gate);
            }
            if let Some(relay) = relay {
                gate_relay.push(relay);
            }
        }

        let gate_csts = ecc_gate_csts
            .into_iter()
            .map(|gate| gate.to_simple_gate_or_relay().0.unwrap())
            .collect();

        let custom_gates_size = <usize as ExpSerde>::deserialize_from(&mut reader).unwrap();
        assert_eq!(custom_gates_size, 0);

        Ok(CrossLayerSegment {
            input_size,
            output_size,
            child_segs,
            gate_muls,
            gate_adds,
            gate_csts,
            gate_relay,
        })
    }
}

const VERSION_NUM: usize = 3914834606642317635; // b'CIRCUIT6'

impl<F: FieldEngine> ExpSerde for CrossLayerRecursiveCircuit<F> {
    fn serialize_into<W: Write>(&self, _writer: W) -> SerdeResult<()> {
        todo!()
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let version_num = <usize as ExpSerde>::deserialize_from(&mut reader)?;
        assert_eq!(version_num, VERSION_NUM);
        let expected_mod = <F::CircuitField as Field>::MODULUS;
        let mut read_mod = [0u8; 32];
        reader.read_exact(&mut read_mod)?;
        assert_eq!(read_mod, expected_mod.to_le_bytes());

        Ok(CrossLayerRecursiveCircuit {
            num_public_inputs: <usize as ExpSerde>::deserialize_from(&mut reader)?,
            num_outputs: <usize as ExpSerde>::deserialize_from(&mut reader)?,
            expected_num_output_zeros: <usize as ExpSerde>::deserialize_from(&mut reader)?,

            segments: Vec::<CrossLayerSegment<F>>::deserialize_from(&mut reader)?,
            layers: <Vec<usize> as ExpSerde>::deserialize_from(&mut reader)?,
        })
    }
}

impl<F: FieldEngine> ExpSerde for Witness<F> {
    fn serialize_into<W: Write>(&self, _writer: W) -> SerdeResult<()> {
        todo!()
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let num_witnesses = <usize as ExpSerde>::deserialize_from(&mut reader)?;
        let num_private_inputs_per_witness = <usize as ExpSerde>::deserialize_from(&mut reader)?;
        let num_public_inputs_per_witness = <usize as ExpSerde>::deserialize_from(&mut reader)?;
        let _modulus = <[u64; 4]>::deserialize_from(&mut reader)?;

        let mut values = vec![];
        for _ in 0..num_witnesses * (num_private_inputs_per_witness + num_public_inputs_per_witness)
        {
            values.push(F::CircuitField::deserialize_from(&mut reader)?);
        }

        Ok(Self {
            num_witnesses,
            num_private_inputs_per_witness,
            num_public_inputs_per_witness,
            values,
        })
    }
}
