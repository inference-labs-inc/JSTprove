use std::io::{Read, Write};

use arith::Field;
use gkr_engine::FieldEngine;
use serdes::{ExpSerde, SerdeResult};

use super::circuit::{Circuit, CircuitLayer, StructureInfo};
use super::gates::{CoefType, Gate, GateAdd, GateConst, GateMul, GateUni};

impl ExpSerde for CoefType {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        match self {
            CoefType::Constant => 0u8.serialize_into(&mut writer)?,
            CoefType::Random => 1u8.serialize_into(&mut writer)?,
            CoefType::PublicInput(idx) => {
                2u8.serialize_into(&mut writer)?;
                idx.serialize_into(&mut writer)?
            }
        };
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let tag = u8::deserialize_from(&mut reader)?;
        match tag {
            0 => Ok(CoefType::Constant),
            1 => Ok(CoefType::Random),
            2 => {
                let idx = usize::deserialize_from(&mut reader)?;
                Ok(CoefType::PublicInput(idx))
            }
            _ => panic!("Invalid tag for CoefType"),
        }
    }
}

impl<C: FieldEngine, const INPUT_NUM: usize> ExpSerde for Gate<C, INPUT_NUM> {
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        for id in &self.i_ids {
            id.serialize_into(&mut writer)?;
        }

        self.o_id.serialize_into(&mut writer)?;

        match self.coef_type {
            CoefType::Constant => {
                1u8.serialize_into(&mut writer)?;
                self.coef.serialize_into(&mut writer)?;
            }
            CoefType::Random => {
                2u8.serialize_into(&mut writer)?;
            }
            CoefType::PublicInput(id) => {
                3u8.serialize_into(&mut writer)?;
                id.serialize_into(&mut writer)?;
            }
        }

        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let mut i_ids = [0usize; INPUT_NUM];
        for id in &mut i_ids {
            *id = <usize as ExpSerde>::deserialize_from(&mut reader)?;
        }

        let o_id = <usize as ExpSerde>::deserialize_from(&mut reader)?;

        let coef_type_u8 = u8::deserialize_from(&mut reader)?;
        let (coef_type, coef) = match coef_type_u8 {
            1 => (
                CoefType::Constant,
                C::CircuitField::deserialize_from(&mut reader)?,
            ),
            2 => (CoefType::Random, C::CircuitField::ZERO),
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
            gate_type: 0,
        })
    }
}

impl<C: FieldEngine> ExpSerde for CircuitLayer<C> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.input_var_num.serialize_into(&mut writer)?;
        self.output_var_num.serialize_into(&mut writer)?;
        self.mul.serialize_into(&mut writer)?;
        self.add.serialize_into(&mut writer)?;
        self.const_.serialize_into(&mut writer)?;
        self.uni.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let input_var_num = usize::deserialize_from(&mut reader)?;
        let output_var_num = usize::deserialize_from(&mut reader)?;
        let mul = Vec::<GateMul<C>>::deserialize_from(&mut reader)?;
        let add = Vec::<GateAdd<C>>::deserialize_from(&mut reader)?;
        let const_ = Vec::<GateConst<C>>::deserialize_from(&mut reader)?;
        let uni = Vec::<GateUni<C>>::deserialize_from(&mut reader)?;
        Ok(CircuitLayer {
            input_var_num,
            output_var_num,

            input_vals: vec![],
            output_vals: vec![],

            mul,
            add,
            const_,
            uni,

            structure_info: StructureInfo::default(),
        })
    }
}

const CIRCUIT_VERSION: usize = 0x4A53545F43495231; // b'JST_CIR1'

impl<C: FieldEngine> ExpSerde for Circuit<C> {
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        CIRCUIT_VERSION.serialize_into(&mut writer)?;
        writer.write_all(&C::SENTINEL)?;
        self.expected_num_output_zeros.serialize_into(&mut writer)?;
        self.layers.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let version = usize::deserialize_from(&mut reader)?;
        assert_eq!(version, CIRCUIT_VERSION, "Circuit file version mismatch");
        let mut sentinel = [0u8; 32];
        reader.read_exact(&mut sentinel)?;
        assert_eq!(sentinel, C::SENTINEL, "Circuit field type mismatch");
        let expected_num_output_zeros = usize::deserialize_from(&mut reader)?;
        let layers = Vec::<CircuitLayer<C>>::deserialize_from(&mut reader)?;
        Ok(Circuit {
            layers,
            expected_num_output_zeros,
            ..Default::default()
        })
    }
}
