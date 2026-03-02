use std::{
    collections::HashMap,
    io::{Error as IoError, Read, Write},
};

use serdes::{ExpSerde, SerdeResult};

use super::{Circuit, IrConfig, RootCircuit};
use crate::circuit::config::Config;

impl<Irc: IrConfig> ExpSerde for Circuit<Irc>
where
    Irc::Instruction: ExpSerde,
    Irc::Constraint: ExpSerde,
{
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.instructions.serialize_into(&mut writer)?;
        self.constraints.serialize_into(&mut writer)?;
        self.outputs.serialize_into(&mut writer)?;
        self.num_inputs.serialize_into(&mut writer)?;
        Ok(())
    }
    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let instructions = Vec::<Irc::Instruction>::deserialize_from(&mut reader)?;
        let constraints = Vec::<Irc::Constraint>::deserialize_from(&mut reader)?;
        let outputs = Vec::<usize>::deserialize_from(&mut reader)?;
        let num_inputs = usize::deserialize_from(&mut reader)?;
        Ok(Circuit {
            instructions,
            constraints,
            outputs,
            num_inputs,
        })
    }
}

impl<Irc: IrConfig> ExpSerde for RootCircuit<Irc>
where
    Irc::Instruction: ExpSerde,
    Irc::Constraint: ExpSerde,
{
    fn serialize_into<W: Write>(&self, mut writer: W) -> SerdeResult<()> {
        Irc::Config::CONFIG_ID.serialize_into(&mut writer)?;
        self.num_public_inputs.serialize_into(&mut writer)?;
        self.expected_num_output_zeroes
            .serialize_into(&mut writer)?;
        self.circuits.serialize_into(&mut writer)?;
        Ok(())
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let config_id = usize::deserialize_from(&mut reader)?;
        if config_id != Irc::Config::CONFIG_ID {
            return Err(IoError::new(
                std::io::ErrorKind::InvalidData,
                "config id mismatch",
            ))?;
        }
        let num_public_inputs = usize::deserialize_from(&mut reader)?;
        let expected_num_output_zeroes = usize::deserialize_from(&mut reader)?;
        let circuits = HashMap::<usize, Circuit<Irc>>::deserialize_from(&mut reader)?;
        Ok(RootCircuit {
            num_public_inputs,
            expected_num_output_zeroes,
            circuits,
        })
    }
}
