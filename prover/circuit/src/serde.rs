use gkr_engine::FieldEngine;
use serdes::{ExpSerde, SerdeResult};
use std::io::Read;

use super::Witness;

impl<C: FieldEngine> ExpSerde for Witness<C> {
    fn serialize_into<W: std::io::Write>(&self, mut _writer: W) -> SerdeResult<()> {
        todo!()
    }

    fn deserialize_from<R: Read>(mut reader: R) -> SerdeResult<Self> {
        let num_witnesses = <usize as ExpSerde>::deserialize_from(&mut reader).unwrap();
        let num_private_inputs_per_witness =
            <usize as ExpSerde>::deserialize_from(&mut reader).unwrap();
        let num_public_inputs_per_witness =
            <usize as ExpSerde>::deserialize_from(&mut reader).unwrap();
        let _modulus = <[u64; 4]>::deserialize_from(&mut reader).unwrap();

        let mut values = vec![];
        for _ in 0..num_witnesses * (num_private_inputs_per_witness + num_public_inputs_per_witness)
        {
            values.push(C::CircuitField::deserialize_from(&mut reader).unwrap());
        }

        Ok(Self {
            num_witnesses,
            num_private_inputs_per_witness,
            num_public_inputs_per_witness,
            values,
        })
    }
}
