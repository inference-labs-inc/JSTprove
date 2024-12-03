use crate::{circuit::layered::witness::Witness, utils::serde::Serde};

use super::*;

pub struct WitnessSolver<C: Config> {
    pub circuit: RootCircuit<C>,
}

impl<C: Config> WitnessSolver<C> {
    fn solve_witness_inner(
        &self,
        vars: Vec<C::CircuitField>,
        public_vars: Vec<C::CircuitField>,
    ) -> Result<(Vec<C::CircuitField>, usize), Error> {
        assert_eq!(vars.len(), self.circuit.input_size());
        assert_eq!(public_vars.len(), self.circuit.num_public_inputs);
        let mut a = self.circuit.eval_with_public_inputs(vars, &public_vars)?;
        let res_len = a.len();
        a.extend(public_vars);
        Ok((a, res_len))
    }

    pub fn solve_witness_from_raw_inputs(
        &self,
        vars: Vec<C::CircuitField>,
        public_vars: Vec<C::CircuitField>,
    ) -> Result<Witness<C>, Error> {
        let (values, num_inputs_per_witness) = self.solve_witness_inner(vars, public_vars)?;
        Ok(Witness {
            num_witnesses: 1,
            num_inputs_per_witness,
            num_public_inputs_per_witness: self.circuit.num_public_inputs,
            values,
        })
    }

    pub fn solve_witnesses_from_raw_inputs<
        F: Fn(usize) -> (Vec<C::CircuitField>, Vec<C::CircuitField>),
    >(
        &self,
        num_witnesses: usize,
        f: F,
    ) -> Result<Witness<C>, Error> {
        let mut values = Vec::new();
        let mut num_inputs_per_witness = 0;
        for i in 0..num_witnesses {
            let (a, b) = f(i);
            let (a, num) = self.solve_witness_inner(a, b)?;
            values.extend(a);
            num_inputs_per_witness = num;
        }
        Ok(Witness {
            num_witnesses,
            num_inputs_per_witness,
            num_public_inputs_per_witness: self.circuit.num_public_inputs,
            values,
        })
    }
}

impl<C: Config> Serde for WitnessSolver<C> {
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> Result<Self, std::io::Error> {
        let circuit = RootCircuit::<C>::deserialize_from(&mut reader)?;
        Ok(Self { circuit })
    }
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        self.circuit.serialize_into(&mut writer)
    }
}
