use crate::circuit::{
    config::{CircuitField, SIMDField},
    layered::witness::{Witness, WitnessValues},
};

use arith::SimdField;
use serdes::ExpSerde;

use super::{Config, Error, FieldArith, HintCaller, RootCircuit};

#[derive(ExpSerde)]
pub struct WitnessSolver<C: Config> {
    pub circuit: RootCircuit<C>,
}

impl<C: Config> WitnessSolver<C> {
    fn solve_witness_inner(
        &self,
        vars: Vec<CircuitField<C>>,
        public_vars: Vec<CircuitField<C>>,
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> Result<(Vec<CircuitField<C>>, usize), Error> {
        assert_eq!(vars.len(), self.circuit.input_size());
        assert_eq!(public_vars.len(), self.circuit.num_public_inputs);
        let mut a = self.circuit.eval_safe(vars, &public_vars, hint_caller)?;
        let res_len = a.len();
        a.extend(public_vars);
        Ok((a, res_len))
    }

    pub fn solve_witness_from_raw_inputs(
        &self,
        vars: Vec<CircuitField<C>>,
        public_vars: Vec<CircuitField<C>>,
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> Result<Witness<C>, Error> {
        let (values, num_inputs_per_witness) =
            self.solve_witness_inner(vars, public_vars, hint_caller)?;
        Ok(Witness {
            num_witnesses: 1,
            num_inputs_per_witness,
            num_public_inputs_per_witness: self.circuit.num_public_inputs,
            values: WitnessValues::Scalar(values),
        })
    }

    pub fn solve_witnesses_from_raw_inputs<
        F: Fn(usize) -> (Vec<CircuitField<C>>, Vec<CircuitField<C>>),
    >(
        &self,
        num_witnesses: usize,
        f: F,
        hint_caller: &impl HintCaller<CircuitField<C>>,
    ) -> Result<Witness<C>, Error> {
        let mut values = Vec::new();
        let mut num_inputs_per_witness = 0;
        let pack_size = SIMDField::<C>::PACK_SIZE;
        let num_blocks = (num_witnesses + pack_size - 1) / pack_size;
        for j in 0..num_blocks {
            let i_start = j * pack_size;
            let i_end = num_witnesses.min((j + 1) * pack_size);
            let b_end = (j + 1) * pack_size;
            let mut tmp_inputs = Vec::new();
            let mut tmp_public_inputs = Vec::new();
            for i in i_start..i_end {
                let (a, b) = f(i);
                assert_eq!(a.len(), self.circuit.input_size());
                assert_eq!(b.len(), self.circuit.num_public_inputs);
                tmp_inputs.push(a);
                tmp_public_inputs.push(b);
            }
            let mut simd_inputs = Vec::with_capacity(self.circuit.input_size());
            let mut simd_public_inputs = Vec::with_capacity(self.circuit.num_public_inputs);
            let mut tmp: Vec<CircuitField<C>> = vec![CircuitField::<C>::zero(); pack_size];
            for k in 0..self.circuit.input_size() {
                for i in i_start..i_end {
                    tmp[i - i_start] = tmp_inputs[i - i_start][k];
                }
                for i in i_end..b_end {
                    tmp[i - i_start] = tmp[i - i_start - 1];
                }
                simd_inputs.push(SIMDField::<C>::pack(&tmp));
            }
            for k in 0..self.circuit.num_public_inputs {
                for i in i_start..i_end {
                    tmp[i - i_start] = tmp_public_inputs[i - i_start][k];
                }
                for i in i_end..b_end {
                    tmp[i - i_start] = tmp[i - i_start - 1];
                }
                simd_public_inputs.push(SIMDField::<C>::pack(&tmp));
            }
            let simd_result =
                self.circuit
                    .eval_safe_simd(simd_inputs, &simd_public_inputs, hint_caller)?;
            num_inputs_per_witness = simd_result.len();
            values.extend(simd_result);
            values.extend(simd_public_inputs);
        }
        Ok(Witness {
            num_witnesses,
            num_inputs_per_witness,
            num_public_inputs_per_witness: self.circuit.num_public_inputs,
            values: WitnessValues::Simd(values),
        })
    }
}
