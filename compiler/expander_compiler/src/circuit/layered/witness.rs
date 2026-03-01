use std::any::{Any, TypeId};
use std::mem;

use arith::{Field, SimdField as ExpSimdField};
use serdes::{ExpSerde, SerdeResult};

use super::{Circuit, InputType};
use crate::circuit::config::{CircuitField, Config, SIMDField};

#[derive(Clone, Debug)]
pub enum WitnessValues<C: Config> {
    Scalar(Vec<CircuitField<C>>),
    Simd(Vec<SIMDField<C>>),
}

#[derive(Clone, Debug)]
pub struct Witness<C: Config> {
    pub num_witnesses: usize,
    pub num_inputs_per_witness: usize,
    pub num_public_inputs_per_witness: usize,
    pub values: WitnessValues<C>,
}

fn unpack_block<F: Field, SF: arith::SimdField<Scalar = F>>(
    s: &[SF],
    a: usize,
    b: usize,
) -> Vec<(Vec<F>, Vec<F>)> {
    let pack_size = SF::PACK_SIZE;
    let mut res = Vec::with_capacity(pack_size);
    for _ in 0..pack_size {
        res.push((Vec::with_capacity(a), Vec::with_capacity(b)));
    }
    for x in s.iter().take(a) {
        let tmp = x.unpack();
        for j in 0..pack_size {
            res[j].0.push(tmp[j]);
        }
    }
    for x in s.iter().skip(a).take(b) {
        let tmp = x.unpack();
        for j in 0..pack_size {
            res[j].1.push(tmp[j]);
        }
    }
    res
}

fn pack_block<F: Field, SF: arith::SimdField<Scalar = F>>(
    s: &[F],
    a: usize,
    b: usize,
) -> (Vec<SF>, Vec<SF>) {
    let pack_size = SF::PACK_SIZE;
    let mut res = Vec::with_capacity(a);
    let mut res2 = Vec::with_capacity(b);
    let s_size = (s.len() / (a + b)).min(pack_size);
    for i in 0..a {
        let mut tmp = Vec::with_capacity(pack_size);
        for j in 0..s_size {
            tmp.push(s[j * (a + b) + i]);
        }
        // fill the rest with the last element
        for _ in s_size..pack_size {
            tmp.push(s[(s_size - 1) * (a + b) + i]);
        }
        res.push(SF::pack(&tmp));
    }
    for i in a..a + b {
        let mut tmp = Vec::with_capacity(pack_size);
        for j in 0..s_size {
            tmp.push(s[j * (a + b) + i]);
        }
        // fill the rest with the last element
        for _ in s_size..pack_size {
            tmp.push(s[(s_size - 1) * (a + b) + i]);
        }
        res2.push(SF::pack(&tmp));
    }
    (res, res2)
}

fn use_simd<C: Config>(num_witnesses: usize) -> bool {
    num_witnesses > 1 && SIMDField::<C>::PACK_SIZE > 1
}

type UnpackedBlock<C> = Vec<(Vec<CircuitField<C>>, Vec<CircuitField<C>>)>;

pub struct WitnessIteratorScalar<'a, C: Config> {
    witness: &'a Witness<C>,
    index: usize,
    buf_unpacked: UnpackedBlock<C>,
}

impl<'a, C: Config> Iterator for WitnessIteratorScalar<'a, C> {
    type Item = (Vec<CircuitField<C>>, Vec<CircuitField<C>>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.witness.num_witnesses {
            return None;
        }
        let a = self.witness.num_inputs_per_witness;
        let b = self.witness.num_public_inputs_per_witness;
        match &self.witness.values {
            WitnessValues::Scalar(values) => {
                let res = (
                    values[self.index * (a + b)..self.index * (a + b) + a].to_vec(),
                    values[self.index * (a + b) + a..self.index * (a + b) + a + b].to_vec(),
                );
                self.index += 1;
                Some(res)
            }
            WitnessValues::Simd(values) => {
                let pack_size = SIMDField::<C>::PACK_SIZE;
                if self.index % pack_size == 0 {
                    self.buf_unpacked =
                        unpack_block(&values[(self.index / pack_size) * (a + b)..], a, b);
                }
                let res = (
                    mem::take(&mut self.buf_unpacked[self.index % pack_size].0),
                    mem::take(&mut self.buf_unpacked[self.index % pack_size].1),
                );
                self.index += 1;
                Some(res)
            }
        }
    }
}

pub struct WitnessIteratorSimd<'a, C: Config> {
    witness: &'a Witness<C>,
    index: usize,
}

impl<'a, C: Config> Iterator for WitnessIteratorSimd<'a, C> {
    type Item = (Vec<SIMDField<C>>, Vec<SIMDField<C>>);
    fn next(&mut self) -> Option<Self::Item> {
        let pack_size = SIMDField::<C>::PACK_SIZE;
        if self.index * pack_size >= self.witness.num_witnesses {
            return None;
        }
        let a = self.witness.num_inputs_per_witness;
        let b = self.witness.num_public_inputs_per_witness;
        match &self.witness.values {
            WitnessValues::Scalar(values) => {
                let (inputs, public_inputs) =
                    pack_block(&values[self.index * pack_size * (a + b)..], a, b);
                self.index += 1;
                Some((inputs, public_inputs))
            }
            WitnessValues::Simd(values) => {
                let inputs = values[self.index * (a + b)..self.index * (a + b) + a].to_vec();
                let public_inputs =
                    values[self.index * (a + b) + a..self.index * (a + b) + a + b].to_vec();
                self.index += 1;
                Some((inputs, public_inputs))
            }
        }
    }
}

impl<C: Config> Witness<C> {
    pub fn iter_scalar(&self) -> WitnessIteratorScalar<'_, C> {
        WitnessIteratorScalar {
            witness: self,
            index: 0,
            buf_unpacked: Vec::new(),
        }
    }

    pub fn iter_simd(&self) -> WitnessIteratorSimd<'_, C> {
        WitnessIteratorSimd {
            witness: self,
            index: 0,
        }
    }

    fn convert_to_simd(&mut self) {
        let values = match &self.values {
            WitnessValues::Scalar(values) => values,
            WitnessValues::Simd(_) => {
                return;
            }
        };
        let mut res = Vec::new();
        let a = self.num_inputs_per_witness + self.num_public_inputs_per_witness;
        let pack_size = SIMDField::<C>::PACK_SIZE;
        let num_blocks = (self.num_witnesses + pack_size - 1) / pack_size;
        for i in 0..num_blocks {
            let tmp =
                pack_block::<CircuitField<C>, SIMDField<C>>(&values[i * pack_size * a..], a, 0);
            res.extend(tmp.0);
        }
        self.values = WitnessValues::Simd(res);
    }
}

impl<C: Config, I: InputType> Circuit<C, I> {
    fn run_inner(
        &self,
        witness: &Witness<C>,
        need_output: bool,
    ) -> (Vec<bool>, Vec<Vec<CircuitField<C>>>) {
        if witness.num_witnesses == 0 {
            panic!("expected at least 1 witness")
        }
        let mut outputs = Vec::new();
        let mut constraints = Vec::new();
        if use_simd::<C>(witness.num_witnesses) {
            for (inputs, public_inputs) in witness.iter_simd() {
                let (out, constraint_result) =
                    self.eval_with_public_inputs_simd(inputs, &public_inputs);
                if need_output {
                    let n = outputs.len();
                    for _ in 0..SIMDField::<C>::PACK_SIZE {
                        outputs.push(Vec::new());
                    }
                    for o in out {
                        for (i, x) in o.unpack().iter().enumerate() {
                            outputs[n + i].push(*x);
                        }
                    }
                }
                constraints.extend(constraint_result);
            }
            outputs.truncate(witness.num_witnesses);
            constraints.truncate(witness.num_witnesses);
        } else {
            for (inputs, public_inputs) in witness.iter_scalar() {
                let (out, constraint_result) = self.eval_with_public_inputs(inputs, &public_inputs);
                outputs.push(out);
                constraints.push(constraint_result);
            }
        }
        (constraints, outputs)
    }

    pub fn run(&self, witness: &Witness<C>) -> Vec<bool> {
        let (constraints, _) = self.run_inner(witness, false);
        constraints
    }

    pub fn run_with_output(&self, witness: &Witness<C>) -> (Vec<bool>, Vec<Vec<CircuitField<C>>>) {
        self.run_inner(witness, true)
    }
}

impl<C: Config> Witness<C> {
    pub fn to_simd<T>(&self) -> (Vec<T>, Vec<T>)
    where
        T: arith::SimdField<Scalar = CircuitField<C>> + 'static,
    {
        match self.num_witnesses.cmp(&T::PACK_SIZE) {
            std::cmp::Ordering::Less => {
                println!(
                    "Warning: not enough witnesses, expect {}, got {}",
                    T::PACK_SIZE,
                    self.num_witnesses
                )
            }
            std::cmp::Ordering::Greater => {
                println!(
                    "Warning: dropping additional witnesses, expect {}, got {}",
                    T::PACK_SIZE,
                    self.num_witnesses
                )
            }
            std::cmp::Ordering::Equal => {}
        }
        let a = self.num_inputs_per_witness;
        let b = self.num_public_inputs_per_witness;
        match &self.values {
            WitnessValues::Scalar(values) => pack_block(values, a, b),
            WitnessValues::Simd(values) => {
                if TypeId::of::<T>() == TypeId::of::<SIMDField<C>>() {
                    let inputs = values[..a].to_vec();
                    let public_inputs = values[a..a + b].to_vec();
                    let tmp: Box<dyn Any> = Box::new((inputs, public_inputs));
                    match tmp.downcast::<(Vec<T>, Vec<T>)>() {
                        Ok(t) => {
                            return *t;
                        }
                        Err(_) => panic!("downcast failed"),
                    }
                }
                let mut tmp = Vec::new();
                for (x, y) in self.iter_scalar().take(T::PACK_SIZE) {
                    tmp.extend(x);
                    tmp.extend(y);
                }
                pack_block(&tmp, a, b)
            }
        }
    }
}

impl<C: Config> ExpSerde for Witness<C> {
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> SerdeResult<Self> {
        let num_witnesses = usize::deserialize_from(&mut reader)?;
        let num_inputs_per_witness = usize::deserialize_from(&mut reader)?;
        let num_public_inputs_per_witness = usize::deserialize_from(&mut reader)?;
        let modulus = ethnum::U256::deserialize_from(&mut reader)?;
        if modulus != CircuitField::<C>::MODULUS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid modulus",
            ))?;
        }
        let mut values = Vec::with_capacity(
            num_witnesses * (num_inputs_per_witness + num_public_inputs_per_witness),
        );
        for _ in 0..num_witnesses * (num_inputs_per_witness + num_public_inputs_per_witness) {
            values.push(CircuitField::<C>::deserialize_from(&mut reader)?);
        }
        let mut res = Self {
            num_witnesses,
            num_inputs_per_witness,
            num_public_inputs_per_witness,
            values: WitnessValues::Scalar(values),
        };
        if use_simd::<C>(num_witnesses) {
            res.convert_to_simd();
        }
        Ok(res)
    }

    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> SerdeResult<()> {
        self.num_witnesses.serialize_into(&mut writer)?;
        self.num_inputs_per_witness.serialize_into(&mut writer)?;
        self.num_public_inputs_per_witness
            .serialize_into(&mut writer)?;
        CircuitField::<C>::MODULUS.serialize_into(&mut writer)?;
        match &self.values {
            WitnessValues::Scalar(values) => {
                for v in values {
                    v.serialize_into(&mut writer)?;
                }
            }
            WitnessValues::Simd(_) => {
                for (a, b) in self.iter_scalar() {
                    for v in a {
                        v.serialize_into(&mut writer)?;
                    }
                    for v in b {
                        v.serialize_into(&mut writer)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{field::M31, frontend::M31Config};
    use arith::Field;

    #[test]
    fn basic_simd() {
        let n = 29;
        let a = 17;
        let b = 5;
        let mut v = Vec::new();
        for _ in 0..n * (a + b) {
            v.push(M31::random_unsafe(&mut rand::thread_rng()));
        }
        let w1: Witness<M31Config> = Witness {
            num_witnesses: n,
            num_inputs_per_witness: a,
            num_public_inputs_per_witness: b,
            values: WitnessValues::<M31Config>::Scalar(v),
        };
        let mut w2 = w1.clone();
        w2.convert_to_simd();
        let w1_iv_sc = w1.iter_scalar().collect::<Vec<_>>();
        let w2_iv_sc = w2.iter_scalar().collect::<Vec<_>>();
        let w1_iv_sm = w1.iter_simd().collect::<Vec<_>>();
        let w2_iv_sm = w2.iter_simd().collect::<Vec<_>>();
        assert_eq!(w1_iv_sc, w2_iv_sc);
        assert_eq!(w1_iv_sm, w2_iv_sm);
    }
}
