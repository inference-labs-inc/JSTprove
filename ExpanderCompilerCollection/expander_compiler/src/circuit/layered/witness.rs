use super::*;
use crate::{circuit::config::Config, field::FieldModulus, utils::serde::Serde};

#[derive(Debug)]
pub struct Witness<C: Config> {
    pub num_witnesses: usize,
    pub num_inputs_per_witness: usize,
    pub num_public_inputs_per_witness: usize,
    pub values: Vec<C::CircuitField>,
}

impl<C: Config> Circuit<C> {
    pub fn run(&self, witness: &Witness<C>) -> Vec<bool> {
        if witness.num_witnesses == 0 {
            panic!("expected at least 1 witness")
        }
        let mut res = Vec::new();
        let a = witness.num_inputs_per_witness;
        let b = witness.num_public_inputs_per_witness;
        for i in 0..witness.num_witnesses {
            let (_, out) = self.eval_with_public_inputs(
                witness.values[i * (a + b)..i * (a + b) + a].to_vec(),
                &witness.values[i * (a + b) + a..i * (a + b) + a + b],
            );
            res.push(out);
        }
        res
    }
}

impl<C: Config> Witness<C> {
    pub fn to_simd<T>(&self) -> (Vec<T>, Vec<T>)
    where
        T: arith::SimdField<Scalar = C::CircuitField>,
    {
        match self.num_witnesses.cmp(&T::pack_size()) {
            std::cmp::Ordering::Less => {
                println!(
                    "Warning: not enough witnesses, expect {}, got {}",
                    T::pack_size(),
                    self.num_witnesses
                )
            }
            std::cmp::Ordering::Greater => {
                println!(
                    "Warning: dropping additional witnesses, expect {}, got {}",
                    T::pack_size(),
                    self.num_witnesses
                )
            }
            std::cmp::Ordering::Equal => {}
        }
        let ni = self.num_inputs_per_witness;
        let np = self.num_public_inputs_per_witness;
        let mut res = Vec::with_capacity(ni);
        let mut res_public = Vec::with_capacity(np);
        for i in 0..ni + np {
            let mut values: Vec<C::CircuitField> = (0..self.num_witnesses.min(T::pack_size()))
                .map(|j| self.values[j * (ni + np) + i])
                .collect();
            values.resize(T::pack_size(), C::CircuitField::zero());
            let simd_value = T::pack(&values);
            if i < ni {
                res.push(simd_value);
            } else {
                res_public.push(simd_value);
            }
        }
        (res, res_public)
    }
}

impl<C: Config> Serde for Witness<C> {
    fn deserialize_from<R: std::io::Read>(mut reader: R) -> Result<Self, std::io::Error> {
        let num_witnesses = usize::deserialize_from(&mut reader)?;
        let num_inputs_per_witness = usize::deserialize_from(&mut reader)?;
        let num_public_inputs_per_witness = usize::deserialize_from(&mut reader)?;
        let modulus = ethnum::U256::deserialize_from(&mut reader)?;
        if modulus != C::CircuitField::modulus() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid modulus",
            ));
        }
        let mut values = Vec::with_capacity(
            num_witnesses * (num_inputs_per_witness + num_public_inputs_per_witness),
        );
        for _ in 0..num_witnesses * (num_inputs_per_witness + num_public_inputs_per_witness) {
            values.push(C::CircuitField::deserialize_from(&mut reader)?);
        }
        Ok(Self {
            num_witnesses,
            num_inputs_per_witness,
            num_public_inputs_per_witness,
            values,
        })
    }
    fn serialize_into<W: std::io::Write>(&self, mut writer: W) -> Result<(), std::io::Error> {
        self.num_witnesses.serialize_into(&mut writer)?;
        self.num_inputs_per_witness.serialize_into(&mut writer)?;
        self.num_public_inputs_per_witness
            .serialize_into(&mut writer)?;
        C::CircuitField::modulus().serialize_into(&mut writer)?;
        for v in &self.values {
            v.serialize_into(&mut writer)?;
        }
        Ok(())
    }
}
