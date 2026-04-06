#![allow(clippy::pedantic)]

use std::fmt::Debug;

use arith::Field;

#[derive(Clone, Debug)]
pub struct MulGate<F: Field> {
    pub o_id: usize,
    pub i_ids: [usize; 2],
    pub coef: F,
}

#[derive(Clone, Debug)]
pub struct AddGate<F: Field> {
    pub o_id: usize,
    pub i_id: usize,
    pub coef: F,
}

#[derive(Clone, Debug)]
pub struct ConstGate<F: Field> {
    pub o_id: usize,
    pub coef: F,
}

#[derive(Clone, Debug)]
pub struct CircuitLayer<F: Field> {
    pub input_var_num: usize,
    pub output_var_num: usize,
    pub mul_gates: Vec<MulGate<F>>,
    pub add_gates: Vec<AddGate<F>>,
    pub const_gates: Vec<ConstGate<F>>,
}

impl<F: Field> CircuitLayer<F> {
    pub fn input_size(&self) -> usize {
        1 << self.input_var_num
    }

    pub fn output_size(&self) -> usize {
        1 << self.output_var_num
    }
}

#[derive(Clone, Debug)]
pub struct LayeredCircuit<F: Field> {
    pub layers: Vec<CircuitLayer<F>>,
}

impl<F: Field> LayeredCircuit<F> {
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn evaluate(&self, input: &[F]) -> Vec<Vec<F>> {
        let depth = self.depth();
        let mut layer_vals: Vec<Vec<F>> = Vec::with_capacity(depth + 1);
        layer_vals.push(input.to_vec());

        for i in (0..depth).rev() {
            let layer = &self.layers[i];
            let prev = &layer_vals[depth - 1 - i];
            let out_size = layer.output_size();
            let mut output = vec![F::ZERO; out_size];

            for g in &layer.mul_gates {
                output[g.o_id] += g.coef * prev[g.i_ids[0]] * prev[g.i_ids[1]];
            }
            for g in &layer.add_gates {
                output[g.o_id] += g.coef * prev[g.i_id];
            }
            for g in &layer.const_gates {
                output[g.o_id] += g.coef;
            }

            layer_vals.push(output);
        }

        layer_vals
    }
}

#[derive(Clone, Debug)]
pub struct Proof {
    pub data: Vec<u8>,
}

impl Proof {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

pub trait FiatShamirTranscript: Clone + Default {
    fn append_field_element<F: Field>(&mut self, element: &F);
    fn append_bytes(&mut self, bytes: &[u8]);
    fn challenge_field_element<F: Field>(&mut self) -> F;
    fn finalize_proof(&self) -> Proof;
    fn parse_proof(proof: &Proof) -> Self;
}

pub trait PolynomialCommitment<F: Field>: Clone {
    type SRS;
    type Commitment: Clone + Debug;
    type Opening: Clone + Debug;

    fn setup(size: usize) -> Self::SRS;
    fn commit(srs: &Self::SRS, evals: &[F]) -> Self::Commitment;
    fn open(srs: &Self::SRS, evals: &[F], point: &[F]) -> Self::Opening;
    fn verify(
        srs: &Self::SRS,
        commitment: &Self::Commitment,
        point: &[F],
        value: F,
        opening: &Self::Opening,
    ) -> bool;
}

pub trait ProofSystem {
    type F: Field;
    type Transcript: FiatShamirTranscript;
}
