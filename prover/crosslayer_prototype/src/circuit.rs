use crate::gates::{SimpleGateAdd, SimpleGateCst, SimpleGateMul};
use arith::Field;
use gkr_engine::FieldEngine;
use rand::RngCore;

use super::CrossLayerRelay;

#[derive(Debug, Clone, Default)]
pub struct GenericLayer<F: FieldEngine> {
    pub layer_id: usize,
    pub layer_size: usize,
    pub input_layer_size: usize,

    pub add_gates: Vec<SimpleGateAdd<F>>,
    pub mul_gates: Vec<SimpleGateMul<F>>,
    pub const_gates: Vec<SimpleGateCst<F>>,
    pub relay_gates: Vec<CrossLayerRelay<F>>,
}

impl<F: FieldEngine> GenericLayer<F> {
    pub fn random_for_bench(
        mut rng: impl RngCore,
        layer_id: usize,
        layer_sizes: &[usize],
        n_gates_per_layer: usize,
    ) -> Self {
        if layer_id == 0 {
            return Self::default();
        }

        let output_size = layer_sizes[layer_id];
        let input_size = layer_sizes[layer_id - 1];
        let mut layer = GenericLayer::<F> {
            layer_id,
            layer_size: output_size,
            input_layer_size: input_size,
            ..Default::default()
        };

        for _ in 0..n_gates_per_layer {
            let gate_type = rng.next_u64() as usize % 9;
            if gate_type < 4 {
                let gate = SimpleGateAdd::random_for_testing(&mut rng, output_size, input_size);
                layer.add_gates.push(gate);
            } else if gate_type < 6 {
                let i_layer = rng.next_u64() as usize % layer_id;
                let gate = CrossLayerRelay::random_for_testing(
                    &mut rng,
                    output_size,
                    layer_sizes[i_layer],
                    i_layer,
                );
                layer.relay_gates.push(gate);
            } else if gate_type < 8 {
                let gate = SimpleGateMul::random_for_testing(&mut rng, output_size, input_size);
                layer.mul_gates.push(gate);
            } else {
                let gate = SimpleGateCst::random_for_testing(&mut rng, output_size, input_size);
                layer.const_gates.push(gate);
            }
        }

        layer
    }

    pub fn random_for_testing(
        mut rng: impl RngCore,
        layer_id: usize,
        layer_sizes: &[usize],
    ) -> Self {
        if layer_id == 0 {
            return Self::default();
        }
        let output_size = layer_sizes[layer_id];
        let input_size = layer_sizes[layer_id - 1];
        let mut layer = GenericLayer::<F> {
            layer_id,
            layer_size: output_size,
            input_layer_size: input_size,
            ..Default::default()
        };

        let n_gates = output_size * 2; // Not necessarily 2, just for testing

        for _ in 0..n_gates {
            let gate_type = rng.next_u64() as usize % 4;
            match gate_type {
                0 => {
                    let gate = SimpleGateAdd::random_for_testing(&mut rng, output_size, input_size);
                    layer.add_gates.push(gate);
                }
                1 => {
                    let gate = SimpleGateMul::random_for_testing(&mut rng, output_size, input_size);
                    layer.mul_gates.push(gate);
                }
                2 => {
                    let gate = SimpleGateCst::random_for_testing(&mut rng, output_size, input_size);
                    layer.const_gates.push(gate);
                }
                3 => {
                    let i_layer = rng.next_u64() as usize % layer_id;
                    let gate = CrossLayerRelay::random_for_testing(
                        &mut rng,
                        output_size,
                        layer_sizes[i_layer],
                        i_layer,
                    );
                    layer.relay_gates.push(gate);
                }
                _ => unreachable!(),
            }
        }

        layer
    }
}

#[derive(Debug, Clone, Default)]
pub struct CrossLayerCircuitEvals<F: FieldEngine> {
    pub vals: Vec<Vec<F::SimdCircuitField>>,
}

#[derive(Debug, Clone, Default)]
pub struct CrossLayerCircuit<F: FieldEngine> {
    pub layers: Vec<GenericLayer<F>>,
}

impl<F: FieldEngine> CrossLayerCircuit<F> {
    pub fn random_for_bench(
        mut rng: impl RngCore,
        n_layers: usize,
        size_of_each_layer: usize,
        n_gates_each_layer: usize,
    ) -> Self {
        let layer_sizes = vec![size_of_each_layer; n_layers];
        let mut circuit = Self::default();
        circuit.layers.push(GenericLayer::<F> {
            layer_id: 0,
            layer_size: layer_sizes[0],
            ..Default::default()
        });

        for i in 1..layer_sizes.len() {
            let layer =
                GenericLayer::<F>::random_for_bench(&mut rng, i, &layer_sizes, n_gates_each_layer);
            circuit.layers.push(layer);
        }
        circuit
    }

    pub fn random_for_testing(mut rng: impl RngCore, n_layers: usize) -> Self {
        let layer_sizes = (0..n_layers)
            .map(|i_layer| 1usize << (n_layers - 1 - i_layer))
            .collect::<Vec<_>>();

        let mut circuit = Self::default();
        circuit.layers.push(GenericLayer::<F> {
            layer_id: 0,
            layer_size: layer_sizes[0],
            ..Default::default()
        }); // layer 0 is input layer, no gates

        for i in 1..layer_sizes.len() {
            let layer = GenericLayer::random_for_testing(&mut rng, i, &layer_sizes);
            circuit.layers.push(layer);
        }
        circuit
    }

    pub fn evaluate(&self, input: &[F::SimdCircuitField]) -> CrossLayerCircuitEvals<F> {
        let mut vals = Vec::with_capacity(self.layers.len());
        vals.push(input.to_vec());

        for i_layer in 1..self.layers.len() {
            let layer = &self.layers[i_layer];
            let mut new_layer_vals = vec![F::SimdCircuitField::zero(); layer.layer_size];

            for gate in &layer.add_gates {
                new_layer_vals[gate.o_id] += vals[i_layer - 1][gate.i_ids[0]] * gate.coef;
            }

            for gate in &layer.mul_gates {
                new_layer_vals[gate.o_id] += (vals[i_layer - 1][gate.i_ids[0]]
                    * vals[i_layer - 1][gate.i_ids[1]])
                    * gate.coef;
            }

            for gate in &layer.const_gates {
                new_layer_vals[gate.o_id] += F::SimdCircuitField::from(gate.coef);
            }

            for gate in &layer.relay_gates {
                new_layer_vals[gate.o_id] += vals[gate.i_layer][gate.i_id] * gate.coef;
            }

            vals.push(new_layer_vals);
        }

        CrossLayerCircuitEvals { vals }
    }

    pub fn max_num_input_var(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                if layer.input_layer_size > 0 {
                    layer.input_layer_size.trailing_zeros() as usize
                } else {
                    0
                }
            })
            .max()
            .unwrap()
    }

    pub fn max_num_output_var(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                if layer.layer_size > 0 {
                    layer.layer_size.trailing_zeros() as usize
                } else {
                    0
                }
            })
            .max()
            .unwrap()
    }

    pub fn print_stats(&self) {
        let mut n_add_gates = 0;
        let mut n_mul_gates = 0;
        let mut n_const_gates = 0;
        let mut n_relay_gates = 0;

        for layer in &self.layers {
            n_add_gates += layer.add_gates.len();
            n_mul_gates += layer.mul_gates.len();
            n_const_gates += layer.const_gates.len();
            n_relay_gates += layer.relay_gates.len();
        }

        println!("Number of layers: {}", self.layers.len());
        println!("Number of add gates: {n_add_gates}");
        println!("Number of mul gates: {n_mul_gates}");
        println!("Number of const gates: {n_const_gates}");
        println!("Number of relay gates: {n_relay_gates}");
    }
}

// CrossLayerConnections is a struct that stores the connections between the gates of different
// layers TODO-Optimization: This does not seem to be memory efficient
#[derive(Debug, Clone, Default)]
pub struct CrossLayerConnections {
    pub connections: Vec<Vec<Vec<(usize, usize)>>>, /* connections[i][j] = (k, l) means output
                                                     * layer i, gate k is connected to input
                                                     * layer j, gate l */
}

impl CrossLayerConnections {
    pub fn parse_circuit<F: FieldEngine>(c: &CrossLayerCircuit<F>) -> Self {
        let mut connections = vec![vec![vec![]; c.layers.len()]; c.layers.len()];

        #[allow(clippy::needless_range_loop)]
        for o_layer in 1..c.layers.len() {
            for gate in &c.layers[o_layer].relay_gates {
                connections[o_layer][gate.i_layer].push((gate.o_id, gate.i_id));
            }
        }

        CrossLayerConnections { connections }
    }
}

// A direct copy of the witness struct from ecc
#[derive(Debug, Clone)]
pub struct Witness<F: FieldEngine> {
    pub num_witnesses: usize,
    pub num_private_inputs_per_witness: usize,
    pub num_public_inputs_per_witness: usize,
    pub values: Vec<F::CircuitField>,
}
