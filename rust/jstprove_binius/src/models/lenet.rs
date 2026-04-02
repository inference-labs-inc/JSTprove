use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

use crate::layers::{conv2d, gemm, maxpool, relu};

pub struct LeNetCircuit {
    pub circuit: binius_frontend::Circuit,
    pub input_wires: Vec<Wire>,
    pub output_wires: Vec<Wire>,
    weight_wires: LeNetWeights,
}

struct LeNetWeights {
    conv1_w: Vec<Wire>,
    conv1_b: Vec<Wire>,
    conv2_w: Vec<Wire>,
    conv2_b: Vec<Wire>,
    fc1_w: Vec<Wire>,
    fc1_b: Vec<Wire>,
    fc2_w: Vec<Wire>,
    fc2_b: Vec<Wire>,
    fc3_w: Vec<Wire>,
    fc3_b: Vec<Wire>,
}

const RESCALE: u32 = 12;
const IN_C: usize = 3;
const IN_H: usize = 32;
const IN_W: usize = 32;

fn build_lenet(b: &CircuitBuilder) -> (Vec<Wire>, Vec<Wire>, LeNetWeights) {
    let input: Vec<Wire> = (0..IN_C * IN_H * IN_W).map(|_| b.add_witness()).collect();

    let conv1_w: Vec<Wire> = (0..6 * IN_C * 5 * 5).map(|_| b.add_witness()).collect();
    let conv1_b: Vec<Wire> = (0..6).map(|_| b.add_witness()).collect();
    let (x, h, w) = conv2d::conv2d(
        b, &input, &conv1_w, &conv1_b, IN_C, 6, IN_H, IN_W, 5, 1, RESCALE,
    );
    let x = relu::relu_batch(b, &x);
    let (x, h, w) = maxpool::maxpool2d(b, &x, 6, h, w, 2, 2);

    let conv2_w: Vec<Wire> = (0..16 * 6 * 5 * 5).map(|_| b.add_witness()).collect();
    let conv2_b: Vec<Wire> = (0..16).map(|_| b.add_witness()).collect();
    let (x, h, w) = conv2d::conv2d(b, &x, &conv2_w, &conv2_b, 6, 16, h, w, 5, 1, RESCALE);
    let x = relu::relu_batch(b, &x);
    let (x, _h, _w) = maxpool::maxpool2d(b, &x, 16, h, w, 2, 2);

    let flat_size = x.len();

    let fc1_w: Vec<Wire> = (0..120 * flat_size).map(|_| b.add_witness()).collect();
    let fc1_b: Vec<Wire> = (0..120).map(|_| b.add_witness()).collect();
    let x = gemm::gemm(b, &x, &fc1_w, &fc1_b, flat_size, 120, RESCALE);
    let x = relu::relu_batch(b, &x);

    let fc2_w: Vec<Wire> = (0..84 * 120).map(|_| b.add_witness()).collect();
    let fc2_b: Vec<Wire> = (0..84).map(|_| b.add_witness()).collect();
    let x = gemm::gemm(b, &x, &fc2_w, &fc2_b, 120, 84, RESCALE);
    let x = relu::relu_batch(b, &x);

    let fc3_w: Vec<Wire> = (0..10 * 84).map(|_| b.add_witness()).collect();
    let fc3_b: Vec<Wire> = (0..10).map(|_| b.add_witness()).collect();
    let output = gemm::gemm(b, &x, &fc3_w, &fc3_b, 84, 10, RESCALE);

    let weights = LeNetWeights {
        conv1_w,
        conv1_b,
        conv2_w,
        conv2_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        fc3_w,
        fc3_b,
    };

    (input, output, weights)
}

pub fn build() -> LeNetCircuit {
    let builder = CircuitBuilder::new();
    let (input_wires, output_wires, weight_wires) = build_lenet(&builder);
    let circuit = builder.build();
    LeNetCircuit {
        circuit,
        input_wires,
        output_wires,
        weight_wires,
    }
}

pub fn fill_dummy_witness(lenet: &LeNetCircuit) -> binius_core::constraint_system::ValueVec {
    let mut filler = lenet.circuit.new_witness_filler();
    for &w in &lenet.input_wires {
        filler[w] = Word(1);
    }
    fill_weight_group(&mut filler, &lenet.weight_wires.conv1_w, 1);
    fill_weight_group(&mut filler, &lenet.weight_wires.conv1_b, 0);
    fill_weight_group(&mut filler, &lenet.weight_wires.conv2_w, 1);
    fill_weight_group(&mut filler, &lenet.weight_wires.conv2_b, 0);
    fill_weight_group(&mut filler, &lenet.weight_wires.fc1_w, 1);
    fill_weight_group(&mut filler, &lenet.weight_wires.fc1_b, 0);
    fill_weight_group(&mut filler, &lenet.weight_wires.fc2_w, 1);
    fill_weight_group(&mut filler, &lenet.weight_wires.fc2_b, 0);
    fill_weight_group(&mut filler, &lenet.weight_wires.fc3_w, 1);
    fill_weight_group(&mut filler, &lenet.weight_wires.fc3_b, 0);
    lenet.circuit.populate_wire_witness(&mut filler).unwrap();
    filler.into_value_vec()
}

fn fill_weight_group(filler: &mut WitnessFiller, wires: &[Wire], val: u64) {
    for &w in wires {
        filler[w] = Word(val);
    }
}
