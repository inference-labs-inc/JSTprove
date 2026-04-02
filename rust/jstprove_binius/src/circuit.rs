use binius_core::verify::verify_constraints;
use binius_frontend::{CircuitBuilder, Wire};

pub struct BiniusCircuit {
    pub circuit: binius_frontend::Circuit,
    pub input_wires: Vec<Wire>,
    pub output_wires: Vec<Wire>,
}

pub fn build_iadd_chain(n_additions: usize) -> BiniusCircuit {
    let builder = CircuitBuilder::new();

    let mut input_wires = Vec::with_capacity(n_additions + 1);
    for _ in 0..=n_additions {
        input_wires.push(builder.add_witness());
    }

    let zero = builder.add_constant_64(0);
    let mut acc = input_wires[0];
    let mut output_wires = Vec::with_capacity(n_additions);
    for &inp in &input_wires[1..] {
        let (sum, _carry) = builder.iadd_cin_cout(acc, inp, zero);
        acc = sum;
        output_wires.push(acc);
    }

    let circuit = builder.build();
    BiniusCircuit {
        circuit,
        input_wires,
        output_wires,
    }
}

pub fn build_bitwise_chain(n_ops: usize) -> BiniusCircuit {
    let builder = CircuitBuilder::new();

    let mut input_wires = Vec::with_capacity(n_ops + 1);
    for _ in 0..=n_ops {
        input_wires.push(builder.add_witness());
    }

    let mut acc = input_wires[0];
    let mut output_wires = Vec::with_capacity(n_ops);
    for (i, &inp) in input_wires[1..].iter().enumerate() {
        acc = if i % 2 == 0 {
            builder.bxor(acc, inp)
        } else {
            builder.band(acc, inp)
        };
        output_wires.push(acc);
    }

    let circuit = builder.build();
    BiniusCircuit {
        circuit,
        input_wires,
        output_wires,
    }
}

pub fn verify_circuit_constraints(
    bc: &BiniusCircuit,
    witness: &binius_core::constraint_system::ValueVec,
) -> anyhow::Result<()> {
    let cs = bc.circuit.constraint_system();
    verify_constraints(cs, witness).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
