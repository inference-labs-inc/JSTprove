use crate::frontend::M31Config as C;
use crate::{
    circuit::{
        config::Config,
        input_mapping::InputMapping,
        ir::{
            common::rand_gen::*,
            dest::{
                Circuit as IrCircuit, Instruction as IrInstruction, RootCircuit as IrRootCircuit,
            },
            expr::{Expression, Term},
        },
        layered::{self, CrossLayerInputType, InputType, NormalInputType},
    },
    frontend::CircuitField,
};

use crate::field::M31 as CField;

use crate::field::FieldArith;

use super::{compile, CompileOptions};

pub fn test_input<C: Config, I: InputType>(
    rc: &IrRootCircuit<C>,
    lc: &layered::Circuit<C, I>,
    input_mapping: &InputMapping,
    input: &Vec<CircuitField<C>>,
) {
    let (rc_output, rc_cond) = rc.eval_unsafe(input.clone());
    let lc_input = input_mapping.map_inputs(input);
    let (lc_output, lc_cond) = lc.eval_unsafe(lc_input);
    assert_eq!(rc_cond, lc_cond);
    assert_eq!(rc_output, lc_output);
}

pub fn compile_and_random_test<C: Config, I: InputType>(
    rc: &IrRootCircuit<C>,
    n_tests: usize,
) -> (layered::Circuit<C, I>, InputMapping) {
    assert!(rc.validate().is_ok());
    let (lc, input_mapping) = compile(
        rc,
        CompileOptions {
            allow_input_reorder: true,
        },
    );
    assert_eq!(lc.validate(), Ok(()));
    assert_eq!(rc.input_size(), input_mapping.cur_size());
    let input_size = rc.input_size();
    for _ in 0..n_tests {
        let input: Vec<CircuitField<C>> = (0..input_size)
            .map(|_| CircuitField::<C>::random_unsafe(&mut rand::thread_rng()))
            .collect();
        test_input(rc, &lc, &input_mapping, &input);
    }
    (lc, input_mapping)
}

#[test]
fn random_circuits_1() {
    let mut config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 10 },
        num_inputs: RandomRange { min: 1, max: 10 },
        num_instructions: RandomRange { min: 1, max: 10 },
        num_constraints: RandomRange { min: 0, max: 10 },
        num_outputs: RandomRange { min: 1, max: 10 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.5,
    };
    for i in 0..1000 {
        config.seed = i;
        let root = IrRootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        compile_and_random_test::<_, NormalInputType>(&root, 5);
        compile_and_random_test::<_, CrossLayerInputType>(&root, 5);
    }
}

#[test]
fn random_circuits_2() {
    let mut config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 100 },
        num_inputs: RandomRange { min: 1, max: 3 },
        num_instructions: RandomRange { min: 5, max: 10 },
        num_constraints: RandomRange { min: 0, max: 5 },
        num_outputs: RandomRange { min: 1, max: 3 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.1,
    };
    for i in 0..1000 {
        config.seed = i + 10000;
        let root = IrRootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        compile_and_random_test::<_, NormalInputType>(&root, 5);
        compile_and_random_test::<_, CrossLayerInputType>(&root, 5);
    }
}

#[test]
fn random_circuits_3() {
    let mut config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 20 },
        num_inputs: RandomRange { min: 1, max: 3 },
        num_instructions: RandomRange { min: 30, max: 50 },
        num_constraints: RandomRange { min: 0, max: 5 },
        num_outputs: RandomRange { min: 1, max: 3 },
        num_terms: RandomRange { min: 1, max: 5 },
        sub_circuit_prob: 0.05,
    };
    for i in 0..1000 {
        config.seed = i + 20000;
        let root = IrRootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        compile_and_random_test::<_, NormalInputType>(&root, 5);
        compile_and_random_test::<_, CrossLayerInputType>(&root, 5);
    }
}

#[test]
fn random_circuits_4() {
    let mut config = RandomCircuitConfig {
        seed: 0,
        num_circuits: RandomRange { min: 1, max: 1 },
        num_inputs: RandomRange { min: 100, max: 200 },
        num_instructions: RandomRange {
            min: 1000,
            max: 2000,
        },
        num_constraints: RandomRange { min: 0, max: 5 },
        num_outputs: RandomRange { min: 1, max: 3 },
        num_terms: RandomRange { min: 1, max: 30 },
        sub_circuit_prob: 0.0,
    };
    for i in 0..200 {
        config.seed = i + 30000;
        let root = IrRootCircuit::<C>::random(&config);
        assert_eq!(root.validate(), Ok(()));
        compile_and_random_test::<_, NormalInputType>(&root, 5);
        compile_and_random_test::<_, CrossLayerInputType>(&root, 5);
    }
}

#[test]
fn cross_layer_circuit() {
    let mut root = IrRootCircuit::<C>::default();
    const N: usize = 1000;
    root.circuits.insert(
        0,
        IrCircuit::<C> {
            instructions: vec![],
            constraints: vec![N * 2 - 1],
            outputs: vec![],
            num_inputs: N,
        },
    );
    for i in 0..N - 1 {
        root.circuits
            .get_mut(&0)
            .unwrap()
            .instructions
            .push(IrInstruction::InternalVariable {
                expr: Expression::from_terms(vec![
                    Term::new_linear(CField::one(), N + i),
                    Term::new_linear(CField::one(), N - i - 1),
                ]),
            });
    }
    assert_eq!(root.validate(), Ok(()));
    let (lc, _) = compile_and_random_test::<_, CrossLayerInputType>(&root, 5);
    assert!((lc.layer_ids.len() as isize - N as isize).abs() <= 10);
    for i in lc.layer_ids.iter() {
        assert!(lc.segments[*i].gate_adds.len() <= 10);
    }
}

#[test]
fn ensure_circuit_is_minimal() {
    let mut root1 = IrRootCircuit::<C>::default();
    root1.circuits.insert(
        0,
        IrCircuit {
            instructions: vec![],
            constraints: vec![1],
            outputs: vec![],
            num_inputs: 1,
        },
    );
    let mut root2 = IrRootCircuit::<C>::default();
    root2.circuits.insert(
        0,
        IrCircuit {
            instructions: vec![IrInstruction::InternalVariable {
                expr: Expression::from_terms(vec![Term::new_random_linear(1)]),
            }],
            constraints: vec![],
            outputs: vec![2],
            num_inputs: 1,
        },
    );
    assert_eq!(root1.validate(), Ok(()));
    assert_eq!(root2.validate(), Ok(()));
    let (lc1n, _) = compile_and_random_test::<_, NormalInputType>(&root1, 5);
    let (lc2n, _) = compile_and_random_test::<_, NormalInputType>(&root2, 0); // random value will mismatch if >0
    let (lc1c, _) = compile_and_random_test::<_, CrossLayerInputType>(&root1, 5);
    let (lc2c, _) = compile_and_random_test::<_, CrossLayerInputType>(&root2, 0);
    assert_eq!(lc1n.layer_ids.len(), 1);
    assert_eq!(lc2n.layer_ids.len(), 1);
    assert_eq!(lc1c.layer_ids.len(), 1);
    assert_eq!(lc2c.layer_ids.len(), 1);
}
