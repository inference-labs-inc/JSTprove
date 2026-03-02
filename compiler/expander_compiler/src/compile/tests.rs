use mersenne31::M31;

use crate::circuit::{ir, layered::NormalInputType};
use crate::frontend::M31Config as C;

type CField = M31;

#[test]
fn simple_div() {
    let mut root = ir::source::RootCircuit::<C>::default();
    root.circuits.insert(
        0,
        ir::source::Circuit {
            instructions: vec![ir::source::Instruction::Div {
                x: 1,
                y: 2,
                checked: true,
            }],
            constraints: vec![ir::source::Constraint {
                typ: ir::source::ConstraintType::Zero,
                var: 3,
            }],
            outputs: vec![3],
            num_inputs: 2,
        },
    );
    assert_eq!(root.validate(), Ok(()));
    let (input_solver, lc) = super::compile::<_, NormalInputType>(&root).unwrap();
    assert_eq!(input_solver.circuits[&0].outputs.len(), 4);
    let (o, cond) = lc.eval_unsafe(vec![
        CField::from(2),
        CField::from(3),
        CField::from(5),
        CField::from(7),
    ]);
    assert_eq!(o[0], CField::from(10));
    assert!(!cond);
}
