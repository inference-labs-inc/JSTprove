use std::vec;

use mersenne31::M31;

use super::{Allocation, Circuit, Coef, GateAdd, GateConst, GateMul, Segment};
use crate::circuit::layered::{NormalInput, NormalInputType, NormalInputUsize};
use crate::field::FieldArith;
use crate::frontend::M31Config as C;

type CField = M31;

#[test]
fn simple() {
    let circuit: Circuit<C, NormalInputType> = Circuit {
        num_public_inputs: 0,
        num_actual_outputs: 2,
        expected_num_output_zeroes: 0,
        segments: vec![
            Segment {
                num_inputs: NormalInputUsize { v: 2 },
                num_outputs: 1,
                child_segs: vec![],
                gate_muls: vec![GateMul {
                    inputs: [NormalInput { offset: 0 }, NormalInput { offset: 1 }],
                    output: 0,
                    coef: Coef::Constant(CField::from(2)),
                }],
                gate_adds: vec![],
                gate_consts: vec![],
                gate_customs: vec![],
            },
            Segment {
                num_inputs: NormalInputUsize { v: 4 },
                num_outputs: 2,
                child_segs: vec![(
                    0,
                    vec![
                        Allocation {
                            input_offset: NormalInputUsize { v: 0 },
                            output_offset: 0,
                        },
                        Allocation {
                            input_offset: NormalInputUsize { v: 2 },
                            output_offset: 1,
                        },
                    ],
                )],
                gate_muls: vec![],
                gate_adds: vec![],
                gate_consts: vec![],
                gate_customs: vec![],
            },
            Segment {
                num_inputs: NormalInputUsize { v: 2 },
                num_outputs: 2,
                child_segs: vec![(
                    0,
                    vec![Allocation {
                        input_offset: NormalInputUsize { v: 0 },
                        output_offset: 0,
                    }],
                )],
                gate_muls: vec![],
                gate_adds: vec![
                    GateAdd {
                        inputs: [NormalInput { offset: 0 }],
                        output: 1,
                        coef: Coef::Constant(CField::from(3)),
                    },
                    GateAdd {
                        inputs: [NormalInput { offset: 1 }],
                        output: 1,
                        coef: Coef::Constant(CField::from(4)),
                    },
                ],
                gate_consts: vec![GateConst {
                    inputs: [],
                    output: 1,
                    coef: Coef::Constant(CField::from(5)),
                }],
                gate_customs: vec![],
            },
        ],
        layer_ids: vec![1, 2],
    };
    assert!(circuit.validate().is_ok());
    for _ in 0..100 {
        let s: Vec<CField> = (0..4)
            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
            .collect();
        let (out, _) = circuit.eval_unsafe(s.clone());
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], s[0] * s[1] * s[2] * s[3] * CField::from(8));
        assert_eq!(
            out[1],
            s[0] * s[1] * CField::from(6) + s[2] * s[3] * CField::from(8) + CField::from(5)
        );
    }
}
