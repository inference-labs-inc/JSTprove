use mersenne31::{M31Ext3, M31};

use crate::FieldType;

use super::FieldEngine;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct M31x1Config;

impl FieldEngine for M31x1Config {
    const FIELD_TYPE: FieldType = FieldType::M31x1;

    const SENTINEL: [u8; 32] = [
        255, 255, 255, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];

    type CircuitField = M31;

    type SimdCircuitField = M31;

    type ChallengeField = M31Ext3;

    type Field = M31Ext3;
}
