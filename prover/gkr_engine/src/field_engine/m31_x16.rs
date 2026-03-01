use mersenne31::{M31Ext3, M31Ext3x16, M31x16, M31};

use crate::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct M31x16Config;

impl FieldEngine for M31x16Config {
    const FIELD_TYPE: FieldType = FieldType::M31x16;

    const SENTINEL: [u8; 32] = [
        255, 255, 255, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];

    type CircuitField = M31;

    type SimdCircuitField = M31x16;

    type ChallengeField = M31Ext3;

    type Field = M31Ext3x16;
}
