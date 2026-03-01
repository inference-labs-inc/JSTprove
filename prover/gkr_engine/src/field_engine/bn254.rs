use crate::{FieldEngine, FieldType};
use arith::Fr;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BN254Config;

impl FieldEngine for BN254Config {
    const FIELD_TYPE: FieldType = FieldType::BN254;

    const SENTINEL: [u8; 32] = [
        1, 0, 0, 240, 147, 245, 225, 67, 145, 112, 185, 121, 72, 232, 51, 40, 93, 88, 129, 129,
        182, 69, 80, 184, 41, 160, 49, 225, 114, 78, 100, 48,
    ];

    type CircuitField = Fr;

    type ChallengeField = Fr;

    type Field = Fr;

    type SimdCircuitField = Fr;
}
