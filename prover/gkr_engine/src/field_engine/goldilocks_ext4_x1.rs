use goldilocks::{Goldilocks, GoldilocksExt4};

use crate::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GoldilocksExt4x1Config;

impl FieldEngine for GoldilocksExt4x1Config {
    const FIELD_TYPE: FieldType = FieldType::GoldilocksExt4x1;

    const SENTINEL: [u8; 32] = [
        1, 0, 0, 0, 255, 255, 255, 255, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];

    type CircuitField = Goldilocks;

    type SimdCircuitField = Goldilocks;

    type ChallengeField = GoldilocksExt4;

    type Field = GoldilocksExt4;
}
