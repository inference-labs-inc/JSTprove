use goldilocks::{Goldilocks, GoldilocksExt3};

use crate::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GoldilocksExt3x1Config;

impl FieldEngine for GoldilocksExt3x1Config {
    const FIELD_TYPE: FieldType = FieldType::GoldilocksExt3x1;

    const SENTINEL: [u8; 32] = [
        1, 0, 0, 0, 255, 255, 255, 255, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];

    type CircuitField = Goldilocks;

    type SimdCircuitField = Goldilocks;

    type ChallengeField = GoldilocksExt3;

    type Field = GoldilocksExt3;
}
