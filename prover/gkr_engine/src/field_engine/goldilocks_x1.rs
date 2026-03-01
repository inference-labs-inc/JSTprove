use goldilocks::{Goldilocks, GoldilocksExt2};

use crate::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Goldilocksx1Config;

impl FieldEngine for Goldilocksx1Config {
    const FIELD_TYPE: FieldType = FieldType::Goldilocksx1;

    const SENTINEL: [u8; 32] = [
        1, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];

    type CircuitField = Goldilocks;

    type SimdCircuitField = Goldilocks;

    type ChallengeField = GoldilocksExt2;

    type Field = GoldilocksExt2;
}
