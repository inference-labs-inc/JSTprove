use babybear::{BabyBear, BabyBearExt3, BabyBearExt3x16, BabyBearx16};

use super::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BabyBearx16Config;

impl FieldEngine for BabyBearx16Config {
    const FIELD_TYPE: FieldType = FieldType::BabyBearx16;

    const SENTINEL: [u8; 32] = [
        1, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];

    type CircuitField = BabyBear;

    type SimdCircuitField = BabyBearx16;

    type ChallengeField = BabyBearExt3;

    type Field = BabyBearExt3x16;
}
