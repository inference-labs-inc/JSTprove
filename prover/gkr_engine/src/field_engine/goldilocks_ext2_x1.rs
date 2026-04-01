use goldilocks::GoldilocksExt2Scalar;

use crate::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GoldilocksExt2x1Config;

impl FieldEngine for GoldilocksExt2x1Config {
    const FIELD_TYPE: FieldType = FieldType::GoldilocksExt2x1;

    // Goldilocks modulus (LE) in bytes [0..8], extension degree 2 in byte [8].
    // Distinguishes from Goldilocksx1Config which has byte [8] = 0.
    const SENTINEL: [u8; 32] = [
        1, 0, 0, 0, 255, 255, 255, 255, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];

    type CircuitField = GoldilocksExt2Scalar;

    type SimdCircuitField = GoldilocksExt2Scalar;

    type ChallengeField = GoldilocksExt2Scalar;

    type Field = GoldilocksExt2Scalar;
}
