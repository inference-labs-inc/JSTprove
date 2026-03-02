use gf2::{GF2x8, GF2};
use gf2_128::{GF2_128x8, GF2_128};

use crate::{FieldEngine, FieldType};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GF2ExtConfig;

impl FieldEngine for GF2ExtConfig {
    const FIELD_TYPE: FieldType = FieldType::GF2Ext128;

    const SENTINEL: [u8; 32] = [
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];

    type CircuitField = GF2;

    type SimdCircuitField = GF2x8;

    type ChallengeField = GF2_128;

    type Field = GF2_128x8;
}
