pub use arith::{Field as FieldArith, Fr as BN254Fr};
use babybear::{BabyBear, BabyBearx16};
pub use gf2::{GF2x8, GF2};
pub use goldilocks::{Goldilocks, Goldilocksx8};
pub use mersenne31::{M31x16, M31};
use serdes::ExpSerde;

pub trait Field: FieldArith + ExpSerde {
    fn optimistic_inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else if *self == Self::ONE {
            Some(Self::ONE)
        } else {
            self.inv()
        }
    }
}

impl Field for BN254Fr {}
impl Field for GF2 {}
impl Field for M31 {}
impl Field for Goldilocks {}
impl Field for BabyBear {}

// This trait exist only for making Rust happy
// If we use arith::Field, Rust says upstream may add more impls
pub trait FieldRaw: FieldArith {}

impl FieldRaw for BN254Fr {}
impl FieldRaw for GF2 {}
impl FieldRaw for M31 {}
impl FieldRaw for M31x16 {}
impl FieldRaw for GF2x8 {}
impl FieldRaw for Goldilocks {}
impl FieldRaw for Goldilocksx8 {}
impl FieldRaw for BabyBear {}
impl FieldRaw for BabyBearx16 {}
