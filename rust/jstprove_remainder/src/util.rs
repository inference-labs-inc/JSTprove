use shared_types::Fr;

pub fn i64_to_fr(val: i64) -> Fr {
    if val >= 0 {
        Fr::from(val as u64)
    } else {
        -Fr::from(val.unsigned_abs())
    }
}
