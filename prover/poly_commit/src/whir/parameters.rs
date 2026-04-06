pub const WHIR_RATE_LOG: usize = 1;
pub const WHIR_FOLDING_FACTOR: usize = 4;
pub const WHIR_OOD_SAMPLES: usize = 0;
pub const WHIR_POW_BITS: usize = 0;
pub const WHIR_TARGET_SECURITY_BITS: f64 = 128.0;

pub fn whir_queries_for_committed_round(_cr: usize) -> usize {
    let rate: f64 = (0.5_f64).powi(WHIR_RATE_LOG as i32);
    let eta = rate.sqrt() / 20.0;
    let per_sample = rate.sqrt() + eta;
    let bits_per_query = -per_sample.log2();
    (WHIR_TARGET_SECURITY_BITS / bits_per_query).ceil().max(1.0) as usize
}

pub fn num_committed_rounds(num_vars: usize) -> usize {
    if num_vars <= WHIR_FOLDING_FACTOR {
        return 0;
    }
    (num_vars - 1) / WHIR_FOLDING_FACTOR
}
