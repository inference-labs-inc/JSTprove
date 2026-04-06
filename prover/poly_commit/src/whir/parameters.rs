pub const WHIR_RATE_LOG: usize = 1;
pub const WHIR_FOLDING_FACTOR: usize = 4;
pub const WHIR_FIELD_SIZE_BITS: f64 = 128.0;
pub const WHIR_OOD_SAMPLES: usize = 2;
pub const WHIR_POW_BITS: usize = 22;

pub fn whir_fold_security() -> f64 {
    let log_inv_rate = WHIR_RATE_LOG as f64;
    let log_k = (1u64 << WHIR_FOLDING_FACTOR) as f64;
    let log_k = log_k.log2();
    let error = 7.0 * std::f64::consts::LOG2_10 + 3.5 * log_inv_rate + 2.0 * log_k;
    WHIR_FIELD_SIZE_BITS - error + WHIR_POW_BITS as f64
}

pub fn whir_queries_for_committed_round(cr: usize) -> usize {
    let log_inv_rate = WHIR_RATE_LOG + (cr + 1) * (WHIR_FOLDING_FACTOR - 1);
    let rate: f64 = (0.5_f64).powi(log_inv_rate as i32);
    let eta = rate.sqrt() / 20.0;
    let per_sample = rate.sqrt() + eta;
    let bits_per_query = -per_sample.log2();
    let target = whir_fold_security();
    (target / bits_per_query).ceil().max(1.0) as usize
}

pub fn num_committed_rounds(num_vars: usize) -> usize {
    if num_vars <= WHIR_FOLDING_FACTOR {
        return 0;
    }
    (num_vars - 1) / WHIR_FOLDING_FACTOR
}
