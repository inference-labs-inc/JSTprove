pub const WHIR_RATE_LOG: usize = 1;

pub const WHIR_FIELD_SIZE_BITS: f64 = 128.0;

pub fn whir_fold_security() -> f64 {
    let log_inv_rate = WHIR_RATE_LOG as f64;
    let log_k: f64 = 2.0;
    let error = 7.0 * std::f64::consts::LOG2_10 + 3.5 * log_inv_rate + 2.0 * log_k;
    WHIR_FIELD_SIZE_BITS - error
}

pub fn whir_queries_for_round(round: usize) -> usize {
    let log_inv_rate = WHIR_RATE_LOG + round;
    let rate: f64 = (0.5_f64).powi(log_inv_rate as i32);
    let eta = rate.sqrt() / 20.0;
    let per_sample = rate.sqrt() + eta;
    let bits_per_query = -per_sample.log2();
    let target = whir_fold_security();
    (target / bits_per_query).ceil().max(1.0) as usize
}
