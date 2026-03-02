use super::config::Config;

pub fn cost_of_compress<C: Config>(deg_cnt: &[usize; 3]) -> usize {
    C::COST_MUL * deg_cnt[2]
        + C::COST_ADD * deg_cnt[1]
        + C::COST_CONST * deg_cnt[0]
        + C::COST_VARIABLE
}

pub fn cost_of_multiply<C: Config>(
    a_deg_0: usize,
    a_deg_1: usize,
    b_deg_0: usize,
    b_deg_1: usize,
) -> usize {
    C::COST_MUL * (a_deg_1 * b_deg_1)
        + C::COST_ADD * (a_deg_0 * b_deg_1 + a_deg_1 * b_deg_0)
        + C::COST_CONST * (a_deg_0 * b_deg_0)
}

pub fn cost_of_possible_references<C: Config>(
    deg_cnt: &[usize; 3],
    ref_add: usize,
    ref_mul: usize,
) -> usize {
    C::COST_CONST * (deg_cnt[0] * ref_add)
        + C::COST_ADD * (deg_cnt[1] * ref_add + deg_cnt[0] * ref_mul)
        + C::COST_MUL * (deg_cnt[2] * ref_add + (deg_cnt[1] + deg_cnt[2] * 2) * ref_mul)
}

pub fn cost_of_relay<C: Config>(v1_layer: usize, v2_layer: usize) -> usize {
    (v1_layer as isize - v2_layer as isize).unsigned_abs() * (C::COST_VARIABLE + C::COST_ADD)
}
