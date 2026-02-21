pub fn next_power_of_two(n: usize) -> usize {
    n.next_power_of_two()
}

pub fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

pub fn pad_to_power_of_two_i64(data: &[i64]) -> Vec<i64> {
    let target = next_power_of_two(data.len());
    let mut padded = Vec::with_capacity(target);
    padded.extend_from_slice(data);
    padded.resize(target, 0);
    padded
}

pub fn num_vars_for(n: usize) -> usize {
    log2_ceil(next_power_of_two(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_ceil() {
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(4), 2);
        assert_eq!(log2_ceil(5), 3);
        assert_eq!(log2_ceil(8), 3);
        assert_eq!(log2_ceil(9), 4);
    }

    #[test]
    fn test_num_vars_for() {
        assert_eq!(num_vars_for(1), 0);
        assert_eq!(num_vars_for(2), 1);
        assert_eq!(num_vars_for(3), 2);
        assert_eq!(num_vars_for(4), 2);
        assert_eq!(num_vars_for(7), 3);
        assert_eq!(num_vars_for(8), 3);
    }
}
