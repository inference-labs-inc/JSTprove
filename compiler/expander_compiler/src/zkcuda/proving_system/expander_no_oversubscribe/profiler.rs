#[cfg(feature = "zkcuda_profile")]
mod profiler_enabled {
    use std::collections::HashMap;

    use arith::Fr;
    use halo2curves::ff::PrimeField;

    #[derive(Clone, Debug, Default)]
    pub struct NBytesProfiler {
        pub bytes_stats: HashMap<usize, usize>,
    }

    impl NBytesProfiler {
        pub fn new() -> Self {
            NBytesProfiler {
                bytes_stats: HashMap::new(),
            }
        }

        pub fn add_bytes(&mut self, n_bytes: usize) {
            *self.bytes_stats.entry(n_bytes).or_insert(0) += 1;
        }

        pub fn add_fr(&mut self, fr: Fr) {
            let le_bytes = fr.to_repr();
            let be_leading_zeros_bytes = le_bytes.into_iter().rev().take_while(|&b| b == 0).count();
            let n_bytes = le_bytes.len() - be_leading_zeros_bytes;
            self.add_bytes(n_bytes);
        }

        pub fn print_stats(&self) {
            for (bytes, count) in &self.bytes_stats {
                println!("{bytes} bytes: {count}");
            }
        }
    }
}

#[cfg(not(feature = "zkcuda_profile"))]
mod profiler_disabled {
    use arith::Fr;

    #[derive(Clone, Debug, Default)]
    pub struct NBytesProfiler;

    impl NBytesProfiler {
        pub fn new() -> Self {
            NBytesProfiler
        }

        pub fn add_bytes(&mut self, _n_bytes: usize) {}

        pub fn add_fr(&mut self, _fr: Fr) {}

        pub fn print_stats(&self) {}
    }
}

#[cfg(not(feature = "zkcuda_profile"))]
pub use profiler_disabled::NBytesProfiler;
#[cfg(feature = "zkcuda_profile")]
pub use profiler_enabled::NBytesProfiler;

#[cfg(feature = "zkcuda_profile")]
mod test {
    #![allow(unused_imports)]
    use super::NBytesProfiler;
    use arith::Fr;

    #[test]
    fn test_n_bytes_profiler() {
        let mut profiler = NBytesProfiler::new();
        profiler.add_bytes(32);
        profiler.add_bytes(64);
        profiler.add_fr(Fr::from(256u64));
        profiler.print_stats();
    }
}
