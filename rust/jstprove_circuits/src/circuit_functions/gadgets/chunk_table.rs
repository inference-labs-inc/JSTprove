struct Entry {
    kappa: u8,
    n_bits: u8,
    max_elements: u32,
    chunk_bits: u8,
}

// Generated from logup_sweep benchmark data.
// To regenerate: cargo run --release --bin logup_sweep | python3 scripts/logup_codegen.py
//
// Entries are sorted by (kappa, n_bits, max_elements).
// Lookup scans for the first entry matching (kappa, n_bits) whose
// max_elements >= estimated_elements. If none match, DEFAULT_LOGUP_CHUNK_BITS is used.
//
// Measured on BN254 (kappa=18, n_bits=64):
//   chunk=10: lenet 9.29s / mini_resnet 12.12s
//   chunk=12: lenet 7.97s / mini_resnet 10.30s  (best)
//   chunk=13: lenet 8.34s / mini_resnet 10.34s
//   chunk=14: lenet 9.02s / mini_resnet 11.45s
static TABLE: &[Entry] = &[
    Entry {
        kappa: 18,
        n_bits: 64,
        max_elements: 10_000,
        chunk_bits: 12,
    },
    Entry {
        kappa: 18,
        n_bits: 64,
        max_elements: 50_000,
        chunk_bits: 12,
    },
    Entry {
        kappa: 18,
        n_bits: 64,
        max_elements: u32::MAX,
        chunk_bits: 12,
    },
    Entry {
        kappa: 18,
        n_bits: 31,
        max_elements: u32::MAX,
        chunk_bits: 12,
    },
];

#[must_use]
pub fn lookup(kappa: usize, n_bits: usize, estimated_elements: usize) -> usize {
    for entry in TABLE {
        if entry.kappa as usize == kappa
            && entry.n_bits as usize == n_bits
            && estimated_elements <= entry.max_elements as usize
        {
            return entry.chunk_bits as usize;
        }
    }
    super::DEFAULT_LOGUP_CHUNK_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bn254_lenet_returns_12() {
        assert_eq!(lookup(18, 64, 6_500), 12);
    }

    #[test]
    fn bn254_large_model_returns_12() {
        assert_eq!(lookup(18, 64, 100_000), 12);
    }

    #[test]
    fn goldilocks_returns_12() {
        assert_eq!(lookup(18, 31, 5_000), 12);
    }

    #[test]
    fn unknown_kappa_falls_back_to_default() {
        assert_eq!(
            lookup(22, 64, 1_000),
            super::super::DEFAULT_LOGUP_CHUNK_BITS
        );
    }
}
