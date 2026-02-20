use frontend::layouter::builder::{CircuitBuilder, InputLayerNodeRef, NodeRef};
use shared_types::Field;


pub struct LogUpRangeCheckContext<F: Field> {
    pub chunk_bits: usize,
    pub table_node: NodeRef<F>,
    pub fiat_shamir_node: frontend::layouter::builder::FSNodeRef<F>,
    pub lookup_table: frontend::layouter::builder::LookupTableNodeRef,
    committed_input: InputLayerNodeRef<F>,
}

impl<F: Field> LogUpRangeCheckContext<F> {
    pub fn new(
        builder: &mut CircuitBuilder<F>,
        chunk_bits: usize,
        committed_input: &InputLayerNodeRef<F>,
        public_input: &InputLayerNodeRef<F>,
    ) -> Self {
        let table_size = 1 << chunk_bits;
        let table_num_vars = chunk_bits;

        let table_node = builder.add_input_shred(
            "range_check_table",
            table_num_vars,
            public_input,
        );

        let fiat_shamir_node = builder.add_fiat_shamir_challenge_node(1);
        let lookup_table = builder.add_lookup_table(&table_node, &fiat_shamir_node);

        Self {
            chunk_bits,
            table_node,
            fiat_shamir_node,
            lookup_table,
            committed_input: committed_input.clone(),
        }
    }

    pub fn add_range_check(
        &mut self,
        builder: &mut CircuitBuilder<F>,
        digits_node: &NodeRef<F>,
        multiplicities_node: &NodeRef<F>,
    ) -> frontend::layouter::builder::LookupConstraintNodeRef {
        builder.add_lookup_constraint(
            &self.lookup_table,
            digits_node,
            multiplicities_node,
        )
    }

    pub fn table_data(chunk_bits: usize) -> Vec<u64> {
        (0..(1u64 << chunk_bits)).collect()
    }

    pub fn chunk_bits(&self) -> usize {
        self.chunk_bits
    }
}

pub fn decompose_to_digits(value: i64, chunk_bits: usize, n_digits: usize) -> Vec<u64> {
    let mask = (1u64 << chunk_bits) - 1;
    let unsigned = if value >= 0 {
        value as u64
    } else {
        let modulus_approx = 1u128 << 64;
        (modulus_approx as i128 + value as i128) as u64
    };
    let mut digits = Vec::with_capacity(n_digits);
    let mut remaining = unsigned;
    for _ in 0..n_digits {
        digits.push(remaining & mask);
        remaining >>= chunk_bits;
    }
    digits
}

pub fn compute_multiplicities(
    all_digits: &[u64],
    chunk_bits: usize,
) -> Vec<u64> {
    let table_size = 1usize << chunk_bits;
    let mut counts = vec![0u64; table_size];
    for &digit in all_digits {
        counts[digit as usize] += 1;
    }
    counts
}

pub fn num_digits_for_bits(total_bits: usize, chunk_bits: usize) -> usize {
    (total_bits + chunk_bits - 1) / chunk_bits
}
