// Copyright 2025 Irreducible Inc.
use std::array;

use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire, WitnessFiller};

// ι round constants
pub const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

// ρ rotation offsets r[x,y] in lane order (i = x + 5*y)
#[rustfmt::skip]
pub const R: [u32; 25] =  [
	 0,  1, 62, 28, 27,
	36, 44,  6, 55, 20,
	 3, 10, 43, 25, 39,
	41, 45, 15, 21,  8,
	18,  2, 61, 56, 14,
];

#[inline(always)]
pub const fn idx(x: usize, y: usize) -> usize {
    x + 5 * y
}

#[derive(Clone, Copy)]
pub struct State {
    pub words: [Wire; 25],
}

/// Keccak f\[1600\] permutation circuit.
pub struct Permutation {
    pub input_state: State,
    pub output_state: State,
}

impl Permutation {
    /// Create a new permutation circuit.
    ///
    /// ## Arguments
    ///
    /// * `b` - The circuit builder to use.
    pub fn new(b: &CircuitBuilder, input_state: State) -> Self {
        let mut output_state = input_state;
        Self::keccak_f1600(b, &mut output_state.words);

        Self {
            input_state,
            output_state,
        }
    }

    /// Populate the state with the given witness.
    ///
    /// ## Arguments
    ///
    /// * `w` - The witness to populate the state with.
    /// * `state` - The state to populate the witness with.
    pub fn populate_state(&self, w: &mut WitnessFiller, state: [u64; 25]) {
        for i in 0..25 {
            w[self.input_state.words[i]] = Word(state[i]);
        }
    }

    /// Perform the Keccak f\[1600\] permutation.
    ///
    /// ## Arguments
    ///
    /// * `b` - The circuit builder to use.
    /// * `state` - The state to perform the permutation on.
    pub fn keccak_f1600(b: &CircuitBuilder, state: &mut [Wire; 25]) {
        for round in 0..24 {
            Self::keccak_permutation_round(b, state, round);
        }
    }

    pub fn keccak_permutation_round(b: &CircuitBuilder, state: &mut [Wire; 25], round: usize) {
        Self::theta(b, state);
        Self::rho_pi(b, state);
        Self::chi(b, state);
        Self::iota(b, state, round);
    }

    fn theta(b: &CircuitBuilder, state: &mut [Wire; 25]) {
        let c0 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(0, y)]));
        let c1 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(1, y)]));
        let c2 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(2, y)]));
        let c3 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(3, y)]));
        let c4 = b.bxor_multi(&array::from_fn::<_, 5, _>(|y| state[idx(4, y)]));

        // D[x] = C[x-1] ^ rotl1(C[x+1])
        let d0 = b.bxor(c4, b.rotl(c1, 1));
        let d1 = b.bxor(c0, b.rotl(c2, 1));
        let d2 = b.bxor(c1, b.rotl(c3, 1));
        let d3 = b.bxor(c2, b.rotl(c4, 1));
        let d4 = b.bxor(c3, b.rotl(c0, 1));

        // Fusing every linear expression into its AND leads to too many distinct shifted value
        // indices and that slows down the shift reduction phase. Empirically we found out that
        // preventing committing those here leads to better performance.
        b.force_commit(d0);
        b.force_commit(d1);
        b.force_commit(d2);
        b.force_commit(d3);
        b.force_commit(d4);

        // A'[x,y] = A[x,y] ^ D[x]
        for y in 0..5 {
            state[idx(0, y)] = b.bxor(state[idx(0, y)], d0);
            state[idx(1, y)] = b.bxor(state[idx(1, y)], d1);
            state[idx(2, y)] = b.bxor(state[idx(2, y)], d2);
            state[idx(3, y)] = b.bxor(state[idx(3, y)], d3);
            state[idx(4, y)] = b.bxor(state[idx(4, y)], d4);
        }
    }

    fn chi(b: &CircuitBuilder, state: &mut [Wire; 25]) {
        for y in 0..5 {
            let a0 = state[idx(0, y)];
            let a1 = state[idx(1, y)];
            let a2 = state[idx(2, y)];
            let a3 = state[idx(3, y)];
            let a4 = state[idx(4, y)];

            state[idx(0, y)] = b.fax(b.bnot(a1), a2, a0);
            state[idx(1, y)] = b.fax(b.bnot(a2), a3, a1);
            state[idx(2, y)] = b.fax(b.bnot(a3), a4, a2);
            state[idx(3, y)] = b.fax(b.bnot(a4), a0, a3);
            state[idx(4, y)] = b.fax(b.bnot(a0), a1, a4);
        }
    }

    fn rho_pi(b: &CircuitBuilder, state: &mut [Wire; 25]) {
        let mut temp = [state[0]; 25];
        for y in 0..5 {
            for x in 0..5 {
                // no need to rotate if rotating by 0
                if R[idx(x, y)] == 0 {
                    continue;
                }
                temp[idx(y, (2 * x + 3 * y) % 5)] = b.rotl(state[idx(x, y)], R[idx(x, y)]);
            }
        }
        *state = temp;
    }

    fn iota(b: &CircuitBuilder, state: &mut [Wire; 25], round: usize) {
        let rc_wire = b.add_constant(Word(RC[round]));
        state[0] = b.bxor(state[0], rc_wire);
    }
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use binius_frontend::CircuitBuilder;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::*;

    mod reference {
        use super::{R, RC, idx};

        pub fn theta(state: &mut [u64; 25]) {
            let mut c = [0u64; 5];
            for x in 0..5 {
                c[x] = state[idx(x, 0)]
                    ^ state[idx(x, 1)]
                    ^ state[idx(x, 2)]
                    ^ state[idx(x, 3)]
                    ^ state[idx(x, 4)];
            }
            let d = [
                c[4] ^ c[1].rotate_left(1),
                c[0] ^ c[2].rotate_left(1),
                c[1] ^ c[3].rotate_left(1),
                c[2] ^ c[4].rotate_left(1),
                c[3] ^ c[0].rotate_left(1),
            ];

            for y in 0..5 {
                for x in 0..5 {
                    state[idx(x, y)] ^= d[x];
                }
            }
        }

        #[inline(always)]
        pub fn rho_pi(state: &mut [u64; 25]) {
            let mut temp = [state[0]; 25];
            for y in 0..5 {
                for x in 0..5 {
                    temp[idx(y, (2 * x + 3 * y) % 5)] = state[idx(x, y)].rotate_left(R[idx(x, y)]);
                }
            }
            *state = temp;
        }

        pub fn iota(state: &mut [u64; 25], round: usize) {
            state[0] ^= RC[round];
        }

        #[inline(always)]
        pub fn chi(state: &mut [u64; 25]) {
            for y in 0..5 {
                let a0 = state[idx(0, y)];
                let a1 = state[idx(1, y)];
                let a2 = state[idx(2, y)];
                let a3 = state[idx(3, y)];
                let a4 = state[idx(4, y)];
                state[idx(0, y)] = a0 ^ ((!a1) & a2);
                state[idx(1, y)] = a1 ^ ((!a2) & a3);
                state[idx(2, y)] = a2 ^ ((!a3) & a4);
                state[idx(3, y)] = a3 ^ ((!a4) & a0);
                state[idx(4, y)] = a4 ^ ((!a0) & a1);
            }
        }

        pub fn keccak_permutation_round(state: &mut [u64; 25], round: usize) {
            theta(state);
            rho_pi(state);
            chi(state);
            iota(state, round);
        }

        pub fn keccak_f1600(state: &mut [u64; 25]) {
            for round in 0..24 {
                keccak_permutation_round(state, round);
            }
        }
    }

    #[test]
    fn test_keccak_permutation() {
        let mut rng = StdRng::seed_from_u64(0);

        let builder = CircuitBuilder::new();

        let input_words = State {
            words: array::from_fn(|_| builder.add_inout()),
        };

        let permutation = Permutation::new(&builder, input_words);

        let circuit = builder.build();

        let mut w = circuit.new_witness_filler();

        let initial_state = rng.random::<[u64; 25]>();
        permutation.populate_state(&mut w, initial_state);

        circuit.populate_wire_witness(&mut w).unwrap();

        let mut expected_output = initial_state;
        reference::keccak_f1600(&mut expected_output);

        for i in 0..25 {
            assert_eq!(
                w[permutation.output_state.words[i]],
                Word(expected_output[i])
            );
        }

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    fn validate_circuit_component(
        circuit_fn: impl FnOnce(&CircuitBuilder, &mut [Wire; 25]),
        reference_fn: impl FnOnce(&mut [u64; 25]),
        input_state: [u64; 25],
    ) {
        let builder = CircuitBuilder::new();

        let input_wires: [Wire; 25] = std::array::from_fn(|_| builder.add_inout());

        let mut state_wires = input_wires;
        circuit_fn(&builder, &mut state_wires);
        let circuit = builder.build();

        let mut expected_output = input_state;
        reference_fn(&mut expected_output);

        let mut w = circuit.new_witness_filler();
        for i in 0..25 {
            w[input_wires[i]] = Word(input_state[i]);
        }
        circuit.populate_wire_witness(&mut w).unwrap();

        for i in 0..25 {
            assert_eq!(
                w[state_wires[i]],
                Word(expected_output[i]),
                "Output mismatch at index {}: circuit={:?}, expected={:?}",
                i,
                w[state_wires[i]],
                Word(expected_output[i])
            );
        }

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn test_keccak_f1600() {
        let mut rng = StdRng::seed_from_u64(0);
        let input_state = rng.random::<[u64; 25]>();

        validate_circuit_component(
            Permutation::keccak_f1600,
            reference::keccak_f1600,
            input_state,
        );
    }

    #[test]
    fn test_keccak_permutation_round() {
        let mut rng = StdRng::seed_from_u64(0);
        let input_state = rng.random::<[u64; 25]>();

        validate_circuit_component(
            |b, state| Permutation::keccak_permutation_round(b, state, 0),
            |state| reference::keccak_permutation_round(state, 0),
            input_state,
        );
    }

    #[test]
    fn test_theta() {
        let mut rng = StdRng::seed_from_u64(0);
        let input_state = rng.random::<[u64; 25]>();

        validate_circuit_component(Permutation::theta, reference::theta, input_state);
    }

    #[test]
    fn test_rho_pi() {
        let mut rng = StdRng::seed_from_u64(0);
        let input_state = rng.random::<[u64; 25]>();

        validate_circuit_component(Permutation::rho_pi, reference::rho_pi, input_state);
    }

    #[test]
    fn test_chi() {
        let mut rng = StdRng::seed_from_u64(0);
        let input_state = rng.random::<[u64; 25]>();

        validate_circuit_component(Permutation::chi, reference::chi, input_state);
    }

    #[test]
    fn test_iota() {
        let mut rng = StdRng::seed_from_u64(0);
        let input_state = rng.random::<[u64; 25]>();

        validate_circuit_component(
            |b, state| Permutation::iota(b, state, 0),
            |state| reference::iota(state, 0),
            input_state,
        );
    }
}
