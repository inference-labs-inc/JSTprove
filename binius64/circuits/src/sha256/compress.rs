// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The internal state of SHA-256.
///
/// The state size is 256 bits. For efficiency reasons it's packed in 8 x 32-bit words, and not
/// 4 x 64-bit words.
///
/// The elements are referred to as a–h or H0–H7.
#[derive(Clone)]
pub struct State(pub [Wire; 8]);

impl State {
    pub fn new(wires: [Wire; 8]) -> Self {
        State(wires)
    }

    pub fn public(builder: &CircuitBuilder) -> Self {
        State(std::array::from_fn(|_| builder.add_inout()))
    }

    pub fn private(builder: &CircuitBuilder) -> Self {
        State(std::array::from_fn(|_| builder.add_witness()))
    }

    pub fn iv(builder: &CircuitBuilder) -> Self {
        State(std::array::from_fn(|i| {
            builder.add_constant(Word(IV[i] as u64))
        }))
    }

    /// Packs the state into 4 x 64-bit words.
    pub fn pack_4x64b(&self, builder: &CircuitBuilder) -> [Wire; 4] {
        fn pack_pair(b: &CircuitBuilder, hi: Wire, lo: Wire) -> Wire {
            b.bxor(lo, b.shl(hi, 32))
        }

        [
            pack_pair(builder, self.0[0], self.0[1]),
            pack_pair(builder, self.0[2], self.0[3]),
            pack_pair(builder, self.0[4], self.0[5]),
            pack_pair(builder, self.0[6], self.0[7]),
        ]
    }
}

/// SHA-256 compress function.
pub struct Compress {
    pub state_in: State,
    pub state_out: State,
    pub m: [Wire; 16],
}

impl Compress {
    pub fn new(builder: &CircuitBuilder, state_in: State, m: [Wire; 16]) -> Self {
        // it is a PRECONDITION of this gadget that the high halves of the wires fed in be empty.
        // that is, `m[i] & 0xffffffff == m[i]` must hold for each wire `i` passed in.
        // it is the caller's responsibility to ensure that this is the case for all wires.
        // if this isn't true, then the behavior becomes undefined / the gadget becomes insecure.

        // ---- message-schedule ----
        // W[0..15] = block_words
        // for t = 16 .. 63:
        //     s0   = σ0(W[t-15])
        //     s1   = σ1(W[t-2])
        //     (p, _)  = Add32(W[t-16], s0)
        //     (q, _)  = Add32(p, W[t-7])
        //     (W[t],_) = Add32(q, s1)

        let mut w: Vec<Wire> = Vec::with_capacity(64);
        // W[0..15] = block_words
        w.extend_from_slice(&m);

        // W[16..63] computed from previous W values
        for t in 16..64 {
            let s0 = small_sigma_0(builder, w[t - 15]);
            let s1 = small_sigma_1(builder, w[t - 2]);
            let p = builder.iadd_32(w[t - 16], s0);
            let q = builder.iadd_32(p, w[t - 7]);
            w.push(builder.iadd_32(q, s1));
        }

        let w: &[Wire; 64] = (&*w).try_into().unwrap();
        let mut state = state_in.clone();
        for t in 0..64 {
            state = round(builder, t, state, w);
        }

        // Add the compressed chunk to the current hash value
        let state_out = State([
            builder.iadd_32(state_in.0[0], state.0[0]),
            builder.iadd_32(state_in.0[1], state.0[1]),
            builder.iadd_32(state_in.0[2], state.0[2]),
            builder.iadd_32(state_in.0[3], state.0[3]),
            builder.iadd_32(state_in.0[4], state.0[4]),
            builder.iadd_32(state_in.0[5], state.0[5]),
            builder.iadd_32(state_in.0[6], state.0[6]),
            builder.iadd_32(state_in.0[7], state.0[7]),
        ]);

        Compress {
            state_in,
            state_out,
            m,
        }
    }

    pub fn populate_m(&self, w: &mut binius_frontend::WitnessFiller, m: [u8; 64]) {
        debug_assert_eq!(self.m.len(), 16);

        for i in 0..16 {
            let j = i * 4;
            // Assemble a 32-bit big-endian word and widen to 64 bits.
            let limb = ((m[j] as u64) << 24)
                | ((m[j + 1] as u64) << 16)
                | ((m[j + 2] as u64) << 8)
                | (m[j + 3] as u64);

            // Write it to the witness.  Word is a thin wrapper around u64.
            w[self.m[i]] = Word(limb);
        }
    }
}

fn round(builder: &CircuitBuilder, round: usize, state: State, w: &[Wire; 64]) -> State {
    let State([a, b, c, d, e, f, g, h]) = state;

    let big_sigma_e = big_sigma_1(builder, e);
    let ch_efg = ch(builder, e, f, g);
    let t1a = builder.iadd_32(h, big_sigma_e);
    let t1b = builder.iadd_32(t1a, ch_efg);
    let rc = builder.add_constant(Word(K[round] as u64));
    let t1c = builder.iadd_32(t1b, rc);
    let t1 = builder.iadd_32(t1c, w[round]);

    let big_sigma_a = big_sigma_0(builder, a);
    let maj_abc = maj(builder, a, b, c);
    let t2 = builder.iadd_32(big_sigma_a, maj_abc);

    let h = g;
    let g = f;
    let f = e;
    let e = builder.iadd_32(d, t1);
    let d = c;
    let c = b;
    let b = a;
    let a = builder.iadd_32(t1, t2);

    State([a, b, c, d, e, f, g, h])
}

/// Ch(x, y, z) = (x AND y) XOR (NOT y AND z)
///             = z XOR (x AND (y XOR z))
fn ch(builder: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    builder.bxor(z, builder.band(x, builder.bxor(y, z)))
}

/// Maj(x, y, z) = (x AND y) XOR (x AND z) XOR (y AND z)
///              = (x XOR z) AND (y XOR z) XOR z.
fn maj(builder: &CircuitBuilder, x: Wire, y: Wire, z: Wire) -> Wire {
    builder.bxor(builder.band(builder.bxor(x, z), builder.bxor(y, z)), z)
}

/// Σ0(a)       = XOR( XOR( ROTR(x,  2), ROTR(x, 13) ), ROTR(x, 22) )
fn big_sigma_0(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr32(x, 2);
    let r2 = b.rotr32(x, 13);
    let r3 = b.rotr32(x, 22);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, r3)
}

/// Σ1(e)       = XOR( XOR( ROTR(x,  6), ROTR(x, 11) ), ROTR(x, 25) )
fn big_sigma_1(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr32(x, 6);
    let r2 = b.rotr32(x, 11);
    let r3 = b.rotr32(x, 25);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, r3)
}

/// σ0(x)       = XOR( XOR( ROTR(x,  7), ROTR(x, 18) ), SHR(x,  3) )
fn small_sigma_0(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr32(x, 7);
    let r2 = b.rotr32(x, 18);
    let s1 = b.srl32(x, 3);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, s1)
}

/// σ1(x)       = XOR( XOR( ROTR(x, 17), ROTR(x, 19) ), SHR(x, 10) )
fn small_sigma_1(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr32(x, 17);
    let r2 = b.rotr32(x, 19);
    let s1 = b.srl32(x, 10);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, s1)
}

#[cfg(test)]
mod tests {
    use binius_core::{verify::verify_constraints, word::Word};
    use binius_frontend::{CircuitBuilder, Wire};

    use super::{Compress, State};

    /// A test circuit that proves a knowledge of preimage for a given state vector S in
    ///
    ///     compress512(preimage) = S
    ///
    /// without revealing the preimage, only S.
    #[test]
    fn proof_preimage() {
        // Use the test-vector for SHA256 single block message: "abc".
        let mut preimage: [u8; 64] = [0; 64];
        preimage[0..3].copy_from_slice(b"abc");
        preimage[3] = 0x80;
        preimage[63] = 0x18;

        #[rustfmt::skip]
		let expected_state: [u32; 8] = [
			0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223,
			0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad,
		];

        let circuit = CircuitBuilder::new();
        let state = State::iv(&circuit);
        let input: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
        let output: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
        let compress = Compress::new(&circuit, state, input);

        // Mask to only low 32-bit.
        let mask32 = circuit.add_constant(Word::MASK_32);
        for (i, (actual_x, expected_x)) in compress.state_out.0.iter().zip(output).enumerate() {
            circuit.assert_eq(
                format!("preimage_eq[{i}]"),
                circuit.band(*actual_x, mask32),
                expected_x,
            );
        }

        let circuit = circuit.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        // Populate the input message for the compression function.
        compress.populate_m(&mut w, preimage);

        for (i, &output) in output.iter().enumerate() {
            w[output] = Word(expected_state[i] as u64);
        }
        circuit.populate_wire_witness(&mut w).unwrap();

        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn sha256_chain() {
        // Tests multiple SHA-256 compress512 invocations where the outputs are linked to the inputs
        // of the following compression function.
        const N: usize = 3;
        let circuit = CircuitBuilder::new();

        let mut compress_vec = Vec::with_capacity(N);

        // First, declare the initial state.
        let mut state = State::iv(&circuit);
        for i in 0..N {
            // Create a new subcircuit builder. This is not necessary but can improve readability
            // and diagnostics.
            let sha256_builder = circuit.subcircuit(format!("sha256[{i}]"));

            // Build a new instance of the sha256 verification subcircuit, passing the inputs `m` to
            // it. For the first compression `m` is public but everything else if private.
            let m: [Wire; 16] = if i == 0 {
                std::array::from_fn(|_| sha256_builder.add_inout())
            } else {
                std::array::from_fn(|_| sha256_builder.add_witness())
            };
            let compress = Compress::new(&sha256_builder, state, m);
            state = compress.state_out.clone();

            compress_vec.push(compress);
        }

        let circuit = circuit.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        for compress in &compress_vec {
            compress.populate_m(&mut w, [0; 64]);
        }
        circuit.populate_wire_witness(&mut w).unwrap();

        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn sha256_parallel() {
        // Test multiple SHA-256 compressions in parallel (no chaining)
        const N: usize = 3;
        let circuit = CircuitBuilder::new();

        let mut compress_vec = Vec::with_capacity(N);

        for i in 0..N {
            // Create a new subcircuit builder
            let sha256_builder = circuit.subcircuit(format!("sha256[{i}]"));

            // Each SHA-256 instance gets its own IV and input (all committed)
            let state = State::iv(&sha256_builder);
            let m: [Wire; 16] = std::array::from_fn(|_| sha256_builder.add_inout());
            let compress = Compress::new(&sha256_builder, state, m);

            compress_vec.push(compress);
        }

        let circuit = circuit.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        for compress in &compress_vec {
            compress.populate_m(&mut w, [0; 64]);
        }
        circuit.populate_wire_witness(&mut w).unwrap();

        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }
}
