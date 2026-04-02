// Copyright 2025 Irreducible Inc.
use binius_core::word::Word;
use binius_frontend::{CircuitBuilder, Wire};

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// The internal state of SHA-512.
///
/// The state size is 512 bits.
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
        State(std::array::from_fn(|i| builder.add_constant(Word(IV[i]))))
    }
}

/// SHA-512 compress function.
pub struct Compress {
    pub state_in: State,
    pub state_out: State,
    pub m: [Wire; 16],
}

impl Compress {
    pub fn new(builder: &CircuitBuilder, state_in: State, m: [Wire; 16]) -> Self {
        // ---- message-schedule ----
        // W[0..15] = block_words
        // for t = 16 .. 79:
        //     s0   = σ0(W[t-15])
        //     s1   = σ1(W[t-2])
        //     (p, _)  = Add(W[t-16], s0)
        //     (q, _)  = Add(p, W[t-7])
        //     (W[t],_) = Add(q, s1)
        let mut w = Vec::with_capacity(80);

        // W[0..15] = block_words
        w.extend_from_slice(&m);

        let zero = builder.add_constant(Word::ZERO);
        // W[16..79] computed from previous W values
        for t in 16..80 {
            let s0 = small_sigma_0(builder, w[t - 15]);
            let s1 = small_sigma_1(builder, w[t - 2]);
            let (p, _carry) = builder.iadd_cin_cout(w[t - 16], s0, zero);
            let (q, _carry) = builder.iadd_cin_cout(p, w[t - 7], zero);
            let (w_t, _carry) = builder.iadd_cin_cout(q, s1, zero);
            w.push(w_t);
        }

        let w: &[Wire; 80] = (&*w).try_into().unwrap();
        let mut state = state_in.clone();
        for t in 0..80 {
            state = round(builder, t, state, w);
        }

        // Add the compressed chunk to the current hash value
        let (a_out, _carry) = builder.iadd_cin_cout(state_in.0[0], state.0[0], zero);
        let (b_out, _carry) = builder.iadd_cin_cout(state_in.0[1], state.0[1], zero);
        let (c_out, _carry) = builder.iadd_cin_cout(state_in.0[2], state.0[2], zero);
        let (d_out, _carry) = builder.iadd_cin_cout(state_in.0[3], state.0[3], zero);
        let (e_out, _carry) = builder.iadd_cin_cout(state_in.0[4], state.0[4], zero);
        let (f_out, _carry) = builder.iadd_cin_cout(state_in.0[5], state.0[5], zero);
        let (g_out, _carry) = builder.iadd_cin_cout(state_in.0[6], state.0[6], zero);
        let (h_out, _carry) = builder.iadd_cin_cout(state_in.0[7], state.0[7], zero);

        let state_out = State([a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out]);

        Compress {
            state_in,
            state_out,
            m,
        }
    }

    pub fn populate_m(&self, w: &mut binius_frontend::WitnessFiller, m: [u8; 128]) {
        debug_assert_eq!(self.m.len(), 16);

        for i in 0..16 {
            let j = i * 8;
            // Assemble a 64-bit big-endian word.
            let limb = ((m[j] as u64) << 56)
                | ((m[j + 1] as u64) << 48)
                | ((m[j + 2] as u64) << 40)
                | ((m[j + 3] as u64) << 32)
                | ((m[j + 4] as u64) << 24)
                | ((m[j + 5] as u64) << 16)
                | ((m[j + 6] as u64) << 8)
                | (m[j + 7] as u64);

            // Write it to the witness.  Word is a thin wrapper around u64.
            w[self.m[i]] = Word(limb);
        }
    }
}

fn round(builder: &CircuitBuilder, round: usize, state: State, w: &[Wire; 80]) -> State {
    let State([a, b, c, d, e, f, g, h]) = state;

    let big_sigma_e = big_sigma_1(builder, e);
    let ch_efg = ch(builder, e, f, g);
    let zero = builder.add_constant(Word::ZERO);
    let (t1a, _carry) = builder.iadd_cin_cout(h, big_sigma_e, zero);
    let (t1b, _carry) = builder.iadd_cin_cout(t1a, ch_efg, zero);
    let rc = builder.add_constant(Word(K[round]));
    let (t1c, _carry) = builder.iadd_cin_cout(t1b, rc, zero);
    let (t1, _carry) = builder.iadd_cin_cout(t1c, w[round], zero);

    let big_sigma_a = big_sigma_0(builder, a);
    let maj_abc = maj(builder, a, b, c);
    let (t2, _carry) = builder.iadd_cin_cout(big_sigma_a, maj_abc, zero);

    let h = g;
    let g = f;
    let f = e;
    let (e, _carry) = builder.iadd_cin_cout(d, t1, zero);
    let d = c;
    let c = b;
    let b = a;
    let (a, _carry) = builder.iadd_cin_cout(t1, t2, zero);

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

/// Σ0(x)       = XOR( XOR( ROTR(x, 28), ROTR(x, 34) ), ROTR(x, 39) )
fn big_sigma_0(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr(x, 28);
    let r2 = b.rotr(x, 34);
    let r3 = b.rotr(x, 39);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, r3)
}

/// Σ1(x)       = XOR( XOR( ROTR(x, 14), ROTR(x, 18) ), ROTR(x, 41) )
fn big_sigma_1(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr(x, 14);
    let r2 = b.rotr(x, 18);
    let r3 = b.rotr(x, 41);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, r3)
}

/// σ0(x)       = XOR( XOR( ROTR(x,  1), ROTR(x,  8)), SHR(x,  7) )
fn small_sigma_0(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr(x, 1);
    let r2 = b.rotr(x, 8);
    let s1 = b.shr(x, 7);
    let x1 = b.bxor(r1, r2);
    b.bxor(x1, s1)
}

/// σ1(x)       = XOR( XOR( ROTR(x, 19), ROTR(x, 61) ), SHR(x,  6) )
fn small_sigma_1(b: &CircuitBuilder, x: Wire) -> Wire {
    let r1 = b.rotr(x, 19);
    let r2 = b.rotr(x, 61);
    let s1 = b.shr(x, 6);
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
    ///     compress1024(preimage) = S
    ///
    /// without revealing the preimage, only S.
    #[test]
    fn proof_preimage() {
        // Use the test-vector for SHA512 single block message: "abc".
        let mut preimage: [u8; 128] = [0; 128];
        preimage[0..3].copy_from_slice(b"abc");
        preimage[3] = 0x80;
        preimage[127] = 0x18;

        #[rustfmt::skip]
		let expected_state: [u64; 8] = [
			0xddaf35a193617aba, 0xcc417349ae204131, 0x12e6fa4e89a97ea2, 0x0a9eeee64b55d39a,
			0x2192992a274fc1a8, 0x36ba3c23a3feebbd, 0x454d4423643ce80e, 0x2a9ac94fa54ca49f,
		];

        let circuit = CircuitBuilder::new();
        let state = State::iv(&circuit);
        let input: [Wire; 16] = std::array::from_fn(|_| circuit.add_witness());
        let output: [Wire; 8] = std::array::from_fn(|_| circuit.add_inout());
        let compress = Compress::new(&circuit, state, input);

        for (i, (actual_x, expected_x)) in compress.state_out.0.iter().zip(output).enumerate() {
            circuit.assert_eq(format!("preimage_eq[{i}]"), *actual_x, expected_x);
        }

        let circuit = circuit.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        // Populate the input message for the compression function.
        compress.populate_m(&mut w, preimage);

        for (i, &output) in output.iter().enumerate() {
            w[output] = Word(expected_state[i]);
        }
        circuit.populate_wire_witness(&mut w).unwrap();

        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn sha512_chain() {
        // Tests multiple SHA-512 compress1024 invocations where the outputs are linked to the
        // inputs of the following compression function.
        const N: usize = 3;
        let circuit = CircuitBuilder::new();

        let mut compress_vec = Vec::with_capacity(N);

        // First, declare the initial state.
        let mut state = State::iv(&circuit);
        for i in 0..N {
            // Create a new subcircuit builder. This is not necessary but can improve readability
            // and diagnostics.
            let sha512_builder = circuit.subcircuit(format!("sha512[{i}]"));

            // Build a new instance of the sha512 verification subcircuit, passing the inputs `m` to
            // it. For the first compression `m` is public but everything else if private.
            let m: [Wire; 16] = if i == 0 {
                std::array::from_fn(|_| sha512_builder.add_inout())
            } else {
                std::array::from_fn(|_| sha512_builder.add_witness())
            };
            let compress = Compress::new(&sha512_builder, state, m);
            state = compress.state_out.clone();

            compress_vec.push(compress);
        }

        let circuit = circuit.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        for compress in &compress_vec {
            compress.populate_m(&mut w, [0; 128]);
        }
        circuit.populate_wire_witness(&mut w).unwrap();

        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    fn sha512_parallel() {
        // Test multiple SHA-512 compressions in parallel (no chaining)
        const N: usize = 3;
        let circuit = CircuitBuilder::new();

        let mut compress_vec = Vec::with_capacity(N);

        for i in 0..N {
            // Create a new subcircuit builder
            let sha512_builder = circuit.subcircuit(format!("sha512[{i}]"));

            // Each SHA-512 instance gets its own IV and input (all committed)
            let state = State::iv(&sha512_builder);
            let m: [Wire; 16] = std::array::from_fn(|_| sha512_builder.add_inout());
            let compress = Compress::new(&sha512_builder, state, m);

            compress_vec.push(compress);
        }

        let circuit = circuit.build();
        let cs = circuit.constraint_system();
        let mut w = circuit.new_witness_filler();

        for compress in &compress_vec {
            compress.populate_m(&mut w, [0; 128]);
        }
        circuit.populate_wire_witness(&mut w).unwrap();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }
}
