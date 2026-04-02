// Copyright 2025 Irreducible Inc.
use binius_core::Word;
use binius_frontend::{CircuitBuilder, Wire, util::all_true};

use super::{
    codeword::codeword,
    hashing::{circuit_chain_hash, circuit_message_hash},
};
use crate::keccak::Keccak256;

/// XMSS standard nonce length in bytes
const NONCE_LENGTH_BYTES: usize = 23;

/// SHA-256/Keccak-256 output size in bytes
const MESSAGE_LENGTH_BYTES: usize = 32;

/// Number of 64-bit wires needed to represent a 32-byte hash (32 bytes / 8 bytes per wire)
const HASH_WIRES_COUNT: usize = 4;

/// Result of Winternitz OTS verification containing hashers that need to be populated.
///
/// This struct contains all the hash circuits and witness data needed for the bounded (pooled)
/// optimization. Instead of creating a full chain for each coordinate, we pool only the remaining
/// steps to the endpoint for every chain and prove existence of valid segments in that pooled
/// array.
pub struct WinternitzOtsHashers {
    /// Hasher for tweaked message: domain_param || TWEAK_MESSAGE (0x02) || nonce || message
    /// This produces the hash from which coordinate values x_i are extracted
    pub message_hasher: Keccak256,

    /// Exactly dimension * (chain_len - 1) - target_sum hashers representing
    /// all pooled hash operations across all chains.
    /// Each enforces a chain-tweaked Keccak step:
    /// `hash_out = H(domain_param || 0x00 || hash_in || chain_idx || position)`.
    pub step_hashers: Vec<Keccak256>,

    /// Input hash values for each pooled step (32 bytes as HASH_WIRES_COUNT wires)
    pub step_hash_inputs: Vec<[Wire; HASH_WIRES_COUNT]>,

    /// Chain index for each pooled step (identifies which coordinate this step belongs to)
    pub step_chain_indices: Vec<Wire>,

    /// Position within chain segment for each pooled step (1-based count)
    /// Note: counts are relative within a pooled segment: 1..r_i
    pub step_counts: Vec<Wire>,

    /// Absolute chain position for each pooled step (x_i + count), passed to the chain hasher
    pub step_positions: Vec<Wire>,
}

/// Verifies a Winternitz One-Time Signature using a pooled hash-chain optimization.
///
/// # Definitions
/// - `dimension`: number of chains = `spec.dimension()`.
/// - `chain_len`: length of each chain = `spec.chain_len()` = `2^{coordinate_resolution_bits}`.
/// - `x_i`: the i-th coordinate extracted by the codeword from the tweaked message hash. These
///   satisfy `0 ≤ x_i ≤ chain_len - 1` with `∑ x_i = target_sum`.
/// - `r_i = (chain_len - 1) - x_i`: remaining steps from `sig_i` to the endpoint `pk_i`.
///   Consequently, `pk_i = H^{r_i}(sig_i) = H^{chain_len - 1 - x_i}(sig_i)`.
///
/// # Circuit layout
///
/// `r_i = (chain_len - 1) - x_i` from `sig_i` to `pk_i` in a pooled table of size
///
/// # Pooled-chain layout
///
/// The circuit verifies a single pooled chain consisting of all hashers required.
///
/// Number of hashers = dimension*(chain_len - 1) - target_sum`.
///
/// Each pooled entry carries and is used as follows:
/// - `hash_in[idx]`   (32 bytes) - Input to hash step
/// - `hash_out[idx]`  (32 bytes) - Output of hash step
/// - `chain_idx[idx]` (u64)      - Which coordinate chain this step belongs to
/// - `count[idx]`     (u64)      - Relative position within a segment (1..r_i), used only for
///   existence checks `(hash_out, count) == (pk_i, r_i)` and to enforce starts `(count == 1)`
/// - `position[idx]`  (u64)      - Absolute chain position used by the chain-tweaked Keccak:
///   `hash_out = H(domain_param || 0x00 || hash_in || chain_idx || position)`. For a chain with
///   coordinate `x_i`, we set `position = x_i + count` so the hashing follows the classic absolute
///   index semantics while we still prove segment existence using relative `count`.
///
/// A concrete layout for three chains with remaining lengths r0=3, r1=2, r2=1:
///
/// ```text
/// idx:      0             1             2             3             4             5
///         +-------------+-------------+-------------+-------------+-------------+-------------+
/// chain:  |    c=0      |    c=0      |    c=0      |    c=1      |    c=1      |    c=2      |
///         +-------------+-------------+-------------+-------------+-------------+-------------+
/// count:  |    k=1      |    k=2      |    k=3      |    k=1      |    k=2      |    k=1      |
///         +-------------+-------------+-------------+-------------+-------------+-------------+
/// link:   | in=H0       | in=out[0]   | in=out[1]   | in=H1       | in=out[3]   | in=H2       |
///         +-------------+-------------+-------------+-------------+-------------+-------------+
/// ```
///
/// If a chain `c` has `x_c = 0`, it contributes no hashers to the pooled-chain
///
/// Within any maximal run of identical `chain_idx`, we enforce a proper chain segment:
/// - `count` increases by exactly 1
/// - `hash_in[i] == hash_out[i-1]`
///
/// When `chain_idx` changes, the next segment must start with `count == 1`.
///
/// # Verification of the pooled-chain
///
/// Verification can be described with the following pseudocode. For each chain `i`, a linear scan
/// checks existence of a row with `(hash_out, count) == (pk_i, r_i)`:
///
/// ```text
/// for i in 0..num_chains:
///   found = 1   # 0 indicates boolean True
///   for idx in 0..pooled_len:
///     found *= (hash_check[idx].hash_out, hash_check[idx].count) XOR (pk_i, r_i)
///   assert found == 0
/// ```
///
/// In our circuit, we express the same condition as an OR of equalities:
/// - For chain `i` we compute `eq_all[idx] = (hash_out[idx] == pk_i) AND (count[idx] == r_i)`
/// - Accumulate `any_match = OR_idx eq_all[idx]`
/// - Assert `any_match` is true (MSB-bool), which is logically equivalent to the product-of-XORs
///   being zero if and only if a match exists.
///
/// Additionally, we assert an existence of a segment start for chain `i` by checking there exists
/// an index `idx` with: `(hash_in[idx] == sig_i) AND (count[idx] == 1) AND (chain_idx[idx] == i)`.
/// (Both the pk and the start existence checks are relaxed when `r_i == 0`, and additionally we
/// assert `pk_i == sig_i` for that chain.)
///
/// # Function Parameters
///
/// - `domain_param`: Domain separation parameter for hash functions
/// - `message`: 32-byte message being signed (as HASH_WIRES_COUNT wires of 8 bytes each)
/// - `nonce`: NONCE_LENGTH_BYTES-byte nonce for tweaked message hash
/// - `signature_hashes`: Starting hash values for each coordinate chain (`sig_i`)
/// - `public_key_hashes`: Expected final hash values for each coordinate chain (`pk_i`)
/// - `spec`: Winternitz parameters (dimension, resolution, target sum, etc.)
///
/// # Returns
///
/// A `WinternitzOtsHashers` containing the hashers that need to be populated
#[allow(clippy::too_many_arguments)]
pub fn circuit_winternitz_ots(
    builder: &CircuitBuilder,
    domain_param: &[Wire],
    message: &[Wire],
    nonce: &[Wire],
    signature_hashes: &[[Wire; HASH_WIRES_COUNT]],
    public_key_hashes: &[[Wire; HASH_WIRES_COUNT]],
    spec: &WinternitzSpec,
) -> WinternitzOtsHashers {
    assert!(
        spec.domain_param_len <= domain_param.len() * 8,
        "domain_param wires must have capacity for {} bytes, but only has capacity for {} bytes",
        spec.domain_param_len,
        domain_param.len() * 8
    );
    assert_eq!(
        message.len(),
        HASH_WIRES_COUNT,
        "message must be 32 bytes as {} wires",
        HASH_WIRES_COUNT
    );

    // Step 1: Compute tweaked message hash
    // Format: domain_param || TWEAK_MESSAGE (0x02) || nonce || message
    let message_hash_output: [Wire; HASH_WIRES_COUNT] =
        std::array::from_fn(|_| builder.add_witness());
    let message_hasher = circuit_message_hash(
        builder,
        domain_param.to_vec(),
        spec.domain_param_len,
        nonce.to_vec(),
        NONCE_LENGTH_BYTES,
        message.to_vec(),
        MESSAGE_LENGTH_BYTES,
        message_hash_output,
    );

    // Step 2: Extract codeword coordinates from message hash
    // Coordinates x_i determine chain lengths: each chain i has x_i hash steps
    let message_hash_bytes = spec.message_hash_len;
    let message_hash_wires_needed = message_hash_bytes.div_ceil(8);
    let message_hash_for_codeword = &message_hash_output[..message_hash_wires_needed];

    let coordinates = codeword(
        builder,
        spec.dimension(),
        spec.coordinate_resolution_bits,
        spec.target_sum,
        message_hash_for_codeword,
    );

    assert_eq!(
        coordinates.len(),
        spec.dimension(),
        "Codeword dimension mismatch"
    );
    assert_eq!(
        signature_hashes.len(),
        spec.dimension(),
        "Signature hashes count mismatch"
    );
    assert_eq!(
        public_key_hashes.len(),
        spec.dimension(),
        "Public key hashes count mismatch"
    );

    // Step 3: Create pooled hash step entries
    // For bounded (top-based) semantics, we pool the remaining steps to the endpoint.
    // Each chain i contributes r_i = (chain_len - 1) - x_i steps,
    // so total pooled_len = dimension*(chain_len - 1) - target_sum.
    let chain_len = spec.chain_len();
    let pooled_len: usize = spec.dimension() * (chain_len - 1) - (spec.target_sum as usize);
    let mut step_hashers = Vec::with_capacity(pooled_len);

    // Each pooled step enforces: hash_out = H(param || TWEAK_CHAIN (0x00) || hash_in || chain_idx
    // || position)
    let zero = builder.add_constant(Word::ZERO);
    let one = builder.add_constant(Word::ONE);

    let mut step_hash_inputs: Vec<[Wire; HASH_WIRES_COUNT]> = Vec::with_capacity(pooled_len);
    let mut step_hash_out: Vec<[Wire; HASH_WIRES_COUNT]> = Vec::with_capacity(pooled_len);
    let mut step_chain_indices: Vec<Wire> = Vec::with_capacity(pooled_len);
    let mut step_counts: Vec<Wire> = Vec::with_capacity(pooled_len);
    let mut step_positions: Vec<Wire> = Vec::with_capacity(pooled_len);

    for _ in 0..pooled_len {
        let in_w: [Wire; HASH_WIRES_COUNT] = std::array::from_fn(|_| builder.add_witness());
        let count = builder.add_witness();
        let chain_idx = builder.add_witness();
        let position = builder.add_witness();
        let out_w: [Wire; HASH_WIRES_COUNT] = std::array::from_fn(|_| builder.add_witness());

        let keccak = circuit_chain_hash(
            builder,
            domain_param.to_vec(),
            spec.domain_param_len,
            in_w,
            chain_idx,
            position,
            out_w,
        );

        step_hash_inputs.push(in_w);
        step_hash_out.push(out_w);
        step_chain_indices.push(chain_idx);
        step_counts.push(count);
        step_positions.push(position);
        step_hashers.push(keccak);
    }

    // Step 3b: Enforce segment linking constraints
    // Within each chain segment (consecutive rows with the same chain_idx),
    // consecutive steps must properly link:
    // - count increases by exactly 1: count[j] = count[j-1] + 1
    // - chained hashes:             hash_in[j] == hash_out[j-1]
    // When chain_idx changes, the new segment must start with count == 1.
    if pooled_len > 0 {
        for j in 1..pooled_len {
            let same_chain = builder.icmp_eq(step_chain_indices[j], step_chain_indices[j - 1]);

            // count[j] - count[j-1] == 1
            let (diff, _borrow) = builder.isub_bin_bout(step_counts[j], step_counts[j - 1], zero);
            let inc_ok = builder.icmp_eq(diff, one);

            // hash_in[j] == hash_out[j-1] for all limbs
            let mut link_terms = Vec::with_capacity(HASH_WIRES_COUNT + 1);
            link_terms.push(inc_ok);
            for limb in 0..HASH_WIRES_COUNT {
                let limb_ok =
                    builder.icmp_eq(step_hash_inputs[j][limb], step_hash_out[j - 1][limb]);
                link_terms.push(limb_ok);
            }
            let link_ok = all_true(builder, link_terms);

            // When chain changes, force a new segment start: count[j] == 1
            let start_ok = builder.icmp_eq(step_counts[j], one);

            // If same_chain then enforce link_ok, else enforce start_ok
            let ok = builder.select(same_chain, link_ok, start_ok);
            builder.assert_true(format!("bounded_link_step[{j}]"), ok);
        }
    }

    // Step 3c: First step must start at count = 1
    builder.assert_eq("bounded_first_count_one", step_counts[0], one);

    // Step 4: Existence proofs for each coordinate chain
    // For each chain i, we must prove two things exist in the pooled steps:
    // 1. Endpoint: some entry has (hash_out, count) == (pk_i, r_i)
    // 2. Start: some entry has (hash_in, count, chain_idx) == (sig_i, 1, i)
    for c in 0..spec.dimension() {
        let mut any_match = zero; // accumulates MSB-bool OR of per-step equality
        let mut any_start = zero; // accumulates MSB-bool OR of step starts matching signature

        for j in 0..pooled_len {
            // Check endpoint match: (hash_out[j], count[j]) == (pk_i, r_i)
            let mut endpoint_terms = Vec::with_capacity(HASH_WIRES_COUNT + 1);
            for limb in 0..HASH_WIRES_COUNT {
                let limb_eq = builder.icmp_eq(step_hash_out[j][limb], public_key_hashes[c][limb]);
                endpoint_terms.push(limb_eq);
            }
            let l_minus_1 = builder.add_constant_64((chain_len as u64) - 1);
            let (remaining, _borrow) = builder.isub_bin_bout(l_minus_1, coordinates[c], zero);
            let count_eq = builder.icmp_eq(step_counts[j], remaining);
            endpoint_terms.push(count_eq);
            let endpoint_match = all_true(builder, endpoint_terms);
            any_match = builder.bor(any_match, endpoint_match);

            // Check start match: (hash_in[j], count[j], chain_idx[j]) == (sig_i, 1, i)
            // count=1 ensures we're at the beginning of a chain segment (1-based)
            let mut start_terms = Vec::with_capacity(HASH_WIRES_COUNT + 2);
            for limb in 0..HASH_WIRES_COUNT {
                let limb_eq = builder.icmp_eq(step_hash_inputs[j][limb], signature_hashes[c][limb]);
                start_terms.push(limb_eq);
            }
            let count_is_one = builder.icmp_eq(step_counts[j], one);
            start_terms.push(count_is_one);
            let chain_idx_eq =
                builder.icmp_eq(step_chain_indices[j], builder.add_constant_64(c as u64));
            start_terms.push(chain_idx_eq);
            let start_match = all_true(builder, start_terms);
            any_start = builder.bor(any_start, start_match);
        }

        // Enforce existence constraints (skip pooled checks if remaining == 0)
        let l_minus_1 = builder.add_constant_64((chain_len as u64) - 1);
        let (remaining, _borrow) = builder.isub_bin_bout(l_minus_1, coordinates[c], zero);
        let is_zero_count = builder.icmp_eq(remaining, zero);
        // If there are zero remaining steps, enforce pk_i == sig_i directly
        for limb in 0..HASH_WIRES_COUNT {
            builder.assert_eq_cond(
                "bounded_zero_remaining_pk_eq_sig",
                public_key_hashes[c][limb],
                signature_hashes[c][limb],
                is_zero_count,
            );
        }

        // Must find endpoint match: (hash_out, count) == (pk_i, r_i)
        let endpoint_satisfied = builder.bor(any_match, is_zero_count);
        builder.assert_true("bounded_wots_pk_found_or_zero", endpoint_satisfied);

        // Must find start match: (hash_in, count, chain_idx) == (sig_i, 1, i)
        let start_satisfied = builder.bor(any_start, is_zero_count);
        builder.assert_true("bounded_wots_sig_start_or_zero", start_satisfied);
    }

    WinternitzOtsHashers {
        message_hasher,
        step_hashers,
        step_hash_inputs,
        step_chain_indices,
        step_counts,
        step_positions,
    }
}

/// Specification for Winternitz OTS parameters
///
/// # Constraints
/// - `message_hash_len` must be <= 32 bytes (Keccak-256 output size)
/// - `coordinate_resolution_bits` must divide evenly into `message_hash_len * 8`
pub struct WinternitzSpec {
    /// Number of bytes from message hash to use (must be <= 32)
    pub message_hash_len: usize,
    /// Number of bits per coordinate in the codeword
    pub coordinate_resolution_bits: usize,
    /// Expected sum of all coordinates
    pub target_sum: u64,
    /// Size of the domain parameter in bytes
    pub domain_param_len: usize,
}

impl WinternitzSpec {
    /// Creates a new WinternitzSpec with validation
    pub fn new(
        message_hash_len: usize,
        coordinate_resolution_bits: usize,
        target_sum: u64,
        domain_param_len: usize,
    ) -> Self {
        assert!(
            message_hash_len <= 32,
            "message_hash_len {} exceeds maximum of 32 bytes (Keccak-256 output size)",
            message_hash_len
        );
        assert!(
            (message_hash_len * 8).is_multiple_of(coordinate_resolution_bits),
            "coordinate_resolution_bits {} must divide evenly into message_hash_len * 8 = {}",
            coordinate_resolution_bits,
            message_hash_len * 8
        );

        Self {
            message_hash_len,
            coordinate_resolution_bits,
            target_sum,
            domain_param_len,
        }
    }

    /// Returns the number of coordinates/chains
    pub fn dimension(&self) -> usize {
        self.message_hash_len * 8 / self.coordinate_resolution_bits
    }

    /// Returns the chain length (2^coordinate_resolution_bits)
    pub fn chain_len(&self) -> usize {
        1 << self.coordinate_resolution_bits
    }

    /// Create a spec matching SPEC_1 from leansig-xmss
    pub fn spec_1() -> Self {
        Self::new(18, 2, 119, 18)
    }

    /// Create a spec matching SPEC_2 from leansig-xmss
    pub fn spec_2() -> Self {
        Self::new(18, 4, 297, 18)
    }
}

/// Result of successfully grinding a nonce that produces a valid target sum.
pub struct GrindResult {
    /// The complete tweaked message: param || 0x02 || nonce || message
    pub tweaked_message: Vec<u8>,
    /// The extracted codeword coordinates from the message hash
    pub coords: Vec<u8>,
    /// The nonce value that achieved the target sum
    pub nonce: Vec<u8>,
}

/// Grind for a nonce that produces codeword coordinates summing to the target value.
///
/// This function repeatedly generates random nonces and computes the tweaked message hash
/// until it finds one where the extracted codeword coordinates sum to the target value
/// specified in the Winternitz specification.
///
/// # Arguments
///
/// * `spec` - The Winternitz OTS specification containing dimension, resolution, and target sum
/// * `rng` - Random number generator for generating nonce candidates
/// * `param` - The cryptographic parameter
/// * `message` - The message to be signed
///
/// # Returns
///
/// * `Some(GrindResult)` - Contains the successful nonce, tweaked message, and coordinates
/// * `None` - Failed to find a valid nonce within 1000 attempts.
pub fn grind_nonce(
    spec: &WinternitzSpec,
    rng: &mut rand::rngs::StdRng,
    param: &[u8],
    message: &[u8],
) -> Option<GrindResult> {
    use rand::RngCore;

    use super::{codeword::extract_coordinates, hashing::hash_message};

    let mut nonce = vec![0u8; 23];
    for _ in 0..1000 {
        rng.fill_bytes(&mut nonce);
        let tweaked_message_hash = hash_message(param, &nonce, message);

        let coords = extract_coordinates(
            &tweaked_message_hash[..spec.message_hash_len],
            spec.dimension(),
            spec.coordinate_resolution_bits,
        );
        let coord_sum: usize = coords.iter().map(|&c| c as usize).sum();
        if coord_sum == spec.target_sum as usize {
            use super::hashing::build_message_hash;
            let tweaked_message = build_message_hash(param, &nonce, message);
            return Some(GrindResult {
                tweaked_message,
                coords,
                nonce,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use binius_core::verify::verify_constraints;
    use binius_frontend::util::pack_bytes_into_wires_le;
    use rand::{RngCore, SeedableRng, rngs::StdRng};
    use sha3::{Digest, Keccak256};

    use super::*;
    use crate::hash_based_sig::hashing::{build_chain_hash, hash_chain_keccak};

    /// Number of 64-bit wires needed to represent the nonce (NONCE_LENGTH_BYTES / 8 bytes per wire,
    /// rounded up)
    const NONCE_WIRES_COUNT: usize = NONCE_LENGTH_BYTES.div_ceil(8);

    #[test]
    fn test_circuit_winternitz_ots() {
        let spec = WinternitzSpec::spec_1();
        let builder = CircuitBuilder::new();

        // Inputs
        let domain_param: Vec<Wire> = (0..(spec.domain_param_len.div_ceil(8)))
            .map(|_| builder.add_inout())
            .collect();
        let message: Vec<Wire> = (0..HASH_WIRES_COUNT).map(|_| builder.add_inout()).collect();
        let nonce: Vec<Wire> = (0..NONCE_WIRES_COUNT)
            .map(|_| builder.add_inout())
            .collect();

        let signature_hashes: Vec<[Wire; HASH_WIRES_COUNT]> = (0..spec.dimension())
            .map(|_| std::array::from_fn(|_| builder.add_inout()))
            .collect();
        let public_key_hashes: Vec<[Wire; HASH_WIRES_COUNT]> = (0..spec.dimension())
            .map(|_| std::array::from_fn(|_| builder.add_inout()))
            .collect();

        let result = circuit_winternitz_ots(
            &builder,
            &domain_param,
            &message,
            &nonce,
            &signature_hashes,
            &public_key_hashes,
            &spec,
        );

        let circuit = builder.build();
        let mut w = circuit.new_witness_filler();

        // Randomize inputs and grind a nonce for valid target sum
        let mut rng = StdRng::seed_from_u64(7);
        let mut domain_param_bytes = vec![0u8; spec.domain_param_len];
        rng.fill_bytes(&mut domain_param_bytes);
        let mut message_bytes = [0u8; MESSAGE_LENGTH_BYTES];
        rng.fill_bytes(&mut message_bytes);

        // Find coordinates via grinding for consistency with codeword sum
        let grind = grind_nonce(&spec, &mut rng, &domain_param_bytes, &message_bytes)
            .expect("Failed to find valid nonce");
        let mut nonce_bytes = grind.nonce;
        let tweaked_message = grind.tweaked_message;
        let tweaked_message_hash = Keccak256::digest(&tweaked_message);
        nonce_bytes.resize(24, 0);

        // Pack fixed inputs
        pack_bytes_into_wires_le(&mut w, &domain_param, &domain_param_bytes);
        pack_bytes_into_wires_le(&mut w, &message, &message_bytes);
        pack_bytes_into_wires_le(&mut w, &nonce, &nonce_bytes);

        // Prepare signature and derived public key per the bounded (top-based) model:
        // pk_i is the endpoint after (chain_len - 1 - x_i) steps starting from sig_i,
        // with positions = x_i+1..(chain_len-1)
        let mut sig_hashes = Vec::with_capacity(spec.dimension());
        let mut pk_hashes = Vec::with_capacity(spec.dimension());

        for chain_idx in 0..spec.dimension() {
            let mut sig = [0u8; MESSAGE_LENGTH_BYTES];
            rng.fill_bytes(&mut sig);
            sig_hashes.push(sig);

            let xi = grind.coords[chain_idx] as usize;
            let pk_hash = hash_chain_keccak(
                &domain_param_bytes,
                chain_idx,
                &sig,
                xi,
                spec.chain_len() - 1 - xi,
            );
            pk_hashes.push(pk_hash);

            pack_bytes_into_wires_le(&mut w, &signature_hashes[chain_idx], &sig);
            pack_bytes_into_wires_le(&mut w, &public_key_hashes[chain_idx], &pk_hashes[chain_idx]);
        }

        // Populate message hasher
        result
            .message_hasher
            .populate_message(&mut w, &tweaked_message);
        result
            .message_hasher
            .populate_digest(&mut w, tweaked_message_hash.into());

        // Populate pooled step hashers sequentially in chain-major order (remaining steps)
        let mut hasher_idx = 0usize;
        for chain_idx in 0..spec.dimension() {
            let mut cur = sig_hashes[chain_idx];
            let xi = grind.coords[chain_idx] as usize;
            let remaining = spec.chain_len() - 1 - xi;
            for step in 0..remaining {
                let next_digest = {
                    let msg = build_chain_hash(
                        &domain_param_bytes,
                        &cur,
                        chain_idx as u64,
                        (xi + step + 1) as u64,
                    );
                    Keccak256::digest(&msg).into()
                };

                let keccak = &result.step_hashers[hasher_idx];
                let chain_msg = build_chain_hash(
                    &domain_param_bytes,
                    &cur,
                    chain_idx as u64,
                    (xi + step + 1) as u64,
                );
                keccak.populate_message(&mut w, &chain_msg);
                keccak.populate_digest(&mut w, next_digest);

                // Populate associated step metadata wires
                pack_bytes_into_wires_le(&mut w, &result.step_hash_inputs[hasher_idx], &cur);
                w[result.step_chain_indices[hasher_idx]] = Word::from_u64(chain_idx as u64);
                w[result.step_counts[hasher_idx]] = Word::from_u64((step + 1) as u64);
                w[result.step_positions[hasher_idx]] = Word::from_u64((xi + step + 1) as u64);

                cur = next_digest;
                hasher_idx += 1;
            }
        }

        // Fill remaining
        circuit.populate_wire_witness(&mut w).unwrap();
        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }
}
