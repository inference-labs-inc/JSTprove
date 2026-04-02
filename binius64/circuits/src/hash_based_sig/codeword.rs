// Copyright 2025 Irreducible Inc.
use binius_frontend::{CircuitBuilder, Wire};

/// Extract codeword coordinates from a message hash.
///
/// # Arguments
///
/// * `builder` - Circuit builder for constructing constraints
/// * `dimension` - Number of coordinates to extract from the message hash
/// * `resolution` - Bits per coordinate (must be 1, 2, 4, or 8)
/// * `target_sum` - Expected sum of all coordinates
/// * `message_hash` - Message hash as LE-packed bytes in 64-bit wires
///
/// # Extraction Process
///
/// The function extracts `dimension` coordinates, each of `resolution` bits, from
/// the message hash wires. The extraction follows these rules:
///
/// 1. **Byte Ordering**: Uses little-endian byte packing in 64-bit wires
///    - Byte 0 occupies bits 0-7 (least significant)
///    - Byte 1 occupies bits 8-15
///    - And so on...
///
/// 2. **Bit Ordering Within Bytes**: Uses LSB ordering
///    - Within each byte, coordinates are extracted from low bits to high bits
///    - For example, with 2-bit resolution in byte 0:
///      - Coordinate 0: bits 0-1 (lowest 2 bits of byte)
///      - Coordinate 1: bits 2-3
///      - Coordinate 2: bits 4-5
///      - Coordinate 3: bits 6-7 (highest 2 bits of byte)
///
/// 3. **Target sum**: Adds a constraint that the sum of all extracted coordinates equals
///    `target_sum`
///
/// # Returns
///
/// A vector of wires, each containing a single coordinate value. The
/// coordinate values are bounded by 2^resolution - 1.
///
/// # Panics
///
/// * If `resolution` > 8 bits
/// * If `resolution` is not a power of 2
/// * If `message_hash` doesn't contain enough wires for the requested `dimension` and `resolution`
///   (needs at least `ceil(dimension * resolution / 64)` wires)
pub fn codeword(
    builder: &CircuitBuilder,
    dimension: usize,
    resolution: usize,
    target_sum: u64,
    message_hash: &[Wire],
) -> Vec<Wire> {
    assert!(resolution <= 8, "Resolution must be at most 8 bits");
    assert!(
        resolution.is_power_of_two(),
        "Resolution must be a power of 2"
    );

    // Verify we have enough wires for the requested dimension
    let required_wires = (dimension * resolution).div_ceil(64);
    assert!(
        message_hash.len() >= required_wires,
        "Not enough message_hash wires: need at least {} wires for {} coordinates with resolution {} bits",
        required_wires,
        dimension,
        resolution
    );

    let mut coordinates = Vec::with_capacity(dimension);
    let coords_per_wire = 64 / resolution;
    let mask = builder.add_constant_64((1u64 << resolution) - 1);

    for coord_global_idx in 0..dimension {
        let wire_idx = coord_global_idx / coords_per_wire;
        let coord_idx = coord_global_idx % coords_per_wire;
        let message_wire = message_hash[wire_idx];

        let bit_offset = coord_idx * resolution;
        let byte_idx = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;

        // The bytes are packed LE into wires and we extract coordinates in LSB
        // ordering.
        //
        // 1. (byte_idx * 8): positions us at the start of the target byte (LE packing).
        //
        // 2. + bit_in_byte: moves us to the coordinate's position within that byte (LSB ordering)
        let shift = (byte_idx * 8) + bit_in_byte;
        let coord = builder.band(builder.shr(message_wire, shift as u32), mask);
        coordinates.push(coord);
    }

    // The sum of the coordinates should be equal to the target sum
    let zero = builder.add_constant_64(0);
    let target = builder.add_constant_64(target_sum);

    let mut codeword_sum = zero;
    for &coord in coordinates.iter() {
        let (sum, _carry) = builder.iadd_cin_cout(codeword_sum, coord, zero);
        codeword_sum = sum;
    }
    builder.assert_eq("codeword_sum_check", codeword_sum, target);
    coordinates
}

/// A function to extract the coordinates from a hash according to the rules
/// specified in the `codeword` circuit
pub fn extract_coordinates(hash: &[u8], dimension: usize, resolution_bits: usize) -> Vec<u8> {
    let mut coords = Vec::new();
    let coords_per_byte = 8 / resolution_bits;
    let mask = (1u8 << resolution_bits) - 1;

    for i in 0..dimension {
        let byte_idx = i / coords_per_byte;
        let coord_idx = i % coords_per_byte;
        let shift = coord_idx * resolution_bits;
        let coord = (hash[byte_idx] >> shift) & mask;
        coords.push(coord);
    }
    coords
}

#[cfg(test)]
mod tests {
    use binius_core::{Word, verify::verify_constraints};
    use binius_frontend::{CircuitBuilder, Wire};

    use super::*;

    #[test]
    fn test_coordinate_extraction() {
        let dimension = 32;
        let resolution = 4;
        let target_sum = 240;
        let builder = CircuitBuilder::new();
        let message_hash: Vec<Wire> = vec![builder.add_inout(), builder.add_inout()];

        let coordinates = codeword(&builder, dimension, resolution, target_sum, &message_hash);

        let circuit = builder.build();

        let mut w = circuit.new_witness_filler();

        // LE-packed with LSB ordering within bytes
        //
        // First wire:
        //
        // Byte 0  (0x10): binary 0001 0000 -> coords [0, 1]
        // Byte 1  (0x32): binary 0011 0010 -> coords [2, 3]
        // Byte 2  (0x54): binary 0101 0100 -> coords [4, 5]
        // Byte 3  (0x76): binary 0111 0110 -> coords [6, 7]
        // Byte 4  (0x98): binary 1001 1000 -> coords [8, 9]
        // Byte 5  (0xBA): binary 1011 1010 -> coords [10, 11]
        // Byte 6  (0xDC): binary 1101 1100 -> coords [12, 13]
        // Byte 7  (0xFE): binary 1111 1110 -> coords [14, 15]
        //
        // Second wire:
        //
        // Byte 8  (0xEF): binary 1110 1111 -> coords [15, 14]
        // Byte 9  (0xCD): binary 1100 1101 -> coords [13, 12]
        // Byte 10 (0xAB): binary 1010 1011 -> coords [11, 10]
        // Byte 11 (0x89): binary 1000 1001 -> coords [9, 8]
        // Byte 12 (0x67): binary 0110 0111 -> coords [7, 6]
        // Byte 13 (0x45): binary 0100 0101 -> coords [5, 4]
        // Byte 14 (0x23): binary 0010 0011 -> coords [3, 2]
        // Byte 15 (0x01): binary 0000 0001 -> coords [1, 0]
        w[message_hash[0]] = Word::from_u64(0xFEDC_BA98_7654_3210);
        w[message_hash[1]] = Word::from_u64(0x0123_4567_89AB_CDEF);

        assert_eq!(coordinates.len(), dimension);

        circuit.populate_wire_witness(&mut w).unwrap();

        let wire_separator_idx = dimension / 2;

        for i in 0..wire_separator_idx {
            assert_eq!(w[coordinates[i]].as_u64(), i as u64);
        }

        for i in 0..wire_separator_idx {
            assert_eq!(
                w[coordinates[wire_separator_idx + i]].as_u64(),
                (wire_separator_idx - i - 1) as u64
            );
        }

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }

    #[test]
    #[should_panic(expected = "Not enough message_hash wires")]
    fn test_insufficient_wires() {
        let builder = CircuitBuilder::new();
        let message_hash = builder.add_inout();
        // This should panic: 72 coords * 2 bits = 144 bits, needs 3 wires but we only provide 1
        codeword(&builder, 72, 2, 0, &[message_hash]);
    }

    #[test]
    fn test_coordinate_truncation() {
        let dimension = 3;
        let resolution = 4;
        let target_sum = 6;
        let builder = CircuitBuilder::new();
        let message_hash: Wire = builder.add_inout();

        let coordinates = codeword(&builder, dimension, resolution, target_sum, &[message_hash]);

        let circuit = builder.build();

        let mut w = circuit.new_witness_filler();

        // LE-packed with LSB ordering within bytes
        //
        // First wire:
        //
        // Byte 0  (0x21): binary 0010 0001 -> coords [1, 2]
        // Byte 1  (0x43): binary 0100 0011 -> coords [3, 4] (but we only use coord 3)
        //
        // Other bits are skipped because dimension = 3
        w[message_hash] = Word::from_u64(0x0000_0000_0000_4321);

        assert_eq!(coordinates.len(), dimension);

        circuit.populate_wire_witness(&mut w).unwrap();

        assert_eq!(w[coordinates[0]], Word::from_u64(1));
        assert_eq!(w[coordinates[1]], Word::from_u64(2));
        assert_eq!(w[coordinates[2]], Word::from_u64(3));

        let cs = circuit.constraint_system();
        verify_constraints(cs, &w.into_value_vec()).unwrap();
    }
}
