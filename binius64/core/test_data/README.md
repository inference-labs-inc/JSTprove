# Test Data for Serialization Compatibility

This directory contains binary reference files used for testing serialization format compatibility.

## Files

- `constraint_system_v2.bin`: Reference binary serialization of a `ConstraintSystem` using serialization version 2.
- `public_witness_v1.bin`: Reference binary serialization of a `PublicWitness` using serialization version 1.
- `proof_v1.bin`: Reference binary serialization of a `Proof` using serialization version 1.

## Purpose

These binary files serve as regression tests to ensure that:

1. **Backward compatibility**: Future changes to the serialization format don't accidentally break the ability to deserialize existing data.

2. **Version enforcement**: If breaking changes are made to the serialization format, developers are forced to increment the `SERIALIZATION_VERSION` constant, which will cause the compatibility tests to fail until the version is updated.

3. **Format validation**: The tests verify both the structure and content of deserialized data to ensure the format remains consistent.

## Updating Reference Files

If you make intentional breaking changes to the serialization format:

### For ConstraintSystem
1. Increment `ConstraintSystem::SERIALIZATION_VERSION`
2. Run the ignored test to regenerate the reference file:
   ```bash
   cargo test -p binius-core -- --ignored create_reference_binary
   ```
3. Rename the new file to include the new version number
4. Update test paths to reference the new file

### For PublicWitness
1. Increment `PublicWitness::SERIALIZATION_VERSION`
2. Run the ignored test to regenerate the reference file:
   ```bash
   cargo test -p binius-core -- --ignored create_public_witness_reference_binary
   ```
3. Rename the new file to include the new version number
4. Update test paths to reference the new file

### For Proof
1. Increment `Proof::SERIALIZATION_VERSION`
2. Run the ignored test to regenerate the reference file:
   ```bash
   cargo test -p binius-core -- --ignored create_proof_reference_binary
   ```
3. Rename the new file to include the new version number
4. Update test paths to reference the new file

## Binary Format

The binary format uses little-endian encoding and follows this structure:

### ConstraintSystem Format
1. **Version header** (4 bytes): `u32` serialization version
2. **ValueVecLayout**: Layout configuration
3. **Constants**: Vector of `Word` values
4. **AND constraints**: Vector of `AndConstraint` structures
5. **MUL constraints**: Vector of `MulConstraint` structures

### PublicWitness Format
1. **Version header** (4 bytes): `u32` serialization version
2. **Data**: Vector of `Word` values representing the public witness

### Proof Format
1. **Version header** (4 bytes): `u32` serialization version
2. **Challenger type**: String identifying the challenger (e.g., "HasherChallenger<Sha256>")
3. **Transcript data**: Vector of bytes containing the proof transcript

All data uses the platform-independent `SerializeBytes`/`DeserializeBytes` traits from `binius-utils`.
