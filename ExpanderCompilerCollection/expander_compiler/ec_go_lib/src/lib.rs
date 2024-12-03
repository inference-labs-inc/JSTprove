use arith::FieldSerde;
use expander_compiler::circuit::layered;
use libc::{c_uchar, c_ulong, malloc};
use std::io::Cursor;
use std::ptr;
use std::slice;

use expander_compiler::{
    circuit::{config, ir},
    utils::serde::Serde,
};

const ABI_VERSION: c_ulong = 4;

#[repr(C)]
pub struct ByteArray {
    data: *mut c_uchar,
    length: c_ulong,
}

#[repr(C)]
pub struct CompileResult {
    ir_witness_gen: ByteArray,
    layered: ByteArray,
    error: ByteArray,
}

fn compile_inner_with_config<C>(ir_source: Vec<u8>) -> Result<(Vec<u8>, Vec<u8>), String>
where
    C: config::Config,
{
    let ir_source = ir::source::RootCircuit::<C>::deserialize_from(&ir_source[..])
        .map_err(|e| format!("failed to deserialize the source circuit: {}", e))?;
    let (ir_witness_gen, layered) =
        expander_compiler::compile::compile(&ir_source).map_err(|e| e.to_string())?;
    let mut ir_wg_s: Vec<u8> = Vec::new();
    ir_witness_gen
        .serialize_into(&mut ir_wg_s)
        .map_err(|e| format!("failed to serialize the witness generator: {}", e))?;
    let mut layered_s: Vec<u8> = Vec::new();
    layered
        .serialize_into(&mut layered_s)
        .map_err(|e| format!("failed to serialize the layered circuit: {}", e))?;
    Ok((ir_wg_s, layered_s))
}

fn compile_inner(ir_source: Vec<u8>, config_id: u64) -> Result<(Vec<u8>, Vec<u8>), String> {
    match config_id {
        1 => compile_inner_with_config::<config::M31Config>(ir_source),
        2 => compile_inner_with_config::<config::BN254Config>(ir_source),
        3 => compile_inner_with_config::<config::GF2Config>(ir_source),
        _ => Err(format!("unknown config id: {}", config_id)),
    }
}

fn to_compile_result(result: Result<(Vec<u8>, Vec<u8>), String>) -> CompileResult {
    match result {
        Ok((ir_witness_gen, layered)) => {
            let ir_wg_len = ir_witness_gen.len();
            let layered_len = layered.len();
            let ir_wg_ptr = if ir_wg_len > 0 {
                unsafe {
                    let ptr = malloc(ir_wg_len) as *mut u8;
                    ptr.copy_from(ir_witness_gen.as_ptr(), ir_wg_len);
                    ptr
                }
            } else {
                ptr::null_mut()
            };
            let layered_ptr = if layered_len > 0 {
                unsafe {
                    let ptr = malloc(layered_len) as *mut u8;
                    ptr.copy_from(layered.as_ptr(), layered_len);
                    ptr
                }
            } else {
                ptr::null_mut()
            };
            CompileResult {
                ir_witness_gen: ByteArray {
                    data: ir_wg_ptr,
                    length: ir_wg_len as c_ulong,
                },
                layered: ByteArray {
                    data: layered_ptr,
                    length: layered_len as c_ulong,
                },
                error: ByteArray {
                    data: ptr::null_mut(),
                    length: 0,
                },
            }
        }
        Err(error) => {
            let error_len = error.len();
            let error_ptr = if error_len > 0 {
                unsafe {
                    let ptr = malloc(error_len) as *mut u8;
                    ptr.copy_from(error.as_ptr(), error_len);
                    ptr
                }
            } else {
                ptr::null_mut()
            };
            CompileResult {
                ir_witness_gen: ByteArray {
                    data: ptr::null_mut(),
                    length: 0,
                },
                layered: ByteArray {
                    data: ptr::null_mut(),
                    length: 0,
                },
                error: ByteArray {
                    data: error_ptr,
                    length: error_len as c_ulong,
                },
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn compile(ir_source: ByteArray, config_id: c_ulong) -> CompileResult {
    let ir_source = unsafe { slice::from_raw_parts(ir_source.data, ir_source.length as usize) };
    let result = compile_inner(ir_source.to_vec(), config_id);
    to_compile_result(result)
}

fn dump_proof_and_claimed_v<F: arith::Field + arith::FieldSerde>(
    proof: &expander_transcript::Proof,
    claimed_v: &F,
) -> Vec<u8> {
    let mut bytes = Vec::new();

    proof.serialize_into(&mut bytes).unwrap(); // TODO: error propagation
    claimed_v.serialize_into(&mut bytes).unwrap(); // TODO: error propagation

    bytes
}

fn load_proof_and_claimed_v<F: arith::Field + arith::FieldSerde>(
    bytes: &[u8],
) -> Result<(expander_transcript::Proof, F), ()> {
    let mut cursor = Cursor::new(bytes);

    let proof = expander_transcript::Proof::deserialize_from(&mut cursor).map_err(|_| ())?;
    let claimed_v = F::deserialize_from(&mut cursor).map_err(|_| ())?;

    Ok((proof, claimed_v))
}

fn prove_circuit_file_inner<C: expander_config::GKRConfig, CC: config::Config>(
    circuit_filename: &str,
    witness: &[u8],
) -> Vec<u8>
where
    C::SimdCircuitField: arith::SimdField<Scalar = CC::CircuitField>,
{
    let config = expander_config::Config::<C>::new(
        expander_config::GKRScheme::Vanilla,
        expander_config::MPIConfig::new(),
    );
    let mut circuit = expander_circuit::Circuit::<C>::load_circuit(circuit_filename);
    let witness = layered::witness::Witness::<CC>::deserialize_from(witness).unwrap();
    let (simd_input, simd_public_input) = witness.to_simd::<C::SimdCircuitField>();
    circuit.layers[0].input_vals = simd_input;
    circuit.public_input = simd_public_input;
    circuit.evaluate();
    let mut prover = gkr::Prover::new(&config);
    prover.prepare_mem(&circuit);
    let (claimed_v, proof) = prover.prove(&mut circuit);
    dump_proof_and_claimed_v(&proof, &claimed_v)
}

fn verify_circuit_file_inner<C: expander_config::GKRConfig, CC: config::Config>(
    circuit_filename: &str,
    witness: &[u8],
    proof_and_claimed_v: &[u8],
) -> u8
where
    C::SimdCircuitField: arith::SimdField<Scalar = CC::CircuitField>,
{
    let config = expander_config::Config::<C>::new(
        expander_config::GKRScheme::Vanilla,
        expander_config::MPIConfig::new(),
    );
    let mut circuit = expander_circuit::Circuit::<C>::load_circuit(circuit_filename);
    let witness = layered::witness::Witness::<CC>::deserialize_from(witness).unwrap();
    let (simd_input, simd_public_input) = witness.to_simd::<C::SimdCircuitField>();
    circuit.layers[0].input_vals = simd_input;
    circuit.public_input = simd_public_input.clone();
    let (proof, claimed_v) = match load_proof_and_claimed_v(proof_and_claimed_v) {
        Ok((proof, claimed_v)) => (proof, claimed_v),
        Err(_) => {
            return 0;
        }
    };
    let verifier = gkr::Verifier::new(&config);
    verifier.verify(&mut circuit, &simd_public_input, &claimed_v, &proof) as u8
}

#[no_mangle]
pub extern "C" fn prove_circuit_file(
    circuit_filename: ByteArray,
    witness: ByteArray,
    config_id: c_ulong,
) -> ByteArray {
    let circuit_filename = unsafe {
        let slice = slice::from_raw_parts(circuit_filename.data, circuit_filename.length as usize);
        std::str::from_utf8(slice).unwrap()
    };
    let witness = unsafe { slice::from_raw_parts(witness.data, witness.length as usize) };
    let proof = match config_id {
        1 => prove_circuit_file_inner::<expander_config::M31ExtConfigSha2, config::M31Config>(
            circuit_filename,
            witness,
        ),
        2 => prove_circuit_file_inner::<expander_config::BN254ConfigSha2, config::BN254Config>(
            circuit_filename,
            witness,
        ),
        3 => prove_circuit_file_inner::<expander_config::GF2ExtConfigSha2, config::GF2Config>(
            circuit_filename,
            witness,
        ),
        _ => panic!("unknown config id: {}", config_id),
    };
    let proof_len = proof.len();
    let proof_ptr = if proof_len > 0 {
        unsafe {
            let ptr = malloc(proof_len) as *mut u8;
            ptr.copy_from(proof.as_ptr(), proof_len);
            ptr
        }
    } else {
        ptr::null_mut()
    };
    ByteArray {
        data: proof_ptr,
        length: proof_len as c_ulong,
    }
}

#[no_mangle]
pub extern "C" fn verify_circuit_file(
    circuit_filename: ByteArray,
    witness: ByteArray,
    proof: ByteArray,
    config_id: c_ulong,
) -> c_uchar {
    let circuit_filename = unsafe {
        let slice = slice::from_raw_parts(circuit_filename.data, circuit_filename.length as usize);
        std::str::from_utf8(slice).unwrap()
    };
    let witness = unsafe { slice::from_raw_parts(witness.data, witness.length as usize) };
    let proof = unsafe { slice::from_raw_parts(proof.data, proof.length as usize) };
    match config_id {
        1 => verify_circuit_file_inner::<expander_config::M31ExtConfigSha2, config::M31Config>(
            circuit_filename,
            witness,
            proof,
        ),
        2 => verify_circuit_file_inner::<expander_config::BN254ConfigSha2, config::BN254Config>(
            circuit_filename,
            witness,
            proof,
        ),
        3 => verify_circuit_file_inner::<expander_config::GF2ExtConfigSha2, config::GF2Config>(
            circuit_filename,
            witness,
            proof,
        ),
        _ => panic!("unknown config id: {}", config_id),
    }
}

#[no_mangle]
pub extern "C" fn abi_version() -> c_ulong {
    ABI_VERSION
}
