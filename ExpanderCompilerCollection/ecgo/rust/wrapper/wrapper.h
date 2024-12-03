#include <stdint.h>
#include <stdlib.h>

typedef struct {
    uint8_t* data;
    uint64_t length;
} ByteArray;

typedef struct {
    ByteArray ir_witness_gen;
    ByteArray layered;
    ByteArray error;
} CompileResult;

typedef CompileResult (*compile_func)(ByteArray ir_source, uint64_t config_id);

CompileResult compile(void *f, ByteArray ir_source, uint64_t config_id) {
    return ((compile_func) f)(ir_source, config_id);
}

typedef ByteArray (*prove_circuit_file_func)(ByteArray circuit_filename, ByteArray witness, uint64_t config_id);

ByteArray prove_circuit_file(void *f, ByteArray circuit_filename, ByteArray witness, uint64_t config_id) {
    return ((prove_circuit_file_func) f)(circuit_filename, witness, config_id);
}

typedef uint8_t (*verify_circuit_file_func)(ByteArray circuit_filename, ByteArray witness, ByteArray proof, uint64_t config_id);

uint8_t verify_circuit_file(void *f, ByteArray circuit_filename, ByteArray witness, ByteArray proof, uint64_t config_id) {
    return ((verify_circuit_file_func) f)(circuit_filename, witness, proof, config_id);
}

typedef uint64_t (*abi_version_func)();
uint64_t abi_version(void *f) {
    return ((abi_version_func) f)();
}