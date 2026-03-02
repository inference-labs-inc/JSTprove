use std::fs;
use std::process::Command;

const DATA_PREFIX: &str = "data/";

// circuit for repeating Keccak for 2 times
pub const KECCAK_M31_CIRCUIT: &str = "data/circuit_m31.txt";
pub const KECCAK_GF2_CIRCUIT: &str = "data/circuit_gf2.txt";
pub const KECCAK_BN254_CIRCUIT: &str = "data/circuit_bn254.txt";
pub const KECCAK_GOLDILOCKS_CIRCUIT: &str = "data/circuit_goldilocks.txt";
pub const KECCAK_BABYBEAR_CIRCUIT: &str = "data/circuit_babybear.txt";

pub const KECCAK_M31_WITNESS: &str = "data/witness_m31.txt";
pub const KECCAK_GF2_WITNESS: &str = "data/witness_gf2.txt";
pub const KECCAK_BN254_WITNESS: &str = "data/witness_bn254.txt";
pub const KECCAK_GOLDILOCKS_WITNESS: &str = "data/witness_goldilocks.txt";
pub const KECCAK_BABYBEAR_WITNESS: &str = "data/witness_babybear.txt";

pub const KECCAK_M31_MPI2_WITNESS: &str = "data/witness_m31_mpi_2.txt";
pub const KECCAK_GF2_MPI2_WITNESS: &str = "data/witness_gf2_mpi_2.txt";
pub const KECCAK_BN254_MPI2_WITNESS: &str = "data/witness_bn254_mpi_2.txt";
pub const KECCAK_GOLDILOCKS_MPI2_WITNESS: &str = "data/witness_goldilocks_mpi_2.txt";
pub const KECCAK_BABYBEAR_MPI2_WITNESS: &str = "data/witness_babybear_mpi_2.txt";

pub const KECCAK_M31_PROOF: &str = "data/proof_m31.txt";
pub const KECCAK_GF2_PROOF: &str = "data/proof_gf2.txt";
pub const KECCAK_BN254_PROOF: &str = "data/proof_bn254.txt";

// URL for Keccak circuit repeated for 2 times
pub const KECCAK_CIRCUIT_M31_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/circuit_m31.txt";
pub const KECCAK_CIRCUIT_GF2_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/circuit_gf2.txt";
pub const KECCAK_CIRCUIT_BN254_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/circuit_bn254.txt";
pub const KECCAK_CIRCUIT_GOLDILOCKS_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/circuit_goldilocks.txt";
pub const KECCAK_CIRCUIT_BABYBEAR_URL: &str =
        "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/circuit_babybear.txt";

pub const KECCAK_WITNESS_M31_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_m31.txt";
pub const KECCAK_WITNESS_GF2_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_gf2.txt";
pub const KECCAK_WITNESS_BN254_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_bn254.txt";
pub const KECCAK_WITNESS_GOLDILOCKS_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_goldilocks.txt";
pub const KECCAK_WITNESS_BABYBEAR_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_babybear.txt";

pub const KECCAK_WITNESS_M31_MPI2_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_m31_mpi_2.txt";
pub const KECCAK_WITNESS_GF2_MPI2_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_gf2_mpi_2.txt";
pub const KECCAK_WITNESS_BN254_MPI2_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_bn254_mpi_2.txt";
pub const KECCAK_WITNESS_GOLDILOCKS_MPI2_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_goldilocks_mpi_2.txt";
pub const KECCAK_WITNESS_BABYBEAR_MPI2_URL: &str =
        "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/witness_babybear_mpi_2.txt";

// circuit for repeating Poseidon for 120 times
pub const POSEIDON_M31_CIRCUIT: &str = "data/poseidon_120_circuit_m31.txt";
// circuit for repeating Poseidon for 120 times
pub const POSEIDON_M31_WITNESS: &str = "data/poseidon_120_witness_m31.txt";
// // circuit for repeating Poseidon for 120 times
// pub const POSEIDON_BN254_CIRCUIT: &str = "data/poseidon_120_circuit_bn254.txt";
// // circuit for repeating Poseidon for 120 times
// pub const POSEIDON_BN254_WITNESS: &str = "data/poseidon_120_witness_bn254.txt";

// URL for Poseidon circuit repeated for 120 times
pub const POSEIDON_CIRCUIT_M31_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/poseidon-ci/poseidon_120_circuit_m31.txt";
// URL for Poseidon circuit repeated for 120 times
pub const POSEIDON_WITNESS_M31_URL: &str =
        "https://storage.googleapis.com/expander-compiled-circuits/poseidon-ci/poseidon_120_witness_m31.txt";
// // URL for Poseidon circuit repeated for 120 times
// pub const POSEIDON_CIRCUIT_BN254_URL: &str =
// "https://storage.googleapis.com/expander-compiled-circuits/poseidon-ci/poseidon_120_circuit_bn254.txt";
// // URL for Poseidon circuit repeated for 120 times
// pub const POSEIDON_WITNESS_BN254_URL: &str =
//         "https://storage.googleapis.com/expander-compiled-circuits/poseidon-ci/poseidon_120_witness_bn254.txt";

// URL for proofs
pub const KECCAK_M31_PROOF_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/proof_m31.txt";
pub const KECCAK_GF2_PROOF_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/proof_gf2.txt";
pub const KECCAK_BN254_PROOF_URL: &str =
    "https://storage.googleapis.com/expander-compiled-circuits/keccak-ci/serialization-v6/proof_bn254.txt";

// NOTE(Hang 08/23/24):
// CI process is unhappy about reqwest as a dependency,
// so we use wget as a backup option.
fn download_and_store(url: &str, file: &str) {
    let download = Command::new("bash")
        .arg("-c")
        .arg(format!("wget {url} -O {file}"))
        .output()
        .expect("Failed to download circuit");

    assert!(download.status.success(), "Circuit download failure")
}

pub fn dev_env_data_setup() {
    fs::create_dir_all(DATA_PREFIX).unwrap();

    // keccak circuit
    download_and_store(KECCAK_CIRCUIT_M31_URL, KECCAK_M31_CIRCUIT);
    download_and_store(KECCAK_CIRCUIT_GF2_URL, KECCAK_GF2_CIRCUIT);
    download_and_store(KECCAK_CIRCUIT_BN254_URL, KECCAK_BN254_CIRCUIT);
    download_and_store(KECCAK_CIRCUIT_GOLDILOCKS_URL, KECCAK_GOLDILOCKS_CIRCUIT);
    download_and_store(KECCAK_CIRCUIT_BABYBEAR_URL, KECCAK_BABYBEAR_CIRCUIT);

    download_and_store(KECCAK_WITNESS_M31_URL, KECCAK_M31_WITNESS);
    download_and_store(KECCAK_WITNESS_GF2_URL, KECCAK_GF2_WITNESS);
    download_and_store(KECCAK_WITNESS_BN254_URL, KECCAK_BN254_WITNESS);
    download_and_store(KECCAK_WITNESS_GOLDILOCKS_URL, KECCAK_GOLDILOCKS_WITNESS);
    download_and_store(KECCAK_WITNESS_BABYBEAR_URL, KECCAK_BABYBEAR_WITNESS);

    download_and_store(KECCAK_M31_PROOF_URL, KECCAK_M31_PROOF);
    download_and_store(KECCAK_GF2_PROOF_URL, KECCAK_GF2_PROOF);
    download_and_store(KECCAK_BN254_PROOF_URL, KECCAK_BN254_PROOF);

    // keccak circuit with MPI
    download_and_store(KECCAK_WITNESS_M31_MPI2_URL, KECCAK_M31_MPI2_WITNESS);
    download_and_store(KECCAK_WITNESS_GF2_MPI2_URL, KECCAK_GF2_MPI2_WITNESS);
    download_and_store(KECCAK_WITNESS_BN254_MPI2_URL, KECCAK_BN254_MPI2_WITNESS);

    // poseidon circuit
    download_and_store(POSEIDON_CIRCUIT_M31_URL, POSEIDON_M31_CIRCUIT);
    download_and_store(POSEIDON_WITNESS_M31_URL, POSEIDON_M31_WITNESS);
}
