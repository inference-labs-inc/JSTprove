use gkr_hashers::poseidon::r#impls::{
    MATRIX_CIRC_MDS_12_SML_ROW, MATRIX_CIRC_MDS_16_SML_ROW, MATRIX_CIRC_MDS_8_SML_ROW,
    POSEIDON_SEED_PREFIX,
};
use tiny_keccak::{Hasher, Keccak};

const FIELD_NAME: &str = "Mersenne 31";

pub fn get_constants(width: usize, round_num: usize) -> Vec<Vec<u32>> {
    let seed = format!("{POSEIDON_SEED_PREFIX}_{FIELD_NAME}_{width}");

    let mut keccak = Keccak::v256();
    let mut buffer = [0u8; 32];
    keccak.update(seed.as_bytes());
    keccak.finalize(&mut buffer);

    let mut res = vec![vec![0u32; width]; round_num];

    (0..round_num).for_each(|i| {
        (0..width).for_each(|j| {
            let mut keccak = Keccak::v256();
            keccak.update(&buffer);
            keccak.finalize(&mut buffer);

            let mut u32_le_bytes = [0u8; 4];
            u32_le_bytes.copy_from_slice(&buffer[..4]);

            res[i][j] = u32::from_le_bytes(u32_le_bytes);
        });
    });

    res
}

pub const POSEIDON_M31X16_FULL_ROUNDS: usize = 8;

pub const POSEIDON_M31X16_PARTIAL_ROUNDS: usize = 14;

pub const POSEIDON_M31X16_RATE: usize = 8;

pub fn get_mds_matrix(width: usize) -> Vec<Vec<u32>> {
    let mds_first_row: &[u32] = match width {
        8 => &MATRIX_CIRC_MDS_8_SML_ROW,
        12 => &MATRIX_CIRC_MDS_12_SML_ROW,
        16 => &MATRIX_CIRC_MDS_16_SML_ROW,
        _ => panic!("unsupported state width for MDS matrix"),
    };

    let mut res = vec![vec![0u32; width]; width];

    (0..width).for_each(|i| (0..width).for_each(|j| res[i][j] = mds_first_row[(i + j) % width]));

    res
}
