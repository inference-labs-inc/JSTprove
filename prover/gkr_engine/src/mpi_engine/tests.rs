use arith::Field;
use ark_std::test_rng;
use gf2::{GF2x128, GF2x64, GF2x8};
use itertools::izip;
use mersenne31::{M31Ext3, M31x16, M31};

use crate::{MPIConfig, MPIEngine};

fn test_gather_vec_helper(mpi_config: &MPIConfig) {
    const TEST_SIZE: usize = (1 << 10) + 1;

    let mut local_vec = vec![M31::ZERO; TEST_SIZE];
    for i in 0..TEST_SIZE {
        local_vec[i] = M31::from((mpi_config.world_rank() * TEST_SIZE + i) as u32);
    }

    let mut global_vec = if mpi_config.is_root() {
        vec![M31::ZERO; TEST_SIZE * mpi_config.world_size()]
    } else {
        vec![]
    };

    mpi_config.gather_vec(&local_vec, &mut global_vec);
    if mpi_config.is_root() {
        for (i, v) in global_vec.iter().enumerate() {
            assert_eq!(M31::from(i as u32), *v);
        }
    }
}

fn test_varlen_gather_vec_helper(mpi_config: &MPIConfig) {
    let msg: Vec<_> = (0..=mpi_config.world_rank()).collect();
    let mut global_elems: Vec<Vec<usize>> = Vec::new();

    mpi_config.gather_varlen_vec(&msg, &mut global_elems);

    dbg!(&global_elems);

    global_elems.iter().enumerate().for_each(|(i, elems)| {
        (0..=i).for_each(|j| {
            assert_eq!(j, elems[j]);
        })
    });
}

fn test_all_to_all_transpose_helper<F: Field>(mpi_config: &MPIConfig) {
    const TEST_MATRIX_LEN: usize = 1 << 23;

    dbg!(F::NAME);
    dbg!(F::SIZE);

    let mut rng = test_rng();
    let global_matrix: Vec<_> = (0..TEST_MATRIX_LEN)
        .map(|_| F::random_unsafe(&mut rng))
        .collect();

    let local_length = TEST_MATRIX_LEN / mpi_config.world_size();

    let local_share_starts = local_length * mpi_config.world_rank();
    let mut local_shares =
        global_matrix[local_share_starts..local_length + local_share_starts].to_vec();

    dbg!(local_share_starts, local_length);

    mpi_config.all_to_all_transpose(&mut local_shares);

    let transpose_slice_len = local_length / mpi_config.world_size();
    izip!(
        global_matrix.chunks(local_length),
        local_shares.chunks(transpose_slice_len)
    )
    .enumerate()
    .for_each(|(i, (c, ls))| {
        let row_starts = transpose_slice_len * mpi_config.world_rank();

        dbg!(i, row_starts);

        izip!(&c[row_starts..transpose_slice_len + row_starts], ls)
            .for_each(|(left, right)| assert_eq!(*left, *right));
    });
}

fn test_scatter_vec_helper(mpi_config: &MPIConfig) {
    const TEST_SIZE: usize = 1024 + 1;

    let send_vec: Vec<_> = if mpi_config.is_root() {
        let mut buf = vec![0u8; TEST_SIZE * mpi_config.world_size()];
        buf.chunks_mut(TEST_SIZE)
            .enumerate()
            .for_each(|(i, chunk)| {
                let fill_var = (i % mpi_config.world_size()) as u8;
                chunk.fill(fill_var);
            });

        buf
    } else {
        Vec::new()
    };

    let mut local_vec = vec![0u8; TEST_SIZE];

    mpi_config.scatter_vec(&send_vec, &mut local_vec);

    let expected = local_vec
        .iter()
        .all(|v| *v == mpi_config.world_rank() as u8);

    assert!(expected);
}

#[test]
fn test_mpi_engine() {
    let mpi_config = MPIConfig::prover_new();

    test_gather_vec_helper(&mpi_config);

    test_all_to_all_transpose_helper::<GF2x128>(&mpi_config);
    test_all_to_all_transpose_helper::<GF2x64>(&mpi_config);
    test_all_to_all_transpose_helper::<GF2x8>(&mpi_config);

    test_all_to_all_transpose_helper::<M31x16>(&mpi_config);
    test_all_to_all_transpose_helper::<M31>(&mpi_config);
    test_all_to_all_transpose_helper::<M31Ext3>(&mpi_config);

    test_varlen_gather_vec_helper(&mpi_config);

    test_scatter_vec_helper(&mpi_config);
}
