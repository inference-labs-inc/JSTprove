use std::fmt::Debug;

use arith::Field;
use serdes::ExpSerde;

use super::MPIEngine;

#[macro_export]
macro_rules! root_println {
    ($config: expr, $($arg:tt)*) => {
        if $config.is_root() {
            println!($($arg)*);
        }
    };
}

#[derive(Clone, PartialEq)]
pub struct MPIConfig {
    pub world_size: i32,
    pub world_rank: i32,
}

impl Default for MPIConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            world_rank: 0,
        }
    }
}

impl Debug for MPIConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MPIConfig")
            .field("world_size", &self.world_size)
            .field("world_rank", &self.world_rank)
            .finish()
    }
}

impl MPIConfig {
    pub fn prover_new() -> Self {
        Self {
            world_size: 1,
            world_rank: 0,
        }
    }

    #[inline]
    pub fn verifier_new(world_size: i32) -> Self {
        Self {
            world_size,
            world_rank: 0,
        }
    }
}

impl MPIEngine for MPIConfig {
    const ROOT_RANK: i32 = 0;

    #[inline(always)]
    fn gather_vec<F: Sized + Clone>(&self, local_vec: &[F], global_vec: &mut Vec<F>) {
        *global_vec = local_vec.to_vec();
    }

    #[inline]
    fn scatter_vec<F: Sized + Clone>(&self, send_vec: &[F], recv_vec: &mut [F]) {
        recv_vec.clone_from_slice(send_vec);
    }

    #[inline]
    fn root_broadcast_f<F: Copy>(&self, _f: &mut F) {}

    #[inline]
    fn root_broadcast_bytes(&self, _bytes: &mut Vec<u8>) {}

    #[inline]
    fn sum_vec<F: Field>(&self, local_vec: &[F]) -> Vec<F> {
        local_vec.to_vec()
    }

    #[inline]
    fn coef_combine_vec<F: Field>(&self, local_vec: &[F], _coef: &[F]) -> Vec<F> {
        local_vec.to_vec()
    }

    #[inline(always)]
    fn all_to_all_transpose<F: Sized>(&self, _row: &mut [F]) {}

    #[inline(always)]
    fn gather_varlen_vec<F: ExpSerde>(&self, elems: &Vec<F>, global_elems: &mut Vec<Vec<F>>) {
        let mut bytes: Vec<u8> = Vec::new();
        elems.serialize_into(&mut bytes).unwrap();
        *global_elems = vec![Vec::deserialize_from(bytes.as_slice()).unwrap()];
    }

    #[inline(always)]
    fn is_single_process(&self) -> bool {
        self.world_size == 1
    }

    #[inline(always)]
    fn world_size(&self) -> usize {
        self.world_size as usize
    }

    #[inline(always)]
    fn world_rank(&self) -> usize {
        self.world_rank as usize
    }

    #[inline(always)]
    fn barrier(&self) {}
}
