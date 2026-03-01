use arith::Field;
use serdes::ExpSerde;

pub trait MPIEngine: Clone + Default + Send + Sync {
    const ROOT_RANK: i32 = 0;

    fn gather_vec<F: Sized + Clone>(&self, local_vec: &[F], global_vec: &mut Vec<F>);

    fn scatter_vec<F: Sized + Clone>(&self, send_vec: &[F], receive_vec: &mut [F]);

    fn root_broadcast_f<F: Copy>(&self, f: &mut F);

    fn root_broadcast_bytes(&self, bytes: &mut Vec<u8>);

    fn sum_vec<F: Field>(&self, local_vec: &[F]) -> Vec<F>;

    fn coef_combine_vec<F: Field>(&self, local_vec: &[F], coef: &[F]) -> Vec<F>;

    fn all_to_all_transpose<F: Sized>(&self, row: &mut [F]);

    #[allow(clippy::ptr_arg)]
    fn gather_varlen_vec<F: ExpSerde>(&self, local_vec: &Vec<F>, global_vec: &mut Vec<Vec<F>>);

    fn is_single_process(&self) -> bool;

    fn world_size(&self) -> usize;

    fn world_rank(&self) -> usize;

    #[inline(always)]
    fn is_root(&self) -> bool {
        self.world_rank() == Self::ROOT_RANK as usize
    }

    fn barrier(&self);
}
