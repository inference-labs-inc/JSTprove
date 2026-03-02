use arith::Field;
use polynomials::{MultilinearExtension, SumOfProductsPoly};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::{IOPProverMessage, IOPProverState};

impl<F: Field> IOPProverState<F> {
    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    pub fn prover_init(polynomials: &SumOfProductsPoly<F>) -> Self {
        let num_vars = polynomials.num_vars();
        Self {
            challenges: Vec::with_capacity(num_vars),
            round: 0,
            init_num_vars: num_vars,
            mle_list: polynomials.clone(),
            init_sum_of_vals: polynomials
                .f_and_g_pairs
                .par_iter()
                .map(|(f, g)| {
                    f.coeffs
                        .iter()
                        .zip(g.coeffs.iter())
                        .map(|(&f, &g)| f * g)
                        .sum::<F>()
                })
                .collect(),
            eq_prefix: vec![F::one(); polynomials.f_and_g_pairs.len()],
        }
    }

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    pub fn prove_round_and_update_state(&mut self, challenge: &Option<F>) -> IOPProverMessage<F> {
        if self.round >= self.init_num_vars {
            panic!("prover is not active")
        }

        // Step 1:
        // fix argument and evaluate f(x) over x_m = r; where r is the challenge
        // for the current round, and m is the round number, indexed from 1
        //
        // i.e.:
        // at round m <= n, for each mle g(x_1, ... x_n) within the flattened_mle
        // which has already been evaluated to
        //
        //    g(r_1, ..., r_{m-1}, x_m ... x_n)
        //
        // eval g over r_m, and mutate g to g(r_1, ... r_m,, x_{m+1}... x_n)

        if let Some(chal) = challenge {
            if self.round == 0 {
                panic!("first round should not have a challenge");
            }
            self.challenges.push(*chal);

            let r = self.challenges[self.round - 1];

            self.fix_top_variable_for_poly_pairs(&r);
        } else if self.round > 0 {
            panic!("verifier message is empty")
        }

        self.round += 1;

        let mut h_0 = F::zero();
        let mut h_1 = F::zero();
        let mut h_2 = F::zero();

        // The following commented code is a sequential version of the computation
        //
        // for (f, g) in self.mle_list.f_and_g_pairs.iter() {
        //     // evaluate the polynomial at 0, 1 and 2
        //     // and obtain f(0)g(0) and f(1)g(1) and f(2)g(2)
        //     let f_coeffs = f.coeffs.as_slice();
        //     let g_coeffs = g.coeffs.as_slice();

        //     h_0 += f_coeffs[..len].iter().sum::<F>() * g_coeffs[..len].iter().sum::<F>();
        //     h_1 += f_coeffs[len..].iter().sum::<F>() * g_coeffs[len..].iter().sum::<F>();

        //     let f_2 = f_coeffs[..len]
        //         .iter()
        //         .zip(f_coeffs[len..].iter())
        //         .map(|(a, b)| -*a + b.double())
        //         .sum::<F>();
        //     let g2 = g_coeffs[..len]
        //         .iter()
        //         .zip(g_coeffs[len..].iter())
        //         .map(|(a, b)| -*a + b.double())
        //         .sum::<F>();
        //     h_2 += f_2 * g2;
        // }

        self.mle_list
            .f_and_g_pairs
            .par_iter()
            .enumerate()
            .map(|(i, (f, g))| {
                // evaluate the polynomial at 0, 1 and 2
                // and obtain f(0)g(0) and f(1)g(1) and f(2)g(2)

                if let Some(sub_idx) =
                    Self::get_sub_idx(self.init_num_vars, self.round, f.num_vars())
                {
                    let len = 1 << (f.num_vars() - sub_idx - 1);
                    let f_coeffs = f.coeffs.as_slice();
                    let g_coeffs = g.coeffs.as_slice();

                    let h_0_local = f_coeffs[..len]
                        .iter()
                        .zip(g_coeffs[..len].iter())
                        .map(|(&f, &g)| f * g)
                        .sum::<F>();

                    let h_1_local = f_coeffs[len..]
                        .iter()
                        .zip(g_coeffs[len..].iter())
                        .map(|(&f, &g)| f * g)
                        .sum::<F>();

                    let h_2_local = f_coeffs[..len]
                        .iter()
                        .zip(f_coeffs[len..].iter())
                        .map(|(a, b)| -*a + b.double())
                        .zip(
                            g_coeffs[..len]
                                .iter()
                                .zip(g_coeffs[len..].iter())
                                .map(|(a, b)| -*a + b.double()),
                        )
                        .map(|(a, b)| a * b)
                        .sum::<F>();

                    let eq_prefix_i = self.eq_prefix[i].square();
                    (
                        h_0_local * eq_prefix_i,
                        h_1_local * eq_prefix_i,
                        h_2_local * eq_prefix_i,
                    )
                } else {
                    let h = self.eq_prefix[i].square() * self.init_sum_of_vals[i];
                    (h, F::zero(), h)
                }
            })
            .collect::<Vec<_>>()
            .iter()
            .for_each(|(h_0_local, h_1_local, h_2_local)| {
                h_0 += h_0_local;
                h_1 += h_1_local;
                h_2 += h_2_local;
            });

        IOPProverMessage {
            evaluations: vec![h_0, h_1, h_2],
        }
    }

    fn get_sub_idx(init_num_vars: usize, round: usize, local_num_vars: usize) -> Option<usize> {
        if round < init_num_vars - local_num_vars + 1 {
            None
        } else {
            Some(round - (init_num_vars - local_num_vars + 1))
        }
    }

    fn fix_top_variable_for_poly_pairs(&mut self, challenge: &F) {
        self.mle_list
            .f_and_g_pairs
            .par_iter_mut()
            .zip(self.eq_prefix.par_iter_mut())
            .for_each(|((f, g), eq_prefix)| {
                if let Some(_sub_idx) =
                    Self::get_sub_idx(self.init_num_vars, self.round, f.num_vars())
                {
                    // fix the top variable for each polynomial pair
                    f.fix_top_variable(*challenge);
                    g.fix_top_variable(*challenge);
                } else {
                    *eq_prefix *= F::one() - *challenge; // eq(challenge, 0)
                }
            });
    }
}
