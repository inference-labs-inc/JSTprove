use arith::Field;

#[derive(Debug, Clone, PartialEq)]
pub struct EqPolynomial<F> {
    _phantom: std::marker::PhantomData<F>,
}

// public functions
impl<F: Field> EqPolynomial<F> {
    #[inline]
    pub fn eq_eval_at(
        r: &[F],
        mul_factor: &F,
        eq_evals: &mut [F],
        sqrt_n_1st: &mut [F],
        sqrt_n_2nd: &mut [F],
    ) {
        let first_half_bits = r.len() / 2;
        let first_half_mask = (1 << first_half_bits) - 1;
        Self::build_eq_x_r_with_buf(&r[0..first_half_bits], mul_factor, sqrt_n_1st);
        Self::build_eq_x_r_with_buf(&r[first_half_bits..], &F::one(), sqrt_n_2nd);

        for (i, eq_eval) in eq_evals.iter_mut().enumerate().take(1 << r.len()) {
            let first_half = i & first_half_mask;
            let second_half = i >> first_half_bits;
            *eq_eval = sqrt_n_1st[first_half] * sqrt_n_2nd[second_half];
        }
    }

    #[inline]
    /// Expander's method of computing Eq(x, r)
    pub fn build_eq_x_r_with_buf(r: &[F], mul_factor: &F, eq_evals: &mut [F]) {
        eq_evals[0] = *mul_factor;
        let mut cur_eval_num = 1;

        for r_i in r.iter() {
            for j in 0..cur_eval_num {
                eq_evals[j + cur_eval_num] = eq_evals[j] * r_i;
                eq_evals[j] -= eq_evals[j + cur_eval_num];
            }
            cur_eval_num <<= 1;
        }
    }

    #[inline(always)]
    /// Evaluate eq polynomial.
    pub fn eq_vec(xs: &[F], ys: &[F]) -> F {
        xs.iter()
            .zip(ys.iter())
            .map(|(x, y)| Self::eq(x, y))
            .product()
    }

    /// Hyperplonk's method of computing Eq(x, r)
    ///
    /// This function build the eq(x, r) polynomial for any given r, and output the
    /// evaluation of eq(x, r) in its vector form.
    ///
    /// Evaluate
    ///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
    /// over r, which is
    ///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
    #[inline]
    pub fn build_eq_x_r(r: &[F]) -> Vec<F> {
        // we build eq(x,r) from its evaluations
        // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
        // for example, with num_vars = 4, x is a binary vector of 4, then
        //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
        //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
        //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
        //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
        //  ....
        //  1 1 1 1 -> r0       * r1        * r2        * r3
        // we will need 2^num_var evaluations

        let mut eval = Vec::new();
        Self::build_eq_x_r_helper(r, &mut eval);

        eval
    }

    /// Jolt's method of compute Eq(x,r)
    ///
    /// Computes evals serially. Uses less memory (and fewer allocations) than `evals_parallel`.
    #[inline]
    pub fn evals_jolt(r: &[F]) -> Vec<F> {
        let mut evals: Vec<F> = vec![F::one(); 1 << r.len()];
        let mut size = 1;
        for j in 0..r.len() {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * r[r.len() - j - 1];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Jolt's method of compute Eq(x,r)
    ///
    /// Computes evals serially. Uses less memory (and fewer allocations) than `evals_parallel`.
    #[inline]
    pub fn scaled_evals_jolt(r: &[F], mul_factor: &F) -> Vec<F> {
        let mut evals: Vec<F> = vec![*mul_factor; 1 << r.len()];
        let mut size = 1;
        for j in 0..r.len() {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * r[r.len() - j - 1];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Given an r for eq(x, r), while x \in {0, 1}^\ell represented by index,
    /// use O(\ell) time to evalutate eq(x, r).
    #[inline]
    pub fn ith_eq_vec_elem(r: &[F], mut index: usize) -> F {
        assert!(index < (1 << r.len()));

        r.iter()
            .map(|r_i| {
                let mut term = *r_i;
                if index & 1 == 0 {
                    term = F::ONE - term;
                }

                index >>= 1;
                term
            })
            .product()
    }
}

// Private functions
impl<F: Field> EqPolynomial<F> {
    #[inline(always)]
    fn eq(x: &F, y: &F) -> F {
        // x * y + (1 - x) * (1 - y)
        let xy = *x * y;
        xy + xy - x - y + F::one()
    }

    // #[inline(always)]
    // fn eq_3(x: &F, y: &F, z: &F) -> F {
    //     // TODO: extend this
    //     *x * y * z + (F::one() - x) * (F::one() - y) * (F::one() - z)
    // }

    /// Hyperplonk's method of computing Eq(x, r)
    ///
    /// A helper function to build eq(x, r) recursively.
    /// This function takes `r.len()` steps, and for each step it requires a maximum
    /// `r.len()-1` multiplications.
    #[inline]
    fn build_eq_x_r_helper(r: &[F], buf: &mut Vec<F>) {
        if r.is_empty() {
            buf.push(F::one());
        } else if r.len() == 1 {
            // initializing the buffer with [1-r_0, r_0]
            buf.push(F::one() - r[0]);
            buf.push(r[0]);
        } else {
            Self::build_eq_x_r_helper(&r[1..], buf);

            // suppose at the previous step we received [b_1, ..., b_k]
            // for the current step we will need
            // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
            // if x_0 = 1:   r0 * [b_1, ..., b_k]
            // let mut res = vec![];
            // for &b_i in buf.iter() {
            //     let tmp = r[0] * b_i;
            //     res.push(b_i - tmp);
            //     res.push(tmp);
            // }
            // *buf = res;

            let mut res = vec![F::zero(); buf.len() << 1];
            res.iter_mut().enumerate().for_each(|(i, val)| {
                let bi = buf[i >> 1];
                let tmp = r[0] * bi;
                if i & 1 == 0 {
                    *val = bi - tmp;
                } else {
                    *val = tmp;
                }
            });
            *buf = res;
        }
    }
}
