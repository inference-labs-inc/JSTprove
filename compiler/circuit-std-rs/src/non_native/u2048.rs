use expander_compiler::frontend::{BN254Config, RootAPI, Variable};

use super::u120;

// we use 18 limbs, each with 120 bits, to store a 2048 bit integer
pub const N_LIMBS: usize = 18;

// Each 120 bits limb needs 30 hex number to store
pub const HEX_PER_LIMB: usize = 30;

#[derive(Debug, Clone, Copy)]
pub struct U2048Variable {
    pub limbs: [Variable; N_LIMBS],
}

impl U2048Variable {
    #[inline]
    pub fn from_raw(limbs: [Variable; N_LIMBS]) -> Self {
        Self { limbs }
    }

    #[inline]
    // generate a bool variable for the comparison of two U2048 variables
    pub fn unconstrained_greater_eq<Builder: RootAPI<BN254Config>>(
        &self,
        other: &Self,
        builder: &mut Builder,
    ) -> Variable {
        // Start from most significant limb (N_LIMBS-1) and work down
        let mut gt_flags = Vec::with_capacity(N_LIMBS);
        let mut eq_flags = Vec::with_capacity(N_LIMBS);

        // Compute comparison flags for all limbs
        for i in (0..N_LIMBS).rev() {
            gt_flags.push(builder.unconstrained_greater(self.limbs[i], other.limbs[i]));
            eq_flags.push(builder.unconstrained_eq(self.limbs[i], other.limbs[i]));
        }

        // Start with the most significant limb comparison
        let mut result = gt_flags[0]; // corresponds to limb N_LIMBS-1
        let mut all_eq_so_far = eq_flags[0];

        // Process remaining limbs from most to least significant
        for i in 1..N_LIMBS {
            // If all previous limbs were equal and current limb is greater
            let curr_greater = builder.unconstrained_bit_and(all_eq_so_far, gt_flags[i]);
            result = builder.unconstrained_bit_or(result, curr_greater);

            // Update equality chain
            all_eq_so_far = builder.unconstrained_bit_and(all_eq_so_far, eq_flags[i]);
        }

        // Result is true if we found a greater limb or if all limbs were equal
        builder.unconstrained_bit_or(result, all_eq_so_far)
    }

    #[inline]
    pub fn assert_is_less_than<Builder: RootAPI<BN254Config>>(
        &self,
        other: &Self,
        builder: &mut Builder,
    ) -> Variable {
        let mut result = builder.constant(0);
        let mut all_eq_so_far = builder.constant(1);

        // Compare limbs from most significant to least significant
        for i in (0..N_LIMBS).rev() {
            // Compare current limbs using u120 comparison
            let curr_less = u120::is_less_than_u120(&self.limbs[i], &other.limbs[i], builder);

            // Check equality for current limbs
            let diff = builder.sub(self.limbs[i], other.limbs[i]);
            let curr_eq = builder.is_zero(diff);

            // If all previous limbs were equal and current limb is less
            let update = builder.and(all_eq_so_far, curr_less);

            // Update result: result = result OR (all_eq_so_far AND curr_less)
            result = builder.or(result, update);

            // Update equality chain: all_eq_so_far = all_eq_so_far AND curr_eq
            all_eq_so_far = builder.and(all_eq_so_far, curr_eq);

            // Assert boolean constraints
            builder.assert_is_bool(result);
            builder.assert_is_bool(all_eq_so_far);
            builder.assert_is_bool(curr_less);
            builder.assert_is_bool(curr_eq);

            // Cannot be both less and equal for current limb
            let both = builder.mul(curr_less, curr_eq);
            builder.assert_is_zero(both);
        }

        // If all limbs were equal, result must be 0
        let equal_case = builder.mul(all_eq_so_far, result);
        builder.assert_is_zero(equal_case);

        result
    }

    // Helper function to check if one U2048 is greater than or equal to another
    #[inline]
    pub fn assert_is_greater_eq<Builder: RootAPI<BN254Config>>(
        &self,
        other: &Self,
        builder: &mut Builder,
    ) -> Variable {
        let less = other.assert_is_less_than(self, builder);
        let eq = self.assert_is_equal(other, builder);

        // result = less OR eq
        let mut result = builder.add(less, eq);
        let tmp = builder.mul(less, eq);
        result = builder.sub(result, tmp);
        builder.assert_is_bool(result);

        result
    }

    // Helper function to check equality
    #[inline]
    pub fn assert_is_equal<Builder: RootAPI<BN254Config>>(
        &self,
        other: &Self,
        builder: &mut Builder,
    ) -> Variable {
        let mut is_equal = builder.constant(1);

        for i in 0..N_LIMBS {
            let diff = builder.sub(self.limbs[i], other.limbs[i]);
            let curr_eq = builder.is_zero(diff);
            is_equal = builder.mul(is_equal, curr_eq);
            builder.assert_is_bool(curr_eq);
        }

        builder.assert_is_bool(is_equal);
        is_equal
    }

    #[inline]
    // add two U2048 variables with mod reductions
    // a + b = result + carry * modulus
    pub fn assert_add<Builder: RootAPI<BN254Config>>(
        x: &U2048Variable,
        y: &U2048Variable,
        result: &U2048Variable,
        carry: &Variable,
        modulus: &U2048Variable,
        two_to_120: &Variable,
        builder: &mut Builder,
    ) {
        // First compute raw sum x + y with carries between limbs
        let mut sum = vec![];
        let mut temp_carry = builder.constant(0);
        for i in 0..N_LIMBS {
            let (r, c) = u120::add_u120(&x.limbs[i], &y.limbs[i], &temp_carry, two_to_120, builder);
            temp_carry = c;
            sum.push(r);
        }
        let sum = U2048Variable::from_raw(sum.try_into().unwrap());

        // Verify carry is boolean
        builder.assert_is_bool(*carry);

        // Now verify: sum = result + carry * modulus

        // First compute carry * modulus
        let mut carry_times_modulus = vec![];
        for i in 0..N_LIMBS {
            carry_times_modulus.push(builder.mul(*carry, modulus.limbs[i]));
        }
        let carry_times_modulus = U2048Variable::from_raw(carry_times_modulus.try_into().unwrap());

        // For each limb, verify: sum[i] = result[i] + (carry * modulus)[i]
        let mut temp_carry = builder.constant(0);
        for i in 0..N_LIMBS {
            let (expected, c) = u120::add_u120(
                &result.limbs[i],
                &carry_times_modulus.limbs[i],
                &temp_carry,
                two_to_120,
                builder,
            );
            temp_carry = c;

            // Assert equality for this limb
            builder.assert_is_equal(sum.limbs[i], expected);
        }

        // Final carry should be 0 since all numbers are within range
        builder.assert_is_zero(temp_carry);

        let lt = Self::assert_is_less_than(result, modulus, builder);
        let one = builder.constant(1);
        builder.assert_is_equal(lt, one);
    }

    #[inline]
    // assert multiplication of two U2048 variables
    // x * y = result + carry * modulus
    pub fn assert_mul<Builder: RootAPI<BN254Config>>(
        x: &U2048Variable,
        y: &U2048Variable,
        result: &U2048Variable,
        carry: &U2048Variable,
        modulus: &U2048Variable,
        two_to_120: &Variable,
        builder: &mut Builder,
    ) {
        let zero = builder.constant(0);
        // x * y
        let left = U2048Variable::mul_without_mod_reduction(x, y, two_to_120, builder);
        // carry * modulus
        let mut right =
            U2048Variable::mul_without_mod_reduction(carry, modulus, two_to_120, builder);
        // result + carry
        let mut right_carry = builder.constant(0);
        for i in 0..N_LIMBS {
            (right[i], right_carry) = u120::add_u120(
                &result.limbs[i],
                &right[i],
                &right_carry,
                two_to_120,
                builder,
            );
        }
        for i in 0..N_LIMBS {
            (right[N_LIMBS + i], right_carry) = u120::add_u120(
                &zero,
                &right[N_LIMBS + i],
                &right_carry,
                two_to_120,
                builder,
            );
        }

        for i in 0..4 {
            builder.assert_is_equal(left[i], right[i]);
        }
    }

    #[inline]
    pub fn mul_without_mod_reduction<Builder: RootAPI<BN254Config>>(
        x: &U2048Variable,
        y: &U2048Variable,
        two_to_120: &Variable,
        builder: &mut Builder,
    ) -> Vec<Variable> {
        let zero = builder.constant(0);
        let mut local_res = vec![zero; 2 * N_LIMBS];
        let mut addition_carries = vec![zero; 2 * N_LIMBS + 1];

        for i in 0..N_LIMBS {
            for j in 0..N_LIMBS {
                let target_position = i + j;

                // prod + mul_carry * 2^120 = x[i] * y[j]
                let (xiyi_lo, xiyi_hi) =
                    u120::mul_u120(&x.limbs[i], &y.limbs[j], &zero, two_to_120, builder);

                // update xiyi_lo to result[target]
                let (sum, new_carry) = u120::add_u120(
                    &local_res[target_position],
                    &xiyi_lo,
                    &zero,
                    two_to_120,
                    builder,
                );

                local_res[target_position] = sum;
                addition_carries[target_position + 1] =
                    builder.add(addition_carries[target_position + 1], new_carry);

                // update mul_carry to result[target+1]
                let (sum, new_carry) = u120::add_u120(
                    &local_res[target_position + 1],
                    &xiyi_hi,
                    &zero,
                    two_to_120,
                    builder,
                );
                local_res[target_position + 1] = sum;
                addition_carries[target_position + 2] =
                    builder.add(addition_carries[target_position + 2], new_carry);
            }
        }
        // for i in 0..2 * N_LIMBS {
        //     println!("{i}");
        //     builder.display(local_res[i]);
        //     builder.display(addition_carries[i]);
        // }

        // integrate carries into result
        let mut cur_carry = builder.constant(0);
        for i in 0..2 * N_LIMBS {
            (local_res[i], cur_carry) = u120::add_u120(
                &local_res[i],
                &addition_carries[i],
                &cur_carry,
                two_to_120,
                builder,
            );
        }
        local_res
    }
}
