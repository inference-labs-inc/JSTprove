// Copyright 2025 Irreducible Inc.
use anyhow::{Result, anyhow};
use binius_core::{
    constraint_system::{ConstraintSystem, ValueVec},
    verify::verify_constraints,
    word::Word,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct RandBlast {
    rng: StdRng,
}

impl RandBlast {
    pub fn new(seed: u64) -> Self {
        RandBlast {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate random values for all witness wires
    fn generate_random_witness(&mut self, cs: &ConstraintSystem) -> ValueVec {
        let mut witness = cs.new_value_vec();

        // Set constants
        for (i, &constant) in cs.constants.iter().enumerate() {
            witness.set(i, constant);
        }

        // Set random values for witness wires
        let start = cs.constants.len() + cs.value_vec_layout.n_inout;
        let end = start + cs.value_vec_layout.n_witness;
        for i in start..end {
            let random_val = self.rng.random::<u64>();
            witness.set(i, Word(random_val));
        }

        witness
    }

    /// Test if two constraint systems are equivalent by running random inputs
    pub fn test_equivalence(
        &mut self,
        lhs: &ConstraintSystem,
        rhs: &ConstraintSystem,
        num_tests: usize,
    ) -> Result<()> {
        // Check that both systems have the same witness size
        if lhs.value_vec_layout.n_witness != rhs.value_vec_layout.n_witness {
            return Err(anyhow!(
                "Constraint systems have different witness sizes: {} vs {}",
                lhs.value_vec_layout.n_witness,
                rhs.value_vec_layout.n_witness
            ));
        }

        for test_num in 0..num_tests {
            // Generate random witness values
            let witness1 = self.generate_random_witness(lhs);

            // Copy witness values to second constraint system
            let mut witness2 = rhs.new_value_vec();

            // Copy constants from rhs
            for (i, &constant) in rhs.constants.iter().enumerate() {
                witness2.set(i, constant);
            }

            // Copy witness values (after constants)
            let start1 = lhs.constants.len() + lhs.value_vec_layout.n_inout;
            let start2 = rhs.constants.len() + rhs.value_vec_layout.n_inout;

            for i in 0..lhs.value_vec_layout.n_witness {
                witness2.set(start2 + i, witness1.get(start1 + i));
            }

            // Verify both constraint systems
            let result1 = verify_constraints(lhs, &witness1);
            let result2 = verify_constraints(rhs, &witness2);

            match (result1, result2) {
                (Ok(()), Ok(())) => {
                    // Both satisfied - good
                }
                (Err(_), Err(_)) => {
                    // Both failed - also acceptable for equivalence
                }
                (Ok(()), Err(e)) => {
                    return Err(anyhow!(
                        "Test {}: LHS satisfied but RHS failed with: {}",
                        test_num,
                        e
                    ));
                }
                (Err(e), Ok(())) => {
                    return Err(anyhow!(
                        "Test {}: RHS satisfied but LHS failed with: {}",
                        test_num,
                        e
                    ));
                }
            }

            if (test_num + 1).is_multiple_of(1000) {
                println!("  Completed {} tests", test_num + 1);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use binius_core::constraint_system::{AndConstraint, ValueIndex, ValueVecLayout};

    use super::*;

    #[test]
    fn test_equivalent_systems() {
        // Create two equivalent constraint systems: v2 = v0 & v1
        let n_const = 2;
        let n_witness = 3;
        let value_vec_layout = ValueVecLayout {
            n_const,
            n_inout: 0,
            n_witness,
            n_internal: 0,
            offset_inout: n_const,
            offset_witness: 4,      // next power of 2 after constants
            committed_total_len: 8, // next power of 2 after all values
            n_scratch: 0,
        };
        // Both implement: v0 & v1 ^ v2 = 0
        let constraint = AndConstraint::plain_abc(
            vec![ValueIndex(2)], // v0
            vec![ValueIndex(3)], // v1
            vec![ValueIndex(4)], // v2
        );

        let lhs = ConstraintSystem::new(
            vec![Word(0), Word(u64::MAX)],
            value_vec_layout.clone(),
            vec![constraint.clone()],
            vec![],
        );
        let rhs = ConstraintSystem::new(
            vec![Word(0), Word(u64::MAX)],
            value_vec_layout,
            vec![constraint],
            vec![],
        );

        let mut blaster = RandBlast::new(0);
        let result = blaster.test_equivalence(&lhs, &rhs, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_inequivalent_systems() {
        // Create two constraint systems with same constants
        let n_const = 2;
        let n_witness = 3;
        let value_vec_layout = ValueVecLayout {
            n_const,
            n_inout: 0,
            n_witness,
            n_internal: 0,
            offset_inout: n_const,
            offset_witness: 4,      // next power of 2 after constants
            committed_total_len: 8, // next power of 2 after all values
            n_scratch: 0,
        };
        // LHS: ZERO & ZERO ^ ZERO = 0 (always satisfied)
        let lhs_constraint = AndConstraint::plain_abc(
            vec![ValueIndex(0)], // ZERO
            vec![ValueIndex(0)], // ZERO
            vec![ValueIndex(0)], // ZERO
        );

        // RHS: ALL_ONE & ALL_ONE ^ ZERO = 0
        // This is never satisfied since:
        //
        // ALL_ONE & ALL_ONE = ALL_ONE, and ALL_ONE ^ ZERO = ALL_ONE != 0
        let rhs_constraint = AndConstraint::plain_abc(
            vec![ValueIndex(1)], // ALL_ONE
            vec![ValueIndex(1)], // ALL_ONE
            vec![ValueIndex(0)], // ZERO
        );

        let lhs = ConstraintSystem::new(
            vec![Word(0), Word(u64::MAX)],
            value_vec_layout.clone(),
            vec![lhs_constraint],
            vec![],
        );
        let rhs = ConstraintSystem::new(
            vec![Word(0), Word(u64::MAX)],
            value_vec_layout,
            vec![rhs_constraint],
            vec![],
        );

        // LHS is always satisfied, RHS is never satisfied
        // They are definitely not equivalent
        let mut blaster = RandBlast::new(0);
        let result = blaster.test_equivalence(&lhs, &rhs, 10);
        assert!(result.is_err(), "Systems should not be equivalent");
    }
}
