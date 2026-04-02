// Copyright 2025 Irreducible Inc.
use anyhow::{Result, anyhow};
use binius_core::constraint_system::{
    AndConstraint, ConstraintSystem, MulConstraint, Operand, ShiftVariant, ShiftedValueIndex,
};
use z3::{
    Context, Model, Solver,
    ast::{self, Ast},
};

pub struct SmtChecker<'ctx> {
    ctx: &'ctx Context,
    solver: Solver<'ctx>,
}

impl<'ctx> SmtChecker<'ctx> {
    pub fn new(ctx: &'ctx Context) -> Self {
        let solver = Solver::new(ctx);
        SmtChecker { ctx, solver }
    }

    /// Create Z3 bitvector variables for all values in the constraint system
    fn create_variables(&self, cs: &ConstraintSystem) -> Vec<ast::BV<'ctx>> {
        let mut vars = Vec::new();

        // Add constants
        for constant in &cs.constants {
            vars.push(ast::BV::from_u64(self.ctx, constant.0, 64));
        }

        // Add input/output variables (currently unused, but reserved)
        for i in 0..cs.value_vec_layout.n_inout {
            let name = format!("io_{i}");
            vars.push(ast::BV::new_const(self.ctx, name, 64));
        }

        // Add witness variables
        for i in 0..cs.value_vec_layout.n_witness {
            let name = format!("w_{i}");
            vars.push(ast::BV::new_const(self.ctx, name, 64));
        }

        // Pad to power of two if needed
        let total_len = cs.value_vec_len();
        while vars.len() < total_len {
            vars.push(ast::BV::from_u64(self.ctx, 0, 64));
        }

        vars
    }

    /// Evaluate a shifted value.
    fn eval_shifted(&self, vars: &[ast::BV<'ctx>], sv: &ShiftedValueIndex) -> ast::BV<'ctx> {
        let val = &vars[sv.value_index.0 as usize];

        if sv.amount == 0 {
            return val.clone();
        }

        match sv.shift_variant {
            ShiftVariant::Sll => val.bvshl(&ast::BV::from_u64(self.ctx, sv.amount as u64, 64)),
            ShiftVariant::Slr => val.bvlshr(&ast::BV::from_u64(self.ctx, sv.amount as u64, 64)),
            ShiftVariant::Sar => val.bvashr(&ast::BV::from_u64(self.ctx, sv.amount as u64, 64)),
            ShiftVariant::Rotr => val.bvrotr(&ast::BV::from_u64(self.ctx, sv.amount as u64, 64)),
            ShiftVariant::Sll32 => {
                let shift = ast::BV::from_u64(self.ctx, sv.amount as u64, 32);
                let hi = val.extract(63, 32);
                let lo = val.extract(31, 0);
                hi.bvshl(&shift).concat(&lo.bvshl(&shift))
            }
            ShiftVariant::Srl32 => {
                let shift = ast::BV::from_u64(self.ctx, sv.amount as u64, 32);
                let hi = val.extract(63, 32);
                let lo = val.extract(31, 0);
                hi.bvlshr(&shift).concat(&lo.bvlshr(&shift))
            }
            ShiftVariant::Sra32 => {
                let shift = ast::BV::from_u64(self.ctx, sv.amount as u64, 32);
                let hi = val.extract(63, 32);
                let lo = val.extract(31, 0);
                hi.bvashr(&shift).concat(&lo.bvashr(&shift))
            }
            ShiftVariant::Rotr32 => {
                let rotate = ast::BV::from_u64(self.ctx, sv.amount as u64, 32);
                let hi = val.extract(63, 32);
                let lo = val.extract(31, 0);
                hi.bvrotr(&rotate).concat(&lo.bvrotr(&rotate))
            }
        }
    }

    /// Evaluate an operand (XOR of shifted values)
    fn eval_operand(&self, vars: &[ast::BV<'ctx>], operand: &Operand) -> ast::BV<'ctx> {
        if operand.is_empty() {
            return ast::BV::from_u64(self.ctx, 0, 64);
        }

        let mut result = self.eval_shifted(vars, &operand[0]);

        for sv in &operand[1..] {
            let shifted = self.eval_shifted(vars, sv);
            result = result.bvxor(&shifted);
        }

        result
    }

    /// Evaluate AND constraint and return the equality boolean expression.
    fn eval_and_constraint(
        &self,
        vars: &[ast::BV<'ctx>],
        constraint: &AndConstraint,
    ) -> ast::Bool<'ctx> {
        let a = self.eval_operand(vars, &constraint.a);
        let b = self.eval_operand(vars, &constraint.b);
        let c = self.eval_operand(vars, &constraint.c);

        let and_result = a.bvand(&b);
        let xor_result = and_result.bvxor(&c);
        let zero = ast::BV::from_u64(self.ctx, 0, 64);

        xor_result._eq(&zero)
    }

    /// Evaluate MUL constraint and return the equality expressions
    fn eval_mul_constraint(
        &self,
        vars: &[ast::BV<'ctx>],
        constraint: &MulConstraint,
    ) -> (ast::Bool<'ctx>, ast::Bool<'ctx>) {
        let a = self.eval_operand(vars, &constraint.a);
        let b = self.eval_operand(vars, &constraint.b);
        let hi = self.eval_operand(vars, &constraint.hi);
        let lo = self.eval_operand(vars, &constraint.lo);

        // Extend to 128 bits for multiplication
        let a_ext = a.zero_ext(64);
        let b_ext = b.zero_ext(64);
        let product = a_ext.bvmul(&b_ext);

        // Extract high and low parts
        let expected_hi = product.extract(127, 64);
        let expected_lo = product.extract(63, 0);

        // Return the equality expressions
        (hi._eq(&expected_hi), lo._eq(&expected_lo))
    }

    /// Given the constraint system, translate the constraints into z3 AST.
    fn translate_constraint_system_to_z3(
        &self,
        vars: &[ast::BV<'ctx>],
        cs: &ConstraintSystem,
    ) -> Vec<ast::Bool<'ctx>> {
        let mut constraints = Vec::new();
        for constraint in &cs.and_constraints {
            let eq = self.eval_and_constraint(vars, constraint);
            constraints.push(eq);
        }
        for constraint in &cs.mul_constraints {
            let (hi_eq, lo_eq) = self.eval_mul_constraint(vars, constraint);
            constraints.push(hi_eq);
            constraints.push(lo_eq);
        }
        constraints
    }

    /// Given two constraint systems find a counterexample that satisfies the reference constraint
    /// system but not the contrast constraint system.
    ///
    /// If no counterexample is found, then the constraint systems are provably equivalent.
    fn find_counterexample(
        &mut self,
        vars: &[ast::BV<'ctx>],
        (reference_cs, reference_cs_name): (&ConstraintSystem, &str),
        (contrast_cs, contrast_cs_name): (&ConstraintSystem, &str),
        n_witness: usize,
        witness_start: usize,
    ) -> Result<()> {
        self.solver.push();

        // Assert that all constraints in the reference system are satisfied
        for constraint in self.translate_constraint_system_to_z3(vars, reference_cs) {
            self.solver.assert(&constraint);
        }

        // Assert that at least one constraint in the contrast system fails
        let contrast_constraints = self.translate_constraint_system_to_z3(vars, contrast_cs);
        if !contrast_constraints.is_empty() {
            let negated_constraints: Vec<_> =
                contrast_constraints.iter().map(|c| c.not()).collect();
            let at_least_one_fails =
                ast::Bool::or(self.ctx, &negated_constraints.iter().collect::<Vec<_>>());
            self.solver.assert(&at_least_one_fails);
        }

        if self.solver.check() == z3::SatResult::Sat {
            let model = self
                .solver
                .get_model()
                .ok_or_else(|| anyhow!("Failed to get Z3 model"))?;
            self.solver.pop(1);
            return Err(anyhow!(
                "Systems are not equivalent: found witness satisfying {} but not {}\n{}",
                reference_cs_name,
                contrast_cs_name,
                format_counterexample(&model, vars, n_witness, witness_start)
            ));
        }

        self.solver.pop(1);
        Ok(())
    }

    /// Check if two constraint systems are equivalent
    pub fn check_equivalence(
        &mut self,
        lhs: &ConstraintSystem,
        rhs: &ConstraintSystem,
    ) -> Result<()> {
        assert_eq!(lhs.value_vec_layout, rhs.value_vec_layout);
        assert_eq!(lhs.constants.len(), rhs.constants.len());

        // Create a single set of variables (both systems share the same variable space)
        let vars = self.create_variables(lhs);
        let witness_start = lhs.constants.len() + lhs.value_vec_layout.n_inout;
        let n_witness = lhs.value_vec_layout.n_witness;

        // Now check for equivalence by looking for counterexamples
        // Case 1: Find witness that satisfies LHS but not RHS
        // Case 2: Find witness that satisfies RHS but not LHS.
        self.find_counterexample(&vars, (lhs, "LHS"), (rhs, "RHS"), n_witness, witness_start)?;
        self.find_counterexample(&vars, (rhs, "RHS"), (lhs, "LHS"), n_witness, witness_start)?;

        Ok(())
    }
}

fn format_counterexample<'ctx>(
    model: &Model<'ctx>,
    vars: &[ast::BV<'ctx>],
    n_witness: usize,
    start: usize,
) -> String {
    let mut result = String::from("Counterexample witness values:\n");

    for i in 0..n_witness {
        if let Some(val) = model.eval(&vars[start + i], true) {
            if let Some(u) = val.as_u64() {
                result.push_str(&format!("  w_{i} = 0x{u:016x}\n"));
            } else {
                result.push_str(&format!("  w_{i} = {val}\n"));
            }
        }
    }

    result
}
