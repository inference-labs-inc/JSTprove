pub use crate::circuit::ir::hint_normalized::witness_solver::WitnessSolver;
use crate::circuit::layered::witness::Witness;

use super::{internal, Config, Error};

impl<C: Config> WitnessSolver<C> {
    pub fn solve_witness<Cir: internal::DumpLoadTwoVariables<C::CircuitField>>(
        &self,
        assignment: &Cir,
    ) -> Result<Witness<C>, Error> {
        let mut vars = Vec::new();
        let mut public_vars = Vec::new();
        assignment.dump_into(&mut vars, &mut public_vars);
        self.solve_witness_from_raw_inputs(vars, public_vars)
    }

    pub fn solve_witnesses<Cir: internal::DumpLoadTwoVariables<C::CircuitField>>(
        &self,
        assignments: &[Cir],
    ) -> Result<Witness<C>, Error> {
        self.solve_witnesses_from_raw_inputs(assignments.len(), |i| {
            let mut vars = Vec::new();
            let mut public_vars = Vec::new();
            assignments[i].dump_into(&mut vars, &mut public_vars);
            (vars, public_vars)
        })
    }
}
