use std::fmt;

use super::{Circuit, Instruction, IrConfig, RootCircuit};

impl<Irc: IrConfig> fmt::Display for Circuit<Irc>
where
    Irc::Instruction: fmt::Display,
    Irc::Constraint: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut var_max = self.get_num_inputs_all();
        for insn in self.instructions.iter() {
            let num_outputs = insn.num_outputs();
            for i in 0..num_outputs {
                var_max += 1;
                write!(f, "v{var_max}")?;
                if i < num_outputs - 1 {
                    write!(f, ",")?;
                }
            }
            writeln!(f, " = {insn}")?;
        }
        write!(f, "Outputs: ")?;
        for (i, out) in self.outputs.iter().enumerate() {
            write!(f, "v{out}")?;
            if i < self.outputs.len() - 1 {
                write!(f, ",")?;
            }
        }
        writeln!(f)?;
        write!(f, "Constraints: ")?;
        for (i, con) in self.constraints.iter().enumerate() {
            write!(f, "v{con}")?;
            if i < self.constraints.len() - 1 {
                write!(f, ",")?;
            }
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<Irc: IrConfig> fmt::Display for RootCircuit<Irc>
where
    Irc::Instruction: fmt::Display,
    Irc::Constraint: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (circuit_id, circuit) in self.circuits.iter() {
            writeln!(
                f,
                "Circuit {} numIn={} numOut={} numCon={}",
                circuit_id,
                circuit.num_inputs,
                circuit.outputs.len(),
                circuit.constraints.len()
            )?;
            write!(f, "{circuit}")?;
        }
        Ok(())
    }
}
