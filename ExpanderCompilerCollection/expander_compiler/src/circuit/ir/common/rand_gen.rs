use rand::{Rng, RngCore, SeedableRng};

use super::*;

pub trait RandomInstruction {
    fn random_no_sub_circuit(
        r: impl RngCore,
        num_terms: &RandomRange,
        num_vars: usize,
        num_public_inputs: usize,
    ) -> Self;
}

pub trait RandomConstraintType {
    fn random(r: impl RngCore) -> Self;
}

impl RandomConstraintType for RawConstraintType {
    fn random(_r: impl RngCore) -> Self {
        ()
    }
}

pub struct RandomRange {
    pub min: usize,
    pub max: usize,
}

impl RandomRange {
    pub fn random(&self, rnd: &mut impl RngCore) -> usize {
        rnd.next_u32() as usize % (self.max - self.min + 1) + self.min
    }
}

pub struct RandomCircuitConfig {
    pub seed: usize,
    pub num_circuits: RandomRange,
    pub num_inputs: RandomRange,
    pub num_instructions: RandomRange,
    pub num_constraints: RandomRange,
    pub num_outputs: RandomRange,
    pub num_terms: RandomRange,
    pub sub_circuit_prob: f64,
}

impl<C: Config, Irc: IrConfig<Config = C>> RootCircuit<Irc>
where
    Irc::Instruction: RandomInstruction,
    <Irc::Constraint as Constraint<C>>::Type: RandomConstraintType,
{
    pub fn random(config: &RandomCircuitConfig) -> Self {
        let mut rnd = rand::rngs::StdRng::seed_from_u64(config.seed as u64);
        let mut root = RootCircuit::<Irc>::default();
        let mut circuit_ids = vec![0];
        let num_circuits = config.num_circuits.random(&mut rnd);
        root.num_public_inputs = config.num_inputs.random(&mut rnd);
        let mut has_constraint: Vec<bool> = vec![false; num_circuits];
        while circuit_ids.len() < num_circuits {
            let next_id = rnd.next_u64() as usize;
            if !circuit_ids.contains(&next_id) {
                circuit_ids.push(next_id);
            }
        }
        for (i, circuit_id) in circuit_ids.iter().enumerate().rev() {
            let num_inputs = config.num_inputs.random(&mut rnd);
            let num_instructions = config.num_instructions.random(&mut rnd);
            let mut num_constraints = config.num_constraints.random(&mut rnd);
            let num_outputs = config.num_outputs.random(&mut rnd);
            let mut instructions = Vec::with_capacity(num_instructions);
            let mut num_vars = num_inputs;
            has_constraint[i] = num_constraints > 0;
            for _ in 0..num_instructions {
                if rnd.gen::<f64>() < config.sub_circuit_prob && i != num_circuits - 1 {
                    let sub_circuit_index =
                        rnd.next_u64() as usize % (num_circuits - i - 1) + i + 1;
                    let sub_circuit_id = circuit_ids[sub_circuit_index];
                    let num_sub_outputs = root.circuits[&sub_circuit_id].outputs.len();
                    let num_sub_inputs = root.circuits[&sub_circuit_id].num_inputs;
                    if Irc::ALLOW_DUPLICATE_SUB_CIRCUIT_INPUTS || num_sub_inputs <= num_vars {
                        has_constraint[i] |= has_constraint[sub_circuit_index];
                        let mut sub_inputs = Vec::with_capacity(num_sub_inputs);
                        while sub_inputs.len() < num_sub_inputs {
                            let input = rnd.next_u64() as usize % num_vars + 1;
                            if Irc::ALLOW_DUPLICATE_SUB_CIRCUIT_INPUTS
                                || !sub_inputs.contains(&input)
                            {
                                sub_inputs.push(input);
                            }
                        }
                        instructions.push(Irc::Instruction::sub_circuit_call(
                            sub_circuit_id,
                            sub_inputs,
                            num_sub_outputs,
                        ));
                        num_vars += num_sub_outputs;
                        continue;
                    }
                }
                let insn = Irc::Instruction::random_no_sub_circuit(
                    &mut rnd,
                    &config.num_terms,
                    num_vars,
                    root.num_public_inputs,
                );
                num_vars += insn.num_outputs();
                instructions.push(insn);
            }
            let mut outputs = Vec::with_capacity(num_outputs);
            while outputs.len() < num_outputs
                && (outputs.len() < num_vars || Irc::ALLOW_DUPLICATE_OUTPUTS)
            {
                let output = rnd.next_u64() as usize % num_vars + 1;
                if Irc::ALLOW_DUPLICATE_OUTPUTS || !outputs.contains(&output) {
                    outputs.push(output);
                }
            }
            if i == 0 && !has_constraint[i] {
                assert!(num_vars > 0);
                num_constraints = 1;
            }
            let mut constraints = Vec::with_capacity(num_constraints);
            let mut constraints_vars = Vec::with_capacity(num_constraints);
            while constraints.len() < num_constraints
                && (constraints.len() < num_vars || Irc::ALLOW_DUPLICATE_CONSTRAINTS)
            {
                let constraint = rnd.next_u64() as usize % num_vars + 1;
                if Irc::ALLOW_DUPLICATE_CONSTRAINTS || !constraints_vars.contains(&constraint) {
                    constraints.push(Irc::Constraint::new(
                        constraint,
                        <Irc::Constraint as Constraint<C>>::Type::random(&mut rnd),
                    ));
                    constraints_vars.push(constraint);
                }
            }
            root.circuits.insert(
                *circuit_id,
                Circuit {
                    instructions,
                    constraints,
                    outputs,
                    num_inputs,
                },
            );
        }
        root
    }
}
