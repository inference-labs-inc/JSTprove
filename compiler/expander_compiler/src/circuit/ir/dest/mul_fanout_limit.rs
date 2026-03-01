use super::*;

// This module contains the implementation of the optimization that reduces the fanout of the input variables in multiplication gates.
// There are two ways to reduce the fanout of a variable:
// 1. Copy the whole expression to a new variable. This will copy all gates, and may increase the number of gates by a lot.
// 2. Create a relay expression of the variable. This may increase the layer of the circuit by 1.

// These are the limits for the first method.
const MAX_COPIES_OF_VARIABLES: usize = 4;
const MAX_COPIES_OF_GATES: usize = 64;

fn compute_max_copy_cnt(num_gates: usize) -> usize {
    if num_gates == 0 {
        return 0;
    }
    MAX_COPIES_OF_VARIABLES.min(MAX_COPIES_OF_GATES / num_gates)
}

struct NewIdQueue {
    queue: Vec<(usize, usize)>,
    next: usize,
    default_id: usize,
}

impl NewIdQueue {
    fn new(default_id: usize) -> Self {
        Self {
            queue: Vec::new(),
            next: 0,
            default_id,
        }
    }

    fn push(&mut self, id: usize, num: usize) {
        self.queue.push((id, num));
    }

    fn get(&mut self) -> usize {
        while self.next < self.queue.len() {
            let (id, num) = self.queue[self.next];
            if num > 0 {
                self.queue[self.next].1 -= 1;
                return id;
            }
            self.next += 1;
        }
        self.default_id
    }
}

impl<C: Config> CircuitRelaxed<C> {
    fn solve_mul_fanout_limit(&self, limit: usize) -> CircuitRelaxed<C> {
        let mut max_copy_cnt = vec![0; self.num_inputs + 1];
        let mut mul_ref_cnt = vec![0; self.num_inputs + 1];
        let mut internal_var_insn_id = vec![None; self.num_inputs + 1];

        for (i, insn) in self.instructions.iter().enumerate() {
            match insn {
                Instruction::ConstantLike { .. } => {
                    mul_ref_cnt.push(0);
                    max_copy_cnt.push(compute_max_copy_cnt(1));
                    internal_var_insn_id.push(None);
                }
                Instruction::SubCircuitCall { num_outputs, .. } => {
                    for _ in 0..*num_outputs {
                        mul_ref_cnt.push(0);
                        max_copy_cnt.push(0);
                        internal_var_insn_id.push(None);
                    }
                }
                Instruction::InternalVariable { expr } => {
                    for term in expr.iter() {
                        if let VarSpec::Quad(x, y) = term.vars {
                            mul_ref_cnt[x] += 1;
                            mul_ref_cnt[y] += 1;
                        }
                    }
                    mul_ref_cnt.push(0);
                    max_copy_cnt.push(compute_max_copy_cnt(expr.len()));
                    internal_var_insn_id.push(Some(i))
                }
            }
        }

        let mut add_copy_cnt = vec![0; max_copy_cnt.len()];
        let mut relay_cnt = vec![0; max_copy_cnt.len()];
        let mut any_new = false;

        for i in (1..max_copy_cnt.len()).rev() {
            let mc = max_copy_cnt[i].max(1);
            if mul_ref_cnt[i] <= mc * limit {
                add_copy_cnt[i] = ((mul_ref_cnt[i] + limit - 1) / limit).max(1) - 1;
                any_new = true;
                if let Some(j) = internal_var_insn_id[i] {
                    if let Instruction::InternalVariable { expr } = &self.instructions[j] {
                        for term in expr.iter() {
                            if let VarSpec::Quad(x, y) = term.vars {
                                mul_ref_cnt[x] += add_copy_cnt[i];
                                mul_ref_cnt[y] += add_copy_cnt[i];
                            }
                        }
                    } else {
                        unreachable!();
                    }
                }
            } else {
                // mul_ref_cnt[i] + relay_cnt[i] <= limit * (1 + relay_cnt[i])
                relay_cnt[i] = (mul_ref_cnt[i] - 2) / (limit - 1);
                any_new = true;
            }
        }

        if !any_new {
            return self.clone();
        }

        let mut new_id = vec![];
        let mut new_insns: Vec<Instruction<C>> = Vec::new();
        let mut new_var_max = self.num_inputs;
        let mut last_solved_id = 0;

        for i in 0..=self.num_inputs {
            new_id.push(NewIdQueue::new(i));
        }

        for insn in self.instructions.iter() {
            while last_solved_id + 1 < new_id.len() {
                last_solved_id += 1;
                let x = last_solved_id;
                if add_copy_cnt[x] == 0 && relay_cnt[x] == 0 {
                    continue;
                }
                let y = new_id[x].default_id;
                new_id[x].push(y, limit);
                for _ in 0..add_copy_cnt[x] {
                    let insn = new_insns.last().unwrap().clone();
                    new_insns.push(insn);
                    new_var_max += 1;
                    new_id[x].push(new_var_max, limit);
                }
                for _ in 0..relay_cnt[x] {
                    let y = new_id[x].get();
                    new_insns.push(Instruction::InternalVariable {
                        expr: Expression::new_linear(CircuitField::<C>::one(), y),
                    });
                    new_var_max += 1;
                    new_id[x].push(new_var_max, limit);
                }
            }
            match insn {
                Instruction::ConstantLike { value } => {
                    new_insns.push(Instruction::ConstantLike { value: *value });
                    new_var_max += 1;
                    new_id.push(NewIdQueue::new(new_var_max));
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    new_insns.push(Instruction::SubCircuitCall {
                        sub_circuit_id: *sub_circuit_id,
                        inputs: inputs.iter().map(|x| new_id[*x].default_id).collect(),
                        num_outputs: *num_outputs,
                    });
                    for _ in 0..*num_outputs {
                        new_var_max += 1;
                        let x = new_id.len();
                        new_id.push(NewIdQueue::new(new_var_max));
                        assert_eq!(add_copy_cnt[x], 0);
                    }
                }
                Instruction::InternalVariable { expr } => {
                    let x = new_id.len();
                    if add_copy_cnt[x] > 0 {
                        assert_eq!(relay_cnt[x], 0);
                    }
                    for _ in 0..=add_copy_cnt[x] {
                        let mut new_terms = vec![];
                        for term in expr.iter() {
                            if let VarSpec::Quad(x, y) = term.vars {
                                new_terms.push(Term {
                                    vars: VarSpec::Quad(new_id[x].get(), new_id[y].get()),
                                    coef: term.coef,
                                });
                            } else {
                                new_terms.push(Term {
                                    vars: term.vars.replace_vars(|x| new_id[x].default_id),
                                    coef: term.coef,
                                });
                            }
                        }
                        new_insns.push(Instruction::InternalVariable {
                            expr: Expression::from_terms(new_terms),
                        });
                        new_var_max += 1;
                    }
                    new_id.push(NewIdQueue::new(new_var_max));
                    if add_copy_cnt[x] > 0 {
                        for i in 0..=add_copy_cnt[x] {
                            new_id[x].push(new_var_max - add_copy_cnt[x] + i, limit);
                        }
                        last_solved_id = x;
                    }
                }
            }
        }

        CircuitRelaxed {
            instructions: new_insns,
            num_inputs: self.num_inputs,
            outputs: self.outputs.iter().map(|x| new_id[*x].default_id).collect(),
            constraints: self
                .constraints
                .iter()
                .map(|x| new_id[*x].default_id)
                .collect(),
        }
    }
}

impl<C: Config> RootCircuitRelaxed<C> {
    pub fn solve_mul_fanout_limit(&self, limit: usize) -> RootCircuitRelaxed<C> {
        if limit <= 1 {
            panic!("limit must be greater than 1");
        }

        let mut circuits = HashMap::new();
        for (id, circuit) in self.circuits.iter() {
            circuits.insert(*id, circuit.solve_mul_fanout_limit(limit));
        }
        RootCircuitRelaxed {
            circuits,
            num_public_inputs: self.num_public_inputs,
            expected_num_output_zeroes: self.expected_num_output_zeroes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::layered::{InputUsize, NormalInputType};
    use crate::field::FieldArith;
    use crate::frontend::M31Config as C;
    use mersenne31::M31;
    use rand::{RngCore, SeedableRng};
    use serdes::ExpSerde;

    type CField = M31;

    fn verify_mul_fanout(rc: &RootCircuitRelaxed<C>, limit: usize) {
        for circuit in rc.circuits.values() {
            let mut mul_ref_cnt = vec![0; circuit.num_inputs + 1];
            for insn in circuit.instructions.iter() {
                match insn {
                    Instruction::ConstantLike { .. } => {}
                    Instruction::SubCircuitCall { .. } => {}
                    Instruction::InternalVariable { expr } => {
                        for term in expr.iter() {
                            if let VarSpec::Quad(x, y) = term.vars {
                                mul_ref_cnt[x] += 1;
                                mul_ref_cnt[y] += 1;
                            }
                        }
                    }
                }
                for _ in 0..insn.num_outputs() {
                    mul_ref_cnt.push(0);
                }
            }
            for x in mul_ref_cnt.iter().skip(1) {
                assert!(*x <= limit);
            }
        }
    }

    fn do_test(root: RootCircuitRelaxed<C>, limits: Vec<usize>) {
        for lim in limits.iter() {
            let new_root = root.solve_mul_fanout_limit(*lim);
            assert_eq!(new_root.validate(), Ok(()));
            assert_eq!(new_root.input_size(), root.input_size());
            verify_mul_fanout(&new_root, *lim);
            let inputs: Vec<CField> = (0..root.input_size())
                .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                .collect();
            let (out1, cond1) = root.eval_unsafe(inputs.clone());
            let (out2, cond2) = new_root.eval_unsafe(inputs);
            assert_eq!(out1, out2);
            assert_eq!(cond1, cond2);
        }
    }

    #[test]
    fn fanout_test1() {
        let mut circuit = CircuitRelaxed {
            instructions: Vec::new(),
            constraints: Vec::new(),
            outputs: Vec::new(),
            num_inputs: 2,
        };
        for i in 3..=1003 {
            circuit.instructions.push(Instruction::InternalVariable {
                expr: Expression::new_quad(CField::one(), 1, 2),
            });
            circuit.constraints.push(i);
            circuit.outputs.push(i);
        }

        let mut root = RootCircuitRelaxed::<C>::default();
        root.circuits.insert(0, circuit);

        do_test(root, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 64, 2000]);
    }

    #[test]
    fn fanout_test2() {
        let mut circuit = CircuitRelaxed {
            instructions: Vec::new(),
            constraints: Vec::new(),
            outputs: Vec::new(),
            num_inputs: 1,
        };
        for _ in 0..2 {
            circuit.instructions.push(Instruction::InternalVariable {
                expr: Expression::new_quad(CField::from(100), 1, 1),
            });
        }
        for i in 4..=1003 {
            circuit.instructions.push(Instruction::InternalVariable {
                expr: Expression::new_quad(CField::from(10), 2, 3),
            });
            circuit.constraints.push(i);
            circuit.outputs.push(i);
        }

        let mut root = RootCircuitRelaxed::<C>::default();
        root.circuits.insert(0, circuit);

        do_test(root, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 64, 2000]);
    }

    #[test]
    fn fanout_test3() {
        let mut circuit = CircuitRelaxed {
            instructions: Vec::new(),
            constraints: Vec::new(),
            outputs: Vec::new(),
            num_inputs: 1,
        };
        for _ in 0..2 {
            circuit.instructions.push(Instruction::SubCircuitCall {
                sub_circuit_id: 1,
                inputs: vec![1],
                num_outputs: 1,
            });
        }
        for i in 4..=1003 {
            circuit.instructions.push(Instruction::InternalVariable {
                expr: Expression::new_quad(CField::from(10), 2, 3),
            });
            circuit.constraints.push(i);
            circuit.outputs.push(i);
        }

        let mut root = RootCircuitRelaxed::<C>::default();
        root.circuits.insert(0, circuit);
        root.circuits.insert(
            1,
            CircuitRelaxed {
                instructions: vec![Instruction::InternalVariable {
                    expr: Expression::new_quad(CField::from(100), 1, 1),
                }],
                constraints: vec![],
                outputs: vec![2],
                num_inputs: 1,
            },
        );

        do_test(root, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 64, 2000]);
    }

    #[test]
    fn fanout_test_random() {
        let mut rnd = rand::rngs::StdRng::seed_from_u64(3);
        let mut circuit = CircuitRelaxed {
            instructions: Vec::new(),
            constraints: Vec::new(),
            outputs: Vec::new(),
            num_inputs: 100,
        };
        let mut q = vec![];
        for i in 1..=100 {
            for _ in 0..5 {
                q.push(i);
            }
            if i % 20 == 0 {
                for _ in 0..100 {
                    q.push(i);
                }
            }
        }

        let n = 10003;

        for i in 101..=n {
            let mut terms = vec![];
            let mut c = q.len() / 2;
            if i != n {
                c = c.min(5);
            }
            for _ in 0..c {
                let x = q.swap_remove(rnd.next_u64() as usize % q.len());
                let y = q.swap_remove(rnd.next_u64() as usize % q.len());
                terms.push(Term {
                    vars: VarSpec::Quad(x, y),
                    coef: CField::one(),
                });
            }
            circuit.instructions.push(Instruction::InternalVariable {
                expr: Expression::from_terms(terms),
            });
            circuit.constraints.push(i);
            circuit.outputs.push(i);
            for _ in 0..5 {
                q.push(i);
            }
            if i % 20 == 0 {
                for _ in 0..100 {
                    q.push(i);
                }
            }
        }

        let mut root = RootCircuitRelaxed::<C>::default();
        root.circuits.insert(0, circuit);

        do_test(root, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 64, 2000]);
    }

    #[test]
    fn full_fanout_test_and_dump() {
        use crate::circuit::ir::common::rand_gen::{RandomCircuitConfig, RandomRange};

        for i in 0..1000 {
            let config = RandomCircuitConfig {
                seed: i + 100000,
                num_circuits: RandomRange { min: 20, max: 20 },
                num_inputs: RandomRange { min: 1, max: 3 },
                num_instructions: RandomRange { min: 30, max: 50 },
                num_constraints: RandomRange { min: 0, max: 5 },
                num_outputs: RandomRange { min: 1, max: 3 },
                num_terms: RandomRange { min: 1, max: 5 },
                sub_circuit_prob: 0.05,
            };
            let root = crate::circuit::ir::source::RootCircuit::<C>::random(&config);
            assert_eq!(root.validate(), Ok(()));
            match crate::compile::compile_with_options::<_, NormalInputType>(
                &root,
                crate::compile::CompileOptions::default().with_mul_fanout_limit(256),
            ) {
                Err(e) => {
                    if e.is_internal() {
                        panic!("{:?}", e);
                    }
                }
                Ok((_, circuit)) => {
                    assert_eq!(circuit.validate(), Ok(()));
                    for segment in circuit.segments.iter() {
                        let mut ref_num = vec![0; segment.num_inputs.get(0)];
                        for m in segment.gate_muls.iter() {
                            ref_num[m.inputs[0].offset] += 1;
                            ref_num[m.inputs[1].offset] += 1;
                        }
                        for x in ref_num.iter() {
                            assert!(*x <= 256);
                        }
                    }

                    let mut buf = Vec::new();
                    circuit.serialize_into(&mut buf).unwrap();
                }
            }
        }
    }
}
