use core::panic;
use std::collections::{HashMap, HashSet};

use crate::{
    circuit::ir::{
        common::Instruction as _,
        dest::{CircuitRelaxed as IrCircuit, Instruction, RootCircuitRelaxed as IrRootCircuit},
        expr::{Expression, Term},
    },
    field::FieldArith,
    frontend::Config,
    utils::pool::Pool,
};

struct SplitContext<'a, C: Config> {
    // the root circuit
    rc: &'a IrRootCircuit<C>,

    new_circuits: Pool<IrCircuit<C>>,
    output_layers: HashMap<(usize, Vec<usize>), Vec<usize>>,
    cc_occured_layers: HashMap<(usize, Vec<usize>), Vec<usize>>,
    splitted_circuits: HashMap<(usize, Vec<usize>, Vec<usize>), SplittedCircuit>,
    new_circuit0: Option<IrCircuit<C>>,
}

struct SplittedCircuit {
    segments: Vec<SplittedCircuitSegment>,
    outputs: Vec<SplitVarRef>,
}

#[derive(Clone, Debug)]
struct SplittedCircuitSegment {
    start_layer: usize,
    end_layer: usize,
    new_id: usize,
    mid_inputs: Vec<SplitVarRef>,
}

#[derive(Clone, Debug)]
enum SplitVarRef {
    Input(usize),
    Internal(usize),
    Unknown,
}

impl<'a, C: Config> SplitContext<'a, C> {
    fn compute_output_layers(&mut self, circuit_id: usize, input_layers: Vec<usize>) {
        let circuit = &self.rc.circuits[&circuit_id];
        let mut var_layers = vec![0];
        var_layers.extend(input_layers.iter().cloned());
        let mut cc_occured_layers = HashSet::new();
        for insn in circuit.instructions.iter() {
            match insn {
                Instruction::ConstantLike { .. } => {
                    var_layers.push(1);
                }
                Instruction::InternalVariable { expr } => {
                    let expr_vars: Vec<usize> = expr.get_vars();
                    var_layers
                        .push(expr_vars.iter().map(|x| var_layers[*x]).max().unwrap_or(0) + 1);
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let sub_input_layers: Vec<usize> =
                        inputs.iter().map(|x| var_layers[*x]).collect();
                    let key = (*sub_circuit_id, sub_input_layers.clone());
                    if !self.output_layers.contains_key(&key) {
                        self.compute_output_layers(*sub_circuit_id, sub_input_layers);
                    }
                    let o = &self.output_layers[&key];
                    assert_eq!(o.len(), *num_outputs);
                    for x in o.iter() {
                        var_layers.push(*x);
                    }
                    for x in self.cc_occured_layers[&key].iter() {
                        cc_occured_layers.insert(*x + 1);
                    }
                }
            }
        }
        for x in circuit.constraints.iter() {
            cc_occured_layers.insert(var_layers[*x] + 1);
        }
        let out_layers = circuit.outputs.iter().map(|x| var_layers[*x]).collect();
        self.output_layers
            .insert((circuit_id, input_layers.clone()), out_layers);
        let mut cc_occured_layers: Vec<usize> = cc_occured_layers.into_iter().collect();
        cc_occured_layers.sort();
        self.cc_occured_layers
            .insert((circuit_id, input_layers), cc_occured_layers);
    }

    fn expand_sub_circuit_calls_phase1(
        &mut self,
        circuit_id: usize,
        input_layers: Vec<usize>,
        split_at: &Vec<usize>,
    ) -> (IrCircuit<C>, Vec<usize>, Vec<usize>, Vec<usize>) {
        let circuit = &self.rc.circuits[&circuit_id];
        let mut var_new_id: Vec<usize> = (0..=circuit.get_num_inputs_all()).collect();
        let mut var_max = circuit.num_inputs;
        let mut new_insns: Vec<Instruction<C>> = Vec::new();
        let mut sub_combined_constraints: Vec<usize> = Vec::new();
        let mut new_var_layers = input_layers;
        let mut sc_layers = Vec::new();
        new_var_layers.insert(0, 0);
        for insn in circuit.instructions.iter() {
            match insn {
                Instruction::ConstantLike { value } => {
                    new_insns.push(Instruction::ConstantLike {
                        value: value.clone(),
                    });
                    var_max += 1;
                    var_new_id.push(var_max);
                    new_var_layers.push(1);
                }
                Instruction::InternalVariable { expr } => {
                    let expr = expr.replace_vars(|x| var_new_id[x]);
                    let expr_vars: Vec<usize> = expr.get_vars();
                    new_insns.push(Instruction::InternalVariable { expr });
                    var_max += 1;
                    var_new_id.push(var_max);
                    new_var_layers.push(
                        expr_vars
                            .iter()
                            .map(|x| new_var_layers[*x])
                            .max()
                            .unwrap_or(0)
                            + 1,
                    );
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let sub_input_layers: Vec<usize> = inputs
                        .iter()
                        .map(|x| new_var_layers[var_new_id[*x]])
                        .collect();
                    let key = (
                        *sub_circuit_id,
                        sub_input_layers.clone(),
                        split_at.to_owned(),
                    );
                    if !self.splitted_circuits.contains_key(&key) {
                        self.split_circuit(*sub_circuit_id, sub_input_layers, split_at);
                    }
                    let sub_spl = &self.splitted_circuits[&key];
                    let pre_var_max = var_max;
                    for sub_seg in sub_spl.segments.iter() {
                        let cur_no = self.new_circuits.get(sub_seg.new_id).outputs.len();
                        var_max += cur_no;
                        new_var_layers.resize(new_var_layers.len() + cur_no, sub_seg.end_layer);
                        let cur_inputs: Vec<usize> = sub_seg
                            .mid_inputs
                            .iter()
                            .map(|x| match x {
                                SplitVarRef::Input(i) => var_new_id[inputs[*i]],
                                SplitVarRef::Internal(i) => pre_var_max + i + 1,
                                SplitVarRef::Unknown => panic!("unexpected situation"),
                            })
                            .collect();
                        for x in cur_inputs.iter() {
                            assert!(new_var_layers[*x] <= sub_seg.start_layer);
                        }
                        new_insns.push(Instruction::SubCircuitCall {
                            sub_circuit_id: sub_seg.new_id,
                            inputs: cur_inputs,
                            num_outputs: cur_no,
                        });
                        sc_layers.push(sub_seg.start_layer);
                    }
                    for (var, exp_layer) in sub_spl
                        .outputs
                        .iter()
                        .take(*num_outputs)
                        .zip(self.output_layers[&(key.0, key.1)].iter())
                    {
                        let t = match var {
                            SplitVarRef::Input(i) => var_new_id[inputs[*i]],
                            SplitVarRef::Internal(i) => pre_var_max + i + 1,
                            SplitVarRef::Unknown => panic!("unexpected situation"),
                        };
                        var_new_id.push(t);
                        assert_eq!(new_var_layers[t], *exp_layer);
                    }
                    for var in sub_spl.outputs.iter().skip(*num_outputs) {
                        match var {
                            SplitVarRef::Input(_) => panic!("unexpected situation"),
                            SplitVarRef::Internal(i) => {
                                sub_combined_constraints.push(pre_var_max + i + 1)
                            }
                            SplitVarRef::Unknown => panic!("unexpected situation"),
                        };
                    }
                }
            }
        }
        assert_eq!(new_var_layers.len(), var_max + 1);
        let new_circuit = IrCircuit {
            num_inputs: circuit.num_inputs,
            instructions: new_insns,
            constraints: circuit.constraints.iter().map(|x| var_new_id[*x]).collect(),
            outputs: circuit.outputs.iter().map(|x| var_new_id[*x]).collect(),
        };
        (
            new_circuit,
            sub_combined_constraints,
            new_var_layers,
            sc_layers,
        )
    }

    fn split_circuit(&mut self, circuit_id: usize, input_layers: Vec<usize>, split_at: &[usize]) {
        let pre_split_at = split_at;
        let mut split_at_set: HashSet<usize> = split_at.iter().cloned().collect();
        for x in input_layers.iter() {
            split_at_set.insert(*x);
        }
        let key = (circuit_id, input_layers);
        for x in self.output_layers[&key].iter() {
            split_at_set.insert(*x);
        }
        for x in self.cc_occured_layers[&key].iter() {
            split_at_set.insert(*x);
        }
        let mut split_at: Vec<usize> = split_at_set.iter().cloned().collect();
        split_at.sort();
        let (_, input_layers) = key;

        let (mut circuit, mut sub_combined_constraints, mut min_layers, sc_layers) =
            self.expand_sub_circuit_calls_phase1(circuit_id, input_layers.clone(), &split_at);
        let mut outputs = std::mem::take(&mut circuit.outputs);
        let mut add_outputs = Vec::new();

        if C::ENABLE_RANDOM_COMBINATION {
            let n_layers = min_layers.iter().max().unwrap() + 3;
            sub_combined_constraints.sort_by(|a, b| min_layers[*a].cmp(&min_layers[*b]));
            let mut constraints = std::mem::take(&mut circuit.constraints);
            constraints.sort_by(|a, b| min_layers[*a].cmp(&min_layers[*b]));
            let mut j = 0;
            let mut k = 0;
            for i in 0..n_layers {
                let mut terms = Vec::new();
                while j < sub_combined_constraints.len()
                    && min_layers[sub_combined_constraints[j]] == i
                {
                    terms.push(Term::new_linear(
                        C::CircuitField::one(),
                        sub_combined_constraints[j],
                    ));
                    j += 1;
                }
                while circuit_id == 0 && !add_outputs.is_empty() {
                    terms.push(Term::new_linear(
                        C::CircuitField::one(),
                        add_outputs[add_outputs.len() - 1],
                    ));
                    add_outputs.pop();
                }
                while k < constraints.len() && min_layers[constraints[k]] == i {
                    terms.push(Term::new_random_linear(constraints[k]));
                    k += 1;
                }
                if !terms.is_empty() {
                    circuit.instructions.push(Instruction::InternalVariable {
                        expr: Expression::from_terms(terms),
                    });
                    min_layers.push(i + 1);
                    add_outputs.push(min_layers.len() - 1);
                }
            }
        }
        if circuit_id == 0 {
            assert_eq!(circuit.constraints.len(), 0);
            add_outputs.extend(outputs);
            circuit.outputs = add_outputs;
            self.new_circuit0 = Some(circuit);
            return;
        }
        outputs.extend(add_outputs);
        for l in min_layers.iter().take(circuit.num_inputs + 1) {
            assert!(split_at_set.contains(l));
        }
        for o in outputs.iter() {
            assert!(split_at_set.contains(&min_layers[*o]));
        }
        assert_eq!(split_at[0], 0);

        let mut max_layers = min_layers.clone();
        let mut cur_var_max = circuit.num_inputs;
        let mut sc_i = 0;
        for insn in circuit.instructions.iter() {
            match insn {
                Instruction::InternalVariable { expr } => {
                    let expr_vars: Vec<usize> = expr.get_vars();
                    for x in expr_vars.iter() {
                        max_layers[*x] = max_layers[*x].max(min_layers[cur_var_max + 1] - 1);
                    }
                }
                Instruction::SubCircuitCall { inputs, .. } => {
                    for x in inputs.iter() {
                        max_layers[*x] = max_layers[*x].max(sc_layers[sc_i]);
                    }
                    sc_i += 1;
                }
                _ => {}
            }
            cur_var_max += insn.num_outputs();
        }
        assert_eq!(sc_i, sc_layers.len());

        let mut var_new_id: Vec<SplitVarRef> = vec![SplitVarRef::Unknown; cur_var_max + 1];
        let mut is_output: Vec<bool> = vec![false; cur_var_max + 1];
        let mut fin_outputs = vec![None; cur_var_max + 1];
        for o in outputs.iter() {
            is_output[*o] = true;
        }
        for i in 0..circuit.num_inputs {
            var_new_id[i + 1] = SplitVarRef::Input(i);
            is_output[i + 1] = false;
            fin_outputs[i + 1] = Some(SplitVarRef::Input(i));
        }
        let mut internal_count = 0;
        let mut segments = Vec::new();
        for i in 0..split_at.len() - 1 {
            let start_layer = split_at[i];
            let end_layer = split_at[i + 1];
            let mut inputs = Vec::new();
            let mut var_local_id = vec![0; cur_var_max + 1];
            for j in 1..=circuit.num_inputs {
                if min_layers[j] == start_layer {
                    inputs.push(SplitVarRef::Input(j - 1));
                    var_local_id[j] = inputs.len();
                }
            }
            for j in 1..cur_var_max {
                if (min_layers[j] < start_layer
                    || (min_layers[j] == start_layer && j > circuit.num_inputs))
                    && max_layers[j] >= start_layer
                {
                    inputs.push(var_new_id[j].clone());
                    var_local_id[j] = inputs.len();
                }
            }
            let mut new_insns = Vec::new();
            let mut cur_var_max = circuit.num_inputs;
            let mut local_var_max = inputs.len();
            for insn in circuit.instructions.iter() {
                match insn {
                    Instruction::ConstantLike { value } => {
                        cur_var_max += 1;
                        if start_layer == 0 {
                            local_var_max += 1;
                            new_insns.push(Instruction::ConstantLike {
                                value: value.clone(),
                            });
                            var_local_id[cur_var_max] = local_var_max;
                        }
                    }
                    Instruction::InternalVariable { expr } => {
                        cur_var_max += 1;
                        if min_layers[cur_var_max] > start_layer
                            && min_layers[cur_var_max] <= end_layer
                        {
                            let expr_vars: Vec<usize> = expr.get_vars();
                            for x in expr_vars.iter() {
                                if var_local_id[*x] == 0 {
                                    panic!("unexpected situation");
                                }
                            }
                            let expr = expr.replace_vars(|x| var_local_id[x]);
                            new_insns.push(Instruction::InternalVariable { expr });
                            local_var_max += 1;
                            var_local_id[cur_var_max] = local_var_max;
                        }
                    }
                    Instruction::SubCircuitCall {
                        sub_circuit_id,
                        inputs,
                        num_outputs,
                    } => {
                        if *num_outputs > 0
                            && min_layers[cur_var_max + 1] > start_layer
                            && min_layers[cur_var_max + 1] <= end_layer
                        {
                            for x in inputs.iter() {
                                if var_local_id[*x] == 0 {
                                    panic!("unexpected situation");
                                }
                            }
                            new_insns.push(Instruction::SubCircuitCall {
                                sub_circuit_id: *sub_circuit_id,
                                inputs: inputs.iter().map(|x| var_local_id[*x]).collect(),
                                num_outputs: *num_outputs,
                            });
                            for _ in 0..*num_outputs {
                                cur_var_max += 1;
                                local_var_max += 1;
                                var_local_id[cur_var_max] = local_var_max;
                            }
                        } else {
                            cur_var_max += *num_outputs;
                        }
                    }
                }
            }
            let mut outputs = Vec::new();
            for j in 1..=cur_var_max {
                if (is_output[j] || (min_layers[j] <= end_layer && max_layers[j] >= end_layer))
                    && var_local_id[j] != 0
                {
                    outputs.push(var_local_id[j]);
                    var_new_id[j] = SplitVarRef::Internal(internal_count);
                    if is_output[j] {
                        fin_outputs[j] = Some(SplitVarRef::Internal(internal_count));
                        is_output[j] = false;
                    }
                    internal_count += 1;
                }
            }
            let new_circuit = IrCircuit {
                num_inputs: inputs.len(),
                instructions: new_insns,
                constraints: vec![],
                outputs,
            };
            let new_id = self.new_circuits.add(&new_circuit);
            assert_ne!(new_id, 0);
            segments.push(SplittedCircuitSegment {
                start_layer,
                end_layer,
                new_id,
                mid_inputs: inputs,
            });
        }
        self.splitted_circuits.insert(
            (circuit_id, input_layers, pre_split_at.to_owned()),
            SplittedCircuit {
                segments,
                outputs: outputs
                    .iter()
                    .map(|x| fin_outputs[*x].as_ref().unwrap().clone())
                    .collect(),
            },
        );
    }
}

pub fn split_to_single_layer<C: Config>(root: &IrRootCircuit<C>) -> IrRootCircuit<C> {
    let mut ctx = SplitContext {
        rc: root,
        new_circuits: Pool::new(),
        output_layers: HashMap::new(),
        cc_occured_layers: HashMap::new(),
        splitted_circuits: HashMap::new(),
        new_circuit0: None,
    };
    ctx.new_circuits.add(&IrCircuit {
        num_inputs: !0,
        instructions: vec![],
        constraints: vec![],
        outputs: vec![],
    });
    ctx.compute_output_layers(0, vec![0; root.circuits[&0].num_inputs]);
    ctx.split_circuit(0, vec![0; root.circuits[&0].num_inputs], &[0]);
    let new_circuit0 = ctx.new_circuit0.take().unwrap();
    let mut new_circuits = HashMap::new();
    new_circuits.insert(0, new_circuit0);
    for (id, cir) in ctx.new_circuits.vec().iter().enumerate().skip(1) {
        new_circuits.insert(id, cir.clone());
    }
    IrRootCircuit {
        circuits: new_circuits,
        num_public_inputs: root.num_public_inputs,
        expected_num_output_zeroes: root.expected_num_output_zeroes
            + C::ENABLE_RANDOM_COMBINATION as usize,
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::config::{GF2Config, M31Config};
    use crate::circuit::ir::common::rand_gen::{RandomCircuitConfig, RandomRange};
    use crate::field::M31;

    use super::*;
    use Instruction::*;

    #[test]
    fn simple1() {
        let mut root = IrRootCircuit::<M31Config>::default();
        root.circuits.insert(
            0,
            IrCircuit {
                instructions: vec![SubCircuitCall {
                    sub_circuit_id: 1,
                    inputs: vec![1],
                    num_outputs: 2,
                }],
                constraints: vec![3],
                outputs: vec![2, 3],
                num_inputs: 1,
            },
        );
        root.circuits.insert(
            1,
            IrCircuit {
                instructions: vec![
                    InternalVariable {
                        expr: Expression::from_terms(vec![Term::new_linear(M31::from(2), 1)]),
                    },
                    InternalVariable {
                        expr: Expression::from_terms(vec![Term::new_linear(M31::from(3), 2)]),
                    },
                    InternalVariable {
                        expr: Expression::from_terms(vec![Term::new_linear(M31::from(5), 3)]),
                    },
                ],
                constraints: vec![],
                outputs: vec![2, 4],
                num_inputs: 1,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let new_root = split_to_single_layer(&root);
        assert_eq!(new_root.validate(), Ok(()));
        let inputs = vec![M31::from(1)];
        let (out, cond) = root.eval_unsafe(inputs.clone());
        let (out2, cond2) = new_root.eval_unsafe(inputs.clone());
        assert_eq!(out, out2);
        assert_eq!(cond, cond2);
        let inputs = vec![M31::from(0)];
        let (out, cond) = root.eval_unsafe(inputs.clone());
        let (out2, cond2) = new_root.eval_unsafe(inputs.clone());
        assert_eq!(out, out2);
        assert_eq!(cond, cond2);
    }

    fn rand_test<C: Config>() {
        let mut config = RandomCircuitConfig {
            seed: 0,
            num_circuits: RandomRange { min: 1, max: 10 },
            num_inputs: RandomRange { min: 1, max: 10 },
            num_instructions: RandomRange { min: 1, max: 10 },
            num_constraints: RandomRange { min: 0, max: 10 },
            num_outputs: RandomRange { min: 1, max: 10 },
            num_terms: RandomRange { min: 1, max: 5 },
            sub_circuit_prob: 0.5,
        };
        for i in 0..1000 {
            config.seed = i;
            let mut root = IrRootCircuit::<C>::random(&config);
            if !C::ENABLE_RANDOM_COMBINATION {
                root = root.export_constraints();
            }
            assert_eq!(root.validate(), Ok(()));
            let new_root = split_to_single_layer(&root);
            assert_eq!(new_root.validate(), Ok(()));
            let input: Vec<C::CircuitField> = (0..root.input_size())
                .map(|_| C::CircuitField::random_unsafe(&mut rand::thread_rng()))
                .collect();
            let (out, cond) = root.eval_unsafe(input.clone());
            let (out2, cond2) = new_root.eval_unsafe(input.clone());
            assert_eq!(out, out2);
            assert_eq!(cond, cond2);
        }
    }

    #[test]
    fn rand_m31() {
        rand_test::<M31Config>();
    }

    #[test]
    fn rand_gf2() {
        rand_test::<GF2Config>();
    }
}
