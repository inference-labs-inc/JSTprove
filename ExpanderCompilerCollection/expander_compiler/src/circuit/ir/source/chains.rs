use expr::{LinComb, LinCombTerm};

use crate::circuit::ir::common::Instruction as _;

use super::*;

impl<C: Config> Circuit<C> {
    pub fn detect_chains(&mut self) {
        let mut var_insn_id = vec![self.instructions.len(); self.num_inputs + 1];
        let mut is_add = vec![false; self.instructions.len() + 1];
        let mut is_mul = vec![false; self.instructions.len() + 1];
        let mut insn_ref_count = vec![0; self.instructions.len() + 1];
        for (i, insn) in self.instructions.iter().enumerate() {
            for x in insn.inputs().iter() {
                insn_ref_count[var_insn_id[*x]] += 1;
            }
            for _ in 0..insn.num_outputs() {
                var_insn_id.push(i);
            }
            match insn {
                Instruction::LinComb(_) => {
                    is_add[i] = true;
                }
                Instruction::Mul(_) => {
                    is_mul[i] = true;
                }
                _ => {}
            }
        }
        for x in self.outputs.iter() {
            insn_ref_count[var_insn_id[*x]] += 1;
        }
        for x in self.constraints.iter() {
            insn_ref_count[var_insn_id[x.var]] += 1;
        }
        for i in 0..self.instructions.len() {
            if !is_add[i] {
                continue;
            }
            let lc = if let Instruction::LinComb(lc) = &self.instructions[i] {
                let mut flag = false;
                for x in lc.terms.iter() {
                    if insn_ref_count[var_insn_id[x.var]] == 1 {
                        flag = true;
                        break;
                    }
                }
                if !flag {
                    continue;
                }
                lc.clone()
            } else {
                unreachable!()
            };
            let mut lcs = vec![];
            let mut rem_terms = vec![];
            let mut constant = lc.constant;
            for x in lc.terms {
                if is_add[var_insn_id[x.var]] && insn_ref_count[var_insn_id[x.var]] == 1 {
                    let x_insn = &mut self.instructions[var_insn_id[x.var]];
                    let x_lc = if let Instruction::LinComb(x_lc) = x_insn {
                        x_lc
                    } else {
                        unreachable!()
                    };
                    if !x_lc.constant.is_zero() {
                        constant += x_lc.constant * x.coef;
                    }
                    if x.coef == C::CircuitField::one() {
                        lcs.push(std::mem::take(&mut x_lc.terms));
                    } else {
                        lcs.push(
                            x_lc.terms
                                .iter()
                                .map(|y| LinCombTerm {
                                    var: y.var,
                                    coef: x.coef * y.coef,
                                })
                                .collect(),
                        );
                        std::mem::take(&mut x_lc.terms);
                    }
                } else {
                    rem_terms.push(x);
                }
            }
            let mut terms = rem_terms;
            for mut cur_terms in lcs {
                if terms.len() < cur_terms.len() {
                    std::mem::swap(&mut terms, &mut cur_terms);
                }
                terms.append(&mut cur_terms);
            }
            self.instructions[i] = Instruction::LinComb(LinComb { terms, constant });
        }
        for i in 0..self.instructions.len() {
            if !is_mul[i] {
                continue;
            }
            let vars = if let Instruction::Mul(vars) = &self.instructions[i] {
                let mut flag = false;
                for x in vars.iter() {
                    if insn_ref_count[var_insn_id[*x]] == 1 {
                        flag = true;
                        break;
                    }
                }
                if !flag {
                    continue;
                }
                vars.clone()
            } else {
                unreachable!()
            };
            let mut var_vecs = vec![];
            let mut rem_vars = vec![];
            for x in vars {
                if is_mul[var_insn_id[x]] && insn_ref_count[var_insn_id[x]] == 1 {
                    let x_insn = &mut self.instructions[var_insn_id[x]];
                    let x_vars = if let Instruction::Mul(x_vars) = x_insn {
                        x_vars
                    } else {
                        unreachable!()
                    };
                    var_vecs.push(std::mem::take(x_vars));
                } else {
                    rem_vars.push(x);
                }
            }
            let mut vars = rem_vars;
            for mut cur_vars in var_vecs {
                if vars.len() < cur_vars.len() {
                    std::mem::swap(&mut vars, &mut cur_vars);
                }
                vars.append(&mut cur_vars);
            }
            self.instructions[i] = Instruction::Mul(vars);
        }
    }
}

impl<C: Config> RootCircuit<C> {
    pub fn detect_chains(&mut self) {
        for (_, circuit) in self.circuits.iter_mut() {
            circuit.detect_chains();
        }
    }
}
