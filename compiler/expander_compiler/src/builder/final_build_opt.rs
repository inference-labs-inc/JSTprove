use std::collections::{BinaryHeap, HashMap};

use crate::{
    circuit::{
        config::Config,
        costs::{cost_of_compress, cost_of_multiply, cost_of_possible_references, cost_of_relay},
        ir::{
            common::Instruction,
            dest::{
                CircuitRelaxed as OutCircuit, Instruction as OutInstruction,
                RootCircuitRelaxed as OutRootCircuit,
            },
            expr::{Expression, LinComb, Term, VarSpec},
            hint_less::{
                Circuit as InCircuit, Instruction as InInstruction, RootCircuit as InRootCircuit,
            },
        },
        layered::Coef,
    },
    field::{Field, FieldArith},
    frontend::CircuitField,
    utils::{error::Error, pool::Pool},
};

use super::basic::LinMeta;

const COMPRESS_THRESHOLD: usize = 64;

struct RootBuilder<C: Config> {
    builders: HashMap<usize, Builder<C>>,
    out_circuits: HashMap<usize, OutCircuit<C>>,
}

struct Builder<C: Config> {
    // in_var ref counts
    in_var_ref_counts: Vec<InVarRefCounts>,

    // in_var mapped to expression of mid_vars
    in_var_exprs: Vec<Expression<C>>,

    // pool of stripped mid_vars
    // for internal variables, the expression is actual expression
    // for in_vars, the expression is a fake expression with only one term
    stripped_mid_vars: Pool<MidVarKey<C>>,
    // mid_var i = k*(expr)+b
    mid_var_coefs: Vec<MidVarCoef<C>>,
    // expected layer of mid_var, input==0
    mid_var_layer: Vec<usize>,

    // (effective mid_var id, insn)
    out_insns: Vec<(usize, OutInstruction<C>)>,

    output_layer: usize,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct MidVarKey<C: Config> {
    expr: Expression<C>,
    is_force_single: bool,
}

#[derive(Debug, Clone)]
struct MidVarCoef<C: Config> {
    k: CircuitField<C>,
    kinv: CircuitField<C>,
    b: CircuitField<C>,
}

#[derive(Debug, Clone, Default)]
struct InVarRefCounts {
    add: usize,
    mul: usize,
    single: usize,
}

impl<C: Config> Default for MidVarCoef<C> {
    fn default() -> Self {
        MidVarCoef {
            k: CircuitField::<C>::one(),
            kinv: CircuitField::<C>::one(),
            b: CircuitField::<C>::zero(),
        }
    }
}

impl<C: Config> Builder<C> {
    fn new() -> Self {
        let mut res = Builder {
            in_var_ref_counts: vec![InVarRefCounts::default()],
            in_var_exprs: vec![Expression::default()],
            stripped_mid_vars: Pool::new(),
            mid_var_coefs: vec![MidVarCoef::default()],
            mid_var_layer: vec![0],
            out_insns: Vec::new(),
            output_layer: 0,
        };
        res.stripped_mid_vars.add(&MidVarKey {
            expr: Expression::invalid(),
            is_force_single: false,
        });
        res
    }

    fn new_var(&mut self, layer: usize) -> usize {
        let id = self.stripped_mid_vars.len();
        assert_eq!(
            self.stripped_mid_vars.add(&MidVarKey {
                expr: Expression::new_linear(CircuitField::<C>::one(), id),
                is_force_single: false
            }),
            id
        );
        self.mid_var_coefs.push(MidVarCoef::default());
        self.mid_var_layer.push(layer);
        id
    }

    fn add_in_vars(&mut self, n: usize, layer: usize) {
        let start = self.stripped_mid_vars.len();
        for i in 0..n {
            self.new_var(layer);
            self.in_var_exprs
                .push(Expression::new_linear(CircuitField::<C>::one(), start + i));
        }
    }

    fn add_const(&mut self, c: CircuitField<C>) {
        self.in_var_exprs.push(Expression::new_const(c));
    }

    fn make_single(&mut self, expr: Expression<C>) -> Expression<C> {
        let (e, coef, constant) = strip_constants(&expr);
        if e.len() == 1 && e.degree() <= 1 {
            return expr;
        }
        let idx = self.stripped_mid_vars.add(&MidVarKey {
            expr: e.clone(),
            is_force_single: false,
        });
        if idx == self.mid_var_coefs.len() {
            self.mid_var_coefs.push(MidVarCoef {
                k: coef,
                kinv: coef.optimistic_inv().unwrap(),
                b: constant,
            });
            self.mid_var_layer.push(self.layer_of_expr(&e) + 1);
            return Expression::new_linear(CircuitField::<C>::one(), idx);
        }
        unstrip_constants_single(idx, coef, constant, &self.mid_var_coefs[idx])
    }

    fn try_make_single(&self, expr: Expression<C>) -> Expression<C> {
        let (e, coef, constant) = strip_constants(&expr);
        if e.len() == 1 && e.degree() <= 1 {
            return expr;
        }
        match self.stripped_mid_vars.try_get_idx(&MidVarKey {
            expr: e,
            is_force_single: false,
        }) {
            Some(idx) => unstrip_constants_single(idx, coef, constant, &self.mid_var_coefs[idx]),
            None => expr,
        }
    }

    fn make_really_single(&mut self, e: Expression<C>) -> usize {
        if e.len() == 1 && e.degree() == 1 && e[0].coef == CircuitField::<C>::one() {
            match e[0].vars {
                VarSpec::Linear(v) => return v,
                _ => unreachable!(),
            }
        }
        let (es, coef, constant) = strip_constants(&e);
        let idx = self.stripped_mid_vars.add(&MidVarKey {
            expr: es,
            is_force_single: false,
        });
        if idx == self.mid_var_coefs.len() {
            self.mid_var_coefs.push(MidVarCoef {
                k: coef,
                kinv: coef.optimistic_inv().unwrap(),
                b: constant,
            });
            self.mid_var_layer.push(self.layer_of_expr(&e) + 1);
            return idx;
        }
        if coef == self.mid_var_coefs[idx].k && constant == self.mid_var_coefs[idx].b {
            return idx;
        }
        let e = self.make_single(e);
        let idx = self.stripped_mid_vars.add(&MidVarKey {
            expr: e.clone(),
            is_force_single: true,
        });
        if idx == self.mid_var_coefs.len() {
            self.mid_var_coefs.push(MidVarCoef {
                k: CircuitField::<C>::one(),
                kinv: CircuitField::<C>::one(),
                b: CircuitField::<C>::zero(),
            });
            self.mid_var_layer.push(self.layer_of_expr(&e) + 1);
        }
        idx
    }

    fn layer_of_varspec(&self, vs: &VarSpec) -> usize {
        match vs {
            VarSpec::Linear(v) => self.mid_var_layer[*v],
            VarSpec::Quad(x, y) => self.mid_var_layer[*x].max(self.mid_var_layer[*y]),
            VarSpec::Const => 0,
            VarSpec::Custom { inputs, .. } => {
                let mut max_layer = 0;
                for input in inputs.iter() {
                    max_layer = max_layer.max(self.mid_var_layer[*input]);
                }
                max_layer
            }
            VarSpec::RandomLinear(_) => panic!("unexpected situation"),
        }
    }

    fn layer_of_expr(&self, e: &Expression<C>) -> usize {
        e.iter()
            .map(|term| self.layer_of_varspec(&term.vars))
            .max()
            .unwrap()
    }

    fn lin_comb(&mut self, lcs: &LinComb<C>) -> Expression<C> {
        let mut vars: Vec<Expression<C>> = lcs
            .terms
            .iter()
            .map(|lc| self.in_var_exprs[lc.var].clone())
            .collect();
        if !lcs.constant.is_zero() {
            vars.push(Expression::new_const(lcs.constant));
        }
        self.lin_comb_inner(vars, |i| {
            if i < lcs.terms.len() {
                lcs.terms[i].coef
            } else {
                CircuitField::<C>::one()
            }
        })
    }

    fn lin_comb_inner<F: Fn(usize) -> CircuitField<C>>(
        &mut self,
        mut vars: Vec<Expression<C>>,
        var_coef: F,
    ) -> Expression<C> {
        if vars.is_empty() {
            return Expression::default();
        }
        let vars: Vec<Expression<C>> = vars.drain(..).map(|e| self.try_make_single(e)).collect();
        if vars.len() == 1 {
            return self.layered_add(vars[0].mul_constant(var_coef(0)).to_terms());
        }
        let mut heap: BinaryHeap<LinMeta> = BinaryHeap::new();
        for (l_id, var) in vars.iter().enumerate() {
            if var_coef(l_id).is_zero() {
                continue;
            }
            heap.push(LinMeta {
                l_id,
                t_id: 0,
                vars: var[0].vars.clone(),
            });
        }
        let mut res: Vec<Term<C>> = Vec::new();
        while let Some(lm) = heap.peek() {
            let l_id = lm.l_id;
            let t_id = lm.t_id;
            let var = &vars[l_id];
            if t_id == var.len() - 1 {
                heap.pop();
            } else {
                let mut lm = heap.peek_mut().unwrap();
                lm.t_id = t_id + 1;
                lm.vars = var[t_id + 1].vars.clone();
            }
            let t = &var[t_id];
            let new_coef = t.coef * var_coef(l_id);
            if new_coef.is_zero() {
                continue;
            }
            if !res.is_empty() && res.last().unwrap().vars == t.vars {
                res.last_mut().unwrap().coef += new_coef;
                if res.last().unwrap().coef.is_zero() {
                    res.pop();
                }
            } else {
                res.push(t.clone());
                res.last_mut().unwrap().coef = new_coef;
            }
        }
        if res.is_empty() {
            Expression::default()
        } else {
            //Expression::from_terms_sorted(res)
            self.layered_add(res)
        }
    }

    fn layered_add(&mut self, mut terms: Vec<Term<C>>) -> Expression<C> {
        if terms.len() <= 1 {
            return Expression::from_terms_sorted(terms);
        }
        let min_layer = terms
            .iter()
            .map(|term| self.layer_of_varspec(&term.vars))
            .min()
            .unwrap();
        let max_layer = terms
            .iter()
            .map(|term| self.layer_of_varspec(&term.vars))
            .max()
            .unwrap();
        if min_layer == max_layer {
            return Expression::from_terms_sorted(terms);
        }
        terms.sort_by(|a, b| {
            let la = self.layer_of_varspec(&a.vars);
            let lb = self.layer_of_varspec(&b.vars);
            if la != lb {
                la.cmp(&lb)
            } else {
                a.vars.cmp(&b.vars)
            }
        });
        let mut cur_terms = Vec::new();
        let mut last_layer = -1;
        for term in terms.iter() {
            let layer = self.layer_of_varspec(&term.vars) as isize;
            if layer != last_layer && last_layer != -1 {
                cur_terms = self
                    .make_single(Expression::from_terms(cur_terms))
                    .to_terms();
            }
            cur_terms.push(term.clone());
            last_layer = layer;
        }
        Expression::from_terms(cur_terms)
    }

    fn cmp_expr_for_mul(&self, a: &Expression<C>, b: &Expression<C>) -> std::cmp::Ordering {
        let la = self.layer_of_expr(a);
        let lb = self.layer_of_expr(b);
        if la != lb {
            return la.cmp(&lb);
        }
        let la = a.len();
        let lb = b.len();
        if la != lb {
            return la.cmp(&lb);
        }
        a.cmp(b)
    }

    fn mul_vec(&mut self, vars: &[usize]) -> Expression<C> {
        use crate::utils::heap::{pop, push};
        assert!(vars.len() >= 2);
        let mut exprs: Vec<Expression<C>> = vars
            .iter()
            .map(|&v| self.try_make_single(self.in_var_exprs[v].clone()))
            .collect();
        let mut exprs_pos_heap: Vec<usize> = vec![];
        let mut next_push_pos = 0;
        loop {
            while next_push_pos != exprs.len() {
                push(&mut exprs_pos_heap, next_push_pos, |a, b| {
                    self.cmp_expr_for_mul(&exprs[a], &exprs[b])
                });
                next_push_pos += 1;
            }
            if exprs_pos_heap.len() == 1 {
                break;
            }
            let pos1 = pop(&mut exprs_pos_heap, |a, b| {
                self.cmp_expr_for_mul(&exprs[a], &exprs[b])
            })
            .unwrap();
            let pos2 = pop(&mut exprs_pos_heap, |a, b| {
                self.cmp_expr_for_mul(&exprs[a], &exprs[b])
            })
            .unwrap();
            let mut expr1 = std::mem::take(&mut exprs[pos1]);
            let mut expr2 = std::mem::take(&mut exprs[pos2]);
            if expr1.len() > expr2.len() {
                std::mem::swap(&mut expr1, &mut expr2);
            }
            let deg1 = expr1.degree();
            let deg2 = expr2.degree();
            if deg1 == 0 {
                exprs.push(expr2.mul_constant(expr1.constant_value().unwrap()));
                continue;
            }
            if deg2 == 0 {
                exprs.push(expr1.mul_constant(expr2.constant_value().unwrap()));
                continue;
            }
            if deg1 == 2 {
                if deg2 == 2 {
                    exprs.push(self.make_single(expr2));
                } else {
                    exprs.push(expr2);
                }
                exprs.push(self.make_single(expr1));
                continue;
            }
            if deg2 == 2 {
                exprs.push(expr1);
                exprs.push(self.make_single(expr2));
                continue;
            }
            let dcnt1 = expr1.count_of_degrees();
            let dcnt2 = expr2.count_of_degrees();
            assert!(dcnt1[2] == 0);
            assert!(dcnt2[2] == 0);
            let v1layer = self.layer_of_expr(&expr1);
            let v2layer = self.layer_of_expr(&expr2);
            let mut cost_direct = cost_of_multiply::<C>(dcnt1[0], dcnt1[1], dcnt2[0], dcnt2[1]);
            let mut cost_compress_v1 =
                cost_of_multiply::<C>(0, 1, dcnt2[0], dcnt2[1]) + cost_of_compress::<C>(&dcnt1);
            let mut cost_compress_v2 =
                cost_of_multiply::<C>(dcnt1[0], dcnt1[1], 0, 1) + cost_of_compress::<C>(&dcnt2);
            let cost_compress_both = cost_of_multiply::<C>(0, 1, 0, 1)
                + cost_of_compress::<C>(&dcnt1)
                + cost_of_compress::<C>(&dcnt2);
            let (compress_some, compress_1) = if v1layer == v2layer {
                (
                    cost_compress_v1
                        .min(cost_compress_v2)
                        .min(cost_compress_both)
                        < cost_direct,
                    cost_compress_v1 < cost_compress_v2.min(cost_compress_both),
                )
            } else {
                cost_direct += cost_of_relay::<C>(v1layer, v2layer);
                cost_compress_v1 += cost_of_relay::<C>(v1layer + 1, v2layer);
                cost_compress_v2 += cost_of_relay::<C>(v1layer, v2layer + 1);
                if cost_compress_v1 < cost_direct {
                    (expr1.len() > 2, true)
                } else if cost_compress_v2 < cost_direct {
                    (expr2.len() > 2, false)
                } else {
                    (false, false)
                }
            };
            if compress_some {
                if compress_1 {
                    exprs.push(self.make_single(expr1));
                    exprs.push(expr2);
                } else {
                    exprs.push(expr1);
                    exprs.push(self.make_single(expr2));
                }
                continue;
            }
            let mut vars: Vec<Expression<C>> = Vec::new();
            for x1 in expr1.iter() {
                vars.push(Expression::from_terms(
                    expr2.iter().map(|x2| x1.mul(x2)).collect(),
                ));
            }
            exprs.push(self.lin_comb_inner(vars, |_| CircuitField::<C>::one()));
        }
        let final_pos = exprs_pos_heap.pop().unwrap();
        exprs.swap_remove(final_pos)
    }

    fn add_and_check_if_should_make_single(&mut self, e: Expression<C>) {
        let ref_count = self.in_var_ref_counts[self.in_var_exprs.len()].clone();
        let degree_count = e.count_of_degrees();
        let mut should_compress = ref_count.single > 0;
        should_compress |= degree_count.iter().sum::<usize>() > COMPRESS_THRESHOLD;
        let cost_no_compress =
            cost_of_possible_references::<C>(&degree_count, ref_count.add, ref_count.mul);
        let cost_compress = cost_of_compress::<C>(&degree_count)
            + cost_of_possible_references::<C>(&[0, 1, 0], ref_count.add, ref_count.mul);
        should_compress |= cost_compress < cost_no_compress;
        should_compress &= e.degree() > 0;
        if should_compress {
            // Currently, this don't consider the cost of relay, so it's not good in some cases
            // TODO: fix this
            let es = self.make_single(e);
            self.in_var_exprs.push(es);
        } else {
            self.in_var_exprs.push(e);
        }
    }
}

fn strip_constants<C: Config>(
    expr: &Expression<C>,
) -> (Expression<C>, CircuitField<C>, CircuitField<C>) {
    let mut e = Vec::new();
    let mut cst = CircuitField::<C>::zero();
    for term in expr.iter() {
        if term.vars == VarSpec::Const {
            cst = term.coef;
        } else {
            e.push(term.clone());
        }
    }
    if e.is_empty() {
        return (Expression::default(), CircuitField::<C>::one(), cst);
    }
    let v = e[0].coef;
    let vi = v.optimistic_inv().unwrap();
    for term in e.iter_mut() {
        term.coef *= vi;
    }
    (Expression::from_terms_sorted(e), v, cst)
}

fn unstrip_constants_single<C: Config>(
    vid: usize,
    coef: CircuitField<C>,
    constant: CircuitField<C>,
    mid_var_coef: &MidVarCoef<C>,
) -> Expression<C> {
    assert_ne!(vid, 0);
    // u=k1x+b1, v=k2x+b2
    // x=(u-b1)*k1inv
    // v=k2*(u-b1)*k1inv+b2
    let new_coef = mid_var_coef.kinv * coef;
    let new_constant = constant - mid_var_coef.b * new_coef;
    let mut e = vec![Term::new_linear(new_coef, vid)];
    if !new_constant.is_zero() {
        e.push(Term::new_const(new_constant));
    }
    Expression::from_terms(e)
}

fn process_circuit<C: Config>(
    root: &mut RootBuilder<C>,
    circuit: &InCircuit<C>,
) -> Result<(OutCircuit<C>, Builder<C>), Error> {
    let mut builder = Builder::new();

    // initialize in_var_ref_counts
    for _ in 0..circuit.get_num_inputs_all() {
        builder.in_var_ref_counts.push(InVarRefCounts::default());
    }
    for insn in circuit.instructions.iter() {
        match insn {
            InInstruction::LinComb(lc) => {
                for term in lc.terms.iter() {
                    builder.in_var_ref_counts[term.var].add += 1;
                }
            }
            InInstruction::Mul(vars) => {
                for var in vars {
                    builder.in_var_ref_counts[*var].mul += 1;
                }
            }
            InInstruction::ConstantLike(_) => {}
            InInstruction::SubCircuitCall { inputs, .. } => {
                for input in inputs {
                    builder.in_var_ref_counts[*input].single += 1;
                }
            }
            InInstruction::CustomGate { inputs, .. } => {
                for input in inputs {
                    builder.in_var_ref_counts[*input].single += 1;
                }
            }
        }
        for _ in 0..insn.num_outputs() {
            builder.in_var_ref_counts.push(InVarRefCounts::default());
        }
    }
    for out in circuit.outputs.iter() {
        builder.in_var_ref_counts[*out].single += 1;
    }
    for con in circuit.constraints.iter() {
        builder.in_var_ref_counts[*con].single += 1;
    }

    // add inputs
    builder.add_in_vars(circuit.get_num_inputs_all(), 0);

    // process insns
    for insn in circuit.instructions.iter() {
        match insn {
            InInstruction::LinComb(lc) => {
                let e = builder.lin_comb(lc);
                builder.add_and_check_if_should_make_single(e);
            }
            InInstruction::Mul(vars) => {
                let e = builder.mul_vec(vars);
                builder.add_and_check_if_should_make_single(e);
            }
            InInstruction::ConstantLike(coef) => match coef {
                Coef::Constant(c) => {
                    builder.add_const(*c); // TODO: this might not work
                }
                Coef::Random => {
                    builder.out_insns.push((
                        builder.stripped_mid_vars.len(),
                        OutInstruction::ConstantLike {
                            value: Coef::Random,
                        },
                    ));
                    builder.add_in_vars(1, 1);
                }
                Coef::PublicInput(i) => {
                    builder.out_insns.push((
                        builder.stripped_mid_vars.len(),
                        OutInstruction::ConstantLike {
                            value: Coef::PublicInput(*i),
                        },
                    ));
                    builder.add_in_vars(1, 1);
                }
            },
            InInstruction::SubCircuitCall {
                sub_circuit_id,
                inputs,
                num_outputs,
            } => {
                let sub_builder = root.builders.get(sub_circuit_id).unwrap();
                let single_inputs: Vec<usize> = inputs
                    .iter()
                    .map(|&var| builder.make_really_single(builder.in_var_exprs[var].clone()))
                    .collect();
                let max_input_layer = single_inputs
                    .iter()
                    .map(|&var| builder.mid_var_layer[var])
                    .max()
                    .unwrap_or(1);
                builder.out_insns.push((
                    builder.stripped_mid_vars.len(),
                    OutInstruction::SubCircuitCall {
                        sub_circuit_id: *sub_circuit_id,
                        inputs: single_inputs,
                        num_outputs: *num_outputs,
                    },
                ));
                builder.add_in_vars(*num_outputs, max_input_layer + sub_builder.output_layer);
            }
            InInstruction::CustomGate { gate_type, inputs } => {
                let single_inputs: Vec<usize> = inputs
                    .iter()
                    .map(|&var| builder.make_really_single(builder.in_var_exprs[var].clone()))
                    .collect();
                builder.add_and_check_if_should_make_single(Expression::new_custom(
                    CircuitField::<C>::one(),
                    *gate_type,
                    single_inputs,
                ));
            }
        }
    }

    // constraints and outputs
    let mut constraints: Vec<usize> = Vec::new();
    let mut cons_keys = circuit.constraints.clone();
    cons_keys.sort();
    for con in cons_keys.iter() {
        let e = builder.in_var_exprs[*con].clone();
        if let Some(c) = e.constant_value() {
            if c.is_zero() {
                continue;
            }
            return Err(Error::UserError("constraint is not zero".to_string()));
        }
        constraints.push(builder.make_really_single(e));
    }
    let outputs: Vec<usize> = circuit
        .outputs
        .iter()
        .map(|&o| builder.make_really_single(builder.in_var_exprs[o].clone()))
        .collect();
    builder.output_layer = outputs
        .iter()
        .map(|&var| builder.mid_var_layer[var])
        .max()
        .unwrap_or(0);

    let mut instructions: Vec<OutInstruction<C>> = Vec::new();
    let mut oinsn_id = 0;
    for mid_var_id in circuit.get_num_inputs_all() + 1..builder.stripped_mid_vars.len() {
        while oinsn_id < builder.out_insns.len() && builder.out_insns[oinsn_id].0 == mid_var_id {
            instructions.push(builder.out_insns[oinsn_id].1.clone());
            oinsn_id += 1;
        }
        let mvk = builder.stripped_mid_vars.get(mid_var_id);
        let non_iv = mvk.expr == Expression::new_linear(CircuitField::<C>::one(), mid_var_id);
        if non_iv {
            continue;
        }
        let e2 = mvk.expr.mul_constant(builder.mid_var_coefs[mid_var_id].k);
        let constant = builder.mid_var_coefs[mid_var_id].b;
        let e3 = if constant.is_zero() {
            e2
        } else {
            let mut terms = e2.to_terms();
            terms.push(Term::new_const(constant));
            Expression::from_terms(terms)
        };
        instructions.push(OutInstruction::InternalVariable { expr: e3 });
    }
    while oinsn_id < builder.out_insns.len() {
        instructions.push(builder.out_insns[oinsn_id].1.clone());
        oinsn_id += 1;
    }

    Ok((
        OutCircuit {
            constraints,
            outputs,
            instructions,
            num_inputs: circuit.num_inputs,
        },
        builder,
    ))
}

pub fn process<C: Config>(rc: &InRootCircuit<C>) -> Result<OutRootCircuit<C>, Error> {
    let mut root: RootBuilder<C> = RootBuilder {
        builders: HashMap::new(),
        out_circuits: HashMap::new(),
    };
    let order = rc.topo_order();
    for &circuit_id in order.iter().rev() {
        let (new_circuit, final_builder) =
            process_circuit(&mut root, rc.circuits.get(&circuit_id).unwrap())?;
        root.out_circuits.insert(circuit_id, new_circuit);
        root.builders.insert(circuit_id, final_builder);
    }
    Ok(OutRootCircuit {
        num_public_inputs: rc.num_public_inputs,
        expected_num_output_zeroes: rc.expected_num_output_zeroes,
        circuits: root.out_circuits,
    })
}

#[cfg(test)]
mod tests {
    use std::vec;

    use mersenne31::M31;

    use crate::field::FieldArith;
    use crate::frontend::M31Config as C;
    use crate::{
        circuit::ir::{
            self,
            common::rand_gen::*,
            expr::{Expression, Term},
        },
        utils::error::Error,
    };

    type CField = M31;

    #[test]
    fn simple_add() {
        let mut root = super::InRootCircuit::<C>::default();
        root.circuits.insert(
            0,
            super::InCircuit::<C> {
                instructions: vec![ir::hint_less::Instruction::LinComb(ir::expr::LinComb {
                    terms: vec![
                        ir::expr::LinCombTerm {
                            coef: CField::one(),
                            var: 1,
                        },
                        ir::expr::LinCombTerm {
                            coef: CField::from(2),
                            var: 2,
                        },
                    ],
                    constant: CField::from(3),
                })],
                constraints: vec![3],
                outputs: vec![],
                num_inputs: 2,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        let c0 = &root_processed.circuits[&0];
        assert_eq!(
            c0.instructions[0],
            ir::dest::Instruction::InternalVariable {
                expr: Expression::from_terms(vec![
                    Term::new_linear(CField::one(), 1),
                    Term::new_linear(CField::from(2), 2),
                    Term::new_const(CField::from(3))
                ])
            }
        );
        assert_eq!(c0.constraints, vec![3]);
    }

    #[test]
    fn simple_mul() {
        let mut root = super::InRootCircuit::<C>::default();
        root.circuits.insert(
            0,
            super::InCircuit::<C> {
                instructions: vec![ir::hint_less::Instruction::Mul(vec![1, 2, 3, 4])],
                constraints: vec![5],
                outputs: vec![5],
                num_inputs: 4,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        let root_fin = root_processed.solve_duplicates();
        assert_eq!(root_fin.validate(), Ok(()));
        let (out, _) = root_fin.eval_unsafe(vec![
            CField::from(2),
            CField::from(3),
            CField::from(5),
            CField::from(7),
        ]);
        assert_eq!(out, vec![CField::from(2 * 3 * 5 * 7)]);
    }

    #[test]
    fn random_circuits_1() {
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
        for i in 0..3000 {
            config.seed = i + 100000;
            let root = super::InRootCircuit::<C>::random(&config);
            assert_eq!(root.validate(), Ok(()));
            let (root, _) = root.remove_unreachable();
            match super::process(&root) {
                Ok(root_processed) => {
                    assert_eq!(root_processed.validate(), Ok(()));
                    assert_eq!(root.input_size(), root_processed.input_size());
                    for _ in 0..5 {
                        let inputs: Vec<CField> = (0..root.input_size())
                            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                            .collect();
                        let e1 = root.eval_unsafe_with_errors(inputs.clone());
                        let e2 = root_processed.eval_unsafe_with_errors(inputs);
                        if e1.is_ok() {
                            assert_eq!(e2, e1);
                        }
                    }
                }
                Err(e) => match e {
                    Error::UserError(_) => {}
                    Error::InternalError(e) => {
                        panic!("{:?}", e);
                    }
                },
            }
        }
    }

    #[test]
    fn random_circuits_2() {
        let mut config = RandomCircuitConfig {
            seed: 0,
            num_circuits: RandomRange { min: 1, max: 20 },
            num_inputs: RandomRange { min: 1, max: 3 },
            num_instructions: RandomRange { min: 30, max: 50 },
            num_constraints: RandomRange { min: 0, max: 5 },
            num_outputs: RandomRange { min: 1, max: 3 },
            num_terms: RandomRange { min: 1, max: 5 },
            sub_circuit_prob: 0.05,
        };
        for i in 0..1000 {
            config.seed = i + 200000;
            let root = super::InRootCircuit::<C>::random(&config);
            assert_eq!(root.validate(), Ok(()));
            match super::process(&root) {
                Ok(root_processed) => {
                    assert_eq!(root_processed.validate(), Ok(()));
                    assert_eq!(root.input_size(), root_processed.input_size());
                    for _ in 0..5 {
                        let inputs: Vec<CField> = (0..root.input_size())
                            .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                            .collect();
                        let e1 = root.eval_unsafe_with_errors(inputs.clone());
                        let e2 = root_processed.eval_unsafe_with_errors(inputs);
                        if e1.is_ok() {
                            assert_eq!(e2, e1);
                        }
                    }
                }
                Err(e) => match e {
                    Error::UserError(_) => {}
                    Error::InternalError(e) => {
                        panic!("{:?}", e);
                    }
                },
            }
        }
    }

    #[test]
    fn large_add() {
        let mut root = super::InRootCircuit::<C>::default();
        let terms = (1..=100000)
            .map(|i| ir::expr::LinCombTerm {
                coef: CField::one(),
                var: i,
            })
            .collect();
        let lc = ir::expr::LinComb {
            terms,
            constant: CField::one(),
        };
        root.circuits.insert(
            0,
            super::InCircuit::<C> {
                instructions: vec![super::InInstruction::<C>::LinComb(lc.clone())],
                constraints: vec![100001],
                outputs: vec![],
                num_inputs: 100000,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        match &root_processed.circuits[&0].instructions[0] {
            ir::dest::Instruction::InternalVariable { expr } => {
                assert_eq!(expr.len(), 100001);
            }
            _ => panic!(),
        }
        let inputs: Vec<CField> = (1..=100000).map(CField::from).collect();
        let (out, ok) = root.eval_unsafe(inputs.clone());
        let (out2, ok2) = root_processed.eval_unsafe(inputs);
        assert_eq!(out, out2);
        assert_eq!(ok, ok2);
    }

    #[test]
    fn large_mul() {
        let mut root = super::InRootCircuit::<C>::default();
        let terms: Vec<usize> = (1..=100000).collect();
        root.circuits.insert(
            0,
            super::InCircuit::<C> {
                instructions: vec![super::InInstruction::<C>::Mul(terms.clone())],
                constraints: vec![100001],
                outputs: vec![],
                num_inputs: 100000,
            },
        );
        assert_eq!(root.validate(), Ok(()));
        let root_processed = super::process(&root).unwrap();
        assert_eq!(root_processed.validate(), Ok(()));
        let inputs: Vec<CField> = (1..=100000).map(CField::from).collect();
        let (out, ok) = root.eval_unsafe(inputs.clone());
        let (out2, ok2) = root_processed.eval_unsafe(inputs);
        assert_eq!(out, out2);
        assert_eq!(ok, ok2);
    }
}
