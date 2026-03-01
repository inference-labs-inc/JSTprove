use std::collections::HashMap;
use std::collections::HashSet;

use crate::circuit::{
    config::Config,
    ir::dest::{Circuit as IrCircuit, Instruction, RootCircuit as IrRootCircuit},
    ir::expr::Expression,
    layered::{Coef, InputType, Segment},
};
use crate::utils::pool::Pool;

use super::layer_layout::merge_layouts;
use super::layer_layout::{LayerLayout, LayerLayoutContext, LayerReq};
use super::CompileOptions;

pub struct CompileContext<'a, C: Config, I: InputType> {
    // the root circuit
    pub rc: &'a IrRootCircuit<C>,

    // for each circuit ir, we need a context to store some intermediate information
    pub circuits: HashMap<usize, IrContext<'a, C>>,

    // topo-sorted order
    pub order: Vec<usize>,

    // all generated layer layouts
    pub layer_layout_pool: Pool<LayerLayout>,
    pub layer_req_to_layout: HashMap<LayerReq, usize>,

    // compiled layered circuits
    pub compiled_circuits: Vec<Segment<C, I>>,
    pub conncected_wires: HashMap<Vec<usize>, usize>,

    // layout id of each layer
    pub layout_ids: Vec<usize>,
    // compiled circuit id of each layer
    pub layers: Vec<usize>,

    // input order
    pub input_order: Vec<usize>,

    pub root_has_constraints: bool,

    pub opts: CompileOptions,
}

pub struct IrContext<'a, C: Config> {
    pub circuit: &'a IrCircuit<C>,

    pub num_var: usize,          // number of variables in the circuit
    pub num_sub_circuits: usize, // number of sub circuits

    // for each variable, we need to find the min and max layer it should exist.
    // we assume input layer = 0, and output layer is at least 1
    // it includes only variables mentioned in instructions, so internal variables in sub circuits are ignored here.
    pub min_layer: Vec<usize>,
    pub max_layer: Vec<usize>,
    pub occured_layers: Vec<Vec<usize>>,
    pub output_layer: usize,

    // for each layer i, the minimum layer j that there exists gate j->i
    pub min_used_layer: Vec<usize>,

    pub output_order: HashMap<usize, usize>, // outputOrder[x] == y -> x is the y-th output

    pub sub_circuit_loc_map: HashMap<usize, usize>,
    pub sub_circuit_insn_ids: Vec<usize>,
    pub sub_circuit_insn_refs: Vec<SubCircuitInsn<'a>>,
    pub sub_circuit_start_layer: Vec<usize>,

    // combined constraints of each layer
    pub combined_constraints: Vec<Option<CombinedConstraint>>,

    pub internal_variable_expr: HashMap<usize, &'a Expression<C>>,
    pub constant_like_variables: HashMap<usize, Coef<C>>,

    // layer layout contexts
    pub lcs: Vec<LayerLayoutContext>,
}

#[derive(Default, Clone, Debug)]
pub struct CombinedConstraint {
    // id of this combined variable
    pub id: usize,
    // id of combined variables
    pub variables: Vec<usize>,
    // id of sub circuits (it will combine their combined constraints)
    // if a sub circuit has a combined output in this layer, it must be unique. So circuit id is sufficient.
    // = {x} means subCircuitInsnIds[x]
    pub sub_circuit_ids: Vec<usize>,
}

pub struct SubCircuitInsn<'a> {
    pub sub_circuit_id: usize,
    pub inputs: &'a Vec<usize>,
    pub outputs: Vec<usize>,
}

const EXTRA_PRE_ALLOC_SIZE: usize = 1000;

impl<'a, C: Config, I: InputType> CompileContext<'a, C, I> {
    pub fn compile(&mut self) {
        // 1. do a toposort of the circuits
        self.dfs_topo_sort(0);

        // 2. compute min and max layers for each circuit
        for id in self.order.clone() {
            self.compute_min_max_layers(id);
        }

        // 3. prepare layer layout contexts
        for id in self.order.clone() {
            self.prepare_layer_layout_context(id);
        }

        // 4. solve layer layout for root circuit (it also recursively solves all required sub-circuits)
        let mut layout_ids = Vec::with_capacity(self.circuits[&0].output_layer + 1);
        for i in 0..=self.circuits[&0].output_layer {
            layout_ids.push(self.solve_layer_layout(&LayerReq {
                circuit_id: 0,
                layer: i,
            }));
        }
        self.layout_ids = layout_ids;
        if !self.opts.allow_input_reorder {
            let layout_vec =
                merge_layouts(vec![], (0..self.circuits[&0].lcs[0].vars.len()).collect());
            let id = self.layer_layout_pool.add(&LayerLayout {
                circuit_id: 0,
                layer: 0,
                size: layout_vec.len(),
                inner: super::layer_layout::LayerLayoutInner::Dense {
                    placement: layout_vec,
                },
            });
            self.layout_ids[0] = id;
        }

        // 5. generate wires
        let layers = self.connect_wires(&self.layout_ids.clone());
        self.layers = layers;

        // 6. record the input order (used to generate witness)
        self.input_order = self.record_input_order();
    }

    fn dfs_topo_sort(&mut self, id: usize) {
        if self.circuits.contains_key(&id) {
            return;
        }

        let circuit: &IrCircuit<C> = &self.rc.circuits[&id];
        let mut nv = circuit.num_inputs + 1;
        let mut ns = 0;
        for insn in circuit.instructions.iter() {
            match insn {
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    num_outputs,
                    ..
                } => {
                    self.dfs_topo_sort(*sub_circuit_id);
                    ns += 1;
                    nv += num_outputs;
                }
                Instruction::InternalVariable { .. } => {
                    nv += 1;
                }
                Instruction::ConstantLike { .. } => {
                    nv += 1;
                }
            }
        }

        // when all children are done, we enqueue the current circuit
        self.order.push(id);
        self.circuits.insert(
            id,
            IrContext {
                circuit,
                num_var: nv,
                num_sub_circuits: ns,
                min_layer: Vec::new(),
                max_layer: Vec::new(),
                occured_layers: Vec::new(),
                output_layer: 0,
                min_used_layer: Vec::new(),
                output_order: HashMap::new(),
                sub_circuit_loc_map: HashMap::new(),
                sub_circuit_insn_ids: Vec::new(),
                sub_circuit_insn_refs: Vec::new(),
                sub_circuit_start_layer: Vec::new(),
                combined_constraints: Vec::new(),
                internal_variable_expr: HashMap::new(),
                constant_like_variables: HashMap::new(),
                lcs: Vec::new(),
            },
        );
    }

    fn compute_min_max_layers(&mut self, circuit_id: usize) {
        // variables
        // 0..nbVariable: normal variables
        // next nbSubCircuits terms: sub circuit virtual variables (in order to lower the number of edges)
        // next ? terms: random sum of constraints
        let mut ic = self.circuits.remove(&circuit_id).unwrap();
        let nv = ic.num_var;
        let ns = ic.num_sub_circuits;
        let mut n = nv + ns;
        let circuit = self.rc.circuits.get(&circuit_id).unwrap();

        let pre_alloc_size = n + EXTRA_PRE_ALLOC_SIZE.min(n);
        ic.min_layer = Vec::with_capacity(pre_alloc_size);
        ic.max_layer = Vec::with_capacity(pre_alloc_size);

        // layer advanced by each variable.
        // for normal variable, it's 1
        // for sub circuit virtual variable, it's output layer - 1
        let mut layer_advance: Vec<usize> = Vec::with_capacity(pre_alloc_size);

        for _ in 0..n {
            ic.min_layer.push(0);
            ic.max_layer.push(0);
            layer_advance.push(0);
        }

        let mut in_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut add_edge = |x: usize, y: usize| {
            in_edges[y].push(x);
            out_edges[x].push(y);
        };

        ic.sub_circuit_insn_ids = Vec::with_capacity(ns);
        ic.sub_circuit_insn_refs = Vec::with_capacity(ns);

        // get all input wires and build the graph
        // also computes the topo order
        let mut q0: Vec<usize> = Vec::with_capacity(pre_alloc_size);
        let mut q1: Vec<usize> = Vec::with_capacity(pre_alloc_size);
        for i in 1..=circuit.num_inputs {
            q0.push(i);
        }
        let mut cur_var_idx = circuit.num_inputs + 1;
        for (i, insn) in circuit.instructions.iter().enumerate() {
            match insn {
                Instruction::InternalVariable { expr } => {
                    let used_var: HashSet<usize> = expr.get_vars();
                    for x in used_var.iter() {
                        add_edge(*x, cur_var_idx);
                    }
                    q1.push(cur_var_idx);
                    layer_advance[cur_var_idx] = 1;
                    ic.min_layer[cur_var_idx] = 1;
                    ic.internal_variable_expr.insert(cur_var_idx, expr);
                    cur_var_idx += 1;
                }
                Instruction::SubCircuitCall {
                    sub_circuit_id,
                    inputs,
                    num_outputs,
                } => {
                    let k = ic.sub_circuit_insn_ids.len() + nv;
                    for x in inputs.iter() {
                        add_edge(*x, k);
                    }
                    q1.push(k);
                    layer_advance[k] = self.circuits[sub_circuit_id].output_layer - 1;
                    let mut outputs = Vec::new();
                    for _ in 0..*num_outputs {
                        add_edge(k, cur_var_idx);
                        q1.push(cur_var_idx);
                        layer_advance[cur_var_idx] = 1;
                        outputs.push(cur_var_idx);
                        cur_var_idx += 1;
                    }
                    ic.sub_circuit_insn_ids.push(i);
                    ic.sub_circuit_insn_refs.push(SubCircuitInsn {
                        sub_circuit_id: *sub_circuit_id,
                        inputs,
                        outputs,
                    });
                }
                Instruction::ConstantLike { value } => {
                    ic.min_layer[cur_var_idx] = 1;
                    q0.push(cur_var_idx);
                    ic.constant_like_variables.insert(cur_var_idx, *value);
                    cur_var_idx += 1;
                }
            }
        }
        assert_eq!(cur_var_idx, nv);
        q0.extend_from_slice(&q1); // the merged topo order
        let mut q = q0;

        // compute the min layer (depth) of each variable
        for x in q.iter().cloned() {
            for y in out_edges[x].iter().cloned() {
                ic.min_layer[y] = ic.min_layer[y].max(ic.min_layer[x] + layer_advance[y]);
            }
        }

        // compute sub circuit start layer
        ic.sub_circuit_start_layer = Vec::with_capacity(ns);
        for i in 0..ic.sub_circuit_insn_ids.len() {
            ic.sub_circuit_start_layer
                .push(ic.min_layer[nv + i] - layer_advance[nv + i]);
        }

        // compute output layer and order
        ic.output_layer = 0;
        for (i, x) in circuit.outputs.iter().cloned().enumerate() {
            ic.output_order.insert(x, i);
        }
        for i in 1..nv {
            ic.output_layer = ic.output_layer.max(ic.min_layer[i]);
        }

        // add combined constraints variables, and also update output layer
        let mut max_occured_layer = 0;
        for i in 0..ic.min_layer.len() {
            max_occured_layer = max_occured_layer.max(ic.min_layer[i]);
        }
        let mut cc: Vec<CombinedConstraint> =
            vec![CombinedConstraint::default(); max_occured_layer + 3];
        for x in circuit.constraints.iter().cloned() {
            let xl = ic.min_layer[x] + 1;
            cc[xl].variables.push(x);
        }
        for i in 0..ic.sub_circuit_insn_ids.len() {
            let sub_circuit = &self.circuits[&ic.sub_circuit_insn_refs[i].sub_circuit_id];
            ic.output_layer = ic
                .output_layer
                .max(ic.sub_circuit_start_layer[i] + sub_circuit.output_layer);
            for (j, x) in sub_circuit.combined_constraints.iter().enumerate() {
                if x.is_some() {
                    let sl = j + ic.sub_circuit_start_layer[i] + 1;
                    cc[sl].sub_circuit_ids.push(i);
                }
            }
        }

        // special check: if this is the root circuit, we will merge them into one
        if circuit_id == 0 {
            let mut first = 0;
            while first < cc.len()
                && (cc[first].variables.is_empty() && cc[first].sub_circuit_ids.is_empty())
            {
                first += 1;
            }
            if first != cc.len() {
                let last = max_occured_layer + 1;
                for (i, cc_i) in cc.iter_mut().enumerate().take(last + 1).skip(first + 1) {
                    cc_i.variables.push(i - 1 - first + n);
                }
                cc.truncate(last + 1);
                self.root_has_constraints = true;
            }
        }
        let mut cc: Vec<Option<CombinedConstraint>> = cc.into_iter().map(Some).collect();
        for (i, cc_i) in cc.iter_mut().enumerate() {
            let x = cc_i.as_mut().unwrap();
            if !x.variables.is_empty() || !x.sub_circuit_ids.is_empty() {
                x.id = n;
                if i + 1 > ic.output_layer {
                    ic.output_layer = if circuit_id == 0 { i } else { i + 1 };
                }
                ic.min_layer.push(i);
                ic.max_layer.push(i);
                layer_advance.push(1);
                out_edges.push(Vec::new());
                cc_i.as_ref().unwrap().variables.iter().for_each(|&v| {
                    out_edges[v].push(n);
                });
                q.push(n);
                n += 1;
            } else {
                *cc_i = None;
            }
        }
        while cc.len() - 1 > ic.output_layer && cc.last().unwrap().is_none() {
            cc.pop();
        }
        if circuit_id == 0 {
            if ic.output_layer < cc.len() {
                ic.output_layer = cc.len() - 1;
            } else {
                panic!("unexpected situation");
            }
        }
        ic.combined_constraints = cc;

        // compute maxLayer
        for i in 0..n {
            ic.max_layer[i] = ic.min_layer[i];
        }
        for x in q.iter().cloned() {
            for y in out_edges[x].iter().cloned() {
                ic.max_layer[x] = ic.max_layer[x].max(ic.min_layer[y] - layer_advance[y]);
            }
        }
        for i in 0..ic.sub_circuit_insn_ids.len() {
            if ic.min_layer[nv + i] != ic.max_layer[nv + i] {
                panic!("unexpected situation: sub-circuit virtual variable should have equal min/max layer");
            }
        }
        for (i, v) in ic.sub_circuit_insn_ids.iter().cloned().enumerate() {
            ic.sub_circuit_loc_map.insert(v, i);
        }

        // force outputLayer to be at least 1
        if ic.output_layer < 1 {
            ic.output_layer = 1;
        }

        // if (the output includes partial output of a sub circuit or the sub circuit has constraints),
        // and the sub circuit also ends at the output layer, we have to increase output layer
        'check_next_circuit: for i in 0..ic.sub_circuit_insn_ids.len() {
            let mut count = 0;
            for y in out_edges[nv + i].iter().cloned() {
                if ic.min_layer[y] == ic.output_layer {
                    if ic.output_order.contains_key(&y) {
                        count += 1;
                    }
                } else {
                    continue 'check_next_circuit;
                }
            }
            let mut any_constraint = false;

            for v in self.circuits[&ic.sub_circuit_insn_refs[i].sub_circuit_id]
                .combined_constraints
                .iter()
            {
                if v.is_some() {
                    any_constraint = true;
                }
            }
            if (count != 0 || any_constraint) && count != out_edges[nv + i].len() {
                ic.output_layer += 1;
                break;
            }
        }

        if ic.output_layer == ic.combined_constraints.len() {
            ic.combined_constraints.push(None);
        }

        // force maxLayer of output to be outputLayer
        for x in circuit.outputs.iter().cloned() {
            ic.max_layer[x] = ic.output_layer;
        }

        // adjust minLayer of GetRandomValue variables to a larger value
        for x in ic.constant_like_variables.keys() {
            if ic.max_layer[*x] == ic.output_layer {
                continue;
            }
            ic.min_layer[*x] = ic.output_layer.min(ic.max_layer[*x]);
            for y in out_edges[*x].iter().cloned() {
                ic.min_layer[*x] = ic.min_layer[*x].min(ic.min_layer[y] - layer_advance[y]);
            }
        }

        // compute occured layers
        if I::CROSS_LAYER_RELAY {
            ic.occured_layers = vec![Vec::new(); ic.max_layer.len()];
            let outputs_set: HashSet<usize> = circuit.outputs.iter().cloned().collect();
            for x in q.iter().cloned() {
                let mut tmp = Vec::with_capacity(out_edges[x].len() + 1);
                tmp.push(ic.min_layer[x]);
                for y in out_edges[x].iter().cloned() {
                    tmp.push(ic.min_layer[y] - layer_advance[y]);
                }
                if outputs_set.contains(&x) {
                    tmp.push(ic.output_layer);
                }
                tmp.sort();
                let mut tmp2 = Vec::with_capacity(tmp.len());
                for &v in tmp.iter() {
                    if tmp2.is_empty() || *tmp2.last().unwrap() != v {
                        tmp2.push(v);
                    }
                }
                assert_eq!(tmp2[0], ic.min_layer[x]);
                assert_eq!(*tmp2.last().unwrap(), ic.max_layer[x]);
                ic.occured_layers[x] = tmp2;
            }
        }

        // compute minUsedLayer
        ic.min_used_layer = Vec::with_capacity(ic.output_layer + 1);
        ic.min_used_layer.push(0);
        ic.min_used_layer.extend(0..ic.output_layer);
        for (i, sc) in ic.sub_circuit_insn_refs.iter().enumerate() {
            let sub_circuit = &self.circuits[&sc.sub_circuit_id];
            let input_layer = ic.sub_circuit_start_layer[i];
            for j in 0..=sub_circuit.output_layer {
                ic.min_used_layer[j + input_layer] = ic.min_used_layer[j + input_layer]
                    .min(sub_circuit.min_used_layer[j] + input_layer);
            }
        }
        if I::CROSS_LAYER_RELAY {
            for x in q.iter().cloned() {
                let t = &ic.occured_layers[x];
                for (u, v) in t.iter().zip(t.iter().skip(1)) {
                    ic.min_used_layer[*v] = ic.min_used_layer[*v].min(*u);
                }
            }
        }

        self.circuits.insert(circuit_id, ic);
    }
}
