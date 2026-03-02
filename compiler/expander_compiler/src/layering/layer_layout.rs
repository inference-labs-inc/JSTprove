use std::{collections::HashMap, mem};

use crate::{
    circuit::{config::Config, input_mapping::EMPTY, layered::InputType},
    utils::{misc::next_power_of_two, pool::Pool},
};

use super::compile::CompileContext;

#[derive(Default, Clone)]
pub struct LayerLayoutContext {
    pub vars: Pool<usize>, // global index of variables occurring in this layer
    pub prev_circuit_insn_ids: HashMap<usize, usize>, // insn id of previous circuit
    pub prev_circuit_num_out: HashMap<usize, usize>, // number of outputs of previous circuit, used to check if all output variables are used
    pub prev_circuit_subc_pos: HashMap<usize, usize>,
    pub placement: HashMap<usize, usize>, // placement group of each variable
    pub parent: Vec<usize>,               // parent placement group of some placement group
    pub req: Vec<PlacementRequest>,

    pub middle_sub_circuits: Vec<usize>, // sub-circuits who have middle layers in this layer (referenced by index in sub_circuit_insn_ids)
}

// we will sort placement requests by size, and then greedy
#[derive(Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlacementRequest {
    pub insn_id: usize,
    pub input_ids: Vec<usize>,
}

// finalized layout of a layer
// dense -> placementDense[i] = variable on slot i (placementDense[i] == j means i-th slot stores varIdx[j])
// sparse -> placementSparse[i] = variable on slot i, and there are subLayouts.
#[derive(Hash, Clone, PartialEq, Eq)]
pub struct LayerLayout {
    pub circuit_id: usize,
    pub layer: usize,
    pub size: usize,
    pub inner: LayerLayoutInner,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LayerLayoutInner {
    Sparse {
        placement: HashMap<usize, usize>,
        sub_layout: Vec<SubLayout>,
    },
    Dense {
        placement: Vec<usize>,
    },
}

impl std::hash::Hash for LayerLayoutInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            LayerLayoutInner::Sparse {
                placement,
                sub_layout,
            } => {
                0.hash(state);
                let mut items: Vec<(usize, usize)> =
                    placement.iter().map(|(k, v)| (*k, *v)).collect();
                items.sort();
                items.hash(state);
                sub_layout.hash(state);
            }
            LayerLayoutInner::Dense { placement } => {
                1.hash(state);
                placement.hash(state);
            }
        }
    }
}

#[derive(Hash, Clone, PartialEq, Eq, Debug)]
pub struct SubLayout {
    pub id: usize,      // unique layout id in a compile context
    pub offset: usize,  // offset in layout
    pub insn_id: usize, // instruction id corresponding to this sub-layout
}

// request for layer layout
#[derive(Hash, Clone, PartialEq, Eq)]
pub struct LayerReq {
    pub circuit_id: usize,
    pub layer: usize, // which layer to solve?
}

impl<'a, C: Config, I: InputType> CompileContext<'a, C, I> {
    pub fn prepare_layer_layout_context(&mut self, circuit_id: usize) {
        let mut ic = self.circuits.remove(&circuit_id).unwrap();

        // find out the variables in each layer
        ic.lcs = vec![LayerLayoutContext::default(); ic.output_layer + 1];
        for i in 0..=ic.output_layer {
            if let Some(constraint) = &ic.combined_constraints[i] {
                ic.lcs[i].vars.add(&constraint.id);
            }
        }
        for v in ic.circuit.outputs.iter() {
            ic.lcs[ic.output_layer].vars.add(v);
        }
        for i in 1..ic.num_var {
            if I::CROSS_LAYER_RELAY {
                for j in ic.occured_layers[i].iter().cloned() {
                    ic.lcs[j].vars.add(&i);
                }
            } else {
                for j in ic.min_layer[i]..=ic.max_layer[i] {
                    ic.lcs[j].vars.add(&i);
                }
            }
        }

        // for each sub-circuit, enqueue the placement request in input layer, and mark prev_circuit_insn_id in output layer
        // also push all middle layers to the layer context
        for (i, insn_id) in ic.sub_circuit_insn_ids.iter().cloned().enumerate() {
            let insn = &ic.sub_circuit_insn_refs[i];
            let input_layer = ic.sub_circuit_start_layer[i];
            let output_layer = self.circuits[&insn.sub_circuit_id].output_layer + input_layer;
            let input_ids = insn.inputs.clone();
            ic.lcs[input_layer]
                .req
                .push(PlacementRequest { insn_id, input_ids });

            for x in insn.outputs.iter().cloned() {
                ic.lcs[output_layer]
                    .prev_circuit_insn_ids
                    .insert(x, insn_id);
            }
            ic.lcs[output_layer]
                .prev_circuit_num_out
                .insert(insn_id, insn.outputs.len());
            ic.lcs[output_layer]
                .prev_circuit_subc_pos
                .insert(insn_id, i);

            for j in input_layer + 1..output_layer {
                ic.lcs[j].middle_sub_circuits.push(i);
            }
        }

        for i in 0..=ic.output_layer {
            let lc = &mut ic.lcs[i];
            for x in lc.vars.vec().iter().cloned() {
                lc.placement.insert(x, 0);
            }
            lc.parent.push(0);
            lc.req.sort();
            // greedy placement
            for req in lc.req.iter() {
                let mut pc_cnt: HashMap<usize, usize> = HashMap::new(); // prev circuit count
                let mut pl_cnt: HashMap<usize, usize> = HashMap::new(); // placement count
                for x in req.input_ids.iter() {
                    if let Some(pc) = lc.prev_circuit_insn_ids.get(x) {
                        pc_cnt.insert(*pc, 0);
                    }
                    pl_cnt.insert(lc.placement[x], 0);
                }
                for x in req.input_ids.iter() {
                    if let Some(pc) = lc.prev_circuit_insn_ids.get(x) {
                        *pc_cnt.get_mut(pc).unwrap() += 1;
                    }
                    *pl_cnt.get_mut(&lc.placement[x]).unwrap() += 1;
                }
                // if all inputs don't split previout circuits, and they are in the same placement group,
                // we can create a new placement group containing them
                let mut flag = pl_cnt.len() == 1;
                for (k, v) in pc_cnt.iter() {
                    if *v != lc.prev_circuit_num_out[k] {
                        flag = false;
                    }
                }
                if flag {
                    let np = lc.parent.len(); // new placement group id
                    for x in req.input_ids.iter().cloned() {
                        lc.placement.insert(x, np);
                    }
                    let mut parent = 0;
                    for x in pl_cnt.keys() {
                        parent = *x;
                    }
                    lc.parent.push(parent);
                }
            }
        }
        self.circuits.insert(circuit_id, ic);
    }

    pub fn solve_layer_layout(&mut self, req: &LayerReq) -> usize {
        if let Some(id) = self.layer_req_to_layout.get(req) {
            return *id;
        }
        let res = self.solve_layer_layout_inner(req);
        let id = self.layer_layout_pool.add(&res);
        self.layer_req_to_layout.insert(req.clone(), id);
        id
    }

    fn solve_layer_layout_inner(&mut self, req: &LayerReq) -> LayerLayout {
        if req.circuit_id == 0 {
            if req.layer == 0 && !self.opts.allow_input_reorder {
                return self.solve_layer_layout_fixed(req);
            }
            if req.layer == self.circuits[&req.circuit_id].output_layer {
                return self.solve_layer_layout_fixed(req);
            }
        }
        self.solve_layer_layout_normal(req)
    }

    fn solve_layer_layout_fixed(&mut self, req: &LayerReq) -> LayerLayout {
        let layout_vec = merge_layouts(
            vec![],
            (0..self.circuits[&req.circuit_id].lcs[req.layer].vars.len()).collect(),
        );
        LayerLayout {
            circuit_id: req.circuit_id,
            layer: req.layer,
            size: layout_vec.len(),
            inner: super::layer_layout::LayerLayoutInner::Dense {
                placement: layout_vec,
            },
        }
    }

    fn solve_layer_layout_normal(&mut self, req: &LayerReq) -> LayerLayout {
        let ic = self.circuits.remove(&req.circuit_id).unwrap();
        let lc = &ic.lcs[req.layer];

        // first iterate prev layer circuits, and solve their output layout
        let mut layouts: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut layouts_subs_arr: HashMap<usize, Vec<usize>> = HashMap::new();
        for &x in lc.prev_circuit_num_out.keys() {
            let subc_pos = lc.prev_circuit_subc_pos[&x];
            let insn = &ic.sub_circuit_insn_refs[subc_pos];
            let sub_layer = self.circuits[&insn.sub_circuit_id].output_layer;
            let layout_id = self.solve_layer_layout(&LayerReq {
                circuit_id: insn.sub_circuit_id,
                layer: sub_layer,
            });
            let layout = self.layer_layout_pool.get(layout_id);
            let mut la = if let LayerLayoutInner::Dense { placement } = &layout.inner {
                placement.clone()
            } else {
                panic!("unexpected situation");
            };
            subs_array(
                &mut la,
                self.circuits[&insn.sub_circuit_id].lcs[sub_layer]
                    .vars
                    .vec(),
            );
            subs_map(&mut la, &self.circuits[&insn.sub_circuit_id].output_order);
            subs_array(&mut la, &insn.outputs);
            layouts_subs_arr.insert(x, insn.outputs.clone());
            subs_map(&mut la, lc.vars.map());
            layouts.insert(x, la);
        }

        // build the tree of placement groups
        let mut children_variables: Vec<Vec<usize>> = vec![Vec::new(); lc.parent.len()];
        for (i, &x) in lc.vars.vec().iter().enumerate() {
            if !lc.prev_circuit_insn_ids.contains_key(&x) {
                children_variables[lc.placement[&x]].push(i);
            }
        }
        let mut children_prev_circuits: Vec<Vec<Vec<usize>>> = vec![Vec::new(); lc.parent.len()];
        let mut layouts_keys: Vec<usize> = layouts.keys().cloned().collect();
        layouts_keys.sort();
        for x in layouts_keys.iter() {
            let layout = layouts.get(x).unwrap();
            let v = layouts_subs_arr.get(x).unwrap();
            if !v.is_empty() {
                let v = &v[0];
                children_prev_circuits[lc.placement[v]].push(layout.clone());
            }
        }
        let mut children_nodes: Vec<Vec<usize>> = vec![Vec::new(); lc.parent.len()];
        for (i, &x) in lc.parent.iter().enumerate() {
            if i == 0 {
                continue;
            }
            children_nodes[x].push(i);
        }
        let mut placements: Vec<Vec<usize>> = vec![Vec::new(); lc.parent.len()];
        for i in (0..lc.parent.len()).rev() {
            let mut s = Vec::new();
            for &x in children_nodes[i].iter() {
                s.push(mem::take(&mut placements[x]));
            }
            s.append(&mut children_prev_circuits[i]);
            placements[i] = merge_layouts(s, mem::take(&mut children_variables[i]));
        }

        if lc.middle_sub_circuits.is_empty() {
            self.circuits.insert(req.circuit_id, ic);
            return LayerLayout {
                circuit_id: req.circuit_id,
                layer: req.layer,
                size: placements[0].len(),
                inner: LayerLayoutInner::Dense {
                    placement: placements.swap_remove(0),
                },
            };
        }

        let mut middle_layouts = Vec::with_capacity(lc.middle_sub_circuits.len());
        for &id in lc.middle_sub_circuits.iter() {
            let start_layer = ic.sub_circuit_start_layer[id];
            let req_layer = req.layer - start_layer;
            middle_layouts.push(self.solve_layer_layout(&LayerReq {
                circuit_id: ic.sub_circuit_insn_refs[id].sub_circuit_id,
                layer: req_layer,
            }));
        }
        let mut sizes = Vec::with_capacity(middle_layouts.len() + 1);
        sizes.push(placements[0].len());
        for x in middle_layouts.iter() {
            sizes.push(self.layer_layout_pool.get(*x).size);
        }
        let mut order = Vec::with_capacity(sizes.len());
        for i in 0..sizes.len() {
            order.push(i);
        }
        order.sort_by(|&i, &j| {
            if sizes[i] != sizes[j] {
                return sizes[j].cmp(&sizes[i]);
            }
            i.cmp(&j)
        });
        let mut cur = 0;
        let mut placement_sparse = HashMap::new();
        let mut sub_layout = Vec::new();
        for &i in order.iter() {
            if i == 0 {
                let mut flag = false;
                for (j, &x) in placements[0].iter().enumerate() {
                    if x != EMPTY {
                        flag = true;
                        placement_sparse.insert(cur + j, x);
                    }
                }
                if !flag {
                    continue;
                }
            } else {
                sub_layout.push(SubLayout {
                    id: middle_layouts[i - 1],
                    offset: cur,
                    insn_id: ic.sub_circuit_insn_ids[lc.middle_sub_circuits[i - 1]],
                });
            }
            cur += sizes[i];
        }
        let size = next_power_of_two(cur);

        self.circuits.insert(req.circuit_id, ic);
        LayerLayout {
            circuit_id: req.circuit_id,
            layer: req.layer,
            size,
            inner: LayerLayoutInner::Sparse {
                placement: placement_sparse,
                sub_layout,
            },
        }
    }
}

pub fn merge_layouts(s: Vec<Vec<usize>>, additional: Vec<usize>) -> Vec<usize> {
    // currently it's a simple greedy algorithm
    // sort groups by size, and then place them one by one
    // since their size are always 2^n, the result is aligned
    // finally we insert the remaining variables to the empty slots
    let mut n = 0;
    for x in s.iter() {
        let m = x.len();
        n += m;
        if m & (m - 1) != 0 {
            panic!("unexpected situation: placement group size should be power of 2");
        }
    }
    n = next_power_of_two(n);
    let mut res = Vec::with_capacity(n);

    let mut order = Vec::with_capacity(s.len());
    for (i, s_i) in s.iter().enumerate() {
        if !s_i.is_empty() {
            order.push(i);
        }
    }
    order.sort_by(|&i, &j| {
        if s[i].len() != s[j].len() {
            return s[j].len().cmp(&s[i].len());
        }
        i.cmp(&j)
    });

    for x_ in order.iter() {
        let pg = &s[*x_];
        if res.len() % pg.len() != 0 {
            panic!("unexpected situation");
        }
        let mut placed = false;
        for i in (0..res.len()).step_by(pg.len()) {
            let mut ok = true;
            for j in 0..pg.len() {
                if res[i + j] != EMPTY && pg[j] != EMPTY {
                    ok = false;
                    break;
                }
            }
            if ok {
                for j in 0..pg.len() {
                    if pg[j] != EMPTY {
                        res[i + j] = pg[j];
                    }
                }
                placed = true;
                break;
            }
        }
        if !placed {
            res.extend_from_slice(pg);
        }
    }

    let mut slot = 0;
    for x in additional.iter() {
        while slot < res.len() && res[slot] != EMPTY {
            slot += 1;
        }
        if slot >= res.len() {
            res.push(*x);
        } else {
            res[slot] = *x;
        }
    }

    res.resize(next_power_of_two(res.len()), EMPTY);

    res
}

fn subs_array(l: &mut [usize], s: &[usize]) {
    for i in 0..l.len() {
        if l[i] != EMPTY {
            l[i] = s[l[i]];
        }
    }
}

pub fn subs_map(l: &mut [usize], m: &HashMap<usize, usize>) {
    for x in l.iter_mut() {
        if *x != EMPTY {
            // when a sub circuit thinks it doesn't need some input variable, it won't occur in map
            if let Some(&v) = m.get(x) {
                *x = v;
            } else {
                *x = EMPTY;
            }
        }
    }
}
