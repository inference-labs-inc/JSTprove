use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
};

use rand::{RngCore, SeedableRng};

use crate::utils::{misc::next_power_of_two, union_find::UnionFind};

use super::*;

impl<C: Config, const INPUT_NUM: usize> PartialOrd for Gate<C, INPUT_NUM> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Config, const INPUT_NUM: usize> Ord for Gate<C, INPUT_NUM> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.output.cmp(&other.output) {
            Ordering::Less => {
                return Ordering::Less;
            }
            Ordering::Greater => {
                return Ordering::Greater;
            }
            Ordering::Equal => {}
        };
        for i in 0..INPUT_NUM {
            match self.inputs[i].cmp(&other.inputs[i]) {
                Ordering::Less => {
                    return Ordering::Less;
                }
                Ordering::Greater => {
                    return Ordering::Greater;
                }
                Ordering::Equal => {}
            };
        }
        self.coef.cmp(&other.coef)
    }
}

impl<C: Config> PartialOrd for GateCustom<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Config> Ord for GateCustom<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.gate_type.cmp(&other.gate_type) {
            Ordering::Less => {
                return Ordering::Less;
            }
            Ordering::Greater => {
                return Ordering::Greater;
            }
            Ordering::Equal => {}
        };
        match self.output.cmp(&other.output) {
            Ordering::Less => {
                return Ordering::Less;
            }
            Ordering::Greater => {
                return Ordering::Greater;
            }
            Ordering::Equal => {}
        };
        match self.inputs.len().cmp(&other.inputs.len()) {
            Ordering::Less => {
                return Ordering::Less;
            }
            Ordering::Greater => {
                return Ordering::Greater;
            }
            Ordering::Equal => {}
        };
        for i in 0..self.inputs.len() {
            match self.inputs[i].cmp(&other.inputs[i]) {
                Ordering::Less => {
                    return Ordering::Less;
                }
                Ordering::Greater => {
                    return Ordering::Greater;
                }
                Ordering::Equal => {}
            };
        }
        self.coef.cmp(&other.coef)
    }
}

trait GateOpt<C: Config>: PartialEq + Ord + Clone {
    fn coef_add(&mut self, coef: Coef<C>);
    fn can_merge_with(&self, other: &Self) -> bool;
    fn get_coef(&self) -> Coef<C>;
    fn add_offset(&self, in_offset: usize, out_offset: usize) -> Self;
}

impl<C: Config, const INPUT_NUM: usize> GateOpt<C> for Gate<C, INPUT_NUM> {
    fn coef_add(&mut self, coef: Coef<C>) {
        self.coef = self.coef.add_constant(coef.get_constant().unwrap());
    }
    fn can_merge_with(&self, other: &Self) -> bool {
        self.inputs == other.inputs
            && self.output == other.output
            && self.coef.is_constant()
            && other.coef.is_constant()
    }
    fn get_coef(&self) -> Coef<C> {
        self.coef.clone()
    }
    fn add_offset(&self, in_offset: usize, out_offset: usize) -> Self {
        let mut inputs = self.inputs;
        for input in inputs.iter_mut() {
            *input += in_offset;
        }
        let output = self.output + out_offset;
        let coef = self.coef.clone();
        Gate {
            inputs,
            output,
            coef,
        }
    }
}

impl<C: Config> GateOpt<C> for GateCustom<C> {
    fn coef_add(&mut self, coef: Coef<C>) {
        self.coef = self.coef.add_constant(coef.get_constant().unwrap());
    }
    fn can_merge_with(&self, other: &Self) -> bool {
        self.gate_type == other.gate_type
            && self.inputs == other.inputs
            && self.output == other.output
            && self.coef.is_constant()
            && other.coef.is_constant()
    }
    fn get_coef(&self) -> Coef<C> {
        self.coef.clone()
    }
    fn add_offset(&self, in_offset: usize, out_offset: usize) -> Self {
        let mut inputs = self.inputs.clone();
        for input in inputs.iter_mut() {
            *input += in_offset;
        }
        let output = self.output + out_offset;
        let coef = self.coef.clone();
        GateCustom {
            gate_type: self.gate_type,
            inputs,
            output,
            coef,
        }
    }
}

fn dedup_gates<C: Config, G: GateOpt<C>>(gates: &mut Vec<G>, trim_zero: bool) {
    gates.sort();
    let mut lst = 0;
    for i in 1..gates.len() {
        if gates[lst].can_merge_with(&gates[i]) {
            let t = gates[i].get_coef();
            gates[lst].coef_add(t);
        } else {
            lst += 1;
            let t = gates[i].clone();
            gates[lst] = t;
        }
    }
    gates.truncate(lst + 1);
    if trim_zero {
        let mut n = 0;
        for i in 0..gates.len() {
            let is_zero = match gates[i].get_coef().get_constant() {
                Some(x) => x.is_zero(),
                None => false,
            };
            if !is_zero {
                let t = gates[i].clone();
                gates[n] = t;
                n += 1;
            }
        }
        gates.truncate(n);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum UniGate<C: Config> {
    Mul(GateMul<C>),
    Add(GateAdd<C>),
    Const(GateConst<C>),
    Custom(GateCustom<C>),
}

impl<C: Config> Segment<C> {
    fn dedup_gates(&mut self) {
        let mut occured_outputs = vec![false; self.num_outputs];
        for gate in self.gate_muls.iter_mut() {
            occured_outputs[gate.output] = true;
        }
        for gate in self.gate_adds.iter_mut() {
            occured_outputs[gate.output] = true;
        }
        for gate in self.gate_consts.iter_mut() {
            occured_outputs[gate.output] = true;
        }
        for gate in self.gate_customs.iter_mut() {
            occured_outputs[gate.output] = true;
        }
        dedup_gates(&mut self.gate_muls, true);
        dedup_gates(&mut self.gate_adds, true);
        dedup_gates(&mut self.gate_consts, false);
        dedup_gates(&mut self.gate_customs, true);
        let mut need_outputs = occured_outputs;
        for gate in self.gate_muls.iter() {
            need_outputs[gate.output] = false;
        }
        for gate in self.gate_adds.iter() {
            need_outputs[gate.output] = false;
        }
        for gate in self.gate_consts.iter() {
            need_outputs[gate.output] = false;
        }
        for gate in self.gate_customs.iter() {
            need_outputs[gate.output] = false;
        }
        for (i, need) in need_outputs.iter().enumerate() {
            if *need {
                self.gate_consts.push(GateConst {
                    inputs: [],
                    output: i,
                    coef: Coef::Constant(C::CircuitField::zero()),
                });
            }
        }
        self.gate_consts.sort();
    }

    fn sample_gates(&self, num_gates: usize, mut rng: impl RngCore) -> HashSet<UniGate<C>> {
        let tot_gates = self.num_all_gates();
        let mut ids: HashSet<usize> = HashSet::new();
        while ids.len() < num_gates && ids.len() < tot_gates {
            ids.insert(rng.next_u64() as usize % tot_gates);
        }
        let mut ids: Vec<usize> = ids.into_iter().collect();
        ids.sort();
        let mut gates = HashSet::new();
        let tot_mul = self.gate_muls.len();
        let tot_add = self.gate_adds.len();
        let tot_const = self.gate_consts.len();
        for id in ids.iter() {
            if *id < tot_mul {
                gates.insert(UniGate::Mul(self.gate_muls[*id].clone()));
            } else if *id < tot_mul + tot_add {
                gates.insert(UniGate::Add(self.gate_adds[*id - tot_mul].clone()));
            } else if *id < tot_mul + tot_add + tot_const {
                gates.insert(UniGate::Const(
                    self.gate_consts[*id - tot_mul - tot_add].clone(),
                ));
            } else {
                gates.insert(UniGate::Custom(
                    self.gate_customs[*id - tot_mul - tot_add - tot_const].clone(),
                ));
            }
        }
        gates
    }

    fn all_gates(&self) -> HashSet<UniGate<C>> {
        let mut gates = HashSet::new();
        for gate in self.gate_muls.iter() {
            gates.insert(UniGate::Mul(gate.clone()));
        }
        for gate in self.gate_adds.iter() {
            gates.insert(UniGate::Add(gate.clone()));
        }
        for gate in self.gate_consts.iter() {
            gates.insert(UniGate::Const(gate.clone()));
        }
        for gate in self.gate_customs.iter() {
            gates.insert(UniGate::Custom(gate.clone()));
        }
        gates
    }

    fn num_all_gates(&self) -> usize {
        self.gate_muls.len()
            + self.gate_adds.len()
            + self.gate_consts.len()
            + self.gate_customs.len()
    }

    fn remove_gates(&mut self, gates: &HashSet<UniGate<C>>) {
        let mut new_gates = Vec::new();
        for gate in self.gate_muls.iter() {
            if !gates.contains(&UniGate::Mul(gate.clone())) {
                new_gates.push(gate.clone());
            }
        }
        self.gate_muls = new_gates;
        let mut new_gates = Vec::new();
        for gate in self.gate_adds.iter() {
            if !gates.contains(&UniGate::Add(gate.clone())) {
                new_gates.push(gate.clone());
            }
        }
        self.gate_adds = new_gates;
        let mut new_gates = Vec::new();
        for gate in self.gate_consts.iter() {
            if !gates.contains(&UniGate::Const(gate.clone())) {
                new_gates.push(gate.clone());
            }
        }
        self.gate_consts = new_gates;
        let mut new_gates = Vec::new();
        for gate in self.gate_customs.iter() {
            if !gates.contains(&UniGate::Custom(gate.clone())) {
                new_gates.push(gate.clone());
            }
        }
        self.gate_customs = new_gates;
    }

    fn from_uni_gates(gates: &HashSet<UniGate<C>>) -> Self {
        let mut gate_muls = Vec::new();
        let mut gate_adds = Vec::new();
        let mut gate_consts = Vec::new();
        let mut gate_customs = Vec::new();
        for gate in gates.iter() {
            match gate {
                UniGate::Mul(g) => gate_muls.push(g.clone()),
                UniGate::Add(g) => gate_adds.push(g.clone()),
                UniGate::Const(g) => gate_consts.push(g.clone()),
                UniGate::Custom(g) => gate_customs.push(g.clone()),
            }
        }
        gate_muls.sort();
        gate_adds.sort();
        gate_consts.sort();
        gate_customs.sort();
        let mut max_input = 0;
        let mut max_output = 0;
        for gate in gate_muls.iter() {
            for input in gate.inputs.iter() {
                max_input = max_input.max(*input);
            }
            max_output = max_output.max(gate.output);
        }
        for gate in gate_adds.iter() {
            for input in gate.inputs.iter() {
                max_input = max_input.max(*input);
            }
            max_output = max_output.max(gate.output);
        }
        for gate in gate_consts.iter() {
            max_output = max_output.max(gate.output);
        }
        for gate in gate_customs.iter() {
            for input in gate.inputs.iter() {
                max_input = max_input.max(*input);
            }
            max_output = max_output.max(gate.output);
        }
        Segment {
            num_inputs: next_power_of_two(max_input + 1),
            num_outputs: next_power_of_two(max_output + 1),
            gate_muls,
            gate_adds,
            gate_consts,
            gate_customs,
            child_segs: Vec::new(),
        }
    }
}

impl<C: Config> Circuit<C> {
    pub fn dedup_gates(&mut self) {
        for segment in self.segments.iter_mut() {
            segment.dedup_gates();
        }
    }

    fn expand_gates<T: GateOpt<C>, F: Fn(usize) -> bool, G: Fn(&Segment<C>) -> &Vec<T>>(
        &self,
        segment_id: usize,
        prev_segments: &[Segment<C>],
        should_expand: F,
        get_gates: G,
    ) -> Vec<T> {
        let segment = &self.segments[segment_id];
        let mut gates: Vec<T> = get_gates(segment).clone();
        for (sub_segment_id, allocations) in segment.child_segs.iter() {
            if should_expand(*sub_segment_id) {
                let sub_segment = &prev_segments[*sub_segment_id];
                let sub_gates = get_gates(sub_segment).clone();
                for allocation in allocations.iter() {
                    let in_offset = allocation.input_offset;
                    let out_offset = allocation.output_offset;
                    for gate in sub_gates.iter() {
                        gates.push(gate.add_offset(in_offset, out_offset));
                    }
                }
            }
        }
        gates
    }

    fn expand_segment<F: Fn(usize) -> bool>(
        &self,
        segment_id: usize,
        prev_segments: &[Segment<C>],
        should_expand: F,
    ) -> Segment<C> {
        let segment = &self.segments[segment_id];
        let gate_muls =
            self.expand_gates(segment_id, prev_segments, &should_expand, |s| &s.gate_muls);
        let gate_adds =
            self.expand_gates(segment_id, prev_segments, &should_expand, |s| &s.gate_adds);
        let gate_consts = self.expand_gates(segment_id, prev_segments, &should_expand, |s| {
            &s.gate_consts
        });
        let gate_customs = self.expand_gates(segment_id, prev_segments, &should_expand, |s| {
            &s.gate_customs
        });
        let mut child_segs_map = HashMap::new();
        for (sub_segment_id, allocations) in segment.child_segs.iter() {
            if !should_expand(*sub_segment_id) {
                if !child_segs_map.contains_key(sub_segment_id) {
                    child_segs_map.insert(*sub_segment_id, Vec::new());
                }
                child_segs_map
                    .get_mut(sub_segment_id)
                    .unwrap()
                    .extend(allocations.iter().cloned());
            } else {
                let sub_segment = &prev_segments[*sub_segment_id];
                for (sub_sub_segment_id, sub_allocations) in sub_segment.child_segs.iter() {
                    if !child_segs_map.contains_key(sub_sub_segment_id) {
                        child_segs_map.insert(*sub_sub_segment_id, Vec::new());
                    }
                    for sub_allocation in sub_allocations.iter() {
                        for allocation in allocations.iter() {
                            let new_allocation = Allocation {
                                input_offset: sub_allocation.input_offset + allocation.input_offset,
                                output_offset: sub_allocation.output_offset
                                    + allocation.output_offset,
                            };
                            child_segs_map
                                .get_mut(sub_sub_segment_id)
                                .unwrap()
                                .push(new_allocation);
                        }
                    }
                }
            }
        }
        for (_, allocations) in child_segs_map.iter_mut() {
            allocations.sort();
        }
        let child_segs = child_segs_map.into_iter().collect();
        Segment {
            num_inputs: segment.num_inputs,
            num_outputs: segment.num_outputs,
            gate_muls,
            gate_adds,
            gate_consts,
            gate_customs,
            child_segs,
        }
    }

    pub fn expand_small_segments(&self) -> Self {
        const EXPAND_USE_COUNT_LIMIT: usize = 1;
        const EXPAND_GATE_COUNT_LIMIT: usize = 4;
        let mut in_layers = vec![false; self.segments.len()];
        let mut used_count = vec![0; self.segments.len()];
        let mut expand_range = HashSet::new();
        for &segment_id in self.layer_ids.iter() {
            used_count[segment_id] += EXPAND_USE_COUNT_LIMIT + 1;
            in_layers[segment_id] = true;
        }
        for i in (0..self.segments.len()).rev() {
            if used_count[i] > 0 {
                for (sub_segment_id, allocations) in self.segments[i].child_segs.iter() {
                    used_count[*sub_segment_id] += allocations.len();
                }
            }
        }
        let mut optimized = false;
        for (segment_id, segment) in self.segments.iter().enumerate() {
            if used_count[segment_id] == 0 {
                optimized = true;
                continue;
            }
            if in_layers[segment_id] {
                continue;
            }
            let mut gate_count = segment.gate_muls.len()
                + segment.gate_adds.len()
                + segment.gate_consts.len()
                + segment.gate_customs.len();
            for (_, allocations) in segment.child_segs.iter() {
                gate_count += allocations.len();
            }
            if used_count[segment_id] <= EXPAND_USE_COUNT_LIMIT
                || gate_count <= EXPAND_GATE_COUNT_LIMIT
            {
                expand_range.insert(segment_id);
                optimized = true;
            }
        }
        if !optimized {
            return self.clone();
        }
        let mut expand_range_vec: Vec<usize> = expand_range.iter().cloned().collect();
        expand_range_vec.sort();
        let mut expanded_segments = Vec::with_capacity(self.segments.len());
        for (segment_id, segment) in self.segments.iter().enumerate() {
            if used_count[segment_id] > 0 {
                let expanded = self.expand_segment(segment_id, &expanded_segments, |x| {
                    expand_range.contains(&x)
                });
                expanded_segments.push(expanded);
            } else {
                expanded_segments.push(segment.clone());
            }
        }
        let mut new_id = vec![!0; self.segments.len()];
        let mut new_segments = Vec::new();
        for (segment_id, segment) in expanded_segments.iter().enumerate() {
            if used_count[segment_id] > 0 && !expand_range.contains(&segment_id) {
                let mut new_child_segs = Vec::new();
                for sub_segment in segment.child_segs.iter() {
                    new_child_segs.push((new_id[sub_segment.0], sub_segment.1.clone()));
                }
                let mut seg = Segment {
                    num_inputs: segment.num_inputs,
                    num_outputs: segment.num_outputs,
                    gate_muls: segment.gate_muls.clone(),
                    gate_adds: segment.gate_adds.clone(),
                    gate_consts: segment.gate_consts.clone(),
                    gate_customs: segment.gate_customs.clone(),
                    child_segs: new_child_segs.into_iter().collect(),
                };
                seg.dedup_gates();
                new_segments.push(seg);
                new_id[segment_id] = new_segments.len() - 1;
            }
        }
        let new_layers = self.layer_ids.iter().map(|x| new_id[*x]).collect();
        Circuit {
            num_public_inputs: self.num_public_inputs,
            num_actual_outputs: self.num_actual_outputs,
            expected_num_output_zeroes: self.expected_num_output_zeroes,
            segments: new_segments,
            layer_ids: new_layers,
        }
    }

    pub fn find_common_parts(&self) -> Self {
        const SAMPLE_PER_SEGMENT: usize = 100;
        const COMMON_THRESHOLD_PERCENT: usize = 5;
        const COMMON_THRESHOLD_VALUE: usize = 10;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123); //for deterministic
        let sampled_gates: Vec<HashSet<UniGate<C>>> = self
            .segments
            .iter()
            .map(|segment| segment.sample_gates(SAMPLE_PER_SEGMENT, &mut rng))
            .collect();
        let all_gates: Vec<HashSet<UniGate<C>>> = self
            .segments
            .iter()
            .map(|segment| segment.all_gates())
            .collect();
        let mut edges = Vec::new();
        //println!("segments: {}", self.segments.len());
        for (i, i_gates) in all_gates.iter().enumerate() {
            for (j, j_gates) in sampled_gates.iter().enumerate().take(i) {
                let mut common_count = 0;
                for gate in j_gates.iter() {
                    if i_gates.contains(gate) {
                        common_count += 1;
                    }
                }
                let num_samples = j_gates.len();
                if num_samples >= COMMON_THRESHOLD_VALUE
                    && common_count * 100 >= num_samples * COMMON_THRESHOLD_PERCENT
                {
                    let expected_common_count =
                        self.segments[j].num_all_gates() * common_count / num_samples;
                    edges.push((-(expected_common_count as isize), i, j));
                }
            }
        }
        edges.sort();
        let mut uf = UnionFind::new(self.segments.len());
        let mut group_gates = all_gates;
        for edge in edges.iter() {
            let (_, i, j) = edge;
            let mut x = uf.find(*i);
            let mut y = uf.find(*j);
            if x == y {
                continue;
            }
            if group_gates[x].len() < group_gates[y].len() {
                std::mem::swap(&mut x, &mut y);
            }
            let mut cnt = 0;
            for gate in group_gates[y].iter() {
                if group_gates[x].contains(gate) {
                    cnt += 1;
                }
            }
            if cnt < COMMON_THRESHOLD_VALUE {
                continue;
            }
            let merged_gates: HashSet<UniGate<C>> = group_gates[x]
                .intersection(&group_gates[y])
                .cloned()
                .collect();
            uf.union(x, y);
            group_gates[uf.find(x)] = merged_gates;
        }
        let mut size = vec![0; self.segments.len()];
        for i in 0..self.segments.len() {
            size[uf.find(i)] += 1;
        }
        let mut rm_id: Vec<Option<usize>> = vec![None; self.segments.len()];
        let mut new_segments: Vec<Segment<C>> = Vec::new();
        let mut new_id = vec![!0; self.segments.len()];
        for i in 0..self.segments.len() {
            if i == uf.find(i) && size[i] > 1 && group_gates[i].len() >= COMMON_THRESHOLD_VALUE {
                let segment = Segment::from_uni_gates(&group_gates[i]);
                new_segments.push(segment);
                rm_id[i] = Some(new_segments.len() - 1);
            }
        }
        for (segment_id, segment) in self.segments.iter().enumerate() {
            let mut new_child_segs = Vec::new();
            for sub_segment in segment.child_segs.iter() {
                new_child_segs.push((new_id[sub_segment.0], sub_segment.1.clone()));
            }
            let mut seg = Segment {
                num_inputs: segment.num_inputs,
                num_outputs: segment.num_outputs,
                gate_muls: segment.gate_muls.clone(),
                gate_adds: segment.gate_adds.clone(),
                gate_consts: segment.gate_consts.clone(),
                gate_customs: segment.gate_customs.clone(),
                child_segs: new_child_segs.into_iter().collect(),
            };
            let parent_id = uf.find(segment_id);
            if let Some(common_id) = rm_id[parent_id] {
                seg.remove_gates(&group_gates[parent_id]);
                seg.child_segs.push((
                    common_id,
                    vec![Allocation {
                        input_offset: 0,
                        output_offset: 0,
                    }],
                ));
            }
            seg.dedup_gates();
            new_segments.push(seg);
            new_id[segment_id] = new_segments.len() - 1;
        }
        let new_layers = self.layer_ids.iter().map(|x| new_id[*x]).collect();
        Circuit {
            num_public_inputs: self.num_public_inputs,
            num_actual_outputs: self.num_actual_outputs,
            expected_num_output_zeroes: self.expected_num_output_zeroes,
            segments: new_segments,
            layer_ids: new_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::layered;
    use crate::field::FieldArith;
    use crate::layering::compile;
    use crate::{
        circuit::{
            config::{Config, GF2Config as C},
            ir::{self, common::rand_gen::*},
        },
        utils::error::Error,
    };

    type CField = <C as Config>::CircuitField;

    fn get_random_layered_circuit(rcc: &RandomCircuitConfig) -> Option<layered::Circuit<C>> {
        let root = ir::dest::RootCircuitRelaxed::<C>::random(&rcc);
        let mut root = root.export_constraints();
        root.reassign_duplicate_sub_circuit_outputs();
        let root = root.remove_unreachable().0;
        let root = root.solve_duplicates();
        assert_eq!(root.validate(), Ok(()));
        match root.validate_circuit_has_inputs() {
            Ok(_) => {}
            Err(e) => match e {
                Error::InternalError(s) => {
                    panic!("{}", s);
                }
                Error::UserError(_) => {
                    return None;
                }
            },
        }
        let (lc, _) = compile(&root);
        assert_eq!(lc.validate(), Ok(()));
        Some(lc)
    }

    #[test]
    fn dedup_gates_random() {
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
            config.seed = i + 400000;
            let lc = match get_random_layered_circuit(&config) {
                Some(lc) => lc,
                None => {
                    continue;
                }
            };
            let mut lc_opt = lc.clone();
            lc_opt.dedup_gates();
            assert_eq!(lc_opt.validate(), Ok(()));
            assert_eq!(lc_opt.input_size(), lc.input_size());
            for _ in 0..5 {
                let input: Vec<<C as Config>::CircuitField> = (0..lc.input_size())
                    .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                    .collect();
                let (lc_output, lc_cond) = lc.eval_unsafe(input.clone());
                let (lc_opt_output, lc_opt_cond) = lc.eval_unsafe(input);
                assert_eq!(lc_cond, lc_opt_cond);
                assert_eq!(lc_output, lc_opt_output);
            }
        }
    }

    #[test]
    fn expand_small_segments_random() {
        let mut config = RandomCircuitConfig {
            seed: 0,
            num_circuits: RandomRange { min: 1, max: 100 },
            num_inputs: RandomRange { min: 1, max: 3 },
            num_instructions: RandomRange { min: 5, max: 10 },
            num_constraints: RandomRange { min: 0, max: 5 },
            num_outputs: RandomRange { min: 1, max: 3 },
            num_terms: RandomRange { min: 1, max: 5 },
            sub_circuit_prob: 0.1,
        };
        for i in 0..3000 {
            config.seed = i + 500000;
            let lc = match get_random_layered_circuit(&config) {
                Some(lc) => lc,
                None => {
                    continue;
                }
            };
            let lc_opt = lc.expand_small_segments();
            assert_eq!(lc_opt.validate(), Ok(()));
            assert_eq!(lc_opt.input_size(), lc.input_size());
            for _ in 0..5 {
                let input: Vec<<C as Config>::CircuitField> = (0..lc.input_size())
                    .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                    .collect();
                let (lc_output, lc_cond) = lc.eval_unsafe(input.clone());
                let (lc_opt_output, lc_opt_cond) = lc_opt.eval_unsafe(input);
                assert_eq!(lc_cond, lc_opt_cond);
                assert_eq!(lc_output, lc_opt_output);
            }
        }
    }

    #[test]
    fn find_common_parts_random() {
        let mut config = RandomCircuitConfig {
            seed: 0,
            num_circuits: RandomRange { min: 1, max: 100 },
            num_inputs: RandomRange { min: 1, max: 3 },
            num_instructions: RandomRange { min: 5, max: 10 },
            num_constraints: RandomRange { min: 0, max: 5 },
            num_outputs: RandomRange { min: 1, max: 3 },
            num_terms: RandomRange { min: 1, max: 5 },
            sub_circuit_prob: 0.1,
        };
        for i in 0..3000 {
            config.seed = i + 600000;
            let lc = match get_random_layered_circuit(&config) {
                Some(lc) => lc,
                None => {
                    continue;
                }
            };
            let lc_opt = lc.find_common_parts();
            assert_eq!(lc_opt.validate(), Ok(()));
            assert_eq!(lc_opt.input_size(), lc.input_size());
            for _ in 0..5 {
                let input: Vec<<C as Config>::CircuitField> = (0..lc.input_size())
                    .map(|_| CField::random_unsafe(&mut rand::thread_rng()))
                    .collect();
                let (lc_output, lc_cond) = lc.eval_unsafe(input.clone());
                let (lc_opt_output, lc_opt_cond) = lc_opt.eval_unsafe(input);
                assert_eq!(lc_cond, lc_opt_cond);
                assert_eq!(lc_output, lc_opt_output);
            }
        }
    }
}
