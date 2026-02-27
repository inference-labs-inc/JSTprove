use std::collections::HashMap;

use arith::Field;
use ethnum::U256;
use expander_compiler::expander_circuit::{
    self as ec, Circuit, CircuitLayer, CoefType, StructureInfo,
};
use expander_compiler::frontend::api::{BasicAPI, RootAPI, UnconstrainedAPI};
use expander_compiler::frontend::builder::{ToVariableOrValue, Variable, VariableOrValue};
use expander_compiler::frontend::{CircuitField, Config};
use expander_compiler::gkr_engine::GKREngine;
use expander_compiler::hints::registry::{HintRegistry, hint_key_to_id};
use expander_compiler::serdes::ExpSerde;

type FC<C> = <C as GKREngine>::FieldConfig;

#[derive(Clone, Copy, Debug)]
struct VarLocation {
    layer: usize,
    index: usize,
}

struct LayerAccumulator<C: Config> {
    mul_gates: Vec<ec::GateMul<FC<C>>>,
    add_gates: Vec<ec::GateAdd<FC<C>>>,
    const_gates: Vec<ec::GateConst<FC<C>>>,
    next_output_index: usize,
}

impl<C: Config> Default for LayerAccumulator<C> {
    fn default() -> Self {
        Self {
            mul_gates: Vec::new(),
            add_gates: Vec::new(),
            const_gates: Vec::new(),
            next_output_index: 0,
        }
    }
}

impl<C: Config> LayerAccumulator<C> {
    fn alloc_output(&mut self) -> usize {
        let idx = self.next_output_index;
        self.next_output_index += 1;
        idx
    }
}

pub struct DirectBuilder<C: Config> {
    layers: Vec<LayerAccumulator<C>>,
    var_locations: Vec<VarLocation>,
    witness_values: Vec<CircuitField<C>>,
    zero_assertions: Vec<usize>,
    output_vars: Vec<usize>,
    input_layer_next_index: usize,
    hint_registry: HintRegistry<CircuitField<C>>,
    sub_circuit_structures: HashMap<usize, Vec<usize>>,
    accum_candidate: Option<usize>,
    relay_cache: HashMap<(usize, usize), usize>,
    known_constants: HashMap<usize, CircuitField<C>>,
}

fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

fn make_gate_mul<C: Config>(
    i0: usize,
    i1: usize,
    o: usize,
    coef: CircuitField<C>,
    coef_type: CoefType,
) -> ec::GateMul<FC<C>> {
    ec::Gate {
        i_ids: [i0, i1],
        o_id: o,
        coef,
        coef_type,
        gate_type: 0,
    }
}

fn make_gate_add<C: Config>(
    i0: usize,
    o: usize,
    coef: CircuitField<C>,
    coef_type: CoefType,
) -> ec::GateAdd<FC<C>> {
    ec::Gate {
        i_ids: [i0],
        o_id: o,
        coef,
        coef_type,
        gate_type: 0,
    }
}

fn make_gate_const<C: Config>(
    o: usize,
    coef: CircuitField<C>,
    coef_type: CoefType,
) -> ec::GateConst<FC<C>> {
    ec::Gate {
        i_ids: [],
        o_id: o,
        coef,
        coef_type,
        gate_type: 0,
    }
}

impl<C: Config> DirectBuilder<C> {
    pub fn new(
        input_values: &[CircuitField<C>],
        hint_registry: HintRegistry<CircuitField<C>>,
    ) -> Self {
        let mut builder = Self {
            layers: vec![LayerAccumulator::default()],
            var_locations: Vec::with_capacity(input_values.len() * 4),
            witness_values: Vec::with_capacity(input_values.len() * 4),
            zero_assertions: Vec::new(),
            output_vars: Vec::new(),
            input_layer_next_index: 0,
            hint_registry,
            sub_circuit_structures: HashMap::new(),
            accum_candidate: None,
            relay_cache: HashMap::new(),
            known_constants: HashMap::new(),
        };

        builder
            .var_locations
            .push(VarLocation { layer: 0, index: 0 });
        builder.witness_values.push(CircuitField::<C>::zero());

        for &val in input_values {
            let idx = builder.alloc_input_slot();
            builder.var_locations.push(VarLocation {
                layer: 0,
                index: idx,
            });
            builder.witness_values.push(val);
        }

        builder
    }

    fn alloc_input_slot(&mut self) -> usize {
        let idx = self.input_layer_next_index;
        self.input_layer_next_index += 1;
        self.layers[0].next_output_index = self.input_layer_next_index;
        idx
    }

    fn ensure_layer(&mut self, layer: usize) {
        while self.layers.len() <= layer {
            self.layers.push(LayerAccumulator::default());
        }
    }

    fn relay_to_layer(&mut self, var_id: usize, target_layer: usize) -> usize {
        let loc = self.var_locations[var_id];
        if loc.layer == target_layer {
            return loc.index;
        }

        if let Some(&cached) = self.relay_cache.get(&(var_id, target_layer)) {
            return cached;
        }

        if self.accum_candidate == Some(var_id) {
            self.accum_candidate = None;
        }

        assert!(
            target_layer > loc.layer,
            "cannot relay backwards: var at layer {} to layer {}",
            loc.layer,
            target_layer
        );

        let mut current_index = loc.index;

        let start_layer = loc.layer;
        for l in (start_layer + 1)..=target_layer {
            if let Some(&cached) = self.relay_cache.get(&(var_id, l)) {
                current_index = cached;
                continue;
            }
            self.ensure_layer(l);
            let out_idx = self.layers[l].alloc_output();
            self.layers[l].add_gates.push(make_gate_add::<C>(
                current_index,
                out_idx,
                CircuitField::<C>::one(),
                CoefType::Constant,
            ));
            current_index = out_idx;
            self.relay_cache.insert((var_id, l), out_idx);
        }

        current_index
    }

    fn resolve_input(
        &mut self,
        input: impl ToVariableOrValue<CircuitField<C>>,
    ) -> (usize, CircuitField<C>) {
        match input.convert_to_variable_or_value() {
            VariableOrValue::Variable(v) => {
                let var_id = v.id();
                (var_id, self.witness_values[var_id])
            }
            VariableOrValue::Value(val) => {
                let var_id = self.create_constant_var(val);
                (var_id, val)
            }
        }
    }

    fn create_constant_var(&mut self, val: CircuitField<C>) -> usize {
        let idx = self.alloc_input_slot();
        let var_id = self.var_locations.len();
        self.var_locations.push(VarLocation {
            layer: 0,
            index: idx,
        });
        self.witness_values.push(val);
        self.layers[0]
            .const_gates
            .push(make_gate_const::<C>(idx, val, CoefType::Constant));
        self.known_constants.insert(var_id, val);
        var_id
    }

    fn new_var(&mut self, layer: usize, index: usize, value: CircuitField<C>) -> Variable {
        let var_id = self.var_locations.len();
        self.var_locations.push(VarLocation { layer, index });
        self.witness_values.push(value);
        Variable::from(var_id)
    }

    #[must_use]
    pub fn output_witness_values(&self) -> Vec<CircuitField<C>> {
        self.output_vars
            .iter()
            .map(|&id| self.witness_values[id])
            .collect()
    }

    #[allow(clippy::too_many_lines)]
    fn add_sub_inner(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
        negate_y: bool,
    ) -> Variable {
        let (x_id, x_val) = self.resolve_input(x);
        let (y_id, y_val) = self.resolve_input(y);

        let x_layer = self.var_locations[x_id].layer;
        let y_layer = self.var_locations[y_id].layer;

        let y_coef = if negate_y {
            CircuitField::<C>::zero() - CircuitField::<C>::one()
        } else {
            CircuitField::<C>::one()
        };
        let result_val = if negate_y {
            x_val - y_val
        } else {
            x_val + y_val
        };

        let x_const = self.known_constants.get(&x_id).copied();
        let y_const = self.known_constants.get(&y_id).copied();

        let (accum_id, other_id, other_coef, other_is_const) = if x_layer > y_layer {
            (x_id, y_id, y_coef, y_const)
        } else if y_layer > x_layer && !negate_y {
            (y_id, x_id, CircuitField::<C>::one(), x_const)
        } else {
            (0, 0, CircuitField::<C>::zero(), None)
        };

        if accum_id != 0 && self.accum_candidate == Some(accum_id) {
            let accum_loc = self.var_locations[accum_id];

            if let Some(cv) = other_is_const {
                let folded = other_coef * cv;
                self.layers[accum_loc.layer]
                    .const_gates
                    .push(make_gate_const::<C>(
                        accum_loc.index,
                        folded,
                        CoefType::Constant,
                    ));
            } else {
                let other_idx = self.relay_to_layer(other_id, accum_loc.layer - 1);
                self.layers[accum_loc.layer]
                    .add_gates
                    .push(make_gate_add::<C>(
                        other_idx,
                        accum_loc.index,
                        other_coef,
                        CoefType::Constant,
                    ));
            }

            self.witness_values[accum_id] = result_val;
            return Variable::from(accum_id);
        }

        if let (Some(x_cv), None) = (x_const, y_const) {
            let out_layer = y_layer + 1;
            self.ensure_layer(out_layer);
            let y_idx = self.relay_to_layer(y_id, out_layer - 1);
            let out_idx = self.layers[out_layer].alloc_output();
            self.layers[out_layer].add_gates.push(make_gate_add::<C>(
                y_idx,
                out_idx,
                y_coef,
                CoefType::Constant,
            ));
            let x_folded = x_cv;
            self.layers[out_layer]
                .const_gates
                .push(make_gate_const::<C>(out_idx, x_folded, CoefType::Constant));
            let var = self.new_var(out_layer, out_idx, result_val);
            self.accum_candidate = Some(var.id());
            return var;
        }

        if let (None, Some(y_cv)) = (x_const, y_const) {
            let out_layer = x_layer + 1;
            self.ensure_layer(out_layer);
            let x_idx = self.relay_to_layer(x_id, out_layer - 1);
            let out_idx = self.layers[out_layer].alloc_output();
            self.layers[out_layer].add_gates.push(make_gate_add::<C>(
                x_idx,
                out_idx,
                CircuitField::<C>::one(),
                CoefType::Constant,
            ));
            let y_folded = y_coef * y_cv;
            self.layers[out_layer]
                .const_gates
                .push(make_gate_const::<C>(out_idx, y_folded, CoefType::Constant));
            let var = self.new_var(out_layer, out_idx, result_val);
            self.accum_candidate = Some(var.id());
            return var;
        }

        let out_layer = x_layer.max(y_layer) + 1;
        self.ensure_layer(out_layer);

        let x_idx = self.relay_to_layer(x_id, out_layer - 1);
        let y_idx = self.relay_to_layer(y_id, out_layer - 1);

        let out_idx = self.layers[out_layer].alloc_output();

        self.layers[out_layer].add_gates.push(make_gate_add::<C>(
            x_idx,
            out_idx,
            CircuitField::<C>::one(),
            CoefType::Constant,
        ));
        self.layers[out_layer].add_gates.push(make_gate_add::<C>(
            y_idx,
            out_idx,
            y_coef,
            CoefType::Constant,
        ));

        let var = self.new_var(out_layer, out_idx, result_val);
        self.accum_candidate = Some(var.id());
        var
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn finalize(mut self) -> (Circuit<FC<C>>, Vec<CircuitField<C>>) {
        let const_covered: std::collections::HashSet<usize> =
            self.layers[0].const_gates.iter().map(|g| g.o_id).collect();
        for idx in 0..self.layers[0].next_output_index {
            if !const_covered.contains(&idx) {
                self.layers[0].add_gates.push(make_gate_add::<C>(
                    idx,
                    idx,
                    CircuitField::<C>::one(),
                    CoefType::Constant,
                ));
            }
        }

        let num_zero_assertions = self.zero_assertions.len();

        if !self.zero_assertions.is_empty() {
            let max_layer = self
                .zero_assertions
                .iter()
                .map(|&vid| self.var_locations[vid].layer)
                .max()
                .unwrap();

            let output_layer = max_layer + 1;
            self.ensure_layer(output_layer);

            let assertions = self.zero_assertions.clone();
            for &var_id in &assertions {
                let input_index = self.relay_to_layer(var_id, output_layer - 1);
                let out_idx = self.layers[output_layer].alloc_output();
                self.layers[output_layer].add_gates.push(make_gate_add::<C>(
                    input_index,
                    out_idx,
                    CircuitField::<C>::one(),
                    CoefType::Constant,
                ));
            }
        }

        let num_layers = self.layers.len();
        let mut circuit_layers: Vec<CircuitLayer<FC<C>>> = Vec::with_capacity(num_layers);

        for (i, acc) in self.layers.iter().enumerate() {
            let actual_size = acc.next_output_index.max(1);
            let var_num = ceil_log2(actual_size).max(1);

            let input_var_num = if i == 0 {
                var_num
            } else {
                let prev_size = self.layers[i - 1].next_output_index.max(1);
                ceil_log2(prev_size).max(1)
            };

            circuit_layers.push(CircuitLayer {
                input_var_num,
                output_var_num: var_num,
                input_vals: Vec::new(),
                output_vals: Vec::new(),
                mul: acc.mul_gates.clone(),
                add: acc.add_gates.clone(),
                const_: acc.const_gates.clone(),
                uni: Vec::new(),
                structure_info: StructureInfo::default(),
            });
        }

        let mut circuit = Circuit {
            layers: circuit_layers,
            public_input: Vec::new(),
            expected_num_output_zeros: num_zero_assertions,
            rnd_coefs_identified: false,
            rnd_coefs: Vec::new(),
        };

        circuit.identify_rnd_coefs();
        circuit.identify_structure_info();

        if !circuit.layers.is_empty() && !circuit.layers[0].structure_info.skip_sumcheck_phase_two {
            circuit.add_input_relay_layer();
        }

        let input_size = if circuit.layers.is_empty() {
            0
        } else {
            1 << circuit.layers[0].input_var_num
        };
        let mut witness_private: Vec<CircuitField<C>> = vec![CircuitField::<C>::zero(); input_size];

        for (var_id, loc) in self.var_locations.iter().enumerate() {
            if loc.layer == 0 && loc.index < input_size && var_id > 0 {
                witness_private[loc.index] = self.witness_values[var_id];
            }
        }

        (circuit, witness_private)
    }
}

impl<C: Config> BasicAPI<C> for DirectBuilder<C> {
    fn add(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        self.add_sub_inner(x, y, false)
    }

    fn sub(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        self.add_sub_inner(x, y, true)
    }

    fn mul(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (x_id, x_val) = self.resolve_input(x);
        let (y_id, y_val) = self.resolve_input(y);

        let x_const = self.known_constants.get(&x_id).copied();
        let y_const = self.known_constants.get(&y_id).copied();

        if let Some(const_val) = x_const {
            if self.accum_candidate == Some(y_id) {
                self.accum_candidate = None;
            }
            let y_layer = self.var_locations[y_id].layer;
            let out_layer = y_layer + 1;
            self.ensure_layer(out_layer);
            let y_idx = self.relay_to_layer(y_id, out_layer - 1);
            let out_idx = self.layers[out_layer].alloc_output();
            self.layers[out_layer].add_gates.push(make_gate_add::<C>(
                y_idx,
                out_idx,
                const_val,
                CoefType::Constant,
            ));
            return self.new_var(out_layer, out_idx, x_val * y_val);
        }

        if let Some(const_val) = y_const {
            if self.accum_candidate == Some(x_id) {
                self.accum_candidate = None;
            }
            let x_layer = self.var_locations[x_id].layer;
            let out_layer = x_layer + 1;
            self.ensure_layer(out_layer);
            let x_idx = self.relay_to_layer(x_id, out_layer - 1);
            let out_idx = self.layers[out_layer].alloc_output();
            self.layers[out_layer].add_gates.push(make_gate_add::<C>(
                x_idx,
                out_idx,
                const_val,
                CoefType::Constant,
            ));
            return self.new_var(out_layer, out_idx, x_val * y_val);
        }

        if self.accum_candidate == Some(x_id) || self.accum_candidate == Some(y_id) {
            self.accum_candidate = None;
        }

        let x_layer = self.var_locations[x_id].layer;
        let y_layer = self.var_locations[y_id].layer;
        let out_layer = x_layer.max(y_layer) + 1;
        self.ensure_layer(out_layer);

        let x_idx = self.relay_to_layer(x_id, out_layer - 1);
        let y_idx = self.relay_to_layer(y_id, out_layer - 1);

        let out_idx = self.layers[out_layer].alloc_output();

        self.layers[out_layer].mul_gates.push(make_gate_mul::<C>(
            x_idx,
            y_idx,
            out_idx,
            CircuitField::<C>::one(),
            CoefType::Constant,
        ));

        self.new_var(out_layer, out_idx, x_val * y_val)
    }

    fn xor(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xy = self.mul(x.clone(), y.clone());
        let x_plus_y = self.add(x, y);
        let two_xy = self.add(xy, xy);
        self.sub(x_plus_y, two_xy)
    }

    fn or(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xy = self.mul(x.clone(), y.clone());
        let x_plus_y = self.add(x, y);
        self.sub(x_plus_y, xy)
    }

    fn and(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        self.mul(x, y)
    }

    fn div(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
        _checked: bool,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x.clone());
        let (_, y_val) = self.resolve_input(y.clone());

        let q_val = match y_val.inv() {
            Some(inv) => x_val * inv,
            None => CircuitField::<C>::zero(),
        };
        let quotient_id = self.create_constant_var(q_val);
        let quotient = Variable::from(quotient_id);

        let check = self.mul(quotient, y);
        let diff = self.sub(check, x);
        self.assert_is_zero(diff);

        quotient
    }

    fn neg(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        self.sub(0u32, x)
    }

    fn is_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        let (_, x_val) = self.resolve_input(x.clone());
        let is_z = if x_val == CircuitField::<C>::zero() {
            CircuitField::<C>::one()
        } else {
            CircuitField::<C>::zero()
        };

        let result_var_id = self.create_constant_var(is_z);
        let result_var = Variable::from(result_var_id);

        let inv_val = if x_val == CircuitField::<C>::zero() {
            CircuitField::<C>::zero()
        } else {
            x_val.inv().unwrap()
        };
        let inverse_id = self.create_constant_var(inv_val);
        let inverse = Variable::from(inverse_id);

        let x_times_inv = self.mul(x.clone(), inverse);
        let one_minus_result = self.sub(1u32, result_var);
        let diff = self.sub(x_times_inv, one_minus_result);
        self.assert_is_zero(diff);

        let x_times_result = self.mul(x, result_var);
        self.assert_is_zero(x_times_result);

        result_var
    }

    fn to_binary(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        num_bits: usize,
    ) -> Vec<Variable> {
        let (_, x_val) = self.resolve_input(x.clone());

        let mut bits_vals = Vec::with_capacity(num_bits);
        let mut current = x_val;
        let two = CircuitField::<C>::from(2u32);
        let two_inv = two.inv().unwrap();
        for _ in 0..num_bits {
            let lsb = field_to_u64(current) & 1;
            let bit_val = if lsb == 1 {
                CircuitField::<C>::one()
            } else {
                CircuitField::<C>::zero()
            };
            bits_vals.push(bit_val);
            current = (current - bit_val) * two_inv;
        }

        let mut bit_vars = Vec::with_capacity(num_bits);
        for &bv in &bits_vals {
            let b_id = self.create_constant_var(bv);
            let b_var = Variable::from(b_id);
            self.assert_is_bool(b_var);
            bit_vars.push(b_var);
        }

        let reconstructed = self.from_binary(&bit_vars);
        let diff = self.sub(x, reconstructed);
        self.assert_is_zero(diff);

        bit_vars
    }

    fn gt(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let width = 253;
        let x_bits = self.to_binary(x, width);
        let y_bits = self.to_binary(y, width);

        let mut result = self.constant(CircuitField::<C>::zero());
        let mut decided = self.constant(CircuitField::<C>::zero());

        for i in (0..width).rev() {
            let not_y = self.sub(1u32, y_bits[i]);
            let x_gt_y_here = self.mul(x_bits[i], not_y);
            let not_x = self.sub(1u32, x_bits[i]);
            let y_gt_x_here = self.mul(y_bits[i], not_x);
            let differ = self.or(x_gt_y_here, y_gt_x_here);

            let not_decided = self.sub(1u32, decided);
            let first_diff_x_wins = self.mul(not_decided, x_gt_y_here);
            result = self.add(result, first_diff_x_wins);

            let first_diff = self.mul(not_decided, differ);
            decided = self.add(decided, first_diff);
        }

        result
    }

    fn geq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let diff = self.sub(x.clone(), y.clone());
        let eq = self.is_zero(diff);
        let gt_val = self.gt(x, y);
        self.or(eq, gt_val)
    }

    fn assert_is_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        let (x_id, _) = self.resolve_input(x);
        if self.accum_candidate == Some(x_id) {
            self.accum_candidate = None;
        }
        self.zero_assertions.push(x_id);
    }

    fn assert_is_non_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        let inv = self.inverse(x);
        let _ = inv;
    }

    fn assert_is_bool(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        let x2 = self.mul(x.clone(), x.clone());
        let diff = self.sub(x2, x);
        self.assert_is_zero(diff);
    }

    fn get_random_value(&mut self) -> Variable {
        let idx = self.alloc_input_slot();
        self.layers[0].const_gates.push(make_gate_const::<C>(
            idx,
            CircuitField::<C>::zero(),
            CoefType::Random,
        ));

        let var_id = self.var_locations.len();
        self.var_locations.push(VarLocation {
            layer: 0,
            index: idx,
        });
        self.witness_values.push(CircuitField::<C>::one());
        Variable::from(var_id)
    }

    fn new_hint(
        &mut self,
        hint_key: &str,
        inputs: &[Variable],
        num_outputs: usize,
    ) -> Vec<Variable> {
        let hint_id = hint_key_to_id(hint_key);

        let input_vals: Vec<CircuitField<C>> =
            inputs.iter().map(|v| self.witness_values[v.id()]).collect();

        let output_vals = self
            .hint_registry
            .call(hint_id, &input_vals, num_outputs)
            .unwrap_or_else(|e| panic!("Hint {hint_key} failed: {e:?}"));

        output_vals
            .into_iter()
            .map(|val| {
                let idx = self.alloc_input_slot();
                let var_id = self.var_locations.len();
                self.var_locations.push(VarLocation {
                    layer: 0,
                    index: idx,
                });
                self.witness_values.push(val);
                Variable::from(var_id)
            })
            .collect()
    }

    fn constant(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        match x.convert_to_variable_or_value() {
            VariableOrValue::Variable(v) => v,
            VariableOrValue::Value(val) => {
                let var_id = self.create_constant_var(val);
                Variable::from(var_id)
            }
        }
    }

    fn constant_value(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Option<CircuitField<C>> {
        match x.convert_to_variable_or_value() {
            VariableOrValue::Variable(v) => Some(self.witness_values[v.id()]),
            VariableOrValue::Value(val) => Some(val),
        }
    }
}

fn field_to_u64<F: Field>(f: F) -> u64 {
    let mut bytes = Vec::new();
    <F as ExpSerde>::serialize_into(&f, &mut bytes).unwrap();
    if bytes.len() >= 8 {
        u64::from_le_bytes(bytes[0..8].try_into().unwrap())
    } else {
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(&bytes);
        u64::from_le_bytes(buf)
    }
}

fn u64_to_field<F: Field>(v: u64) -> F {
    F::from_u256(U256::from(v))
}

fn bool_to_field<F: Field>(b: bool) -> F {
    if b { F::one() } else { F::zero() }
}

impl<C: Config> UnconstrainedAPI<C> for DirectBuilder<C> {
    fn unconstrained_identity(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        let (_, val) = self.resolve_input(x);
        let var_id = self.create_constant_var(val);
        Variable::from(var_id)
    }

    fn unconstrained_add(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id = self.create_constant_var(x_val + y_val);
        Variable::from(var_id)
    }

    fn unconstrained_mul(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id = self.create_constant_var(x_val * y_val);
        Variable::from(var_id)
    }

    fn unconstrained_div(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let result = match y_val.inv() {
            Some(inv) => x_val * inv,
            None => CircuitField::<C>::zero(),
        };
        let var_id = self.create_constant_var(result);
        Variable::from(var_id)
    }

    fn unconstrained_pow(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let exp = field_to_u64(y_val);
        let mut result = CircuitField::<C>::one();
        let mut base = x_val;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result *= base;
            }
            base = base * base;
            e >>= 1;
        }
        let var_id = self.create_constant_var(result);
        Variable::from(var_id)
    }

    fn unconstrained_int_div(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let xn = field_to_u64(x_val);
        let yn = field_to_u64(y_val);
        let q = if yn == 0 { 0 } else { xn / yn };
        let var_id = self.create_constant_var(u64_to_field(q));
        Variable::from(var_id)
    }

    fn unconstrained_mod(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let xn = field_to_u64(x_val);
        let yn = field_to_u64(y_val);
        let r = if yn == 0 { 0 } else { xn % yn };
        let var_id = self.create_constant_var(u64_to_field(r));
        Variable::from(var_id)
    }

    fn unconstrained_shift_l(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let xn = field_to_u64(x_val);
        #[allow(clippy::cast_possible_truncation)]
        let shift = field_to_u64(y_val) as u32;
        let result = xn.checked_shl(shift).unwrap_or(0);
        let var_id = self.create_constant_var(u64_to_field(result));
        Variable::from(var_id)
    }

    fn unconstrained_shift_r(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let xn = field_to_u64(x_val);
        #[allow(clippy::cast_possible_truncation)]
        let shift = field_to_u64(y_val) as u32;
        let result = xn.checked_shr(shift).unwrap_or(0);
        let var_id = self.create_constant_var(u64_to_field(result));
        Variable::from(var_id)
    }

    fn unconstrained_lesser_eq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id =
            self.create_constant_var(bool_to_field(field_to_u64(x_val) <= field_to_u64(y_val)));
        Variable::from(var_id)
    }

    fn unconstrained_greater_eq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id =
            self.create_constant_var(bool_to_field(field_to_u64(x_val) >= field_to_u64(y_val)));
        Variable::from(var_id)
    }

    fn unconstrained_lesser(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id =
            self.create_constant_var(bool_to_field(field_to_u64(x_val) < field_to_u64(y_val)));
        Variable::from(var_id)
    }

    fn unconstrained_greater(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id =
            self.create_constant_var(bool_to_field(field_to_u64(x_val) > field_to_u64(y_val)));
        Variable::from(var_id)
    }

    fn unconstrained_eq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id = self.create_constant_var(bool_to_field(x_val == y_val));
        Variable::from(var_id)
    }

    fn unconstrained_not_eq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let var_id = self.create_constant_var(bool_to_field(x_val != y_val));
        Variable::from(var_id)
    }

    fn unconstrained_bool_or(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let xb = x_val != CircuitField::<C>::zero();
        let yb = y_val != CircuitField::<C>::zero();
        let var_id = self.create_constant_var(bool_to_field(xb || yb));
        Variable::from(var_id)
    }

    fn unconstrained_bool_and(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let xb = x_val != CircuitField::<C>::zero();
        let yb = y_val != CircuitField::<C>::zero();
        let var_id = self.create_constant_var(bool_to_field(xb && yb));
        Variable::from(var_id)
    }

    fn unconstrained_bit_or(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let result = field_to_u64(x_val) | field_to_u64(y_val);
        let var_id = self.create_constant_var(u64_to_field(result));
        Variable::from(var_id)
    }

    fn unconstrained_bit_and(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let result = field_to_u64(x_val) & field_to_u64(y_val);
        let var_id = self.create_constant_var(u64_to_field(result));
        Variable::from(var_id)
    }

    fn unconstrained_bit_xor(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let (_, x_val) = self.resolve_input(x);
        let (_, y_val) = self.resolve_input(y);
        let result = field_to_u64(x_val) ^ field_to_u64(y_val);
        let var_id = self.create_constant_var(u64_to_field(result));
        Variable::from(var_id)
    }
}

impl<C: Config> RootAPI<C> for DirectBuilder<C> {
    fn memorized_simple_call<F: Fn(&mut Self, &Vec<Variable>) -> Vec<Variable> + 'static>(
        &mut self,
        f: F,
        inputs: &[Variable],
    ) -> Vec<Variable> {
        f(self, &inputs.to_vec())
    }

    fn hash_to_sub_circuit_id(&mut self, hash: &[u8; 32]) -> usize {
        let mut id_bytes = [0u8; 8];
        id_bytes.copy_from_slice(&hash[0..8]);
        usize::from_le_bytes(id_bytes)
    }

    fn call_sub_circuit<F: FnOnce(&mut Self, &Vec<Variable>) -> Vec<Variable>>(
        &mut self,
        _circuit_id: usize,
        inputs: &[Variable],
        f: F,
    ) -> Vec<Variable> {
        f(self, &inputs.to_vec())
    }

    fn register_sub_circuit_output_structure(&mut self, circuit_id: usize, structure: Vec<usize>) {
        self.sub_circuit_structures.insert(circuit_id, structure);
    }

    fn get_sub_circuit_output_structure(&self, circuit_id: usize) -> Vec<usize> {
        self.sub_circuit_structures
            .get(&circuit_id)
            .cloned()
            .unwrap_or_default()
    }

    fn set_outputs(&mut self, outputs: Vec<Variable>) {
        self.accum_candidate = None;
        self.output_vars = outputs.iter().map(Variable::id).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arith::SimdField;
    use expander_compiler::expander_binary::executor;
    use expander_compiler::frontend::{BN254Config, ChallengeField};
    use expander_compiler::gkr_engine::MPIConfig;

    type BN254Field = CircuitField<BN254Config>;

    fn prove_verify_direct(circuit: &mut Circuit<FC<BN254Config>>, witness: &[BN254Field]) -> bool {
        type Simd = <<BN254Config as GKREngine>::FieldConfig
            as expander_compiler::gkr_engine::FieldEngine>::SimdCircuitField;
        let ps = <Simd as SimdField>::PACK_SIZE;
        circuit.layers[0].input_vals = witness.iter().map(|&v| Simd::pack(&vec![v; ps])).collect();
        circuit.evaluate();

        let output = &circuit.layers.last().unwrap().output_vals;
        let n_zeros = circuit.expected_num_output_zeros;
        for (i, v) in output[..n_zeros].iter().enumerate() {
            assert!(
                v.is_zero(),
                "output[{i}] is not zero (expected_num_output_zeros={n_zeros})"
            );
        }

        let mpi = MPIConfig::prover_new();
        let (claimed_v, proof) = executor::prove::<BN254Config>(circuit, mpi);
        let proof_bytes = executor::dump_proof_and_claimed_v(&proof, &claimed_v).unwrap();

        let (proof2, cv2) =
            executor::load_proof_and_claimed_v::<ChallengeField<BN254Config>>(&proof_bytes)
                .unwrap();
        let mpi_v = MPIConfig::verifier_new(1);
        executor::verify::<BN254Config>(circuit, mpi_v, &proof2, &cv2)
    }

    #[test]
    fn test_coalesce_then_consume() {
        let vals: Vec<BN254Field> = (1..=5u32).map(BN254Field::from).collect();
        let hint_registry = HintRegistry::new();
        let mut b = DirectBuilder::<BN254Config>::new(&vals, hint_registry);

        let v1 = Variable::from(1);
        let v2 = Variable::from(2);
        let v3 = Variable::from(3);
        let v4 = Variable::from(4);
        let v5 = Variable::from(5);

        let sum12 = b.add(v1, v2);
        let sum123 = b.add(sum12, v3);
        let sum1234 = b.add(sum123, v4);

        let product = b.mul(sum1234, v5);

        let expected_sum = BN254Field::from(10u32);
        let expected_prod = BN254Field::from(50u32);
        let d1 = b.sub(sum1234, expected_sum);
        b.assert_is_zero(d1);
        let d2 = b.sub(product, expected_prod);
        b.assert_is_zero(d2);

        b.set_outputs(vec![product]);
        let (mut circuit, witness) = b.finalize();
        assert!(prove_verify_direct(&mut circuit, &witness));
    }

    #[test]
    fn test_coalesce_interleaved_with_mul() {
        let hint_registry = HintRegistry::new();
        let inputs: Vec<BN254Field> = (0..6).map(|i| BN254Field::from(i as u32 + 1)).collect();
        let mut b = DirectBuilder::<BN254Config>::new(&inputs, hint_registry);

        let a0 = Variable::from(1);
        let b0 = Variable::from(2);
        let a1 = Variable::from(3);
        let b1 = Variable::from(4);
        let a2 = Variable::from(5);
        let b2 = Variable::from(6);

        let p0 = b.mul(a0, b0);
        let p1 = b.mul(a1, b1);
        let acc = b.add(p0, p1);
        let p2 = b.mul(a2, b2);
        let acc = b.add(acc, p2);

        let expected = BN254Field::from(2u32 + 3 * 4 + 5 * 6);
        let d = b.sub(acc, expected);
        b.assert_is_zero(d);
        b.set_outputs(vec![acc]);
        let (mut circuit, witness) = b.finalize();
        assert!(prove_verify_direct(&mut circuit, &witness));
    }

    #[test]
    fn test_sub_with_accum_as_subtrahend() {
        let hint_registry = HintRegistry::new();
        let inputs: Vec<BN254Field> = vec![
            BN254Field::from(10u32),
            BN254Field::from(3u32),
            BN254Field::from(2u32),
            BN254Field::from(100u32),
        ];
        let mut b = DirectBuilder::<BN254Config>::new(&inputs, hint_registry);

        let v10 = Variable::from(1);
        let v3 = Variable::from(2);
        let v2 = Variable::from(3);
        let v100 = Variable::from(4);

        let acc = b.add(v10, v3);
        let acc = b.add(acc, v2);
        let result = b.sub(v100, acc);

        let expected = BN254Field::from(100u32 - 10 - 3 - 2);
        let d = b.sub(result, expected);
        b.assert_is_zero(d);
        b.set_outputs(vec![result]);
        let (mut circuit, witness) = b.finalize();
        assert!(prove_verify_direct(&mut circuit, &witness));
    }

    #[test]
    fn test_add_mul_ecc_baseline() {
        use expander_compiler::frontend::{CompileOptions, Define, compile, declare_circuit};

        declare_circuit!(TestCirc {
            a: Variable,
            b: Variable,
            expected_sum: Variable,
            expected_prod: Variable,
        });

        impl Define<BN254Config> for TestCirc<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let sum = api.add(self.a, self.b);
                let prod = api.mul(self.a, self.b);
                let d1 = api.sub(sum, self.expected_sum);
                api.assert_is_zero(d1);
                let d2 = api.sub(prod, self.expected_prod);
                api.assert_is_zero(d2);
                api.set_outputs(vec![sum, prod]);
            }
        }

        let t0 = std::time::Instant::now();
        let circuit = TestCirc::default();
        let cr = compile(&circuit, CompileOptions::default()).unwrap();

        let assignment = TestCirc::<BN254Field> {
            a: BN254Field::from(3u32),
            b: BN254Field::from(5u32),
            expected_sum: BN254Field::from(8u32),
            expected_prod: BN254Field::from(15u32),
        };

        let hint_registry = HintRegistry::new();
        let witness = cr
            .witness_solver
            .solve_witness_with_hints(&assignment, &hint_registry)
            .unwrap();
        let mut exp_circuit = cr.layered_circuit.export_to_expander_flatten();
        let (simd_in, simd_pub) = witness.to_simd();
        exp_circuit.layers[0].input_vals = simd_in;
        exp_circuit.public_input = simd_pub;
        exp_circuit.evaluate();

        let mpi = MPIConfig::prover_new();
        let (claimed_v, proof) = executor::prove::<BN254Config>(&mut exp_circuit, mpi);
        let proof_bytes = executor::dump_proof_and_claimed_v(&proof, &claimed_v).unwrap();

        let (proof2, cv2) =
            executor::load_proof_and_claimed_v::<ChallengeField<BN254Config>>(&proof_bytes)
                .unwrap();
        let mpi_v = MPIConfig::verifier_new(1);
        let ok = executor::verify::<BN254Config>(&mut exp_circuit, mpi_v, &proof2, &cv2);
        let elapsed = t0.elapsed();
        println!("ECC IR pipeline: {elapsed:?}");
        assert!(ok);
    }

    fn approx_heap_bytes_direct(n: usize) -> usize {
        let a_vals: Vec<BN254Field> = (0..n).map(|i| BN254Field::from(i as u32 + 1)).collect();
        let b_vals: Vec<BN254Field> = (0..n).map(|i| BN254Field::from(i as u32 + 2)).collect();
        let mut all_inputs = Vec::with_capacity(2 * n);
        all_inputs.extend_from_slice(&a_vals);
        all_inputs.extend_from_slice(&b_vals);

        let hint_registry = HintRegistry::new();
        let mut builder = DirectBuilder::<BN254Config>::new(&all_inputs, hint_registry);
        let mut acc = builder.mul(Variable::from(1), Variable::from(n + 1));
        for i in 1..n {
            let prod = builder.mul(Variable::from(i + 1), Variable::from(n + i + 1));
            acc = builder.add(acc, prod);
        }
        builder.set_outputs(vec![acc]);
        let (circuit, witness) = builder.finalize();

        let layers = circuit.layers.len();
        let witness_bytes = witness.len() * std::mem::size_of::<BN254Field>();
        println!("  DirectBuilder: {layers} layers, witness {witness_bytes} bytes");
        for (i, l) in circuit.layers.iter().enumerate() {
            println!(
                "    layer {i}: mul={} add={} const={} uni={} in_var={} out_var={} skip_p2={}",
                l.mul.len(),
                l.add.len(),
                l.const_.len(),
                l.uni.len(),
                l.input_var_num,
                l.output_var_num,
                l.structure_info.skip_sumcheck_phase_two
            );
        }
        0
    }

    fn approx_heap_bytes_ecc(n: usize) -> usize {
        use expander_compiler::frontend::{CompileOptions, Define, compile, declare_circuit};

        declare_circuit!(DotCircEcc {
            a: [Variable],
            b: [Variable]
        });
        impl Define<BN254Config> for DotCircEcc<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let n = self.a.len();
                let mut acc = api.mul(self.a[0], self.b[0]);
                for i in 1..n {
                    let prod = api.mul(self.a[i], self.b[i]);
                    acc = api.add(acc, prod);
                }
                api.set_outputs(vec![acc]);
            }
        }

        let circuit = DotCircEcc {
            a: vec![Variable::default(); n],
            b: vec![Variable::default(); n],
        };
        let cr = compile(&circuit, CompileOptions::default()).unwrap();
        let exp = cr.layered_circuit.export_to_expander_flatten();
        let layers = exp.layers.len();
        println!("  ECC pipeline:  {layers} layers");
        for (i, l) in exp.layers.iter().enumerate() {
            println!(
                "    layer {i}: mul={} add={} const={} uni={} in_var={} out_var={} skip_p2={}",
                l.mul.len(),
                l.add.len(),
                l.const_.len(),
                l.uni.len(),
                l.input_var_num,
                l.output_var_num,
                l.structure_info.skip_sumcheck_phase_two
            );
        }
        0
    }

    #[test]
    #[ignore]
    fn test_memory_comparison() {
        for &n in &[1000, 10000] {
            println!("--- n={n} ---");
            let ecc = approx_heap_bytes_ecc(n);
            let direct = approx_heap_bytes_direct(n);
            println!("  ECC ~{} KiB, Direct ~{} KiB\n", ecc, direct);
        }
    }

    fn dot_product_direct(n: usize) -> std::time::Duration {
        let a_vals: Vec<BN254Field> = (0..n).map(|i| BN254Field::from(i as u32 + 1)).collect();
        let b_vals: Vec<BN254Field> = (0..n).map(|i| BN254Field::from(i as u32 + 2)).collect();
        let mut all_inputs = Vec::with_capacity(2 * n);
        all_inputs.extend_from_slice(&a_vals);
        all_inputs.extend_from_slice(&b_vals);

        let t0 = std::time::Instant::now();
        let hint_registry = HintRegistry::new();
        let mut builder = DirectBuilder::<BN254Config>::new(&all_inputs, hint_registry);

        let mut acc = builder.mul(Variable::from(1), Variable::from(n + 1));
        for i in 1..n {
            let prod = builder.mul(Variable::from(i + 1), Variable::from(n + i + 1));
            acc = builder.add(acc, prod);
        }
        builder.set_outputs(vec![acc]);

        let (mut circuit, witness) = builder.finalize();
        let build_time = t0.elapsed();

        assert!(prove_verify_direct(&mut circuit, &witness));
        println!(
            "  DirectBuilder (n={n}): build={build_time:?} total={:?}",
            t0.elapsed()
        );
        t0.elapsed()
    }

    fn dot_product_ecc(n: usize) -> std::time::Duration {
        use expander_compiler::frontend::{CompileOptions, Define, compile, declare_circuit};

        declare_circuit!(DotCirc {
            a: [Variable],
            b: [Variable]
        });

        impl Define<BN254Config> for DotCirc<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let n = self.a.len();
                let mut acc = api.mul(self.a[0], self.b[0]);
                for i in 1..n {
                    let prod = api.mul(self.a[i], self.b[i]);
                    acc = api.add(acc, prod);
                }
                api.set_outputs(vec![acc]);
            }
        }

        let t0 = std::time::Instant::now();
        let circuit = DotCirc {
            a: vec![Variable::default(); n],
            b: vec![Variable::default(); n],
        };
        let cr = compile(&circuit, CompileOptions::default()).unwrap();
        let compile_time = t0.elapsed();

        let assignment = DotCirc::<BN254Field> {
            a: (0..n).map(|i| BN254Field::from(i as u32 + 1)).collect(),
            b: (0..n).map(|i| BN254Field::from(i as u32 + 2)).collect(),
        };
        let hint_registry = HintRegistry::new();
        let witness = cr
            .witness_solver
            .solve_witness_with_hints(&assignment, &hint_registry)
            .unwrap();
        let mut exp_circuit = cr.layered_circuit.export_to_expander_flatten();
        let (simd_in, simd_pub) = witness.to_simd();
        exp_circuit.layers[0].input_vals = simd_in;
        exp_circuit.public_input = simd_pub;
        exp_circuit.evaluate();

        let mpi = MPIConfig::prover_new();
        let (claimed_v, proof) = executor::prove::<BN254Config>(&mut exp_circuit, mpi);
        let proof_bytes = executor::dump_proof_and_claimed_v(&proof, &claimed_v).unwrap();
        let (proof2, cv2) =
            executor::load_proof_and_claimed_v::<ChallengeField<BN254Config>>(&proof_bytes)
                .unwrap();
        let mpi_v = MPIConfig::verifier_new(1);
        let ok = executor::verify::<BN254Config>(&mut exp_circuit, mpi_v, &proof2, &cv2);
        assert!(ok);
        println!(
            "  ECC pipeline   (n={n}): compile={compile_time:?} total={:?}",
            t0.elapsed()
        );
        t0.elapsed()
    }

    #[test]
    #[ignore]
    fn test_dot_product_scaling() {
        for &n in &[100, 1000, 10000] {
            println!("--- dot product n={n} ---");
            let ecc = dot_product_ecc(n);
            let direct = dot_product_direct(n);
            println!(
                "  speedup: {:.1}x\n",
                ecc.as_secs_f64() / direct.as_secs_f64()
            );
        }
    }

    #[test]
    fn test_add_mul_prove_verify() {
        let t0 = std::time::Instant::now();
        let a = BN254Field::from(3u32);
        let b = BN254Field::from(5u32);
        let expected_sum = BN254Field::from(8u32);
        let expected_prod = BN254Field::from(15u32);

        let hint_registry = HintRegistry::new();
        let mut builder = DirectBuilder::<BN254Config>::new(&[a, b], hint_registry);

        let va = Variable::from(1);
        let vb = Variable::from(2);
        let sum = builder.add(va, vb);
        let prod = builder.mul(va, vb);

        let diff_sum = builder.sub(sum, expected_sum);
        builder.assert_is_zero(diff_sum);
        let diff_prod = builder.sub(prod, expected_prod);
        builder.assert_is_zero(diff_prod);

        builder.set_outputs(vec![sum, prod]);
        let (mut circuit, witness) = builder.finalize();
        assert!(prove_verify_direct(&mut circuit, &witness));
        println!("DirectBuilder: {:?}", t0.elapsed());
    }
}
