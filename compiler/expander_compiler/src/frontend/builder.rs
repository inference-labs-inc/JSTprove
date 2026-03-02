use std::collections::HashMap;
use std::convert::From;

use ethnum::U256;
use tiny_keccak::Hasher;

use crate::{
    circuit::{
        config::Config,
        ir::{
            expr::{LinComb, LinCombTerm},
            source::{self, Constraint as SourceConstraint, Instruction as SourceInstruction},
        },
        layered::Coef,
    },
    field::{Field, FieldArith},
    hints::{self, registry::hint_key_to_id},
    utils::function_id::get_function_id,
};

use super::{
    api::{BasicAPI, RootAPI, UnconstrainedAPI},
    CircuitField,
};

#[derive(Clone)]
pub struct Builder<C: Config> {
    instructions: Vec<SourceInstruction<C>>,
    constraints: Vec<SourceConstraint>,
    var_const_id: Vec<usize>,
    const_values: Vec<CircuitField<C>>,
    num_inputs: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Variable {
    id: usize,
}

impl Variable {
    pub fn id(&self) -> usize {
        self.id
    }
}

pub fn new_variable(id: usize) -> Variable {
    Variable { id }
}

// impl Variable for From<usize> trait
impl From<usize> for Variable {
    fn from(id: usize) -> Self {
        Variable { id }
    }
}

pub fn get_variable_id(v: Variable) -> usize {
    v.id
}

pub fn ensure_variable_valid(v: Variable) {
    if v.id == 0 {
        panic!("Variable(0) is not allowed in API calls");
    }
}

pub fn ensure_variables_valid(vs: &[Variable]) {
    for v in vs {
        ensure_variable_valid(*v);
    }
}

pub enum VariableOrValue<F: Field> {
    Variable(Variable),
    Value(F),
}

pub trait ToVariableOrValue<F: Field>: Clone {
    fn convert_to_variable_or_value(self) -> VariableOrValue<F>;
}

trait NotVariable {}
impl NotVariable for u32 {}
impl NotVariable for U256 {}
impl<F: Field> NotVariable for F {}

impl<F: Field, T: Into<F> + NotVariable + Clone> ToVariableOrValue<F> for T {
    fn convert_to_variable_or_value(self) -> VariableOrValue<F> {
        VariableOrValue::Value(self.into())
    }
}

impl<F: Field> ToVariableOrValue<F> for Variable {
    fn convert_to_variable_or_value(self) -> VariableOrValue<F> {
        // In almost all API functions, the argument is impl ToVariableOrValue<CircuitField<C>>.
        // (Actually it's all but new_hint and memorized_simple_call)
        // We need to prevent invalid (default) Variables from passing into the functions.
        // And here's the best location to do it.
        ensure_variable_valid(self);
        VariableOrValue::Variable(self)
    }
}

impl<F: Field> ToVariableOrValue<F> for &Variable {
    fn convert_to_variable_or_value(self) -> VariableOrValue<F> {
        (*self).convert_to_variable_or_value()
    }
}

impl<C: Config> Builder<C> {
    pub fn new(num_inputs: usize) -> (Self, Vec<Variable>) {
        (
            Builder {
                instructions: Vec::new(),
                constraints: Vec::new(),
                num_inputs,
                var_const_id: vec![0; num_inputs + 1],
                const_values: vec![CircuitField::<C>::zero()],
            },
            (1..=num_inputs).map(|id| Variable { id }).collect(),
        )
    }

    pub fn build(self, outputs: &[Variable]) -> source::Circuit<C> {
        source::Circuit {
            instructions: self.instructions,
            constraints: self.constraints,
            num_inputs: self.num_inputs,
            outputs: outputs.iter().map(|v| v.id).collect(),
        }
    }

    fn convert_to_variable<T: ToVariableOrValue<CircuitField<C>>>(&mut self, value: T) -> Variable {
        match value.convert_to_variable_or_value() {
            VariableOrValue::Variable(v) => v,
            VariableOrValue::Value(v) => {
                self.instructions
                    .push(SourceInstruction::ConstantLike(Coef::Constant(v)));
                self.var_const_id.push(self.const_values.len());
                self.const_values.push(v);
                Variable {
                    id: self.var_const_id.len() - 1,
                }
            }
        }
    }

    fn new_var(&mut self) -> Variable {
        self.var_const_id.push(0);
        Variable {
            id: self.var_const_id.len() - 1,
        }
    }
}

impl<C: Config> BasicAPI<C> for Builder<C> {
    fn add(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                return self.constant(xv + yv);
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions.push(SourceInstruction::LinComb(LinComb {
            terms: vec![
                LinCombTerm {
                    var: x.id,
                    coef: CircuitField::<C>::one(),
                },
                LinCombTerm {
                    var: y.id,
                    coef: CircuitField::<C>::one(),
                },
            ],
            constant: CircuitField::<C>::zero(),
        }));
        self.new_var()
    }

    fn sub(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                return self.constant(xv - yv);
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions.push(SourceInstruction::LinComb(LinComb {
            terms: vec![
                LinCombTerm {
                    var: x.id,
                    coef: CircuitField::<C>::one(),
                },
                LinCombTerm {
                    var: y.id,
                    coef: -CircuitField::<C>::one(),
                },
            ],
            constant: CircuitField::<C>::zero(),
        }));
        self.new_var()
    }

    fn neg(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        let xc = self.constant_value(x.clone());
        if let Some(xv) = xc {
            return self.constant(-xv);
        }
        let x = self.convert_to_variable(x);
        self.instructions.push(SourceInstruction::LinComb(LinComb {
            terms: vec![LinCombTerm {
                var: x.id,
                coef: -CircuitField::<C>::one(),
            }],
            constant: CircuitField::<C>::zero(),
        }));
        self.new_var()
    }

    fn mul(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                return self.constant(xv * yv);
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions
            .push(SourceInstruction::Mul(vec![x.id, y.id]));
        self.new_var()
    }

    fn div(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
        checked: bool,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                let res = if yv.is_zero() {
                    if checked || !xv.is_zero() {
                        panic!("division by zero");
                    }
                    CircuitField::<C>::zero()
                } else {
                    xv * yv.inv().unwrap()
                };
                return self.constant(res);
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions.push(SourceInstruction::Div {
            x: x.id,
            y: y.id,
            checked,
        });
        self.new_var()
    }

    fn xor(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                self.assert_is_bool(xv);
                self.assert_is_bool(yv);
                return self.constant(CircuitField::<C>::from((xv != yv) as u32));
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions.push(SourceInstruction::BoolBinOp {
            x: x.id,
            y: y.id,
            op: source::BoolBinOpType::Xor,
        });
        self.new_var()
    }

    fn or(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                self.assert_is_bool(xv);
                self.assert_is_bool(yv);
                return self.constant(CircuitField::<C>::from(
                    (!xv.is_zero() || !yv.is_zero()) as u32,
                ));
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions.push(SourceInstruction::BoolBinOp {
            x: x.id,
            y: y.id,
            op: source::BoolBinOpType::Or,
        });
        self.new_var()
    }

    fn and(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let xc = self.constant_value(x.clone());
        let yc = self.constant_value(y.clone());
        if let Some(xv) = xc {
            if let Some(yv) = yc {
                self.assert_is_bool(xv);
                self.assert_is_bool(yv);
                return self.constant(CircuitField::<C>::from(
                    (!xv.is_zero() && !yv.is_zero()) as u32,
                ));
            }
        }
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        self.instructions.push(SourceInstruction::BoolBinOp {
            x: x.id,
            y: y.id,
            op: source::BoolBinOpType::And,
        });
        self.new_var()
    }

    fn is_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        let xc = self.constant_value(x.clone());
        if let Some(xv) = xc {
            return self.constant(CircuitField::<C>::from(xv.is_zero() as u32));
        }
        let x = self.convert_to_variable(x);
        self.instructions.push(SourceInstruction::IsZero(x.id));
        self.new_var()
    }

    fn to_binary(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        num_bits: usize,
    ) -> Vec<Variable> {
        let xc = self.constant_value(x.clone());
        if let Some(xv) = xc {
            let values = hints::to_binary(xv, num_bits).unwrap();
            return values.iter().map(|v| self.constant(*v)).collect();
        }
        let x = self.convert_to_variable(x);
        self.instructions
            .push(SourceInstruction::ToBinary { x: x.id, num_bits });
        (0..num_bits).map(|_| self.new_var()).collect()
    }

    fn assert_is_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        let xc = self.constant_value(x.clone());
        if let Some(xv) = xc {
            if !xv.is_zero() {
                panic!("assert_is_zero failed");
            }
        }
        let x = self.convert_to_variable(x);
        self.constraints.push(SourceConstraint {
            typ: source::ConstraintType::Zero,
            var: x.id,
        });
    }

    fn assert_is_non_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        let xc = self.constant_value(x.clone());
        if let Some(xv) = xc {
            if xv.is_zero() {
                panic!("assert_is_zero failed");
            }
        }
        let x = self.convert_to_variable(x);
        self.constraints.push(SourceConstraint {
            typ: source::ConstraintType::NonZero,
            var: x.id,
        });
    }

    fn assert_is_bool(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        let xc = self.constant_value(x.clone());
        if let Some(xv) = xc {
            if !xv.is_zero() && xv != CircuitField::<C>::one() {
                panic!("assert_is_bool failed");
            }
        }
        let x = self.convert_to_variable(x);
        self.constraints.push(SourceConstraint {
            typ: source::ConstraintType::Bool,
            var: x.id,
        });
    }

    fn get_random_value(&mut self) -> Variable {
        self.instructions
            .push(SourceInstruction::ConstantLike(Coef::Random));
        self.new_var()
    }

    fn new_hint(
        &mut self,
        hint_key: &str,
        inputs: &[Variable],
        num_outputs: usize,
    ) -> Vec<Variable> {
        ensure_variables_valid(inputs);
        self.instructions.push(SourceInstruction::Hint {
            hint_id: hint_key_to_id(hint_key),
            inputs: inputs.iter().map(|v| v.id).collect(),
            num_outputs,
        });
        (0..num_outputs).map(|_| self.new_var()).collect()
    }

    fn constant(&mut self, value: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        self.convert_to_variable(value)
    }

    fn constant_value(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Option<CircuitField<C>> {
        match x.convert_to_variable_or_value() {
            VariableOrValue::Variable(v) => {
                let t = self.var_const_id[v.id];
                if t != 0 {
                    Some(self.const_values[t])
                } else {
                    None
                }
            }
            VariableOrValue::Value(v) => Some(v),
        }
    }

    // return 1 if x > y; 0 otherwise
    //
    fn gt(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let one = self.constant(CircuitField::<C>::one());

        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);

        // Decompose both numbers to bits
        let num_bits = match CircuitField::<C>::FIELD_SIZE {
            256 => 254, // BN254's field size was set to 256
            32 => 31,   // M31's field size was set to 31
            _ => panic!("Unsupported field size"),
        };
        let x_bits = self.to_binary(x, num_bits);
        let y_bits = self.to_binary(y, num_bits);

        for (i, x_bit) in x_bits.iter().enumerate() {
            self.display(format!("x[{i}]").as_ref(), x_bit);
        }

        // Compare bit by bit from MSB to LSB
        let mut result = self.constant(CircuitField::<C>::zero());
        let mut still_equal = self.constant(CircuitField::<C>::one());

        for i in (0..num_bits).rev() {
            let x_bit = x_bits[i];
            let y_bit = y_bits[i];

            // x_bit > y_bit when x_bit=1 and y_bit=0
            let not_y_bit = self.sub(one, y_bit);
            let x_greater_at_bit = self.and(x_bit, not_y_bit);

            // Set result to 1 if still equal and x > y at this bit
            let should_set_result = self.and(still_equal, x_greater_at_bit);
            result = self.or(result, should_set_result);

            // Check if bits are equal
            let bits_different = self.xor(x_bit, y_bit);
            let bits_equal = self.sub(one, bits_different);

            // Update still_equal: remains 1 only if bits are equal
            still_equal = self.and(still_equal, bits_equal);
        }

        result
    }

    // return 1 if x >= y; 0 otherwise
    fn geq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        // x >= y is equivalent to NOT(y > x)
        let y_gt_x = self.gt(y, x);
        let one = self.constant(CircuitField::<C>::one());
        self.sub(one, y_gt_x)
    }
}

// write macro rules for unconstrained binary op definition
macro_rules! unconstrained_binary_op {
    ($name:ident,$op_name:ident) => {
        fn $name(
            &mut self,
            x: impl ToVariableOrValue<CircuitField<C>>,
            y: impl ToVariableOrValue<CircuitField<C>>,
        ) -> Variable {
            let x = self.convert_to_variable(x);
            let y = self.convert_to_variable(y);
            self.instructions
                .push(SourceInstruction::UnconstrainedBinOp {
                    x: x.id,
                    y: y.id,
                    op: source::UnconstrainedBinOpType::$op_name,
                });
            self.new_var()
        }
    };
}

impl<C: Config> UnconstrainedAPI<C> for Builder<C> {
    fn unconstrained_identity(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        let x = self.convert_to_variable(x);
        self.instructions.push(SourceInstruction::Hint {
            hint_id: hints::BuiltinHintIds::Identity as u64 as usize,
            inputs: vec![x.id],
            num_outputs: 1,
        });
        self.new_var()
    }
    fn unconstrained_add(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        let z = self.add(x, y);
        self.unconstrained_identity(z)
    }
    fn unconstrained_mul(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        let x = self.convert_to_variable(x);
        let y = self.convert_to_variable(y);
        let z = self.mul(x, y);
        self.unconstrained_identity(z)
    }
    unconstrained_binary_op!(unconstrained_div, Div);
    unconstrained_binary_op!(unconstrained_pow, Pow);
    unconstrained_binary_op!(unconstrained_int_div, IntDiv);
    unconstrained_binary_op!(unconstrained_mod, Mod);
    unconstrained_binary_op!(unconstrained_shift_l, ShiftL);
    unconstrained_binary_op!(unconstrained_shift_r, ShiftR);
    unconstrained_binary_op!(unconstrained_lesser_eq, LesserEq);
    unconstrained_binary_op!(unconstrained_greater_eq, GreaterEq);
    unconstrained_binary_op!(unconstrained_lesser, Lesser);
    unconstrained_binary_op!(unconstrained_greater, Greater);
    unconstrained_binary_op!(unconstrained_eq, Eq);
    unconstrained_binary_op!(unconstrained_not_eq, NotEq);
    unconstrained_binary_op!(unconstrained_bool_or, BoolOr);
    unconstrained_binary_op!(unconstrained_bool_and, BoolAnd);
    unconstrained_binary_op!(unconstrained_bit_or, BitOr);
    unconstrained_binary_op!(unconstrained_bit_and, BitAnd);
    unconstrained_binary_op!(unconstrained_bit_xor, BitXor);
}

#[derive(Clone)]
pub struct RootBuilder<C: Config> {
    num_public_inputs: usize,
    current_builders: Vec<(usize, Builder<C>)>,
    sub_circuits: HashMap<usize, source::Circuit<C>>,
    sub_circuit_output_structure: HashMap<usize, Vec<usize>>,
    full_hash_id: HashMap<usize, [u8; 32]>,
    outputs: Vec<Variable>,
}

macro_rules! root_binary_op {
    ($name:ident) => {
        fn $name(
            &mut self,
            x: impl ToVariableOrValue<CircuitField<C>>,
            y: impl ToVariableOrValue<CircuitField<C>>,
        ) -> Variable {
            self.last_builder().$name(x, y)
        }
    };
}

impl<C: Config> BasicAPI<C> for RootBuilder<C> {
    root_binary_op!(add);
    root_binary_op!(sub);
    root_binary_op!(mul);
    root_binary_op!(xor);
    root_binary_op!(or);
    root_binary_op!(and);

    fn neg(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        self.last_builder().neg(x)
    }
    fn div(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
        checked: bool,
    ) -> Variable {
        self.last_builder().div(x, y, checked)
    }

    fn is_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        self.last_builder().is_zero(x)
    }

    fn to_binary(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        num_bits: usize,
    ) -> Vec<Variable> {
        self.last_builder().to_binary(x, num_bits)
    }

    fn assert_is_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        self.last_builder().assert_is_zero(x)
    }

    fn assert_is_non_zero(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        self.last_builder().assert_is_non_zero(x)
    }

    fn assert_is_bool(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) {
        self.last_builder().assert_is_bool(x)
    }

    fn get_random_value(&mut self) -> Variable {
        self.last_builder().get_random_value()
    }

    fn new_hint(
        &mut self,
        hint_key: &str,
        inputs: &[Variable],
        num_outputs: usize,
    ) -> Vec<Variable> {
        self.last_builder().new_hint(hint_key, inputs, num_outputs)
    }

    fn constant(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        self.last_builder().constant(x)
    }

    fn constant_value(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Option<CircuitField<C>> {
        self.last_builder().constant_value(x)
    }

    fn gt(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        self.last_builder().gt(x, y)
    }

    fn geq(
        &mut self,
        x: impl ToVariableOrValue<CircuitField<C>>,
        y: impl ToVariableOrValue<CircuitField<C>>,
    ) -> Variable {
        self.last_builder().geq(x, y)
    }
}

impl<C: Config> RootAPI<C> for RootBuilder<C> {
    fn memorized_simple_call<F: Fn(&mut Self, &Vec<Variable>) -> Vec<Variable> + 'static>(
        &mut self,
        f: F,
        inputs: &[Variable],
    ) -> Vec<Variable> {
        ensure_variables_valid(inputs);
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(b"simple");
        hasher.update(&inputs.len().to_le_bytes());
        hasher.update(&get_function_id::<F>().to_le_bytes());
        let mut hash = [0u8; 32];
        hasher.finalize(&mut hash);

        let circuit_id = self.hash_to_sub_circuit_id(&hash);

        self.call_sub_circuit(circuit_id, inputs, f)
    }

    fn hash_to_sub_circuit_id(&mut self, hash: &[u8; 32]) -> usize {
        let circuit_id = usize::from_le_bytes(hash[0..8].try_into().unwrap());
        if let Some(prev_hash) = self.full_hash_id.get(&circuit_id) {
            if *prev_hash != *hash {
                panic!("subcircuit id collision");
            }
        } else {
            self.full_hash_id.insert(circuit_id, *hash);
        }
        circuit_id
    }

    fn call_sub_circuit<F: FnOnce(&mut Self, &Vec<Variable>) -> Vec<Variable>>(
        &mut self,
        circuit_id: usize,
        inputs: &[Variable],
        f: F,
    ) -> Vec<Variable> {
        if !self.sub_circuits.contains_key(&circuit_id) {
            self.actually_call_sub_circuit(circuit_id, inputs.len(), f);
        }
        let sub = self.sub_circuits.get(&circuit_id).unwrap();
        let outputs: Vec<Variable> = (0..sub.outputs.len())
            .map(|_| self.last_builder().new_var())
            .collect();
        self.last_builder()
            .instructions
            .push(SourceInstruction::SubCircuitCall {
                sub_circuit_id: circuit_id,
                inputs: inputs.iter().map(|v| v.id).collect(),
                num_outputs: outputs.len(),
            });
        outputs
    }

    fn register_sub_circuit_output_structure(&mut self, circuit_id: usize, structure: Vec<usize>) {
        if self
            .sub_circuit_output_structure
            .insert(circuit_id, structure)
            .is_some()
        {
            panic!("subcircuit output structure already registered");
        }
    }

    fn get_sub_circuit_output_structure(&self, circuit_id: usize) -> Vec<usize> {
        self.sub_circuit_output_structure
            .get(&circuit_id)
            .unwrap()
            .clone()
    }

    fn set_outputs(&mut self, outputs: Vec<Variable>) {
        ensure_variables_valid(&outputs);
        self.outputs = outputs;
    }
}

impl<C: Config> RootBuilder<C> {
    pub fn new(
        num_inputs: usize,
        num_public_inputs: usize,
    ) -> (Self, Vec<Variable>, Vec<Variable>) {
        let (mut builder0, inputs) = Builder::new(num_inputs);
        let public_inputs = (0..num_public_inputs).map(|_| builder0.new_var()).collect();
        for i in 0..num_public_inputs {
            builder0
                .instructions
                .push(SourceInstruction::ConstantLike(Coef::PublicInput(i)));
        }
        (
            RootBuilder {
                num_public_inputs,
                current_builders: vec![(0, builder0)],
                sub_circuits: HashMap::new(),
                full_hash_id: HashMap::new(),
                sub_circuit_output_structure: HashMap::new(),
                outputs: Vec::new(),
            },
            inputs,
            public_inputs,
        )
    }

    pub fn build(self) -> source::RootCircuit<C> {
        let mut circuits = self.sub_circuits;
        assert_eq!(self.current_builders.len(), 1);
        for (circuit_id, builder) in self.current_builders {
            circuits.insert(circuit_id, builder.build(&self.outputs));
        }
        source::RootCircuit {
            circuits,
            num_public_inputs: self.num_public_inputs,
            expected_num_output_zeroes: 0,
        }
    }

    pub fn last_builder(&mut self) -> &mut Builder<C> {
        &mut self.current_builders.last_mut().unwrap().1
    }

    fn actually_call_sub_circuit<F: FnOnce(&mut Self, &Vec<Variable>) -> Vec<Variable>>(
        &mut self,
        circuit_id: usize,
        n: usize,
        f: F,
    ) {
        let (sub_builder, sub_inputs) = Builder::new(n);
        self.current_builders.push((circuit_id, sub_builder));
        let sub_outputs = f(self, &sub_inputs);
        let (_, sub_builder) = self.current_builders.pop().unwrap();
        let sub = sub_builder.build(&sub_outputs);
        self.sub_circuits.insert(circuit_id, sub);
    }
}

impl<C: Config> UnconstrainedAPI<C> for RootBuilder<C> {
    fn unconstrained_identity(&mut self, x: impl ToVariableOrValue<CircuitField<C>>) -> Variable {
        self.last_builder().unconstrained_identity(x)
    }
    root_binary_op!(unconstrained_add);
    root_binary_op!(unconstrained_mul);
    root_binary_op!(unconstrained_div);
    root_binary_op!(unconstrained_pow);
    root_binary_op!(unconstrained_int_div);
    root_binary_op!(unconstrained_mod);
    root_binary_op!(unconstrained_shift_l);
    root_binary_op!(unconstrained_shift_r);
    root_binary_op!(unconstrained_lesser_eq);
    root_binary_op!(unconstrained_greater_eq);
    root_binary_op!(unconstrained_lesser);
    root_binary_op!(unconstrained_greater);
    root_binary_op!(unconstrained_eq);
    root_binary_op!(unconstrained_not_eq);
    root_binary_op!(unconstrained_bool_or);
    root_binary_op!(unconstrained_bool_and);
    root_binary_op!(unconstrained_bit_or);
    root_binary_op!(unconstrained_bit_and);
    root_binary_op!(unconstrained_bit_xor);
}
