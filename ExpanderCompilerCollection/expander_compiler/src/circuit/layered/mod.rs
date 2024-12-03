use std::{fmt, hash::Hash};

use arith::FieldForECC;

use crate::{field::FieldArith, hints, utils::error::Error};

use super::config::Config;

#[cfg(test)]
mod tests;

pub mod export;
pub mod opt;
pub mod serde;
pub mod stats;
pub mod witness;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Coef<C: Config> {
    Constant(C::CircuitField),
    Random,
    PublicInput(usize),
}

impl<C: Config> Coef<C> {
    pub fn get_value_unsafe(&self) -> C::CircuitField {
        match self {
            Coef::Constant(c) => *c,
            Coef::Random => C::CircuitField::random_unsafe(&mut rand::thread_rng()),
            Coef::PublicInput(id) => {
                // stub implementation
                let t = id * id % 1000000007;
                let t = t * id % 1000000007;
                C::CircuitField::from(t as u32)
            }
        }
    }

    pub fn get_value_with_public_inputs(
        &self,
        public_inputs: &[C::CircuitField],
    ) -> C::CircuitField {
        match self {
            Coef::Constant(c) => *c,
            Coef::Random => C::CircuitField::random_unsafe(&mut rand::thread_rng()),
            Coef::PublicInput(id) => {
                if *id >= public_inputs.len() {
                    panic!("public input id {} out of range", id);
                }
                public_inputs[*id]
            }
        }
    }

    pub fn validate(&self, num_public_inputs: usize) -> Result<(), Error> {
        match self {
            Coef::Constant(_) => Ok(()),
            Coef::Random => Ok(()),
            Coef::PublicInput(id) => {
                if *id >= num_public_inputs {
                    Err(Error::UserError(format!(
                        "public input id {} out of range",
                        id
                    )))
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Coef::Constant(_))
    }

    pub fn add_constant(&self, c: C::CircuitField) -> Self {
        match self {
            Coef::Constant(x) => Coef::Constant(*x + c),
            _ => panic!("add_constant called on non-constant"),
        }
    }

    pub fn get_constant(&self) -> Option<C::CircuitField> {
        match self {
            Coef::Constant(x) => Some(*x),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn random_no_random(mut rnd: impl rand::RngCore, num_public_inputs: usize) -> Self {
        use rand::Rng;
        if rnd.gen::<f64>() < 0.94 {
            Coef::Constant(C::CircuitField::from(rnd.next_u32()))
        } else {
            Coef::PublicInput(rnd.next_u64() as usize % num_public_inputs)
        }
    }

    pub fn export_to_expander(&self) -> (C::CircuitField, expander_circuit::CoefType) {
        match self {
            Coef::Constant(c) => (*c, expander_circuit::CoefType::Constant),
            Coef::Random => (C::CircuitField::zero(), expander_circuit::CoefType::Random),
            Coef::PublicInput(x) => (
                C::CircuitField::zero(),
                expander_circuit::CoefType::PublicInput(*x),
            ),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Gate<C: Config, const INPUT_NUM: usize> {
    pub inputs: [usize; INPUT_NUM],
    pub output: usize,
    pub coef: Coef<C>,
}

impl<C: Config, const INPUT_NUM: usize> Gate<C, INPUT_NUM> {
    pub fn export_to_expander<
        DestConfig: expander_config::GKRConfig<CircuitField = C::CircuitField>,
    >(
        &self,
    ) -> expander_circuit::Gate<DestConfig, INPUT_NUM> {
        let (c, r) = self.coef.export_to_expander();
        expander_circuit::Gate {
            i_ids: self.inputs,
            o_id: self.output,
            coef: c,
            coef_type: r,
            gate_type: 2 - INPUT_NUM, // TODO: check this
        }
    }
}

pub type GateMul<C> = Gate<C, 2>;
pub type GateAdd<C> = Gate<C, 1>;
pub type GateConst<C> = Gate<C, 0>;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GateCustom<C: Config> {
    pub gate_type: usize,
    pub inputs: Vec<usize>,
    pub output: usize,
    pub coef: Coef<C>,
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Allocation {
    pub input_offset: usize,
    pub output_offset: usize,
}

pub type ChildSpec = (usize, Vec<Allocation>);

#[derive(Default, Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Segment<C: Config> {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub child_segs: Vec<ChildSpec>,
    pub gate_muls: Vec<GateMul<C>>,
    pub gate_adds: Vec<GateAdd<C>>,
    pub gate_consts: Vec<GateConst<C>>,
    pub gate_customs: Vec<GateCustom<C>>,
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Circuit<C: Config> {
    pub num_public_inputs: usize,
    pub num_actual_outputs: usize,
    pub expected_num_output_zeroes: usize,
    pub segments: Vec<Segment<C>>,
    pub layer_ids: Vec<usize>,
}

impl<C: Config> Circuit<C> {
    pub fn validate(&self) -> Result<(), Error> {
        for (i, seg) in self.segments.iter().enumerate() {
            if seg.num_inputs == 0 || (seg.num_inputs & (seg.num_inputs - 1)) != 0 {
                return Err(Error::InternalError(format!(
                    "segment {} inputlen {} not power of 2",
                    i, seg.num_inputs
                )));
            }
            if seg.num_outputs == 0 || (seg.num_outputs & (seg.num_outputs - 1)) != 0 {
                return Err(Error::InternalError(format!(
                    "segment {} outputlen {} not power of 2",
                    i, seg.num_outputs
                )));
            }
            for m in seg.gate_muls.iter() {
                if m.inputs[0] >= seg.num_inputs
                    || m.inputs[1] >= seg.num_inputs
                    || m.output >= seg.num_outputs
                {
                    return Err(Error::InternalError(format!(
                        "segment {} mul gate ({}, {}, {}) out of range",
                        i, m.inputs[0], m.inputs[1], m.output
                    )));
                }
            }
            for a in seg.gate_adds.iter() {
                if a.inputs[0] >= seg.num_inputs || a.output >= seg.num_outputs {
                    return Err(Error::InternalError(format!(
                        "segment {} add gate ({}, {}) out of range",
                        i, a.inputs[0], a.output
                    )));
                }
            }
            for cs in seg.gate_consts.iter() {
                if cs.output >= seg.num_outputs {
                    return Err(Error::InternalError(format!(
                        "segment {} const gate {} out of range",
                        i, cs.output
                    )));
                }
            }
            for cu in seg.gate_customs.iter() {
                for &input in cu.inputs.iter() {
                    if input >= seg.num_inputs {
                        return Err(Error::InternalError(format!(
                            "segment {} custom gate {} input out of range",
                            i, input
                        )));
                    }
                }
                if cu.output >= seg.num_outputs {
                    return Err(Error::InternalError(format!(
                        "segment {} custom gate {} out of range",
                        i, cu.output
                    )));
                }
            }
            for (sub_id, allocs) in seg.child_segs.iter() {
                if *sub_id >= i {
                    return Err(Error::InternalError(format!(
                        "segment {} subcircuit {} out of range",
                        i, sub_id
                    )));
                }
                let subc = &self.segments[*sub_id];
                for a in allocs.iter() {
                    if a.input_offset % subc.num_inputs != 0 {
                        return Err(Error::InternalError(format!(
                            "segment {} subcircuit {} input offset {} not aligned to {}",
                            i, sub_id, a.input_offset, subc.num_inputs
                        )));
                    }
                    if a.input_offset + subc.num_inputs > seg.num_inputs {
                        return Err(Error::InternalError(format!(
                            "segment {} subcircuit {} input offset {} out of range",
                            i, sub_id, a.input_offset
                        )));
                    }
                    if a.output_offset % subc.num_outputs != 0 {
                        return Err(Error::InternalError(format!(
                            "segment {} subcircuit {} output offset {} not aligned to {}",
                            i, sub_id, a.output_offset, subc.num_outputs
                        )));
                    }
                    if a.output_offset + subc.num_outputs > seg.num_outputs {
                        return Err(Error::InternalError(format!(
                            "segment {} subcircuit {} output offset {} out of range",
                            i, sub_id, a.output_offset
                        )));
                    }
                }
            }
        }
        for x in self.layer_ids.iter() {
            if *x >= self.segments.len() {
                return Err(Error::InternalError(format!("layer id {} out of range", x)));
            }
        }
        if self.layer_ids.is_empty() {
            return Err(Error::InternalError("empty layer".to_string()));
        }
        for i in 1..self.layer_ids.len() {
            let cur = &self.segments[self.layer_ids[i]];
            let prev = &self.segments[self.layer_ids[i - 1]];
            if cur.num_inputs != prev.num_outputs {
                return Err(Error::InternalError(format!(
                    "segment {} inputlen {} not equal to segment {} outputlen {}",
                    self.layer_ids[i],
                    cur.num_inputs,
                    self.layer_ids[i - 1],
                    prev.num_outputs
                )));
            }
        }
        let (input_mask, output_mask) = self.compute_masks();
        for i in 1..self.layer_ids.len() {
            for j in 0..self.segments[self.layer_ids[i]].num_inputs {
                if input_mask[self.layer_ids[i]][j] && !output_mask[self.layer_ids[i - 1]][j] {
                    return Err(Error::InternalError(format!(
                        "circuit {} input {} not initialized by circuit {} output",
                        self.layer_ids[i],
                        j,
                        self.layer_ids[i - 1]
                    )));
                }
            }
        }
        Ok(())
    }

    fn compute_masks(&self) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let mut input_mask: Vec<Vec<bool>> = Vec::with_capacity(self.segments.len());
        let mut output_mask: Vec<Vec<bool>> = Vec::with_capacity(self.segments.len());
        for seg in self.segments.iter() {
            let mut input_mask_seg = vec![false; seg.num_inputs];
            let mut output_mask_seg = vec![false; seg.num_outputs];
            for m in seg.gate_muls.iter() {
                input_mask_seg[m.inputs[0]] = true;
                input_mask_seg[m.inputs[1]] = true;
                output_mask_seg[m.output] = true;
            }
            for a in seg.gate_adds.iter() {
                input_mask_seg[a.inputs[0]] = true;
                output_mask_seg[a.output] = true;
            }
            for cs in seg.gate_consts.iter() {
                output_mask_seg[cs.output] = true;
            }
            for cu in seg.gate_customs.iter() {
                for &input in cu.inputs.iter() {
                    input_mask_seg[input] = true;
                }
                output_mask_seg[cu.output] = true;
            }
            for (sub_id, allocs) in seg.child_segs.iter() {
                let subc = &self.segments[*sub_id];
                for a in allocs.iter() {
                    for j in 0..subc.num_inputs {
                        input_mask_seg[a.input_offset + j] =
                            input_mask_seg[a.input_offset + j] || input_mask[*sub_id][j];
                    }
                    for j in 0..subc.num_outputs {
                        output_mask_seg[a.output_offset + j] =
                            output_mask_seg[a.output_offset + j] || output_mask[*sub_id][j];
                    }
                }
            }
            input_mask.push(input_mask_seg);
            output_mask.push(output_mask_seg);
        }
        (input_mask, output_mask)
    }

    pub fn input_size(&self) -> usize {
        self.segments[self.layer_ids[0]].num_inputs
    }

    pub fn eval_unsafe(&self, inputs: Vec<C::CircuitField>) -> (Vec<C::CircuitField>, bool) {
        if inputs.len() != self.input_size() {
            panic!("input length mismatch");
        }
        let mut cur = inputs;
        for &id in self.layer_ids.iter() {
            let mut next = vec![C::CircuitField::zero(); self.segments[id].num_outputs];
            self.apply_segment_unsafe(&self.segments[id], &cur, &mut next);
            cur = next;
        }
        let mut constraints_satisfied = true;
        for out in cur.iter().take(self.expected_num_output_zeroes) {
            if !out.is_zero() {
                constraints_satisfied = false;
                break;
            }
        }
        (
            cur[self.expected_num_output_zeroes..self.num_actual_outputs].to_vec(),
            constraints_satisfied,
        )
    }

    fn apply_segment_unsafe(
        &self,
        seg: &Segment<C>,
        cur: &[C::CircuitField],
        nxt: &mut [C::CircuitField],
    ) {
        for m in seg.gate_muls.iter() {
            nxt[m.output] += cur[m.inputs[0]] * cur[m.inputs[1]] * m.coef.get_value_unsafe();
        }
        for a in seg.gate_adds.iter() {
            nxt[a.output] += cur[a.inputs[0]] * a.coef.get_value_unsafe();
        }
        for cs in seg.gate_consts.iter() {
            nxt[cs.output] += cs.coef.get_value_unsafe();
        }
        for cu in seg.gate_customs.iter() {
            let mut inputs = Vec::with_capacity(cu.inputs.len());
            for &input in cu.inputs.iter() {
                inputs.push(cur[input]);
            }
            let outputs = hints::stub_impl(cu.gate_type, &inputs, 1);
            for (i, &output) in outputs.iter().enumerate() {
                nxt[cu.output + i] += output * cu.coef.get_value_unsafe();
            }
        }
        for (sub_id, allocs) in seg.child_segs.iter() {
            let subc = &self.segments[*sub_id];
            for a in allocs.iter() {
                self.apply_segment_unsafe(
                    subc,
                    &cur[a.input_offset..a.input_offset + subc.num_inputs],
                    &mut nxt[a.output_offset..a.output_offset + subc.num_outputs],
                );
            }
        }
    }

    pub fn eval_with_public_inputs(
        &self,
        inputs: Vec<C::CircuitField>,
        public_inputs: &[C::CircuitField],
    ) -> (Vec<C::CircuitField>, bool) {
        if inputs.len() != self.input_size() {
            panic!("input length mismatch");
        }
        let mut cur = inputs;
        for &id in self.layer_ids.iter() {
            let mut next = vec![C::CircuitField::zero(); self.segments[id].num_outputs];
            self.apply_segment_with_public_inputs(
                &self.segments[id],
                &cur,
                &mut next,
                public_inputs,
            );
            cur = next;
        }
        let mut constraints_satisfied = true;
        for out in cur.iter().take(self.expected_num_output_zeroes) {
            if !out.is_zero() {
                constraints_satisfied = false;
                break;
            }
        }
        (
            cur[self.expected_num_output_zeroes..self.num_actual_outputs].to_vec(),
            constraints_satisfied,
        )
    }

    fn apply_segment_with_public_inputs(
        &self,
        seg: &Segment<C>,
        cur: &[C::CircuitField],
        nxt: &mut [C::CircuitField],
        public_inputs: &[C::CircuitField],
    ) {
        for m in seg.gate_muls.iter() {
            nxt[m.output] += cur[m.inputs[0]]
                * cur[m.inputs[1]]
                * m.coef.get_value_with_public_inputs(public_inputs);
        }
        for a in seg.gate_adds.iter() {
            nxt[a.output] += cur[a.inputs[0]] * a.coef.get_value_with_public_inputs(public_inputs);
        }
        for cs in seg.gate_consts.iter() {
            nxt[cs.output] += cs.coef.get_value_with_public_inputs(public_inputs);
        }
        for cu in seg.gate_customs.iter() {
            let mut inputs = Vec::with_capacity(cu.inputs.len());
            for &input in cu.inputs.iter() {
                inputs.push(cur[input]);
            }
            let outputs = hints::stub_impl(cu.gate_type, &inputs, 1);
            for (i, &output) in outputs.iter().enumerate() {
                nxt[cu.output + i] += output * cu.coef.get_value_unsafe();
            }
        }
        for (sub_id, allocs) in seg.child_segs.iter() {
            let subc = &self.segments[*sub_id];
            for a in allocs.iter() {
                self.apply_segment_with_public_inputs(
                    subc,
                    &cur[a.input_offset..a.input_offset + subc.num_inputs],
                    &mut nxt[a.output_offset..a.output_offset + subc.num_outputs],
                    public_inputs,
                );
            }
        }
    }

    pub fn sort_everything(&mut self) {
        for seg in self.segments.iter_mut() {
            seg.gate_muls.sort();
            seg.gate_adds.sort();
            seg.gate_consts.sort();
            seg.gate_customs.sort();
            seg.child_segs.sort();
        }
    }
}

impl<C: Config> fmt::Display for Coef<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Coef::Constant(c) => write!(f, "{}", c.to_u256()),
            Coef::Random => write!(f, "Random"),
            Coef::PublicInput(id) => write!(f, "PublicInput({})", id),
        }
    }
}

impl<C: Config> fmt::Display for Segment<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "input={} output={}", self.num_inputs, self.num_outputs)?;
        for (sub_id, allocs) in self.child_segs.iter() {
            writeln!(f, "apply circuit {} at:", sub_id)?;
            for a in allocs.iter() {
                writeln!(
                    f,
                    "    input_offset={} output_offset={}",
                    a.input_offset, a.output_offset
                )?;
            }
        }
        for m in self.gate_muls.iter() {
            writeln!(
                f,
                "out{} += in{} * in{} * {}",
                m.output, m.inputs[0], m.inputs[1], m.coef
            )?;
        }
        for a in self.gate_adds.iter() {
            writeln!(f, "out{} += in{} * {}", a.output, a.inputs[0], a.coef)?;
        }
        for cs in self.gate_consts.iter() {
            writeln!(f, "out{} += {}", cs.output, cs.coef)?;
        }
        for cu in self.gate_customs.iter() {
            write!(f, "out{} += custom{}(", cu.output, cu.gate_type)?;
            for (i, input) in cu.inputs.iter().enumerate() {
                write!(f, "in{}", input)?;
                if i < cu.inputs.len() - 1 {
                    write!(f, ",")?;
                }
            }
            writeln!(f, ") * {}", cu.coef)?;
        }
        Ok(())
    }
}

impl<C: Config> fmt::Display for Circuit<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, seg) in self.segments.iter().enumerate() {
            write!(f, "Circuit {}: {}", i, seg)?;
            writeln!(f, "================================")?;
        }
        writeln!(f, "Layers: {:?}", self.layer_ids)?;
        Ok(())
    }
}
