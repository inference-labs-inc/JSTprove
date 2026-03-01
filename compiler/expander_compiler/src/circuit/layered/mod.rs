use std::{fmt, hash::Hash};

use serdes::ExpSerde;

use crate::{field::FieldArith, hints, utils::error::Error};

use super::config::{CircuitField, Config};

#[cfg(test)]
mod tests;

pub mod export;
pub mod opt;
pub mod serde;
pub mod stats;
pub mod witness;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Coef<C: Config> {
    Constant(CircuitField<C>),
    Random,
    PublicInput(usize),
}

impl<C: Config> Coef<C> {
    pub fn get_value_unsafe(&self) -> CircuitField<C> {
        match self {
            Coef::Constant(c) => *c,
            Coef::Random => CircuitField::<C>::random_unsafe(&mut rand::thread_rng()),
            Coef::PublicInput(id) => {
                // stub implementation
                let t = id * id % 1000000007;
                let t = t * id % 1000000007;
                CircuitField::<C>::from(t as u32)
            }
        }
    }

    pub fn get_value_with_public_inputs(
        &self,
        public_inputs: &[CircuitField<C>],
    ) -> CircuitField<C> {
        match self {
            Coef::Constant(c) => *c,
            Coef::Random => CircuitField::<C>::random_unsafe(&mut rand::thread_rng()),
            Coef::PublicInput(id) => {
                if *id >= public_inputs.len() {
                    panic!("public input id {id} out of range");
                }
                public_inputs[*id]
            }
        }
    }

    pub fn get_value_with_public_inputs_simd<SF: arith::SimdField<Scalar = CircuitField<C>>>(
        &self,
        public_inputs: &[SF],
    ) -> SF {
        match self {
            Coef::Constant(c) => SF::one().scale(c),
            Coef::Random => SF::random_unsafe(&mut rand::thread_rng()),
            Coef::PublicInput(id) => {
                if *id >= public_inputs.len() {
                    panic!("public input id {id} out of range");
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
                        "public input id {id} out of range"
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

    pub fn add_constant(&self, c: CircuitField<C>) -> Self {
        match self {
            Coef::Constant(x) => Coef::Constant(*x + c),
            _ => panic!("add_constant called on non-constant"),
        }
    }

    pub fn get_constant(&self) -> Option<CircuitField<C>> {
        match self {
            Coef::Constant(x) => Some(*x),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn random_no_random(mut rnd: impl rand::RngCore, num_public_inputs: usize) -> Self {
        use rand::Rng;
        if rnd.gen::<f64>() < 0.94 {
            Coef::Constant(CircuitField::<C>::from(rnd.next_u32()))
        } else {
            Coef::PublicInput(rnd.next_u64() as usize % num_public_inputs)
        }
    }

    pub fn export_to_expander(&self) -> (CircuitField<C>, expander_circuit::CoefType) {
        match self {
            Coef::Constant(c) => (*c, expander_circuit::CoefType::Constant),
            Coef::Random => (
                CircuitField::<C>::zero(),
                expander_circuit::CoefType::Random,
            ),
            Coef::PublicInput(x) => (
                CircuitField::<C>::zero(),
                expander_circuit::CoefType::PublicInput(*x),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord, ExpSerde)]
pub struct CrossLayerInput {
    // the actual layer of the input is (output_layer-1-layer)
    pub layer: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord, ExpSerde)]
pub struct NormalInput {
    pub offset: usize,
}

pub trait Input:
    std::fmt::Debug
    + std::fmt::Display
    + Clone
    + Copy
    + Default
    + Hash
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + ExpSerde
{
    fn layer(&self) -> usize;
    fn offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
    fn new(layer: usize, offset: usize) -> Self;
}

impl Input for CrossLayerInput {
    fn layer(&self) -> usize {
        self.layer
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }
    fn new(layer: usize, offset: usize) -> Self {
        CrossLayerInput { layer, offset }
    }
}

impl Input for NormalInput {
    fn layer(&self) -> usize {
        0
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }
    fn new(layer: usize, offset: usize) -> Self {
        if layer != 0 {
            panic!("new called on non-zero layer");
        }
        NormalInput { offset }
    }
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, ExpSerde)]
pub struct CrossLayerInputUsize {
    v: Vec<usize>,
}

#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, ExpSerde)]
pub struct NormalInputUsize {
    pub v: usize,
}

pub trait InputUsize:
    std::fmt::Debug + Default + Clone + Hash + PartialEq + Eq + PartialOrd + Ord + ExpSerde
{
    type Iter<'a>: Iterator<Item = usize>
    where
        Self: 'a;
    fn len(&self) -> usize;
    fn iter(&self) -> Self::Iter<'_>;
    fn get(&self, i: usize) -> usize {
        self.iter().nth(i).unwrap()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn from_vec(v: Vec<usize>) -> Self;
    fn to_vec(&self) -> Vec<usize> {
        self.iter().collect()
    }
}

impl InputUsize for CrossLayerInputUsize {
    type Iter<'a> = std::iter::Copied<std::slice::Iter<'a, usize>>;
    fn len(&self) -> usize {
        self.v.len()
    }
    fn iter(&self) -> Self::Iter<'_> {
        self.v.iter().copied()
    }
    fn from_vec(v: Vec<usize>) -> Self {
        CrossLayerInputUsize { v }
    }
}

impl InputUsize for NormalInputUsize {
    type Iter<'a> = std::iter::Once<usize>;
    fn len(&self) -> usize {
        1
    }
    fn iter(&self) -> Self::Iter<'_> {
        std::iter::once(self.v)
    }
    fn from_vec(v: Vec<usize>) -> Self {
        if v.len() != 1 {
            panic!("from_vec called on non-singleton vec");
        }
        NormalInputUsize { v: v[0] }
    }
}

pub trait InputType:
    std::fmt::Debug + Default + Clone + Hash + PartialEq + Eq + PartialOrd + Ord
{
    type Input: Input;
    type InputUsize: InputUsize;
    const CROSS_LAYER_RELAY: bool;
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CrossLayerInputType;

impl InputType for CrossLayerInputType {
    type Input = CrossLayerInput;
    type InputUsize = CrossLayerInputUsize;
    const CROSS_LAYER_RELAY: bool = true;
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NormalInputType;

impl InputType for NormalInputType {
    type Input = NormalInput;
    type InputUsize = NormalInputUsize;
    const CROSS_LAYER_RELAY: bool = false;
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Gate<C: Config, I: InputType, const INPUT_NUM: usize> {
    pub inputs: [I::Input; INPUT_NUM],
    pub output: usize,
    pub coef: Coef<C>,
}

impl<C: Config, const INPUT_NUM: usize> Copy for Gate<C, NormalInputType, INPUT_NUM> {}

impl<C: Config, const INPUT_NUM: usize> Gate<C, NormalInputType, INPUT_NUM> {
    pub fn export_to_expander<
        DestConfig: gkr_engine::FieldEngine<CircuitField = CircuitField<C>>,
    >(
        &self,
    ) -> expander_circuit::Gate<DestConfig, INPUT_NUM> {
        let (c, r) = self.coef.export_to_expander();
        let mut i_ids: [usize; INPUT_NUM] = [0; INPUT_NUM];
        for (x, y) in self.inputs.iter().zip(i_ids.iter_mut()) {
            *y = x.offset();
        }
        expander_circuit::Gate {
            i_ids,
            o_id: self.output,
            coef: c,
            coef_type: r,
            gate_type: 2 - INPUT_NUM,
        }
    }
}

impl<C: Config, const INPUT_NUM: usize> Gate<C, CrossLayerInputType, INPUT_NUM> {
    pub fn export_to_crosslayer_simple<
        DestConfig: gkr_engine::FieldEngine<CircuitField = CircuitField<C>>,
    >(
        &self,
    ) -> crosslayer_prototype::SimpleGate<DestConfig, INPUT_NUM> {
        let (c, r) = self.coef.export_to_expander();
        let mut i_ids: [usize; INPUT_NUM] = [0; INPUT_NUM];
        for (x, y) in self.inputs.iter().zip(i_ids.iter_mut()) {
            assert_eq!(x.layer(), 0);
            *y = x.offset();
        }
        crosslayer_prototype::SimpleGate {
            i_ids,
            o_id: self.output,
            coef: c,
            coef_type: match r {
                expander_circuit::CoefType::Constant => crosslayer_prototype::CoefType::Constant,
                expander_circuit::CoefType::Random => crosslayer_prototype::CoefType::Random,
                expander_circuit::CoefType::PublicInput(x) => {
                    crosslayer_prototype::CoefType::PublicInput(x)
                }
            },
        }
    }
}

pub type GateMul<C, I> = Gate<C, I, 2>;
pub type GateAdd<C, I> = Gate<C, I, 1>;
pub type GateConst<C, I> = Gate<C, I, 0>;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct GateCustom<C: Config, I: InputType> {
    pub gate_type: usize,
    pub inputs: Vec<I::Input>,
    pub output: usize,
    pub coef: Coef<C>,
}

#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq, ExpSerde)]
pub struct Allocation<I: InputType> {
    pub input_offset: I::InputUsize,
    pub output_offset: usize,
}

impl Copy for Allocation<NormalInputType> {}

pub type ChildSpec<I> = (usize, Vec<Allocation<I>>);

#[derive(Default, Debug, Hash, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Segment<C: Config, I: InputType> {
    pub num_inputs: I::InputUsize,
    pub num_outputs: usize,
    pub child_segs: Vec<ChildSpec<I>>,
    pub gate_muls: Vec<GateMul<C, I>>,
    pub gate_adds: Vec<GateAdd<C, I>>,
    pub gate_consts: Vec<GateConst<C, I>>,
    pub gate_customs: Vec<GateCustom<C, I>>,
}

#[derive(Debug, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Circuit<C: Config, I: InputType> {
    pub num_public_inputs: usize,
    pub num_actual_outputs: usize,
    pub expected_num_output_zeroes: usize,
    pub segments: Vec<Segment<C, I>>,
    pub layer_ids: Vec<usize>,
}

impl<C: Config, I: InputType> Circuit<C, I> {
    pub fn validate(&self) -> Result<(), Error> {
        for (i, seg) in self.segments.iter().enumerate() {
            for (j, x) in seg.num_inputs.iter().enumerate() {
                if x == 0 || (x & (x - 1)) != 0 {
                    return Err(Error::InternalError(format!(
                        "segment {i} input {j} len {x} not power of 2"
                    )));
                }
            }
            if seg.num_inputs.len() == 0 {
                return Err(Error::InternalError(format!("segment {i} inputlen 0")));
            }
            if seg.num_outputs == 0 || (seg.num_outputs & (seg.num_outputs - 1)) != 0 {
                return Err(Error::InternalError(format!(
                    "segment {} outputlen {} not power of 2",
                    i, seg.num_outputs
                )));
            }
            for m in seg.gate_muls.iter() {
                if m.inputs[0].layer() >= self.layer_ids.len() {
                    return Err(Error::InternalError(format!(
                        "segment {} mul gate ({:?}, {:?}, {}) input 0 layer out of range",
                        i, m.inputs[0], m.inputs[1], m.output
                    )));
                }
                if m.inputs[1].layer() >= self.layer_ids.len() {
                    return Err(Error::InternalError(format!(
                        "segment {} mul gate ({:?}, {:?}, {}) input 1 layer out of range",
                        i, m.inputs[0], m.inputs[1], m.output
                    )));
                }
                if m.inputs[0].offset() >= seg.num_inputs.get(m.inputs[0].layer()) {
                    return Err(Error::InternalError(format!(
                        "segment {} mul gate ({:?}, {:?}, {}) input 0 out of range",
                        i, m.inputs[0], m.inputs[1], m.output
                    )));
                }
                if m.inputs[1].offset() >= seg.num_inputs.get(m.inputs[1].layer()) {
                    return Err(Error::InternalError(format!(
                        "segment {} mul gate ({:?}, {:?}, {}) input 1 out of range",
                        i, m.inputs[0], m.inputs[1], m.output
                    )));
                }
                if m.output >= seg.num_outputs {
                    return Err(Error::InternalError(format!(
                        "segment {} mul gate ({:?}, {:?}, {}) out of range",
                        i, m.inputs[0], m.inputs[1], m.output
                    )));
                }
            }
            for a in seg.gate_adds.iter() {
                if a.inputs[0].layer() >= self.layer_ids.len() {
                    return Err(Error::InternalError(format!(
                        "segment {} add gate ({:?}, {}) input layer out of range",
                        i, a.inputs[0], a.output
                    )));
                }
                if a.inputs[0].offset() >= seg.num_inputs.get(a.inputs[0].layer()) {
                    return Err(Error::InternalError(format!(
                        "segment {} add gate ({:?}, {}) input out of range",
                        i, a.inputs[0], a.output
                    )));
                }
                if a.output >= seg.num_outputs {
                    return Err(Error::InternalError(format!(
                        "segment {} add gate ({:?}, {}) out of range",
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
                for input in cu.inputs.iter() {
                    if input.layer() >= self.layer_ids.len() {
                        return Err(Error::InternalError(format!(
                            "segment {} custom gate {} input layer out of range",
                            i, cu.output
                        )));
                    }
                    if input.offset() >= seg.num_inputs.get(input.layer()) {
                        return Err(Error::InternalError(format!(
                            "segment {} custom gate {} input out of range",
                            i, cu.output
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
                        "segment {i} subcircuit {sub_id} out of range"
                    )));
                }
                let subc = &self.segments[*sub_id];
                if subc.num_inputs.len() > seg.num_inputs.len() {
                    return Err(Error::InternalError(format!(
                        "segment {} subcircuit {} input length {} larger than {}",
                        i,
                        sub_id,
                        subc.num_inputs.len(),
                        seg.num_inputs.len()
                    )));
                }
                for a in allocs.iter() {
                    if a.input_offset.len() != subc.num_inputs.len() {
                        return Err(Error::InternalError(format!(
                            "segment {} subcircuit {} input offset {:?} length not equal to {}",
                            i,
                            sub_id,
                            a.input_offset,
                            subc.num_inputs.len()
                        )));
                    }
                    for ((x, y), z) in a
                        .input_offset
                        .iter()
                        .zip(subc.num_inputs.iter())
                        .zip(seg.num_inputs.iter())
                    {
                        if x % y != 0 {
                            return Err(Error::InternalError(format!(
                                "segment {i} subcircuit {sub_id} input offset {x} not aligned to {y}"
                            )));
                        }
                        if x + y > z {
                            return Err(Error::InternalError(format!(
                                "segment {i} subcircuit {sub_id} input offset {x} out of range"
                            )));
                        }
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
                return Err(Error::InternalError(format!("layer id {x} out of range")));
            }
        }
        if self.layer_ids.is_empty() {
            return Err(Error::InternalError("empty layer".to_string()));
        }
        let mut layer_sizes = Vec::with_capacity(self.layer_ids.len() + 1);
        layer_sizes.push(self.segments[self.layer_ids[0]].num_inputs.get(0));
        for l in self.layer_ids.iter() {
            layer_sizes.push(self.segments[*l].num_outputs);
        }
        for (i, l) in self.layer_ids.iter().enumerate() {
            let cur = &self.segments[*l];
            if cur.num_inputs.len() > i + 1 {
                return Err(Error::InternalError(format!(
                    "layer {} input length {} larger than {}",
                    i,
                    cur.num_inputs.len(),
                    i + 1
                )));
            }
            for (j, x) in cur.num_inputs.iter().enumerate() {
                if x != layer_sizes[i - j] {
                    return Err(Error::InternalError(format!(
                        "layer {} input {} length {} not equal to {}",
                        i,
                        j,
                        x,
                        layer_sizes[i - j]
                    )));
                }
            }
        }
        let (input_mask, output_mask) = self.compute_masks();
        for i in 1..self.layer_ids.len() {
            for (l, len) in self.segments[self.layer_ids[i]]
                .num_inputs
                .iter()
                .enumerate()
            {
                if i == l {
                    // if this is also the global input, it's always initialized
                    continue;
                }
                for j in 0..len {
                    if input_mask[self.layer_ids[i]][l][j]
                        && !output_mask[self.layer_ids[i - 1 - l]][j]
                    {
                        return Err(Error::InternalError(format!(
                            "circuit {} (layer {}) input {} not initialized by circuit {} (layer {}) output",
                            self.layer_ids[i],
                            i,
                            j,
                            self.layer_ids[i - 1 - l],
                            i - 1 - l
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn compute_masks(&self) -> (Vec<Vec<Vec<bool>>>, Vec<Vec<bool>>) {
        let mut input_mask: Vec<Vec<Vec<bool>>> = Vec::with_capacity(self.segments.len());
        let mut output_mask: Vec<Vec<bool>> = Vec::with_capacity(self.segments.len());
        for seg in self.segments.iter() {
            let mut input_mask_seg: Vec<Vec<bool>> =
                seg.num_inputs.iter().map(|x| vec![false; x]).collect();
            let mut output_mask_seg = vec![false; seg.num_outputs];
            for m in seg.gate_muls.iter() {
                input_mask_seg[m.inputs[0].layer()][m.inputs[0].offset()] = true;
                input_mask_seg[m.inputs[1].layer()][m.inputs[1].offset()] = true;
                output_mask_seg[m.output] = true;
            }
            for a in seg.gate_adds.iter() {
                input_mask_seg[a.inputs[0].layer()][a.inputs[0].offset()] = true;
                output_mask_seg[a.output] = true;
            }
            for cs in seg.gate_consts.iter() {
                output_mask_seg[cs.output] = true;
            }
            for cu in seg.gate_customs.iter() {
                for input in cu.inputs.iter() {
                    input_mask_seg[input.layer()][input.offset()] = true;
                }
                output_mask_seg[cu.output] = true;
            }
            for (sub_id, allocs) in seg.child_segs.iter() {
                let subc = &self.segments[*sub_id];
                for a in allocs.iter() {
                    for (l, (off, len)) in a
                        .input_offset
                        .iter()
                        .zip(subc.num_inputs.iter())
                        .enumerate()
                    {
                        for i in 0..len {
                            input_mask_seg[l][off + i] =
                                input_mask_seg[l][off + i] || input_mask[*sub_id][l][i];
                        }
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
        self.segments[self.layer_ids[0]].num_inputs.get(0)
    }

    pub fn eval_unsafe(&self, inputs: Vec<CircuitField<C>>) -> (Vec<CircuitField<C>>, bool) {
        if inputs.len() != self.input_size() {
            panic!("input length mismatch");
        }
        let mut cur = vec![inputs];
        for id in self.layer_ids.iter() {
            let mut next = vec![CircuitField::<C>::zero(); self.segments[*id].num_outputs];
            let mut inputs: Vec<&[CircuitField<C>]> = Vec::new();
            for i in 0..self.segments[*id].num_inputs.len() {
                inputs.push(&cur[cur.len() - i - 1]);
            }
            self.apply_segment_unsafe(&self.segments[*id], &inputs, &mut next);
            cur.push(next);
        }
        let cur = cur.last().unwrap();
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
        seg: &Segment<C, I>,
        cur: &[&[CircuitField<C>]],
        nxt: &mut [CircuitField<C>],
    ) {
        for m in seg.gate_muls.iter() {
            nxt[m.output] += cur[m.inputs[0].layer()][m.inputs[0].offset()]
                * cur[m.inputs[1].layer()][m.inputs[1].offset()]
                * m.coef.get_value_unsafe();
        }
        for a in seg.gate_adds.iter() {
            nxt[a.output] +=
                cur[a.inputs[0].layer()][a.inputs[0].offset()] * a.coef.get_value_unsafe();
        }
        for cs in seg.gate_consts.iter() {
            nxt[cs.output] += cs.coef.get_value_unsafe();
        }
        for cu in seg.gate_customs.iter() {
            let mut inputs = Vec::with_capacity(cu.inputs.len());
            for input in cu.inputs.iter() {
                inputs.push(cur[input.layer()][input.offset()]);
            }
            let outputs = hints::stub_impl(cu.gate_type, &inputs, 1);
            for (i, output) in outputs.iter().enumerate() {
                nxt[cu.output + i] += *output * cu.coef.get_value_unsafe();
            }
        }
        for (sub_id, allocs) in seg.child_segs.iter() {
            let subc = &self.segments[*sub_id];
            for a in allocs.iter() {
                let inputs = a
                    .input_offset
                    .iter()
                    .zip(subc.num_inputs.iter())
                    .enumerate()
                    .map(|(l, (off, len))| &cur[l][off..off + len])
                    .collect::<Vec<_>>();
                self.apply_segment_unsafe(
                    subc,
                    &inputs,
                    &mut nxt[a.output_offset..a.output_offset + subc.num_outputs],
                );
            }
        }
    }

    pub fn eval_with_public_inputs(
        &self,
        inputs: Vec<CircuitField<C>>,
        public_inputs: &[CircuitField<C>],
    ) -> (Vec<CircuitField<C>>, bool) {
        if inputs.len() != self.input_size() {
            panic!("input length mismatch");
        }
        let mut cur = vec![inputs];
        for id in self.layer_ids.iter() {
            let mut next = vec![CircuitField::<C>::zero(); self.segments[*id].num_outputs];
            let mut inputs: Vec<&[CircuitField<C>]> = Vec::new();
            for i in 0..self.segments[*id].num_inputs.len() {
                inputs.push(&cur[cur.len() - i - 1]);
            }
            self.apply_segment_with_public_inputs(
                &self.segments[*id],
                &inputs,
                &mut next,
                public_inputs,
            );
            cur.push(next);
        }
        let cur = cur.last().unwrap();
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

    pub fn eval_constraint_values(
        &self,
        inputs: Vec<CircuitField<C>>,
        public_inputs: &[CircuitField<C>],
    ) -> Vec<CircuitField<C>> {
        if inputs.len() != self.input_size() {
            panic!(
                "input length mismatch: expected {}, got {}",
                self.input_size(),
                inputs.len()
            );
        }
        let mut cur = vec![inputs];
        for id in self.layer_ids.iter() {
            let mut next = vec![CircuitField::<C>::zero(); self.segments[*id].num_outputs];
            let mut inputs: Vec<&[CircuitField<C>]> = Vec::new();
            for i in 0..self.segments[*id].num_inputs.len() {
                inputs.push(&cur[cur.len() - i - 1]);
            }
            self.apply_segment_with_public_inputs(
                &self.segments[*id],
                &inputs,
                &mut next,
                public_inputs,
            );
            cur.push(next);
        }
        cur.last().unwrap()[..self.expected_num_output_zeroes].to_vec()
    }

    fn apply_segment_with_public_inputs(
        &self,
        seg: &Segment<C, I>,
        cur: &[&[CircuitField<C>]],
        nxt: &mut [CircuitField<C>],
        public_inputs: &[CircuitField<C>],
    ) {
        for m in seg.gate_muls.iter() {
            nxt[m.output] += cur[m.inputs[0].layer()][m.inputs[0].offset()]
                * cur[m.inputs[1].layer()][m.inputs[1].offset()]
                * m.coef.get_value_with_public_inputs(public_inputs);
        }
        for a in seg.gate_adds.iter() {
            nxt[a.output] += cur[a.inputs[0].layer()][a.inputs[0].offset()]
                * a.coef.get_value_with_public_inputs(public_inputs);
        }
        for cs in seg.gate_consts.iter() {
            nxt[cs.output] += cs.coef.get_value_with_public_inputs(public_inputs);
        }
        for cu in seg.gate_customs.iter() {
            let mut inputs = Vec::with_capacity(cu.inputs.len());
            for input in cu.inputs.iter() {
                inputs.push(cur[input.layer()][input.offset()]);
            }
            let outputs = hints::stub_impl(cu.gate_type, &inputs, 1);
            for (i, output) in outputs.iter().enumerate() {
                nxt[cu.output + i] += *output * cu.coef.get_value_with_public_inputs(public_inputs);
            }
        }
        for (sub_id, allocs) in seg.child_segs.iter() {
            let subc = &self.segments[*sub_id];
            for a in allocs.iter() {
                let inputs = a
                    .input_offset
                    .iter()
                    .zip(subc.num_inputs.iter())
                    .enumerate()
                    .map(|(l, (off, len))| &cur[l][off..off + len])
                    .collect::<Vec<_>>();
                self.apply_segment_with_public_inputs(
                    subc,
                    &inputs,
                    &mut nxt[a.output_offset..a.output_offset + subc.num_outputs],
                    public_inputs,
                );
            }
        }
    }

    pub fn eval_with_public_inputs_simd<SF: arith::SimdField<Scalar = CircuitField<C>>>(
        &self,
        inputs: Vec<SF>,
        public_inputs: &[SF],
    ) -> (Vec<SF>, Vec<bool>) {
        if inputs.len() != self.input_size() {
            panic!("input length mismatch");
        }
        let mut cur = vec![inputs];
        for id in self.layer_ids.iter() {
            let mut next = vec![SF::zero(); self.segments[*id].num_outputs];
            let mut inputs: Vec<&[SF]> = Vec::new();
            for i in 0..self.segments[*id].num_inputs.len() {
                inputs.push(&cur[cur.len() - i - 1]);
            }
            self.apply_segment_with_public_inputs_simd(
                &self.segments[*id],
                &inputs,
                &mut next,
                public_inputs,
            );
            cur.push(next);
        }
        let cur = cur.last().unwrap();
        let mut constraints_satisfied = vec![true; SF::PACK_SIZE];
        for out in cur.iter().take(self.expected_num_output_zeroes) {
            let tmp = out.unpack();
            for i in 0..SF::PACK_SIZE {
                if !tmp[i].is_zero() {
                    constraints_satisfied[i] = false;
                }
            }
        }
        (
            cur[self.expected_num_output_zeroes..self.num_actual_outputs].to_vec(),
            constraints_satisfied,
        )
    }

    fn apply_segment_with_public_inputs_simd<SF: arith::SimdField<Scalar = CircuitField<C>>>(
        &self,
        seg: &Segment<C, I>,
        cur: &[&[SF]],
        nxt: &mut [SF],
        public_inputs: &[SF],
    ) {
        for m in seg.gate_muls.iter() {
            nxt[m.output] += cur[m.inputs[0].layer()][m.inputs[0].offset()]
                * cur[m.inputs[1].layer()][m.inputs[1].offset()]
                * m.coef.get_value_with_public_inputs_simd(public_inputs);
        }
        for a in seg.gate_adds.iter() {
            nxt[a.output] += cur[a.inputs[0].layer()][a.inputs[0].offset()]
                * a.coef.get_value_with_public_inputs_simd(public_inputs);
        }
        for cs in seg.gate_consts.iter() {
            nxt[cs.output] += cs.coef.get_value_with_public_inputs_simd(public_inputs);
        }
        for cu in seg.gate_customs.iter() {
            let mut inputs = vec![Vec::with_capacity(cu.inputs.len()); SF::PACK_SIZE];
            for input in cu.inputs.iter() {
                let tmp = cur[input.layer()][input.offset()].unpack();
                for i in 0..SF::PACK_SIZE {
                    inputs[i].push(tmp[i]);
                }
            }
            let mut outputs = Vec::with_capacity(SF::PACK_SIZE);
            for x in inputs.iter() {
                outputs.push(hints::stub_impl(cu.gate_type, x, 1));
            }
            for i in 0..outputs[0].len() {
                let mut tmp = Vec::with_capacity(SF::PACK_SIZE);
                for x in outputs.iter() {
                    tmp.push(x[i]);
                }
                let output = SF::pack(&tmp);
                nxt[cu.output + i] +=
                    output * cu.coef.get_value_with_public_inputs_simd(public_inputs);
            }
        }
        for (sub_id, allocs) in seg.child_segs.iter() {
            let subc = &self.segments[*sub_id];
            for a in allocs.iter() {
                let inputs = a
                    .input_offset
                    .iter()
                    .zip(subc.num_inputs.iter())
                    .enumerate()
                    .map(|(l, (off, len))| &cur[l][off..off + len])
                    .collect::<Vec<_>>();
                self.apply_segment_with_public_inputs_simd(
                    subc,
                    &inputs,
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
            Coef::PublicInput(id) => write!(f, "PublicInput({id})"),
        }
    }
}

impl fmt::Display for CrossLayerInput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(layer={}, offset={})", self.layer, self.offset)
    }
}

impl fmt::Display for NormalInput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.offset)
    }
}

impl<C: Config, I: InputType> fmt::Display for Segment<C, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "input={:?} output={}", self.num_inputs, self.num_outputs)?;
        for (sub_id, allocs) in self.child_segs.iter() {
            writeln!(f, "apply circuit {sub_id} at:")?;
            for a in allocs.iter() {
                writeln!(
                    f,
                    "    input_offset={:?} output_offset={}",
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
                write!(f, "in{input}")?;
                if i < cu.inputs.len() - 1 {
                    write!(f, ",")?;
                }
            }
            writeln!(f, ") * {}", cu.coef)?;
        }
        Ok(())
    }
}

impl<C: Config, I: InputType> fmt::Display for Circuit<C, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, seg) in self.segments.iter().enumerate() {
            write!(f, "Circuit {i}: {seg}")?;
            writeln!(f, "================================")?;
        }
        writeln!(f, "Layers: {:?}", self.layer_ids)?;
        Ok(())
    }
}
