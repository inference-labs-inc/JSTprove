use crate::gnark::emparam::FieldParams;
use crate::gnark::limbs::decompose;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::any::Any;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct Element<T: FieldParams> {
    pub limbs: Vec<Variable>,
    pub overflow: u32,
    pub internal: bool,
    pub mod_reduced: bool,
    pub is_evaluated: bool,
    pub evaluation: Variable,
    pub _marker: std::marker::PhantomData<T>,
}

impl<T: FieldParams> Element<T> {
    pub fn new(
        limbs: Vec<Variable>,
        overflow: u32,
        internal: bool,
        mod_reduced: bool,
        is_evaluated: bool,
        evaluation: Variable,
    ) -> Self {
        Self {
            limbs,
            overflow,
            internal,
            mod_reduced,
            is_evaluated,
            evaluation,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.limbs.is_empty()
    }
}

impl<T: FieldParams> Clone for Element<T> {
    fn clone(&self) -> Self {
        let mut r = Element::new(Vec::new(), 0, false, false, false, Variable::default());
        r.limbs = self.limbs.clone();
        r.overflow = self.overflow;
        r.internal = self.internal;
        r.mod_reduced = self.mod_reduced;
        r
    }
}

impl<T: FieldParams> Default for Element<T> {
    fn default() -> Self {
        Element::new(Vec::new(), 0, false, false, false, Variable::default())
    }
}

pub fn value_of<C: Config, B: RootAPI<C>, T: FieldParams>(
    api: &mut B,
    constant: Box<dyn Any>,
) -> Element<T> {
    let r: Element<T> = new_const_element::<C, B, T>(api, constant);
    r
}

pub fn new_const_element<C: Config, B: RootAPI<C>, T: FieldParams>(
    api: &mut B,
    v: Box<dyn Any>,
) -> Element<T> {
    let fp = T::modulus();
    // convert to big.Int
    let mut b_value = from_interface(v);
    //if neg, add modulus
    if b_value < BigInt::from(0) {
        b_value += &fp;
    }
    // mod reduce
    if fp.cmp(&b_value) != Ordering::Equal {
        b_value %= fp;
    }

    // decompose into limbs
    let mut blimbs = vec![BigInt::default(); T::nb_limbs() as usize];
    let mut limbs = vec![Variable::default(); blimbs.len()];
    if let Err(err) = decompose(&b_value, T::bits_per_limb(), &mut blimbs) {
        panic!("decompose value: {err}");
    }
    // assign limb values
    for i in 0..limbs.len() {
        limbs[i] = api.constant(blimbs[i].to_u64().unwrap() as u32);
    }
    Element::new(limbs, 0, true, false, false, Variable::default())
}
pub fn new_internal_element<T: FieldParams>(limbs: Vec<Variable>, overflow: u32) -> Element<T> {
    Element::new(limbs, overflow, true, false, false, Variable::default())
}
pub fn copy<T: FieldParams>(e: &Element<T>) -> Element<T> {
    let mut r = Element::new(Vec::new(), 0, false, false, false, Variable::default());
    r.limbs = e.limbs.clone();
    r.overflow = e.overflow;
    r.internal = e.internal;
    r.mod_reduced = e.mod_reduced;
    r
}
pub fn from_interface(input: Box<dyn Any>) -> BigInt {
    let r;
    if let Some(v) = input.downcast_ref::<BigInt>() {
        r = v.clone();
    } else if let Some(v) = input.downcast_ref::<u8>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<u16>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<u32>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<u64>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<usize>() {
        r = BigInt::from(*v as u64);
    } else if let Some(v) = input.downcast_ref::<i8>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<i16>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<i32>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<i64>() {
        r = BigInt::from(*v);
    } else if let Some(v) = input.downcast_ref::<isize>() {
        r = BigInt::from(*v as i64);
    } else if let Some(v) = input.downcast_ref::<String>() {
        r = BigInt::parse_bytes(v.as_bytes(), 10).unwrap_or_else(|| {
            panic!("unable to set BigInt from string: {v}");
        });
    } else if let Some(v) = input.downcast_ref::<Vec<u8>>() {
        r = BigInt::from_bytes_be(num_bigint::Sign::Plus, v);
    } else {
        panic!("value to BigInt not supported");
    }
    r
}
